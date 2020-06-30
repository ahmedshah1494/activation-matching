import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from traitlets.config.loader import PyFileConfigLoader
import torchvision
from torch.utils.data import DataLoader, Subset
from models import ModelWrapper, ModelWrapper2
from distillation import StudentModelWrapper, StudentModelWrapper2
import argparse
import re
import utils
from advertorch_examples.utils import get_test_loader
import os

import sys
from pytorch_lightning import Trainer

from advertorch.attacks import LinfPGDAttack, L2PGDAttack, FGSM, CarliniWagnerL2Attack, JacobianSaliencyMapAttack, ChooseBestAttack
from advertorch.attacks.utils import attack_whole_dataset
from advertorch_examples.utils import get_cifar10_test_loader, get_test_loader
from advertorch.utils import get_accuracy
from advertorch.loss import CWLoss

class NullAdversary:
    def __init__(self,model,**kwargs):
        self.model = model
    def perturb(self,x,y):
        return x
    def predict(self, x):
        return self.model(x)

class NormalizationWrapper(nn.Module):
    def __init__(self, model):
        super(NormalizationWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        x = utils.normalize_image_tensor(x)
        return self.model(x)

class EnsembleWrapper(nn.Module):
    def __init__(self, models, concensus_pc=1.):
        super(EnsembleWrapper, self).__init__()
        self.models = models
        self.concensus_pc = concensus_pc
    
    def forward(self, x):
        preds = [nn.functional.gumbel_softmax(m(x), hard=True) for m in self.models]
        preds = torch.stack(preds, dim=0).sum(0)
        preds = torch.cat((preds, torch.zeros((preds.shape[0], 1), device=preds.device)), dim=1)

        for i,p in enumerate(preds):
            if int(p.max()) < len(self.models) * self.concensus_pc:
                preds[i] *= 0
                preds[i, -1] = 1
        return preds

class GaussianSmoothingWrapper(nn.Module):
    def __init__(self, model, sigma, concensus_pc=0.5):
        super(GaussianSmoothingWrapper, self).__init__()
        self.model = model
        self.sigma = sigma
        self.concensus_pc = concensus_pc
    def _forward(self, x):
        eps = torch.normal(mean=0, std=self.sigma,size=x.shape).to(x.device)
        x += eps
        return self.model(x)

    def forward(self, x, n_samples=1):
        eps = torch.normal(mean=0, std=self.sigma,size=(n_samples, *(x.shape))).to(x.device)
        x = x.unsqueeze(0) + eps
        x = x.view(n_samples*x.shape[1], *(x.shape[2:]))        
        preds = nn.functional.gumbel_softmax(self._forward(x), hard=True, dim=1)        
        preds = preds.view(n_samples, -1, *(preds.shape[1:])).sum(0)        
        preds = torch.cat((preds, torch.zeros((preds.shape[0], 1), device=preds.device)), dim=1)

        for i,p in enumerate(preds):
            if int(p.max()) < n_samples * self.concensus_pc:
                preds[i] *= 0
                preds[i, -1] = 1
        return preds

class Attacker:
    def __init__(self,source_model, dataloader, attack_class, *args, binary_classification=False, max_instances=-1, **kwargs):
        self.model = source_model
        self.model = self.model.cuda()
        self.adversary = attack_class(self.model, *args, **kwargs)
        self.loader = dataloader
        self.perturbed_dataset = []
        self.perturbed_dataset_length = 0
        self.max_instances=max_instances
        self.binary_classification = binary_classification
        self.targeted = False
    

    def generate_examples(self, force_attack = True):
        if (not self.perturbed_dataset) or force_attack:
            self.perturbed_dataset = []
            self.perturbed_dataset_length = min(max(self.max_instances,self.loader.batch_size),len(self.loader.dataset)) if self.max_instances>0 else len(self.loader.dataset)
            max_attacks = (self.perturbed_dataset_length+self.loader.batch_size-1)//self.loader.batch_size
            print("Generating %d adversarial examples"%self.perturbed_dataset_length)
            for i,(x,y) in enumerate(self.loader):                
                if self.binary_classification:
                    y = (y == 0).float().view(-1,1)
                advimg = self.adversary.perturb(x.cuda(),y.cuda())                
                self.perturbed_dataset.append((advimg,y))
                if i+1>=max_attacks:
                    break

    def eval(self, attacked_model = None, force_attack = False):
        self.generate_examples(force_attack = force_attack)
        confusion_matrix = None
        if not attacked_model:
            attacked_model = self.model
        attacked_model = attacked_model.cuda()
        
        correct = 0
        for x,y in self.perturbed_dataset:
            logits = attacked_model(x.cuda())
            
            if confusion_matrix is None:
                nclasses = logits.shape[1]
                confusion_matrix = np.zeros((nclasses, nclasses))
            
            pred = torch.argmax(logits, dim=1)
            correct += (pred.cpu() == y).sum()
            
            for t,p in zip(y, pred):
                t = t.int()
                confusion_matrix[t,p] += 1
        
        confusion_matrix /= np.sum(confusion_matrix, axis=1, keepdims=True)
        accuracy = correct.float()/self.perturbed_dataset_length
        
        return accuracy.item(), confusion_matrix

def extract_attack(args):
    if args.binary_classification:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(9), reduction='sum')
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
    if args.attack =="fgsm":
        attack_class = FGSM
        attack_kwargs = {"loss_fn":loss_fn,"eps":args.eps}
        print("Using FGSM attack with eps=%f"%args.eps)
    elif args.attack =="pgdinf":
        attack_class = LinfPGDAttack
        attack_kwargs = {"loss_fn":loss_fn,"eps":args.eps,"nb_iter":args.nb_iter,"eps_iter":args.eps_iter}
        print("Using PGD attack with %d iterations of step %f on Linf ball of radius=%f"%(args.nb_iter,args.eps_iter,args.eps))
    elif args.attack =="pgdl2":
        attack_class = L2PGDAttack
        attack_kwargs = {"loss_fn":loss_fn,"eps":args.eps,"nb_iter":args.nb_iter,"eps_iter":args.eps_iter}
        print("Using PGD attack with %d iterations of step %f on L2 ball of radius=%f"%(args.nb_iter,args.eps_iter,args.eps))
    elif args.attack =="cwl2":
        attack_class = CarliniWagnerL2Attack
        attack_kwargs = {"loss_fn":None,"num_classes":10,"confidence":args.conf,"max_iterations":args.max_iters,"learning_rate":args.lr}
        print("Using Carlini&Wagner attack with %d iterations of step %f and confidence %f"%(args.max_iters,args.lr,args.conf))
    elif args.attack =="jsma":
        attack_class = JacobianSaliencyMapAttack
        attack_kwargs = {"loss_fn":None,"num_classes":10,"theta":args.eps,"gamma":args.gamma,}
        print("Using JSMA attack with %d theta %f and gamma %f"%(args.max_iters,args.eps,args.gamma))
    else:
        print("No known attack specified : test set will be used")
        attack_class = NullAdversary
        attack_kwargs={}

    return attack_class,attack_kwargs

def whitebox_attack(model, args):
    outfile = args.model_path + 'advdata_%s_eps=%f_%drestarts.pt' % (args.attack, args.eps, args.nb_restarts)
    if os.path.exists(outfile):
        return
        
    print("Using a white box attack")       
    test_loader = get_test_loader(args.dataset, batch_size=args.batch_size)   
    print("Model configuration")
    
    attack_class,attack_kwargs = extract_attack(args)
    prefix = "%s-%f" % (args.attack, args.eps)
    # attacker = Attacker(model,test_loader, attack_class=attack_class, max_instances=args.max_instances, 
    #                     clip_min=0., clip_max=1., targeted=False, binary_classification=args.binary_classification, 
    #                     **attack_kwargs)
    # accuracy, confusion_matrix = attacker.eval()
    # print("Accuracy under attack : %f"%accuracy)
    # print('Confusion Matrix:')    
    # print(np.diag(confusion_matrix))

    attackers = [attack_class(model, **attack_kwargs) for i in range(args.nb_restarts)]
    if len(attackers) > 1:
        attacker = ChooseBestAttack(model, attackers, targeted=attackers[0].targeted)
    else:
        attacker = attackers[0]
    adv, label, pred, advpred = attack_whole_dataset(attacker, test_loader)
    print(prefix, 'clean accuracy:',get_accuracy(pred, label))
    print(prefix, 'robust accuracy:',get_accuracy(advpred, label))
    detection_TPR = (advpred == label.max() + 1).float().mean()
    detection_FPR = (pred == label.max() + 1).float().mean()
    print(prefix, 'attack success rate:', 1 - ((advpred == label) | (advpred == label.max() + 1)).float().mean())
    print(prefix, 'attack detection TPR:', detection_TPR)
    print(prefix, 'attack detection FPR:', detection_FPR)

    outfile = args.model_path + 'advdata_%s_eps=%f_%drestarts.pt' % (args.attack, args.eps, args.nb_restarts)
    torch.save({
        'args': dict(vars(args)),
        'data': adv,
        'preds': advpred,
        'clean_preds': pred,
        'labels': label
    }, outfile)


def transfer_attack(model, args):    
    # args.dataset must be path to a that file loadable by torch.load and that contains a dictionary:
    # {    
    #   data: (adversarially perturbed) data samples,
    #   preds: the predictions of the source model on the data
    #   labels: the true labels of the data
    # }
    print('Running transfer attack...')
    print('source:', args.dataset)
    print('target:', args.model_path)

    source_data = torch.load(args.dataset)
    loader = DataLoader(source_data['data'], batch_size=args.batch_size, shuffle=False)

    preds = []
    for x_adv in loader:
        x_adv = x_adv.cuda()        
        logits = model(x_adv)
        preds.append(logits.argmax(1))
    preds = torch.cat(preds)

    print('accuracy:',get_accuracy(preds, source_data['labels']))
    print('agreement:',get_accuracy(preds, source_data['preds']))

    outfile = "logs/transfer_attack_outputs/%s/%s.pt" % (os.path.basename(args.model_path).split('.')[0], os.path.basename(args.dataset))
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    torch.save({
        'sourc_attack_args': source_data['args'],
        'source_adv_data': source_data['data'],
        'source_preds': source_data['preds'],
        'target_preds': preds,
        'labels': source_data['labels']
    }, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('model_path', type=str)
    parser.add_argument('--target_model_path', type=str)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--datafolder', type=str, default='/home/mshah1/workhorse3/')
    parser.add_argument('--attack', type=str, default="none")
    parser.add_argument('--max_instances', type=int, default=-1)
    parser.add_argument('--nb_iter', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nb_restarts', type=int, default=1)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--eps_iter', type=float, default=0.01)
    parser.add_argument('--conf', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--max_iters', type=float, default=100)
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--transfer_attack', action='store_true')
    parser.add_argument('--use_gs_wrapper', action='store_true')
    parser.add_argument('--gs_sigma', type=float, default=0.12)
    parser.add_argument('--no_normalize', action='store_true')

    args = parser.parse_args()
    
    model = torch.load(args.model_path)    
    if not args.no_normalize:
        model = NormalizationWrapper(model)
    if args.use_gs_wrapper:
        model = GaussianSmoothingWrapper(model, args.gs_sigma)
    model.eval()

    if args.transfer_attack:
        transfer_attack(model, args)
    else:
        args.dataset = args.dataset.upper()
        whitebox_attack(model, args)
