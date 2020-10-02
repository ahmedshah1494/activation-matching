import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
from trainer import Trainer, compute_accuracy
from activation_invariance_trainer import ActivationInvarianceTrainer, AdamMoM, SGDMoM
from attack_classifier import extract_attack
from models import LayeredModel, VGG16, WideResnet, GaussianNoiseLayer
from copy import deepcopy
import argparse
from utils import get_cifar10_dataset
import os
from trades import trades_loss
  

class TRADESTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args):
        super().__init__(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args)
        attack_class, kwargs = extract_attack(args)
        self.adversary = attack_class(self.model, **kwargs)

    def compute_adversarial(self, x, y):
        model = self.model
        model.requires_grad_(False)
        if model.training:            
            model = model.train(False)
            xadv = self.adversary.perturb(x, y)    
            model = model.train(True)            
        else:
            xadv = self.adversary.perturb(x, y)
        model.requires_grad_(True)
        return xadv

    def train_step(self, batch, batch_idx):
        x,y = batch
        x = x.to(self.device)
        y = y.to(self.device)
                
        if self.model.training:
            loss, logits, adv_logits = trades_loss(self.model, x, y, self.optimizer, self.args.eps_iter, self.args.eps, self.args.nb_iter, self.args.adv_loss_wt)
        else:
            xadv = self.compute_adversarial(x, y)
            logits = self.model(x)
            adv_logits = self.model(xadv)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss += torch.nn.functional.cross_entropy(adv_logits, y)
        
        cln_acc, _ = compute_accuracy(logits, y)
        adv_acc, _ = compute_accuracy(adv_logits, y)
        if self.model.training:
            for p in self.model.parameters():
                assert (p.grad is None) or (p.grad == 0).all()
        return {'loss': loss}, {'train_clean_accuracy': cln_acc,
                             'train_adv_accuracy': adv_acc,
                             'train_accuracy': (cln_acc + adv_acc) / 2,
                             'train_loss': float(loss.detach().cpu()),}

def add_gaussian_smoothing_layers(model:LayeredModel):
    for i,l in enumerate(model.layers):
        model.layers[i] = nn.Sequential(
            GaussianNoiseLayer(),
            l
        )
def train(args):
    train_dataset, val_dataset, test_dataset, num_classes = get_cifar10_dataset(args.datafolder)
    # train_dataset = [train_dataset[i] for i in range(10000)]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_dict = {
        'vgg16': lambda : VGG16(args, num_classes),
        'wide_resnet': lambda : WideResnet(args, 28, 4, 0, num_classes),
        'wide_resnet-10': lambda : WideResnet(args, 28, 10, 0, num_classes)
    }
    model = model_dict[args.model_name]().to(device)
    # print(model)
    if len(args.layer_idxs) == 0:
        model.args.layer_idxs = list(range(len(model.layers) - 1))
        if args.include_logits:
            model.args.layer_idxs.append(len(model.layers)-1)
    model.args.layer_idxs = torch.tensor(model.args.layer_idxs)
    print(model.args.layer_idxs)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9, nesterov=True)
    elif args.optimizer == 'MoM-Adam':
        optimizer = AdamMoM(model, args.init_c, lr=args.lr, weight_decay=0.0005)        
    elif args.optimizer == 'MoM-SGD':
        optimizer = SGDMoM(model, args.init_c, lr=args.lr, weight_decay=0.0005, momentum=0.9, nesterov=True)
    args.use_MoM = 'MoM' in args.optimizer

    args.logdir = os.path.join(args.logdir, "%s_%s" % (model.name, args.exp_name))
    exp_num = 1 + max([int(x.split('_')[1]) for x in os.listdir(args.logdir)]) if (os.path.exists(args.logdir) and len(os.listdir(args.logdir)) > 0) else 0
    args.logdir = os.path.join(args.logdir, 'exp_%d' % exp_num)
    print('logging to ', args.logdir)

    if args.attack == 'none':
        trainer_class = Trainer
        mode = 'max'
        metric = 'val_accuracy'
    elif args.TRADES:
        trainer_class = TRADESTrainer
        mode = 'max'
        metric = 'val_accuracy'
    else:
        trainer_class = ActivationInvarianceTrainer
        mode = 'max'
        metric = 'val_accuracy'
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=args.patience, verbose=True, factor=args.decay_factor)

    trainer = trainer_class(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args)
    trainer.track_metric(metric, mode)
    trainer.train()
    trainer.test()
    trainer.logger.flush()
    trainer.logger.close()

def test(args):
    _, _, test_dataset, nclasses = get_cifar10_dataset(args.datafolder)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(args.model_path)
    model = model.to(device)

    args.logdir = os.path.join(*([x for x in os.path.dirname(args.model_path).split('/') if x != 'checkpoints']))
    print(args.logdir)
    args.use_MoM = 'MoM' in args.optimizer

    trainer = ActivationInvarianceTrainer(model, None, None, test_loader, None, None, device, args)
    trainer.track_metric('train_accuracy', 'max')
    trainer.test()
    trainer.logger.flush()
    trainer.logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--datafolder', default='/home/mshah1/workhorse3')
    
    parser.add_argument('--model_name')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--normalize_input', action='store_true')

    parser.add_argument('--logdir', default='logs/activation_invariance/')
    parser.add_argument('--exp_name', default='training')
    
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--loss_type', type=str, default='xent')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--decay_factor', type=float, default=0.5)
    parser.add_argument('--init_c', type=float, default=1)
    parser.add_argument('--c_step_size', type=float, default=1.0005)
    
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--attack', type=str, default="none", choices=('none', 'pgdinf', 'pgdl2', 'gs'))
    parser.add_argument('--max_instances', type=int, default=-1)
    parser.add_argument('--nb_iter', type=int, default=7)
    parser.add_argument('--nb_restarts', type=int, default=1)
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--eps_iter', type=float, default=8/(255*4))
    parser.add_argument('--max_iters', type=int, default=7)
    parser.add_argument('--std', type=float, default=0.25)
    parser.add_argument('--maximize_logit_divergence', action='store_true')
    parser.add_argument('--T', type=float, default=1.)
    parser.add_argument('--adv_loss_fn', type=str, choices=('xent', 'z_cos'), default='xent')
    parser.add_argument('--accumulate_adv_grad_prob', type=float, default=0.)

    parser.add_argument('--z_wt', type=float, default=1e-3)
    parser.add_argument('--zz_wt', type=float, default=1e-3)
    parser.add_argument('--adv_loss_wt', type=float, default=1)
    parser.add_argument('--cln_loss_wt', type=float, default=1)
    parser.add_argument('--adv_ratio', type=float, default=1.)
    parser.add_argument('--detach_adv_logits', action='store_true')
    parser.add_argument('--layer_idxs', type=int, nargs='+', default=[])
    parser.add_argument('--include_logits', action='store_true')
    parser.add_argument('--z_criterion', type=str, default='diff', choices=('diff', 'diff-spread', 'cosine', 'cosine-spread'))
    parser.add_argument('--use_preactivation', action='store_true')

    parser.add_argument('--match_logits', action='store_true')
    parser.add_argument('--logit_matching_fn', type=str, choices=('KL', 'cosine', 'L2'), default='KL')
    parser.add_argument('--logit_loss_wt', type=float, default=1)
    parser.add_argument('--normalize_activations', action='store_true')

    parser.add_argument('--layer_weighting', type=str, choices=('const', 'linear', 'exp'), default='const')
    parser.add_argument('--min_layer_wt', type=float, default=0.1)
    parser.add_argument('--max_layer_wt', type=float, default=1.)

    parser.add_argument('--random_seed', type=int, default=9999)

    parser.add_argument('--TRADES', action='store_true')

    parser.add_argument('--debug', action='store_true')

    np.random.seed(9999)
    torch.random.manual_seed(9999)
    torch.cuda.manual_seed(9999)

    args = parser.parse_args()
    args.layer_idxs = torch.tensor(args.layer_idxs)
    if args.test:
        test(args)
    else:
        train(args)