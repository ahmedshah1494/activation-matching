import torch
from torch import nn
from anpgd import ANPGD, ANPGDForTest, bisection_search
from wide_resnet import Wide_ResNet
from advertorch_examples.utils import get_test_loader
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.utils import clamp, batch_multiply
from advertorch.loss import elementwise_margin, CWLoss
from attack_classifier import NormalizationWrapper, extract_attack
from advertorch.attacks.utils import attack_whole_dataset
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from pprint import pprint
from tabulate import tabulate
import utils
import numpy as np
import models
import argparse
import matplotlib.pyplot as plt
import pickle

class AdversaryWrapper(object):
    def __init__(self, model):
        super(AdversaryWrapper).__init__()
        self.predict = model

class ANPGDWrapper(ANPGDForTest):
    def __init__(self, pgdadv, maxeps, num_search_steps):
        super(ANPGDWrapper, self).__init__(pgdadv, maxeps, num_search_steps)
    
    def _get_unitptb_and_eps(self, xadv, x, y, prev_eps):
        unitptb = batch_multiply(1. / (prev_eps + 1e-12), (xadv - x))
        adv_logits = self.predict(xadv)
        logit_margin = elementwise_margin(adv_logits, y)        
        ones = torch.ones_like(y).float()
        # maxeps = self.maxeps * ones        
        maxeps = torch.norm((xadv-x).view(x.shape[0],-1), p=self.pgdadv.ord, dim=1)
        
        adv_pred = adv_logits.argmax(1)
        # print(1 - (adv_pred == y).float().mean())
        # print(maxeps.min(), maxeps.max())
        pred = adv_pred.clone()
        i=0
        # print(i, self.pgdadv.eps, float((adv_pred == pred).float().mean()), float((pred == self.target_y).float().mean()), float(maxeps.min()), float(maxeps.max()))
        while i < 10:            
            if self.pgdadv.targeted:
                unsuccessful_adv_idx = (adv_pred != self.target_y) & (pred != self.target_y)
                if not unsuccessful_adv_idx.any():
                    break
            else:
                unsuccessful_adv_idx = (adv_pred == y) & (pred == y)
            maxeps[unsuccessful_adv_idx] *= 1.5
            maxeps_ = maxeps[unsuccessful_adv_idx]
            unitptb_ = unitptb[unsuccessful_adv_idx]
            x_ = x[unsuccessful_adv_idx]            
                        
            x_ = clamp(x_ + batch_multiply(maxeps_, unitptb_),
                        min=0., max=1.)
            pred[unsuccessful_adv_idx] = self.predict(x_).argmax(1)            
            i += 1
            # print(i, self.pgdadv.eps, float((adv_pred == pred).float().mean()), float((pred == self.target_y).float().mean()), float(maxeps.min()), float(maxeps.max()))
        # print(logit_margin)
        curr_eps = bisection_search(
            maxeps * 0.5, unitptb, self.predict, x, y, elementwise_margin,
            logit_margin, maxeps, self.num_search_steps)
        if self.pgdadv.targeted:
            curr_eps[pred != self.target_y] = np.inf
        return unitptb, curr_eps

    def perturb(self, x, y, target_y=None):
        with ctx_noparamgrad_and_eval(self.predict):
            if self.pgdadv.targeted:
                self.target_y = target_y
                xadv = self.pgdadv.perturb(x, target_y)
                adv_pred = self.pgdadv.predict(xadv).argmax(1)
                # print((adv_pred == target_y).float().mean())
            else:
                xadv = self.pgdadv.perturb(x, y)        
        # print(self.pgdadv.eps, x.shape, xadv.shape, torch.norm((x-xadv).view(x.shape[0],-1), p=float('inf'), dim=1).mean())
        unitptb, curr_eps = self._get_unitptb_and_eps(
            xadv, x, y, self.pgdadv.eps)
        xadv = clamp(x + batch_multiply(curr_eps, unitptb),
                        min=self.pgdadv.clip_min, max=self.pgdadv.clip_max)
        # print('')
        return xadv

def get_layers(model, normalize_input):
    layers = []
    if normalize_input:
        layers.append(models.Normalize())
    for m in model.modules():
        if len(list(m.modules())) == 1:
            layers.append(m)
    for i,l in enumerate(layers):
        if isinstance(l, nn.AdaptiveAvgPool2d) and i < len(layers)-1 and not isinstance(layers[i+1], models.Flatten):
            layers.insert(i+1, models.Flatten())
    layers = nn.Sequential(*layers)
    return layers

def margin_loss(x,y):
    loss = elementwise_margin(x,y)
    return (loss * (loss < 0).float()).mean()

def uniform_distribution_loss(x, y):
    loss = -torch.log_softmax(x, dim=1).sum(1).mean()    
    return loss

def main(args):
    model = torch.load(args.model, map_location=torch.device('cpu'))
    if args.layer_idx is not None and not hasattr(model, 'layers'):
        model = get_layers(model, args.normalize_input)
        fe_model = model[:args.layer_idx+1].to(args.device).eval()
        model = model[args.layer_idx+1:]
        print(fe_model)
        print(model)
    elif args.normalize_input:
        model = NormalizationWrapper(model)

    args.attack = args.denoising_attack    
    adversaries = []
    eps = args.eps
    eps_iter = args.eps_iter
    for e in eps:            
        args.eps = e
        if eps_iter is None:
            args.eps_iter = e / 10
        else:
            args.eps_iter = eps_iter
        attack_class,attack_kwargs = extract_attack(args)
        if args.uniform_anpgd_target:
            attack_kwargs['loss_fn'] = uniform_distribution_loss
        else:
            attack_kwargs['loss_fn'] = torch.nn.CrossEntropyLoss(reduction='sum')
        if args.layer_idx is not None:
            attack_kwargs['clip_min'] = -np.inf
            attack_kwargs['clip_max'] = np.inf
        adversaries.append(attack_class(model, targeted=args.target_closest_boundary, **attack_kwargs))    

    max_eps = 0.08
    adversaries = [ANPGDWrapper(attack, max_eps, 20) for attack in adversaries]
    args.eps = eps
    adversaries = adversaries*args.nb_adversaries

    # print(vars(adversaries[0]))    
    model = AdversarialDenoisingWrapper(model, 
                                        adversaries, 
                                        args.consensus_pc,
                                        args.target_closest_boundary)
    model = model.to(args.device)
    model = model.eval()
    
    data = [torch.load(df) for df in args.data_file]
    adv_data = [d['data'] for d in data]    
    adv_data = torch.stack(adv_data, dim=1).to(torch.device('cpu'))
    adv_preds = [d['preds'] for d in data]    
    adv_preds = torch.stack(adv_preds, dim=1).to(torch.device('cpu'))
    cln_preds = [d['clean_preds'] for d in data]
    cln_preds = torch.stack(cln_preds, dim=1).to(torch.device('cpu'))
    labels = [d['labels'] for d in data]    
    labels = torch.stack(labels, dim=1).to(torch.device('cpu'))    
    adv_loader = torch.utils.data.DataLoader(list(zip(adv_data, adv_preds, cln_preds, labels)), batch_size=args.batch_size, shuffle=False)
    clean_loader = get_test_loader(args.dataset.upper(), batch_size=args.batch_size)
    
    # adv, label, pred, advpred = attack_whole_dataset(attack_class(model.model, **attack_kwargs), clean_loader)
    # print('clean accuracy:',(pred == label).float().mean())
    # print('robust accuracy:',(advpred == label).float().mean())
    
    margins = []
    adv_logits = []
    cln_logits = []
    adv_deltas = []
    cln_deltas = []
    adv_preds = []
    cln_preds = []
    norm_inf_delta = []
    norm_2_delta = []
    labels = []
    
    t = tqdm(zip(adv_loader, clean_loader))
    for (xadv, advpred, clnpred, L), (x, y) in t:        
        if not (L == y.view(-1,1)).all():
            print('label mismatch')
            print(L)
            print(y)
            exit(0)
        if ((clnpred == y.view(-1,1)) & (advpred != y.view(-1,1))).all(1).any():
            idx_mask = ((clnpred == y.view(-1,1)) & (advpred != y.view(-1,1))).all(1)           
            xadv = xadv[idx_mask]
            x = x[idx_mask]
            y = y[idx_mask]
        else:
            print('skipping instance...')
            continue
        xadv = xadv.to(args.device)
        x = x.to(args.device)
        y = y.to(args.device)

        xadv_shape = xadv.shape
        xadv = xadv.view(xadv.shape[0]*xadv.shape[1], *(xadv.shape[2:]))
        if args.layer_idx is not None:
            x = fe_model(x)
            xadv = fe_model(xadv)

        adv_votes, preds = model(xadv) 
        adv_logit, adv_pred = torch.max(adv_votes, dim=1)
        adv_logit = adv_logit.view(xadv_shape[0],xadv_shape[1], *(adv_logit.shape[1:]))
        adv_pred = adv_pred.view(xadv_shape[0],xadv_shape[1], *(adv_pred.shape[1:]))

        d = np.stack(model.deltas, axis=1)
        d = d.reshape(xadv_shape[0],xadv_shape[1], *(d.shape[1:]))
        adv_deltas.append(d)
        
        adv_votes, preds = model(x)
        cln_logit, cln_pred = torch.max(adv_votes, dim=1)
        cln_deltas.append(np.stack(model.deltas, axis=1))
        adv_logits.append(adv_logit.detach().cpu().numpy())
        cln_logits.append(cln_logit.detach().cpu().numpy())
        adv_preds.append(adv_pred.detach().cpu().numpy())
        cln_preds.append(cln_pred.detach().cpu().numpy())
        labels.append(y.cpu().numpy())

        if len(labels) >= 2500/args.batch_size:
            break

    adv_preds = np.concatenate(adv_preds, axis=0)
    cln_preds = np.concatenate(cln_preds, axis=0)
    adv_deltas = np.concatenate(adv_deltas, axis=0)
    cln_deltas = np.concatenate(cln_deltas, axis=0)
    labels = np.concatenate(labels, axis=0)
    adv_logits = np.concatenate(adv_logits, axis=0)
    cln_logits = np.concatenate(cln_logits, axis=0)  

    outfile = '%s_advDenoising_%d-%s-eps=%s_consesus-pc=%.1f' % (args.model, len(adversaries), args.attack, '-'.join(["%.4f" % e for e in args.eps]), args.consensus_pc)
    if args.target_closest_boundary:
        outfile += '_targeted'
    if args.uniform_anpgd_target:
        outfile += '_uniformTargets'
    if args.layer_idx is not None:
        outfile += '_layer-%d' % args.layer_idx
    outfile += '_data.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump([adv_preds,
                        cln_preds,
                        adv_deltas,
                        cln_deltas,
                        labels,
                        adv_logits,
                        cln_logits], f)

    print(adv_preds.shape, adv_logits.shape, adv_deltas.shape, cln_deltas.shape)
    exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--data_file', nargs='+')
    parser.add_argument('--consensus_pc', default=1., type=float)
    parser.add_argument('--nb_adversaries', type=int, default=1)
    parser.add_argument('--denoising_attack', type=str, choices=('pgdinf', 'pgdl2'))
    parser.add_argument('--nb_iter', type=int, default=10)
    parser.add_argument('--eps', type=float, nargs='+')
    parser.add_argument('--eps_iter', type=float)    
    parser.add_argument('--dataset', default='cifar10')    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--normalize_input', action='store_true')
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--target_closest_boundary', action='store_true')
    parser.add_argument('--uniform_anpgd_target', action='store_true')
    parser.add_argument('--layer_idx', type=int)
    args = parser.parse_args()

    np.random.seed(9999)
    torch.random.manual_seed(9999)
    torch.cuda.manual_seed(9999)

    if args.cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    main(args)
