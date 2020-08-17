import torch
from torch import nn
import numpy as np
from trainer import Trainer, compute_accuracy
from attack_classifier import extract_attack
from models import LayeredModel, VGG16, WideResnet, GaussianNoiseLayer

def reshape2D(x):
    return x.view(x.shape[0], -1)

def normalize_vector(x, dim):
    return x / torch.norm(x, p=2, dim=dim, keepdim=True)

def compute_diff_spread(z, zadv):
    _z_diff = z-zadv
                
    z = reshape2D(z)
    pw_dist = (z.unsqueeze(1) - z.unsqueeze(0)).view(-1, z.shape[1])
    pw_dist_std = torch.std(pw_dist, dim=0).detach()

    _z_diff = reshape2D(_z_diff)
    _z_diff = _z_diff/pw_dist_std

    return _z_diff

def compute_cosine_metric(z, zadv, dim=1):
    return 1 - nn.functional.cosine_similarity(z, zadv, dim=dim)

def compute_cosine_spread(z, zadv, label_mask):
    z = reshape2D(z)
    zadv = reshape2D(zadv)

    normed_z = normalize_vector(z, 1)
    normed_zadv = normalize_vector(zadv, 1)
    
    zz_cos = normed_z.mm(normed_z.transpose(0,1)) * label_mask                
    zz_cos[label_mask == -1] += 1
    zz_cos[label_mask == 1] = torch.abs(zz_cos[label_mask == 1])                
    zz_cos = (zz_cos).mean(1, keepdim=True)
    
    zzadv_cos = compute_cosine_metric(z, zadv, 1).unsqueeze(1)

    return zz_cos, zzadv_cos

class MoM(object):
    def __init__(self):
        super().__init__()
        self.c = None
        self. L = None

    def compute_h_term(self, h):
        h_norm = h.pow(2).sum(1)
        if isinstance(self.L, int):
                self.L = torch.zeros((h.shape[1],1), device=h.device) + self.L
        h_term = h.mm(self.L).squeeze() + (self.c * (h_norm))/2
        return h_term.mean()

class AdamMoM(torch.optim.Adam):
    def __init__(self, model, init_c, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        super().__init__(model.parameters(), lr, betas, eps, weight_decay, amsgrad)
        self.c = init_c        
        self.L = 0       

class SGDMoM(torch.optim.SGD):
    def __init__(self, model, init_c, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(model.parameters(), lr, momentum, dampening, weight_decay, nesterov)
        self.c = init_c
        self.L = 0

def logit_cross_entropy(logits, target_logits, T=1):
    p = torch.softmax(target_logits/T, dim=1)
    q = torch.log_softmax(logits/T, dim=1)
    return nn.KLDivLoss(size_average=False)(q, p)

class ActivationInvarianceTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args):
        super().__init__(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args)
        
        attack_class, kwargs = extract_attack(args)
        self.test_adversary = attack_class(self.model, **kwargs)
        if self.args.maximize_logit_divergence:
            loss_fn = {
                'KL': logit_cross_entropy,
                'cosine': lambda x,y: compute_cosine_metric(x, y, 1).sum()
            }[self.args.adv_logit_loss]
            attack_class, kwargs = extract_attack(args, loss_fn=loss_fn)
            self.training_adversary = attack_class(self.model, **kwargs)
        else:
            self.training_adversary = self.test_adversary

    def compute_z_criterion(self, Z, Zadv, y):
        y_ = (y + 1).unsqueeze(1).float()
        label_mask = ((y_.mm(1/y_.transpose(0,1))) != 1).float()
        label_mask[label_mask == 0] = -1

        zzadv_diff = []
        zz_diff = []
        for i, (z, zadv) in enumerate(zip(Z, Zadv)):           
            if self.args.z_criterion == 'diff':
                _z_diff = reshape2D(z-zadv)
                zzadv_diff.append(_z_diff)
            elif self.args.z_criterion == 'diff-spread':
                _z_diff = compute_diff_spread(z, zadv)
                zzadv_diff.append(_z_diff)
            elif self.args.z_criterion == 'cosine':
                z = reshape2D(z)
                zadv = reshape2D(zadv)
                _z_diff = compute_cosine_metric(z, zadv, 1)
                zzadv_diff.append(_z_diff.view(z.shape[0], -1))
            elif self.args.z_criterion == 'cosine-spread':
                zz_cos, zzadv_cos = compute_cosine_spread(z, zadv, label_mask)
                zz_diff.append(zz_cos)
                zzadv_diff.append(zzadv_cos)
        if len(zz_diff) > 0:
            zz_diff = torch.cat(zz_diff, dim=1)
        zzadv_diff = torch.cat(zzadv_diff, dim=1)
        return zzadv_diff, zz_diff

    def compute_adversarial(self, x, y, logits):
        model = self.model
        model.requires_grad_(False)
        if model.training:            
            model = model.train(False)
            if self.args.maximize_logit_divergence:                    
                xadv = self.training_adversary.perturb(x, logits)
            else:
                xadv = self.training_adversary.perturb(x, y)    
            model = model.train(True)            
        else:
            xadv = self.test_adversary.perturb(x, y)
        model.requires_grad_(True)
        return xadv
    
    def compute_outputs(self, x, y, xadv=None):
        model = self.model
        logits, interm_Z = model(x, store_intermediate=True)
        xadv = self.compute_adversarial(x, y, logits)
        adv_logits, interm_Zadv = model(xadv, store_intermediate=True)
        zzadv_diff, zz_diff = self.compute_z_criterion(interm_Z, interm_Zadv, y)
        return xadv, logits, adv_logits, zzadv_diff, zz_diff

    def MoM_update(self, x, y, xadv, zzadv_diff, zz_diff, loss):
        zzadv_diff_norm = zzadv_diff.pow(2).sum(1)
        zz_diff_norm = zz_diff.pow(2).sum(1)
        z_diff_norm = zzadv_diff_norm + zz_diff_norm

        if len(zz_diff) > 0:
            z_diff = torch.cat((zzadv_diff, zz_diff), dim=1)
        else:
            z_diff = zzadv_diff
        
        if isinstance(self.optimizer.L, int):
                self.optimizer.L = torch.zeros((z_diff.shape[1],1), device=z_diff.device) + self.optimizer.L
        z_term = z_diff.mm(self.optimizer.L).squeeze() + (self.optimizer.c * (z_diff_norm))/2
        loss += z_term.mean()

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        with torch.no_grad():
            _, _, _, z_diff, zz_diff = self.compute_outputs(x, y, xadv)
            if len(zz_diff) > 0:
                z_diff = torch.cat((z_diff, zz_diff), dim=1)
            self.optimizer.L += self.optimizer.c * z_diff.mean(0).unsqueeze(1)
        self.optimizer.c *= self.args.c_step_size
        loss = loss.detach()
        return loss, zzadv_diff_norm, zz_diff_norm

    def compute_loss(self, logits, adv_logits, y, z_diff, zz_diff):
        if self.model.training and self.args.adv_ratio < 1:
            selected_adv = np.random.binomial(1, p=self.args.adv_ratio, size=len(logits)).astype(bool)
            while selected_adv.all() or (not selected_adv.any()):
                selected_adv = np.random.binomial(1, p=self.args.adv_ratio, size=len(logits)).astype(bool)
            selected_cln = (~ selected_adv)
        else:
            selected_adv = selected_cln = np.ones((len(logits,))).astype(bool)
        
        loss = torch.nn.functional.cross_entropy(logits[selected_cln], y[selected_cln])
        if self.args.adv_loss_wt > 0:
            if self.args.maximize_logit_divergence:
                adv_classification_loss = self.training_adversary.loss_fn(adv_logits[selected_adv], logits[selected_adv])
                if self.training_adversary.loss_fn == logit_cross_entropy:
                    adv_classification_loss /= len(adv_logits[selected_adv])
            else:
                adv_classification_loss = self.training_adversary.loss_fn(adv_logits[selected_adv], y[selected_adv])
        else:
            adv_classification_loss = 0
        loss += self.args.adv_loss_wt * adv_classification_loss
        
        if self.args.z_criterion == 'cosine-spread':                
            zz_diff_norm = zz_diff.sum(1)
            z_diff_norm = z_diff.sum(1)
            z_term = self.args.z_wt*z_diff_norm + self.args.zz_wt*zz_diff_norm
        else:
            z_diff_norm = z_diff.pow(2).sum(1)
            zz_diff_norm = 0
            z_term = self.args.z_wt*(z_diff_norm)

        if (not self.args.use_MoM) and self.args.z_wt > 0:
            loss += z_term.mean()
        return loss, z_diff_norm, zz_diff_norm

    def train_step(self, batch, batch_idx):
        x,y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        xadv, logits, adv_logits, zzadv_diff, zz_diff = self.compute_outputs(x, y)

        cln_acc, _ = compute_accuracy(logits, y)
        adv_acc, _ = compute_accuracy(adv_logits, y)

        loss, zzadv_diff_norm, zz_diff_norm = self.compute_loss(logits, adv_logits, y, zzadv_diff, zz_diff)

        if self.model.training and self.args.use_MoM:
            loss, zzadv_diff_norm, zz_diff_norm = self.MoM_update(x, y, xadv, zzadv_diff, zz_diff, loss)
        
        if self.model.training:
            for p in self.model.parameters():
                assert (p.grad is None) or (p.grad == 0).all()
        return {'loss':loss}, {'train_clean_accuracy': cln_acc,
                             'train_adv_accuracy': adv_acc,
                             'train_accuracy': (cln_acc + adv_acc) / 2,
                             'train_loss': float(loss.detach().cpu()),
                             'train_Z_loss': float(zzadv_diff_norm.max().detach().cpu()),
                             'train_ZZ_loss': float(zz_diff_norm.max().detach().cpu()) if len(zz_diff) > 0 else 0}
    
    def _optimization_wrapper(self, func):        
        if self.args.use_MoM:
            return func
        else:
            def wrapper(*args, **kwargs):
                self.optimizer.zero_grad()
                output, logs = func(*args, **kwargs)                
                output['loss'].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                return output, logs
            return wrapper