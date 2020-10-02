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

def compute_diff_spread(z, zadv, label_mask):
    _z_diff = reshape2D(z-zadv).pow(2).sum(1)
                
    z = reshape2D(z)
    batch_size = z.shape[0]

    rand_idx = np.random.choice(np.arange(z.shape[0]), 32, replace=False)
    _z = z[rand_idx]
    label_mask = label_mask[:, rand_idx]    

    pw_dist = (z.unsqueeze(1) - _z.unsqueeze(0)).pow(2).sum(2) * label_mask
    # print(z.shape, _z.shape, label_mask.shape, pw_dist.shape)
    # triu_idx = torch.triu_indices(batch_size, batch_size, 1)
    # pw_dist = pw_dist[triu_idx[0], triu_idx[1]]
    pw_dist_mean = pw_dist.mean(1)    
    return _z_diff.view(-1,1), pw_dist_mean.view(-1, 1)

def compute_cosine_metric(z, zadv, dim=1):
    return 1 - nn.functional.cosine_similarity(z, zadv, dim=dim)

def compute_cosine_spread(z, zadv, label_mask, do_abs=True):
    z = reshape2D(z)
    zadv = reshape2D(zadv)

    normed_z = normalize_vector(z, 1)
    normed_zadv = normalize_vector(zadv, 1)
    
    zz_cos = normed_z.mm(normed_z.transpose(0,1)) * label_mask                
    zz_cos[label_mask == -1] += 1
    # if do_abs:
    zz_cos[label_mask == 1] = 1 + zz_cos[label_mask == 1]
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

class ActivationExtractorWrapper(nn.Module):
    def __init__(self, model:LayeredModel):
        super(ActivationExtractorWrapper, self).__init__()
        self.model = model        
    def forward(self, x):
        _, Z = self.model(x, store_intermediate=True)
        Z = torch.cat([reshape2D(z) for z in Z], dim=1)
        return Z

class ActivationInvarianceTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args):
        super().__init__(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args)
        
        attack_class, kwargs = extract_attack(args)
        self.test_adversary = attack_class(self.model, **kwargs)
        if self.args.maximize_logit_divergence:
            loss_fn = lambda x, y: logit_cross_entropy(x, y)
            adv_model = model            
        elif self.args.adv_loss_fn == 'z_cos':
            def z_cos(Zadv, Z):
                cos = torch.bmm(Zadv.unsqueeze(1), Z.unsqueeze(2)).squeeze()                
                return (len(args.layer_idxs) - cos).sum(0)
            loss_fn = z_cos
            adv_model = ActivationExtractorWrapper(model)
        elif self.args.adv_loss_fn == 'z_diff':
            def z_diff(Zadv, Z):
                diff = (Zadv - Z).pow(2).sum()
                return diff
            loss_fn = z_diff
            adv_model = ActivationExtractorWrapper(model)
        elif self.args.adv_loss_fn == 'xent':
            loss_fn = nn.CrossEntropyLoss(reduction='sum')
            adv_model = model
        else:
            raise NotImplementedError
        self.logit_matching_fn = {
            'KL': lambda x, y: logit_cross_entropy(x, y, args.T)/len(x),
            'L2': lambda x, y: (x-y).pow(2).mean(),
            'cosine':lambda x,y: compute_cosine_metric(x, y).mean()
        }[self.args.logit_matching_fn]
        attack_class, kwargs = extract_attack(args, loss_fn=loss_fn)
        self.training_adversary = attack_class(adv_model, **kwargs)

    def compute_z_criterion(self, Z, Zadv, y):
        y_ = (y + 1).unsqueeze(1).float()
        label_mask = ((y_.mm(1/y_.transpose(0,1))) != 1).float()
        label_mask[label_mask == 0] = -1

        zzadv_diff = []
        zz_diff = []
        if self.args.layer_weighting == 'const':
            layer_wts = np.ones((len(Z),))
        elif self.args.layer_weighting == 'linear':
            layer_wts = np.linspace(self.args.min_layer_wt, self.args.max_layer_wt, num=len(Z))
        elif self.args.layer_weighting == 'exp':
            layer_wts = np.geomspace(self.args.min_layer_wt, self.args.max_layer_wt, num=len(Z))

        for i, (z, zadv, w) in enumerate(zip(Z, Zadv, layer_wts)):
            if self.args.z_criterion == 'diff':
                _z_diff = reshape2D(z-zadv)
                zzadv_diff.append(w * _z_diff)
            elif self.args.z_criterion == 'diff-spread':
                _z_diff, _zz_diff = compute_diff_spread(z, zadv, label_mask)
                zz_diff.append(w * _zz_diff)
                zzadv_diff.append(w * _z_diff)
            elif self.args.z_criterion == 'cosine':
                z = reshape2D(z)
                zadv = reshape2D(zadv)
                _z_diff = compute_cosine_metric(z, zadv, 1)
                zzadv_diff.append(w * _z_diff.view(z.shape[0], -1))
            elif self.args.z_criterion == 'cosine-spread':
                zz_cos, zzadv_cos = compute_cosine_spread(z, zadv, label_mask, not self.args.use_MoM)
                zz_diff.append(w * zz_cos)
                zzadv_diff.append(w * zzadv_cos)
            else:
                raise NotImplementedError(self.args.z_criterion)
        if len(zz_diff) > 0:
            zz_diff = torch.cat(zz_diff, dim=1)
        zzadv_diff = torch.cat(zzadv_diff, dim=1)
        return zzadv_diff, zz_diff

    def compute_adversarial(self, x, y, logits=None, Z=None):
        model = self.model
        if model.training:            
            model = model.train(False)
            if self.args.maximize_logit_divergence:                    
                xadv = self.training_adversary.perturb(x, logits)
            elif self.args.adv_loss_fn == 'z_cos':
                Z = torch.cat([reshape2D(z) for z in Z], dim=1)
                xadv = self.training_adversary.perturb(x, Z)
            elif self.args.adv_loss_fn == 'xent':
                xadv = self.training_adversary.perturb(x, y)
            else:
                raise NotImplementedError
            model = model.train(True)            
        else:
            xadv = self.test_adversary.perturb(x, y)
        return xadv
    
    def compute_outputs(self, x, y, xadv=None):
        model = self.model
        logits, interm_Z = model(x, store_intermediate=True)
        if xadv is None:
            xadv = self.compute_adversarial(x, y, logits, interm_Z)
        adv_logits, interm_Zadv = model(xadv, store_intermediate=True)
        zzadv_diff, zz_diff = self.compute_z_criterion(interm_Z, interm_Zadv, y)
        return xadv, logits, adv_logits, zzadv_diff, zz_diff

    def MoM_update(self, x, y, xadv, zzadv_diff, zz_diff, loss):
        zzadv_diff_norm = zzadv_diff.pow(2).sum(1)
        zz_diff_norm = zz_diff.sum(1).mean(0) if len(zz_diff) > 0 else 0
        loss += self.args.zz_wt * zz_diff_norm
        
        z_diff_norm = zzadv_diff_norm
        z_diff = zzadv_diff
        
        if isinstance(self.optimizer.L, int):
            self.optimizer.L = torch.zeros((z_diff.shape[1],1), device=z_diff.device) + self.optimizer.L
        z_term = z_diff.mm(self.optimizer.L).squeeze() + (self.optimizer.c * (z_diff_norm))/2
        loss += z_term.mean()

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

        with torch.no_grad():
            _, _, _, z_diff, _ = self.compute_outputs(x, y, xadv)
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
        
        if self.args.cln_loss_wt > 0:
            loss = self.args.cln_loss_wt * torch.nn.functional.cross_entropy(logits[selected_cln], y[selected_cln])
        else:
            loss = 0
        if self.args.adv_loss_wt > 0:
            if self.args.maximize_logit_divergence:
                adv_classification_loss = self.training_adversary.loss_fn(adv_logits[selected_adv], logits[selected_adv])
            else:
                adv_classification_loss = self.training_adversary.loss_fn(adv_logits[selected_adv], y[selected_adv])
            loss += self.args.adv_loss_wt * adv_classification_loss / selected_adv.sum()            
        else:
            adv_classification_loss = 0
        
        if self.args.z_criterion == 'cosine-spread':                
            zz_diff_norm = zz_diff.sum(1)
            z_diff_norm = z_diff.sum(1)
            z_term = self.args.z_wt*z_diff_norm + self.args.zz_wt*zz_diff_norm
        elif self.args.z_criterion == 'cosine':
            z_diff_norm = z_diff.sum(1)
            zz_diff_norm = 0
            z_term = self.args.z_wt*(z_diff_norm)
        elif self.args.z_criterion == 'diff-spread':
            z_diff_norm = z_diff.sum(1)
            zz_diff_norm = zz_diff.sum(1)
            z_term = self.args.z_wt*z_diff_norm + self.args.zz_wt*zz_diff_norm
        else:
            z_diff_norm = z_diff.pow(2).sum(1)
            zz_diff_norm = 0
            z_term = self.args.z_wt*(z_diff_norm)

        if (not self.args.use_MoM) and self.args.z_wt > 0:
            loss += z_term.mean()

        if self.args.match_logits:
            logit_loss = self.logit_matching_fn(adv_logits[selected_adv], logits[selected_adv])
            loss += self.args.logit_loss_wt * logit_loss
        else:
            logit_loss = 0        
        return loss, z_diff_norm, zz_diff_norm, logit_loss

    def train_step(self, batch, batch_idx):
        x,y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        xadv, logits, adv_logits, zzadv_diff, zz_diff = self.compute_outputs(x, y)

        cln_acc, _ = compute_accuracy(logits, y)
        adv_acc, _ = compute_accuracy(adv_logits, y)

        loss, zzadv_diff_norm, zz_diff_norm, logit_loss = self.compute_loss(logits, adv_logits, y, zzadv_diff, zz_diff)

        if self.model.training and self.args.use_MoM:
            loss, zzadv_diff_norm, zz_diff_norm = self.MoM_update(x, y, xadv, zzadv_diff, zz_diff, loss)
        
        if self.model.training and not self.args.use_MoM:
            for p in self.model.parameters():
                assert (p.grad is None) or (p.grad == 0).all()
        return {'loss':loss}, {'train_clean_accuracy': cln_acc,
                             'train_adv_accuracy': adv_acc,
                             'train_accuracy': (cln_acc + adv_acc) / 2,
                             'train_loss': float(loss.detach().cpu()),
                             'logit_loss': float(logit_loss.detach().cpu()) if isinstance(logit_loss, torch.Tensor) else logit_loss,
                             'train_Z_loss': float(zzadv_diff_norm.max().detach().cpu()),
                             'train_ZZ_loss': float(zz_diff_norm.max().detach().cpu()) if len(zz_diff) > 0 else 0}
    
    def _optimization_wrapper(self, func):        
        if self.args.use_MoM:
            def wrapper(*args, **kwargs):
                self.optimizer.zero_grad()
                return func(*args, **kwargs)            
        else:
            def wrapper(*args, **kwargs):
                self.optimizer.zero_grad()
                output, logs = func(*args, **kwargs)                
                output['loss'].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                return output, logs
        return wrapper