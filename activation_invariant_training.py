import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
from trainer import Trainer, compute_accuracy
from attack_classifier import extract_attack
from models import LayeredModel, VGG16, WideResnet, GaussianNoiseLayer
import argparse
import os

class ActivationInvarianceTrainer(Trainer):
    def __init__(self, model:LayeredModel, train_loader, val_loader, test_loader, optimizer, scheduler, device, args):
        super(ActivationInvarianceTrainer, self).__init__(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args)
        attack_class, kwargs = extract_attack(args)
        self.adversary = attack_class(self.model, **kwargs)

    def compute_outputs(self, x, xadv):
        logits, interm_Z = self.model(x, store_intermediate=True)        
        adv_logits, interm_Zadv = self.model(xadv, store_intermediate=True)        
        if self.args.detach_adv_logits:
            C = self.model.layers[-2].requires_grad_(False)
            adv_logits = C(interm_Zadv[-1])
            C.requires_grad_(True)
        
        if self.args.regress_on_logits:
            interm_Z.append(logits)
            interm_Zadv.append(adv_logits)
        z_diff = []
        for i, (z, zadv) in enumerate(zip(interm_Z, interm_Zadv)):            
            _z_diff = z-zadv
            z_diff.append(_z_diff.view(z.shape[0], -1))
        z_diff = torch.cat(z_diff, dim=1)
        return logits, adv_logits, z_diff

    def train_step(self, batch, batch_idx):
        x,y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        xadv = self.adversary.perturb(x, y)
        logits, adv_logits, z_diff = self.compute_outputs(x, xadv)

        cln_classification_loss = torch.nn.functional.cross_entropy(logits, y)
        adv_classification_loss = torch.nn.functional.cross_entropy(adv_logits, y)
        
        loss = cln_classification_loss + self.args.adv_loss_wt*adv_classification_loss
        z_diff_norm = torch.norm(z_diff, p=2, dim=1)**2
        if self.args.optimizer == 'MoM' and self.model.training:
            if isinstance(self.optimizer.L, int):
                self.optimizer.L = torch.zeros((z_diff.shape[1],1), device=z_diff.device) + self.optimizer.L
            # print(z_diff.shape, self.optimizer.L.shape, z_diff_norm.shape)
            z_term = z_diff.mm(self.optimizer.L).squeeze() + (self.optimizer.c * z_diff_norm)/2            
        else:
            z_term = self.args.z_wt*(z_diff_norm)
        loss += z_term.mean()

        cln_acc, _ = compute_accuracy(logits.detach().cpu(), y.detach().cpu())
        adv_acc, _ = compute_accuracy(adv_logits.detach().cpu(), y.detach().cpu())

        if self.args.optimizer == 'MoM' and self.model.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                _, _, z_diff = self.compute_outputs(x, xadv)
                self.optimizer.L += self.optimizer.c * z_diff.mean(0).unsqueeze(1)
            self.optimizer.c *= self.args.c_step_size
            loss.detach()
        return {'loss':loss}, {'train_clean_accuracy': cln_acc,
                             'train_adv_accuracy': adv_acc,
                             'train_loss': float(loss.detach().cpu()),
                             'train_Z_loss': float(z_diff_norm.max().detach().cpu())}

    def _optimization_wrapper(self, func):        
        if self.args.optimizer == 'MoM':
            return func
        else:
            return super(ActivationInvarianceTrainer, self)._optimization_wrapper(func)

    # def admm_train_step(self, batch, batch_idx):
        x,y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        (model1, model2, u) = self.model

        def get_loss():
            xadv = self.adversary.perturb(x, y)
            logits, interm_Z = model1(x, store_intermediate=True)        
            adv_logits, interm_Zadv = model2(xadv, store_intermediate=True)
            if self.args.detach_adv_logits:
                C = self.model.layers[-2].requires_grad_(False)
                adv_logits = C(interm_Zadv[-1])
                C.requires_grad_(True)
            
            if self.args.regress_on_logits:
                interm_Z.append(logits)
                interm_Zadv.append(adv_logits)

            cln_classification_loss = torch.nn.functional.cross_entropy(logits, y)
            adv_classification_loss = torch.nn.functional.cross_entropy(adv_logits, y)
            z_loss = 0
            for i, (z, zadv) in enumerate(zip(interm_Z, interm_Zadv)):            
                _z_loss = torch.nn.functional.mse_loss(z, zadv, reduction='none')
                z_loss += _z_loss.view(z.shape[0], -1).sum(1).mean()        

            quad_term = 0
            for p1,p2 in zip(model1.parameters(), model2.parameters()):
                quad_term += torch.nn.functional.mse_loss(p1, p2, reduction='sum')

            loss = cln_classification_loss + self.args.adv_loss_wt*adv_classification_loss + u * z_loss + self.args.rho*quad_term/2
            return loss, logits, adv_logits, z_loss

        model2_grad_state = []
        for p in model2.parameters():
            model2_grad_state.append(p.requires_grad)
            p.requires_grad_(False)

        loss = get_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for p, s in zip(model2.parameters(), model2_grad_state):
            p.requires_grad_(s)
        
        model1_grad_state = []
        for p in model1.parameters():
            model1_grad_state.append(p.requires_grad)
            p.requires_grad_(False)
        
        loss = get_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        quad_term = 0
        for p1,p2 in zip(model1.parameters(), model2.parameters()):
            quad_term += torch.nn.functional.mse_loss(p1, p2, reduction='sum')

        u += args.rho*quad_term

        cln_acc, _ = compute_accuracy(logits.detach().cpu(), y.detach().cpu())
        adv_acc, _ = compute_accuracy(adv_logits.detach().cpu(), y.detach().cpu())
        return {'loss':loss}, {'train_clean_accuracy': cln_acc,
                             'train_adv_accuracy': adv_acc,
                             'train_loss': float(loss.detach().cpu()),
                             'train_Z_loss': float(z_loss.detach().cpu())}
def make_val_dataset(dataset, num_classes):
    train_idxs = []        
    val_idxs = []
    class_counts = {i:0 for i in range(num_classes)}
    train_class_counts = 4500

    for i,sample in enumerate(dataset):
        y = sample[1]
        if class_counts[y] < train_class_counts:      
            train_idxs.append(i)
            class_counts[y] += 1
        else:
            val_idxs.append(i)
    print(len(train_idxs), len(val_idxs))
    return train_idxs, val_idxs

def get_cifar10_dataset(datafolder, custom_transforms=None):
    common_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),        
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),        
    ])

    if custom_transforms is not None:
        train_transform, test_transform = custom_transforms
    else:
        train_transform = common_transform    

    print(train_transform)
    print(test_transform)
    
    nclasses = 10
    train_dataset = torchvision.datasets.CIFAR10('%s/'%datafolder, 
            transform=train_transform, download=True)
    train_idxs, val_idxs = make_val_dataset(train_dataset, nclasses)
    val_dataset = Subset(train_dataset, val_idxs)
    train_dataset = Subset(train_dataset, train_idxs)
   
    test_dataset = torchvision.datasets.CIFAR10('%s/'%datafolder, train=False,
            transform=test_transform, download=True)        
    
    return train_dataset, val_dataset, test_dataset, nclasses

def add_gaussian_smoothing_layers(model:LayeredModel):
    for i,l in enumerate(model.layers):
        model.layers[i] = nn.Sequential(
            GaussianNoiseLayer(),
            l
        )

class MoM(torch.optim.Adam):
    def __init__(self, model, init_c, h, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        super().__init__(model.parameters(), lr, betas, eps, weight_decay, amsgrad)
        self.c = init_c
        self.h = h
        self.L = 0       

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
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9, nesterov=True)
    elif args.optimizer == 'MoM':
        optimizer = MoM(model, args.init_c, ActivationInvarianceTrainer.compute_outputs, lr=args.lr, weight_decay=0.0005)

    args.logdir = os.path.join(args.logdir, "%s_%s" % (model.name, args.exp_name))
    exp_num = len(os.listdir(args.logdir)) if os.path.exists(args.logdir) else 0
    args.logdir = os.path.join(args.logdir, 'exp_%d' % exp_num)
    print('logging to ', args.logdir)

    if args.attack == 'none':
        trainer_class = Trainer
        mode = 'max'
        metric = 'val_accuracy'
    else:
        trainer_class = ActivationInvarianceTrainer
        mode = 'max'
        metric = 'val_adv_accuracy'
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
    model = model.to(args.device)

    args.logdir = os.path.join(*([x for x in os.path.dirname(args.model_path).split('/') if x != 'checkpoints']))
    print(args.logdir)
    trainer = ActivationInvarianceTrainer(model, None, None, test_loader, None, None, device, args)        
    trainer.test()
    trainer.logger.flush()
    trainer.logger.close()

if __name__ == '__main__':
    np.random.seed(9999)
    torch.random.manual_seed(9999)
    torch.cuda.manual_seed(9999)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--datafolder', default='/home/mshah1/workhorse3')
    
    parser.add_argument('--model_name')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--normalize_input', action='store_true')

    parser.add_argument('--logdir', default='logs/activation_invariance/')
    parser.add_argument('--exp_name', default='')
    
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

    parser.add_argument('--attack', type=str, default="none", choices=('none', 'pgdinf', 'pgdl2'))
    parser.add_argument('--max_instances', type=int, default=-1)
    parser.add_argument('--nb_iter', type=int, default=7)
    parser.add_argument('--nb_restarts', type=int, default=1)
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--eps_iter', type=float, default=8/(255*4))
    parser.add_argument('--max_iters', type=int, default=7)

    parser.add_argument('--z_wt', type=float, default=1e-3)
    parser.add_argument('--adv_loss_wt', type=float, default=1)
    parser.add_argument('--detach_adv_logits', action='store_true')
    parser.add_argument('--layer_idxs', type=int, nargs='+', default=[])

    parser.add_argument('--gaussian_smoothing', action='store_true')
    parser.add_argument('--regress_on_logits', action='store_true')

    args = parser.parse_args()
    args.layer_idxs = torch.tensor(args.layer_idxs)
    if args.test:
        test(args)
    else:
        train(args)