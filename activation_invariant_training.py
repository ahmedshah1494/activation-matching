import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
from trainer import Trainer, compute_accuracy
from attack_classifier import extract_attack
from models import LayeredModel, VGG16, WideResnet
import argparse
import os

class ActivationInvarianceTrainer(Trainer):
    def __init__(self, model:LayeredModel, train_loader, val_loader, test_loader, optimizer, scheduler, device, args):
        super(ActivationInvarianceTrainer, self).__init__(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args)
        attack_class, kwargs = extract_attack(args)
        self.adversary = attack_class(self.model, **kwargs)
        
    def train_step(self, batch, batch_idx):
        x,y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        xadv = self.adversary.perturb(x, y)        
        logits, interm_Z = self.model(x, store_intermediate=True)        
        adv_logits, interm_Zadv = self.model(xadv, store_intermediate=True)
        if self.args.detach_adv_logits:
            C = self.model.layers[-1].requires_grad_(False)
            adv_logits = C(interm_Zadv[-1])
            C.requires_grad_(True)
            

        cln_classification_loss = torch.nn.functional.cross_entropy(logits, y)
        adv_classification_loss = torch.nn.functional.cross_entropy(adv_logits, y)
        interm_z_loss = 0        
        for z, zadv in zip(interm_Z, interm_Zadv):
            interm_z_loss += torch.nn.functional.mse_loss(z, zadv, reduction='sum')        
        loss = cln_classification_loss + self.args.adv_loss_wt*adv_classification_loss + self.args.z_wt * interm_z_loss

        cln_acc, _ = compute_accuracy(logits.detach().cpu(), y.detach().cpu())
        adv_acc, _ = compute_accuracy(adv_logits.detach().cpu(), y.detach().cpu())
        return {'loss':loss}, {'train_clean_accuracy': cln_acc,
                             'train_adv_accuracy': adv_acc,
                             'train_loss': float(loss.detach().cpu()),
                             'train_Z_loss': float(interm_z_loss.detach().cpu())}        

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

def train(args):
    train_dataset, val_dataset, test_dataset, num_classes = get_cifar10_dataset(args.datafolder)
    # train_dataset = [train_dataset[i] for i in range(10000)]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_dict = {
        'vgg16': lambda : VGG16(args, num_classes),
        'wide_resnet': lambda : WideResnet(args, 28, 10, 0, num_classes)
    }
    model = model_dict[args.model_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, verbose=True, factor=0.5)

    args.logdir = os.path.join(args.logdir, "%s_%s" % (model.name, args.exp_name))
    exp_num = len(os.listdir(args.logdir)) if os.path.exists(args.logdir) else 0
    args.logdir = os.path.join(args.logdir, 'exp_%d' % exp_num)
    print('logging to ', args.logdir)

    trainer = ActivationInvarianceTrainer(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args)
    trainer.track_metric('val_loss', 'min')
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

    parser.add_argument('--logdir', default='logs/activation_invariance/')
    parser.add_argument('--exp_name', default='')
    
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    
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

    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        train(args)