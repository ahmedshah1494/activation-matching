import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
from trainer import Trainer, compute_accuracy
from models import VGG16
import argparse
import os

class ResidualRegularizedTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args):
        super(ResidualRegularizedTrainer, self).__init__(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args)
    
    def _optimization_wrapper(self, func):
        def wrapper(*args, **kwargs):
            self.optimizer.zero_grad()
            output, logs = func(*args, **kwargs)
            output['loss'].backward()
            self.optimizer.step()
            return output, logs
        return wrapper

    def criterion(self, logits, y):
        loss = 1.0 * nn.functional.cross_entropy(logits, y, reduction='none')
        return loss

    def train_step(self, batch, batch_idx):
        x,y = batch        
        x = x.to(self.device)
        y = y.to(self.device)        
        logits, residuals = self.model(x, compute_residuals=True)
        loss = self.criterion(logits, y)
        acc, correct = compute_accuracy(logits.detach().cpu(), y.detach().cpu())        
        residual_norm = [torch.norm(r, dim=0).mean() for r in residuals]
        total_residual = 0
        for rn in residual_norm:
            total_residual += rn
        loss = loss.mean() 
        loss -= args.regularization_weight*total_residual
        total_residual = float(total_residual.detach().cpu())

        return {'loss':loss}, {
                             'train_accuracy': acc,
                             'train_loss': float(loss.detach().cpu()),
                             'train_residual': total_residual
                             }
    
    def val_step(self, batch, batch_idx):
        x,y = batch        
        x = x.to(self.device)
        y = y.to(self.device)        
        logits, _ = self.model(x, compute_residuals=False)
        loss = self.criterion(logits, y)
        acc, correct = compute_accuracy(logits.detach().cpu(), y.detach().cpu())                
        loss = loss.mean().detach().cpu()

        return {'loss':loss}, {
                             'val_accuracy': acc,
                             'val_loss': float(loss.detach().cpu())
                             }

    def epoch_end(self, epoch_idx, train_outputs, val_outputs, train_metrics, val_metrics):
        self.model.reset()
        return super().epoch_end(epoch_idx, train_outputs, val_outputs, train_metrics, val_metrics)

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
    # common_transform = torchvision.transforms.Compose([
    #         torchvision.transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
    #         torchvision.transforms.RandomCrop(32, padding=4),
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.RandomGrayscale(p=0.1),
    #         torchvision.transforms.ToTensor()
    #     ])
    # test_transform = torchvision.transforms.ToTensor()

    common_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),        
    ]) # meanstd transformation

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

def main(args):
    train_dataset, val_dataset, test_dataset, num_classes = get_cifar10_dataset(args.datafolder)
    # train_dataset = [train_dataset[i] for i in range(10000)]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = VGG16(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, verbose=True, factor=0.5)

    args.logdir = os.path.join(args.logdir, "%s_%s" % (model.name, args.exp_name))
    exp_num = len(os.listdir(args.logdir)) if os.path.exists(args.logdir) else 0
    args.logdir = os.path.join(args.logdir, 'exp_%d' % exp_num)

    trainer = ResidualRegularizedTrainer(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args)
    trainer.track_metric('val_loss', 'min')
    trainer.train()
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
    parser.add_argument('--logdir', default='logs/')
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--regularization_weight', type=float, default=1e-4)
    args = parser.parse_args()


    main(args)