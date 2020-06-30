import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import utils
import sys

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        return x.view(x.shape[0], -1)

class MINet(nn.Module):
    def __init__(self, x_size, y_size, hidden_size, outsize, device):
        super(MINet, self).__init__()
        self.fc_x = nn.Linear(x_size, hidden_size)
        self.fc_y = nn.Linear(y_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, outsize),
        )
        self.device = device
        self.to(device)

    def forward(self, x, y):
        proj_x = self.fc_x(x)
        proj_y = self.fc_y(y)

        cat = torch.cat((proj_x, proj_y), dim=1)
        joint = self.mlp(cat)

        rand_idxs = np.arange(len(y))
        np.random.shuffle(rand_idxs)
        cat = torch.cat((proj_x, proj_y[rand_idxs]), dim=1)
        marginal = self.mlp(cat)        
        return joint, marginal
        
    def compute_nim(self, joint, marginal):
        return joint.mean(0) - (torch.logsumexp(marginal, dim=0) - np.log(len(marginal)))

    def fit(self, loader, epochs, optimizer, scheduler, verbose=True):
        patience = 10
        bad_iters = 0
        prev_error = sys.maxsize

        for e in range(epochs):
            avg_loss = 0            
            for bi,(x,y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                joint, marginal = self.forward(x, y)
                loss = -self.compute_nim(joint, marginal).mean() 
                loss.backward()                
                optimizer.step()
                avg_loss = (bi*avg_loss + loss)/(bi+1)
            scheduler.step(avg_loss)            
            if verbose:
                print('%d/%d' % (e+1,epochs), float(avg_loss), bad_iters)
            # t.set_postfix(loss=avg_loss, bad_iters=bad_iters)
            if np.isclose(float(avg_loss), float(prev_error)) or avg_loss > prev_error:
                if bad_iters >= patience -1:
                    break
                else:
                    bad_iters += 1
            else:
                bad_iters = 0
                prev_error = avg_loss


def compute_A(S, N=1):    
    S_inv = torch.inverse(S)    

    I = torch.eye(S.shape[0], device=S_inv.device)
    A = I - S_inv / torch.diag(S_inv).view(-1,1)
    
    return A

def compute_S(Z):
    N, d = Z.shape

    Z = Z.unsqueeze(2)
    S = torch.bmm(Z, Z.transpose(1,2)).sum(0)
    # assert (S == S.transpose(0,1)).all()
    return S

class ResidualRegularizedModel(nn.Module):
    def __init__(self, cov_update_alpha = 0.75):    
        super(ResidualRegularizedModel, self).__init__()
        self.layers = []        
        self.cov = []
        self.cov_update_alpha = cov_update_alpha

    def reset(self):
        self.cov = []

    def forward(self, x, compute_residuals=False):
        residuals = []
        z = x
        for i,l in enumerate(self.layers):
            z = l(z)
            
            if compute_residuals:
                z_shape = z.shape
                z_dim_order = range(len(z.shape))
                if len(z_shape) == 4:
                    z = z.permute(0,2,3,1)
                    z_shape = z.shape
                    z_dim_order = (0,3,1,2)
                    z = z.reshape(-1, z.shape[3])
                
                S = compute_S(z)
                # if self.training:                    
                    # if i >= len(self.cov):
                    #     self.cov.append(S)
                    # else:
                    #     self.cov[i] = self.cov_update_alpha*self.cov[i].detach() + (1-self.cov_update_alpha)*S
                    # S = self.cov[i]
                A = compute_A(S, z.shape[0])  
                # assert (torch.diag(A) == 0).all()

                r = torch.mm(A, z.transpose(0,1)) - z.transpose(0,1)
                residuals.append(r) 
                            
                z = z.reshape(*z_shape)
                z = z.permute(*z_dim_order).contiguous()
        return z, residuals

class VGG16(ResidualRegularizedModel):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.name = 'VGG16'

        self.layers = nn.ModuleList([
            self.conv_bn_relu(3, 64, 3, 1),
            self.conv_bn_relu(64, 64, 3, 1),
            self.conv_bn_relu_pooling(64, 64, 3, 1, 2, 2),
            self.conv_bn_relu(64, 128, 3, 1),
            self.conv_bn_relu_pooling(128, 128, 3, 1, 2, 2),
            self.conv_bn_relu(128, 256, 3, 1),            
            self.conv_bn_relu_pooling(256, 256, 3, 1, 2, 2),
            self.conv_bn_relu(256, 512, 3, 1),
            self.conv_bn_relu(512, 512, 3, 1),
            self.conv_bn_relu_pooling(512, 512, 1, 0, 2, 2),
            self.conv_bn_relu(512, 512, 3, 1),
            self.conv_bn_relu(512, 512, 3, 1),
            nn.Sequential(
                self.conv_bn_relu_pooling(512, 512, 1, 0, 2, 2),
                nn.AdaptiveAvgPool2d((1,1)),
                Flatten(),
            ),
            nn.Linear(512, num_classes),
        ])
    
    def conv_bn_relu(self, in_channels, out_channels, kernel_size, padding):
        module = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return module
    
    def conv_bn_relu_pooling(self, in_channels, out_channels, kernel_size, padding, pooling_kernel_size, pooling_stride):
        module = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride),
        )
        return module

if __name__ == '__main__':
    model = VGG16(10).cuda()
    x = torch.rand(128, 3, 32, 32).cuda()
    logits, residuals = model(x)
    for r in residuals:
        print(r.shape, torch.norm(r, dim=0).mean())