import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from iterative_projected_gradient import perturb_iterative
from utils import get_cifar10_dataset
from trainer import compute_accuracy
from extract_and_plot_embeddings import fit_transform, _get_embeddings
import seaborn as sns
import pandas as pd
import argparse
import hashlib
import os
import pickle

def plot_embeddings(embeddings, labels, adv_embeddings, adv_labels, adv_idx):
    fig = plt.figure(figsize=(18,25))
    sns.set(font_scale=2, style='white')

    for k, ((lidx, emb), (_, adv_emb), L) in enumerate(zip(embeddings.items(), adv_embeddings.items(), labels)):
        ax = plt.subplot((len(embeddings) + 1)//2, min(2, len(embeddings)), k+1)
        # sns.set_context("paper", rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":5})    
        df = pd.DataFrame()
        df['x1'] = emb[:,0]
        df['x2'] = emb[:,1]
        df['Label'] = labels+1    
        g = sns.scatterplot(
            x="x1", y="x2", 
            hue="Label",
            palette=sns.color_palette("hls", labels.max()+1),
            data=df,
            legend=False,
            alpha=0.3
        )
        # ax.set_yticklabels([])
        # ax.set_ylabel('')
        # ax.set_xticklabels([])
        # ax.set_xlabel('')
        cmap = plt.cm.get_cmap('Set1', args.n_points)
        for i,(x_, x) in enumerate(zip(adv_emb, emb[adv_idx])):
            for j, x_j in enumerate(x_):
                plt.plot(*(x_j), marker='^', c=cmap(i), markersize=5 + ((j+1) * 5/len(x_)))
            plt.plot(*(x), marker='^', c=cmap(i), markersize=5, mew=2)
                
        plt.title('layer %d' % (lidx+1))
    fig.tight_layout(rect=[0, 0, 1, .9])

def get_embeddings(args):
    # train_dataset, val_dataset, test_dataset, num_classes = get_cifar10_dataset(args.datafolder)
    test_dataset = torchvision.datasets.CIFAR10(args.datafolder, False, torchvision.transforms.ToTensor())
    model = torch.load(args.model_path).cuda()
    model.args.layer_idxs = []

    eps = 8/255
    eps_iter = eps/4
    nb_iter = 10
    
    loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)
    z, labels = _get_embeddings(model, loader, torch.device('cuda'), range(len(model.layers)))
    print([(k,v.shape) for k,v in z.items()])

    x = []
    y = []
    adv_idx = range(args.n_points)
    for i in adv_idx:
        x_, y_ = test_dataset[i]
        x.append(x_)
        y.append(y_)
    x = torch.stack(x, 0).cuda()
    y = torch.LongTensor(y).cuda()
        
    xadv = perturb_iterative(x, y, model, nb_iter, eps, eps_iter, 
                                torch.nn.CrossEntropyLoss(reduction='sum'), 
                                return_itermediate=True)
    xadv_shape = xadv.shape
    xadv = xadv.view(-1, *(xadv_shape[2:]))
    print(xadv.shape)
    
    zadv, _ = _get_embeddings(model, [(xadv,y.repeat_interleave(nb_iter))], torch.device('cuda'), range(len(model.layers)))
    print([(k,v.shape) for k,v in zadv.items()])

    for (lidx, emb), (_, adv_emb) in zip(z.items(), zadv.items()):
        if emb.shape[1] > 2:
            print('performing projection...')
            print(emb.shape, adv_emb.shape)
            z[lidx], t = fit_transform(emb, args.proj_type, y)
            _adv_emb, _ = fit_transform(adv_emb.reshape(-1,adv_emb.shape[-1]), t)
            _adv_emb = _adv_emb.reshape(args.n_points, nb_iter, _adv_emb.shape[1])
            print(_adv_emb.shape)
            zadv[lidx] = _adv_emb
            print(zadv[lidx].shape)
        print(emb.shape, adv_emb.shape)

    return z, labels, zadv, y, adv_idx

def main(args):    
    h = hashlib.md5(str(vars(args)).encode('utf-8')).hexdigest()
    if os.path.exists(os.path.join('tmp', h+'.pkl')):
        with open(os.path.join('tmp', h+'.pkl'), 'rb') as f:
            emb_data = pickle.load(f)
    else:
        emb_data = get_embeddings(args)
        with open(os.path.join('tmp', h+'.pkl'), 'wb') as f:
            pickle.dump(emb_data, f)
    plot_embeddings(*emb_data)
    outdir = os.path.join(args.outdir, args.proj_type)
    
    outdir = os.path.join(outdir, *(args.model_path.split('/')[1:-2]))
    outfile = '%s_embeddings.png'%(args.proj_type)
    # plt.title(outfile.split('.')[0])    
    outfile = os.path.join(outdir, outfile)
    print(outfile)
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    plt.savefig(outfile)

    
if __name__ == '__main__':
    np.random.seed(9999)
    torch.random.manual_seed(9999)
    torch.cuda.manual_seed(9999)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--datafolder', default='/home/mshah1/workhorse3')
    parser.add_argument('--model_path')
    parser.add_argument('--n_points', type=int)
    parser.add_argument('--proj_type', type=str, default='pca', choices=('pca', 'lda', 'tsne'))
    parser.add_argument('--outdir', type=str, default='embedding_plots/')
    args = parser.parse_args()

    main(args)