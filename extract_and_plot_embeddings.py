import sys
sys.path.append('../')
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.markers as mmark
import seaborn as sns
import pandas as pd
import utils
import argparse
from multiprocessing import cpu_count
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os
import io
import hashlib
import pickle

def _get_embeddings(model, loader,device, layer_idx, normalize=False):
    embeddings = {}
    labels = []
    model = model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    for x,y in loader:        
        x = x.to(device)
        logits, Z = model(x, store_intermediate=True)        
        for i in layer_idx:            
            z = Z[i].detach().cpu().numpy()
            z = z.reshape(z.shape[0], -1)
            embeddings.setdefault(i, []).append(z)
        
        logits = logits.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        correct += (y == logits.argmax(1)).astype(int).sum()
        total += y.shape[0]

        labels.append(y)
    print('accuracy=%.4f' % float(correct/total))
    for k,v in embeddings.items():
        embeddings[k] = np.concatenate(v, 0)
    labels = np.concatenate(labels, 0)
    return embeddings, labels
        
def fit_transform(embeddings, transform, labels=None):
    print(embeddings.shape, transform)
    if type(transform) == str:
        if embeddings.shape[1] > 2:            
            print('performing projection...')
            if transform == 'tsne':
                T = TSNE(n_jobs=-1, verbose=2)
                embeddings = T.fit_transform(embeddings)
            if transform == 'pca':
                T = PCA(2)
                # embeddings -= min(0, embeddings.min())                
                # embeddings[embeddings == 0] = 1e-10
                embeddings = T.fit_transform(embeddings)
            if transform == 'lda':
                T = LinearDiscriminantAnalysis(n_components=2)
                # embeddings[embeddings == 0] = 1e-10
                embeddings = T.fit_transform(embeddings, labels)
            return embeddings, T
        else:
            return embeddings, None
    else:
        embeddings = transform.transform(embeddings)
        return embeddings, transform

def modify_model(model):
    layers = list(model.layers.children())[:-1]
    model.layers = torch.nn.Sequential(*layers)

def get_embeddings(args):
    model = torch.load(args.model_file)
    model.args.layer_idxs = []
    if args.layer_idx == []:
        args.layer_idx = range(len(model.layers)-1)
    # print(model)
    train_dataset, val_dataset, test_dataset, nclasses = utils.get_cifar10_dataset(args.datafolder, [torchvision.transforms.ToTensor()]*2)
    if args.use_train_data:
        rand_idx = np.arange(len(train_dataset))[:10000]
        train_dataset = Subset(train_dataset, rand_idx)
        print(len(train_dataset))
        test_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
        test_dataset = train_dataset
    else:
        test_loader = DataLoader(test_dataset, 128, shuffle=False, num_workers=(cpu_count())//2)
    
    if args.layer_idx == -1:
        test_embeddings, test_labels = zip(*(list(test_loader)))
        test_embeddings = torch.cat(test_embeddings, 0).cpu().numpy()
        test_embeddings.reshape(test_embeddings.shape[0], -1)
        test_labels = torch.cat(test_labels, 0).cpu().numpy()
    else:
        test_embeddings, test_labels = _get_embeddings(model, test_loader, args.device, args.layer_idx)

    embeddings = test_embeddings
    labels = test_labels

    if args.perturbed_dataset_path is not None:
        pertrubed_data = [torch.load(df) for df in args.perturbed_dataset_path]
        adv_data = [d['data'] for d in pertrubed_data]    
        adv_data = torch.stack(adv_data, dim=0).to(torch.device('cpu'))
        adv_preds = [d['preds'] for d in pertrubed_data]    
        adv_preds = torch.stack(adv_preds, dim=0).to(torch.device('cpu'))
        cln_preds = [d['clean_preds'] for d in pertrubed_data]    
        cln_preds = torch.stack(cln_preds, dim=0).to(torch.device('cpu'))
        L = [d['labels'] for d in pertrubed_data]            
        L = torch.stack(L, dim=0).cpu()
        
        print(adv_data.shape, adv_preds.shape, L.shape)

        successful_idx = np.arange(adv_preds.shape[1])[((cln_preds == L) & (adv_preds != L)).all(0)]        
        print(len(successful_idx)/adv_preds.shape[1])
        adv_idx = successful_idx[:args.n_adv]
        
        cln_images = [test_dataset[i][0].permute(1,2,0).cpu().numpy() for i in adv_idx]
        cln_images = np.stack(cln_images, 0)
        adv_images = []
        adv_embeddings = []
        for xadv, adv_labels in zip(adv_data, L):
            xadv = xadv[adv_idx]
            adv_labels = adv_labels[adv_idx]
            if args.layer_idx == -1:
                fxadv = xadv.cpu().numpy()
                fxadv = fxadv.reshape(fxadv.shape[0], -1)
                adv_labels = adv_labels.cpu().numpy()
            else:
                fxadv, adv_labels = _get_embeddings(model, [(xadv, adv_labels)], args.device, args.layer_idx)
            adv_embeddings.append(fxadv)
            adv_images.append(xadv.permute(0,2,3,1).cpu().numpy())

        adv_embeddings = {k: np.stack([x[k] for x in adv_embeddings], 1) for k in adv_embeddings[0].keys()}
        print([(k,v.shape) for k,v in adv_embeddings.items()])

        for a in adv_images:
            diff = (a - cln_images).reshape(a.shape[0], -1)
            diff_inf_norm = np.linalg.norm(diff, ord=np.inf, axis=1)
            diff_2_norm = np.linalg.norm(diff, ord=2, axis=1)
            # print(diff_inf_norm, diff_2_norm)

        pairwise_distance = lambda x,y: np.sqrt(((np.expand_dims(x, 1) - np.expand_dims(y, 0)) ** 2).sum(-1))
        lidx = 0
        print(test_embeddings[lidx].shape, adv_embeddings[lidx].reshape(-1, adv_embeddings[lidx].shape[-1]).shape)        
        cln_dist = pairwise_distance(test_embeddings[lidx][:500], test_embeddings[lidx][:500])
        adv_dist = pairwise_distance(adv_embeddings[lidx].reshape(-1, adv_embeddings[lidx].shape[-1]), adv_embeddings[lidx].reshape(-1, adv_embeddings[lidx].shape[-1]))
        cln_adv_dist = pairwise_distance(adv_embeddings[lidx].reshape(-1, adv_embeddings[lidx].shape[-1]), test_embeddings[lidx][:500])
        print(cln_dist.shape, adv_dist.shape, cln_adv_dist.shape)
        print('clean_dist:', cln_dist.mean())
        print('adv_dist:', adv_dist.mean())
        print('cln_adv_dist:', cln_adv_dist.mean())
        # exit()


        images = np.concatenate([cln_images] + adv_images, 0)
        plt.figure(figsize=(20,20))
        f, grid = plt.subplots(len(args.perturbed_dataset_path)+1, args.n_adv)
        for ax, im in zip(grid.flatten(), images):
            ax.imshow(im, interpolation='bilinear')
        outfile = '_'.join(args.model_file.split('/')[1:-2]).replace('.pt','_wAdv_images.png')
        plt.savefig(outfile)
        plt.clf()

    for (lidx, emb), (_, adv_emb) in zip(embeddings.items(), adv_embeddings.items()):
        if emb.shape[1] > 2:
            print('performing projection...')
            print(emb.shape, adv_emb.shape)
            embeddings[lidx], t = fit_transform(emb, args.proj_type, labels)
            _adv_emb, _ = fit_transform(adv_emb.reshape(-1,adv_emb.shape[-1]), t)
            print(_adv_emb.shape)
            adv_embeddings[lidx] = _adv_emb.reshape(*(adv_emb.shape[:-1]),2)
            print(adv_embeddings[lidx].shape)
        print(emb.shape, adv_emb.shape)

    cln_dist = pairwise_distance(test_embeddings[lidx][:500], test_embeddings[lidx][:500])
    adv_dist = pairwise_distance(adv_embeddings[lidx].reshape(-1, adv_embeddings[lidx].shape[-1]), adv_embeddings[lidx].reshape(-1, adv_embeddings[lidx].shape[-1]))
    cln_adv_dist = pairwise_distance(adv_embeddings[lidx].reshape(-1, adv_embeddings[lidx].shape[-1]), test_embeddings[lidx][:500])
    print(cln_dist.shape, adv_dist.shape, cln_adv_dist.shape)
    print('clean_dist:', cln_dist.mean())
    print('adv_dist:', adv_dist.mean())
    print('cln_adv_dist:', cln_adv_dist.mean())

    return embeddings, labels, adv_embeddings, adv_labels, adv_idx

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
        cmap = plt.cm.get_cmap('Set1', args.n_adv)
        for i,(x_, x) in enumerate(zip(adv_emb, emb[adv_idx])):
            for j, x_j in enumerate(x_):
                plt.plot(*(x_j), marker=Line2D.filled_markers[j], c=cmap(i), markersize=10)
            plt.plot(*(x), marker='o', c=cmap(i), fillstyle='none', markersize=10, mew=2)
        
        handles = []
        for j, x_j in enumerate(x_):
            l = os.path.basename(args.perturbed_dataset_path[j]).replace('.pt','').split('_')
            l = "-".join([l[1], l[2]])
            plt.plot([],[], marker=Line2D.filled_markers[j], markersize=10, c=cmap(i), label=l)
        plt.plot([],[], marker='o', c=cmap(i), fillstyle='none', markersize=10, mew=2, label='clean')
        if k == 0:
            fig.legend(loc='upper left', bbox_to_anchor=(0.02,1), borderaxespad=0, ncol=2)  
        plt.title('layer %d' % (lidx+1))
    fig.tight_layout(rect=[0, 0, 1, .9])

def main(args):    
    h = hashlib.md5(str(vars(args)).encode('utf-8')).hexdigest()
    # if os.path.exists(os.path.join('tmp', h+'.pkl')):
    #     with open(os.path.join('tmp', h+'.pkl'), 'rb') as f:
    #         emb_data = pickle.load(f)
    # else:
    emb_data = get_embeddings(args)
        # with open(os.path.join('tmp', h+'.pkl'), 'wb') as f:
        #     pickle.dump(emb_data, f)
    plot_embeddings(*emb_data)
    if args.use_train_data:
        args.outdir = os.path.join(args.outdir,'training_data')
    outdir = os.path.join(args.outdir, args.proj_type)
    
    outdir = os.path.join(outdir, *(args.model_file.split('/')[1:-2]))
    if args.perturbed_dataset_path is not None:
        outfile = '%s_wAdv_embeddings.png'%(args.proj_type)
    else:
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
    parser.add_argument('--datafolder', type=str,default="/home/mshah1/workhorse3")
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--perturbed_dataset_path', type=str, nargs='+')
    parser.add_argument('--model_file')
    parser.add_argument('--layer_idx', type=int, nargs='+', default=[])
    parser.add_argument('--proj_type', type=str, default='tsne')
    parser.add_argument('--n_adv', type=int, default=1)
    parser.add_argument('--normalize_input', action='store_true')
    parser.add_argument('--use_train_data', action='store_true')
    parser.add_argument('--outdir', type=str, default='embedding_plots/')
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    main(args)