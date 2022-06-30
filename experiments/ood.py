# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir("..")
from src import *
import time
import argparse
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score as avpr


colors = list(mcolors.TABLEAU_COLORS)

parser = argparse.ArgumentParser(description='OoD experiment')
parser.add_argument('--model', type=str, default='HHVAEM',
                    help='model to use (VAE, HVAE, HMCVAE, HHVAE, VAEM, HVAEM, HMCVAEM, HHVAEM)')
parser.add_argument('--version', type=str, default='version_0', 
                    help='name for the log in Tensorboard (defaul None for "version_0")')
parser.add_argument('--dataset', type=str, default='boston',
                    help='dataset to train (boston, mnist, ...)')
parser.add_argument('--dataset_ood', type=str, default="fashion_mnist",
                    help='dataset OUT OF distribution')                   
parser.add_argument('--split', type=int, default=0,
                    help='index of the train/test partition')
parser.add_argument('--samples', type=int, default=100,
                    help='repetitions for each sample')
parser.add_argument('--gpu', type=int, default=1,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()

args.cuda = args.gpu and torch.cuda.is_available()
args.cuda = args.cuda == True and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.dataset = clean_dataset(args.dataset)

if str(device) == "cuda":
    print('cuda activated')

if __name__ == '__main__':
    start = time.time()
    with torch.no_grad():
        f = plt.figure()

        results = []
        ckpt_path = find_path(args.dataset, args.model, args.split, args.version)
        args.model = clean_model(args.model)
        model = load_model(args.model, ckpt_path, device=device).eval()

        mixed = None
        if args.model.__contains__('VAEM'):
            mixed = True
        
        dataloader = get_dataset_loader(model.dataset, 'test', path=model.data_path, batch_size=model.batch_size, split_idx=args.split, dim=None, mixed=mixed)
        dataloader_ood = get_dataset_loader(args.dataset_ood, 'test', path=model.data_path, batch_size=model.batch_size, split_idx=args.split, dim=None, mixed=mixed)
        
        nbatches = np.minimum(len(dataloader), len(dataloader_ood))

        dim_x = model.dim_x
        dim_y = model.dim_y
        dim_x_ood = dataloader_ood.dataset.data.shape[-1]
        dim_y_ood = dataloader_ood.dataset.labels.shape[-1]

        # ===== ON distribution dataset ===== #
        tqdm_batch = tqdm(total=len(dataloader), desc='Batch', position=0, leave=False)
        score = []

        iterator = iter(dataloader)
        for b in range(nbatches):
            batch = next(iterator)
            batch = [b.to(device) for b in batch]
            # Get data
            x, observed_x, y, observed_y = batch
            xn = model.normalize_x(x)
            xt, yt, xy, observed = model.preprocess_batch(batch) 
            mu_z, logvar_z = model.encoder(xy)
            z = model.sample_z(mu_z, logvar_z, samples=args.samples)
            theta_x = model.decoder(z)
            x_hat = model.build_x_hat(xn, observed_x, theta_x)
            zx = torch.cat([z,x_hat],dim=-1)

            rec_x = model.decoder.logp(xt, observed_x, z=z, theta=theta_x).sum(-1)
            rec_x = torch.logsumexp(rec_x, dim=-1) - np.log(args.samples)

            rec_y = model.predictor.logp(yt, observed_y, z=zx).sum(-1)
            rec_y = torch.logsumexp(rec_y, dim=-1) - np.log(args.samples)

            score_i = rec_x + rec_y

            score.append(score_i)

            tqdm_batch.update()
        
        score = torch.cat(score, 0)     # test_samples x 1
        
        # ===== OUT OF distribution dataset ===== #
        tqdm_batch = tqdm(total=len(dataloader), desc='Batch', position=0, leave=False)

        ckpt_path_ood = find_path(args.dataset_ood, args.model,args.split,  'version_0')
        model_ood = load_model(args.model, ckpt_path_ood, device=device).eval()

        score_ood = []

        iterator = iter(dataloader_ood)
        for b in range(nbatches):
            batch = next(iterator)
            batch = [b.to(device) for b in batch]

            """# Get data
            x, observed_x, y, observed_y = batch
            x = x[:, :dim_x]
            observed_x = observed_x[:, :dim_x]
            y = y[:, :dim_y]
            observed_y = observed_y[:, :dim_y]

            batch = x, observed_x, y, observed_y"""

            #xn = model_ood.normalize_x(x)
            #xt, yt, xy, observed = model.preprocess_batch(batch)
            
            # Get data
            x, observed_x, y, observed_y = batch
            

            if not args.dataset.__contains__('mnist'):
                xn = model_ood.normalize_x(x)
                xt, yt, xy, observed = model_ood.preprocess_batch(batch)
                # shuffle batch
                #inds = torch.randperm(batch[0].shape[-1])
                xt = xt[:, :dim_x]
                xn = xn[:, :dim_x]
                observed_x = observed_x[:, :dim_x]

                #inds = torch.randperm(batch[2].shape[-1])
                #inds = np.arange(batch[2].shape[-1])
                yt = yt[:, :dim_y]
                observed_y = observed_y[:, :dim_y]

                x_tilde = torch.cat([xt*observed_x, observed_x], axis=1)
                y_tilde = torch.cat([yt*observed_y, observed_y], axis=1)
                if args.model.__contains__('SHVAE'):
                    model.HMC.logp = model.logp_func(xt*observed_x, observed_x, yt*observed_x, observed_y, xn)  

                xy = torch.cat([x_tilde, y_tilde], axis=1)
            else:
                xn = model.normalize_x(x)
                xt, yt, xy, observed = model.preprocess_batch(batch)
            

            mu_z, logvar_z = model.encoder(xy)
            # If the logvar is too big, the exp(logvar) might be infinite
            #logvar_z = torch.clamp(logvar_z, -50, 50)
            z = model.sample_z(mu_z, logvar_z, samples=args.samples)
            theta_x = model.decoder(z)
            x_hat = model.build_x_hat(xn, observed_x, theta_x)
            zx = torch.cat([z,x_hat],dim=-1)

            rec_x = model.decoder.logp(xt, observed_x, z=z, theta=theta_x).sum(-1)
            rec_x = torch.logsumexp(rec_x, dim=-1) - np.log(args.samples)

            rec_y = model.predictor.logp(yt, observed_y, z=zx).sum(-1)
            rec_y = torch.logsumexp(rec_y, dim=-1) - np.log(args.samples)
            
            score_i = rec_x + rec_y
            score_ood.append(score_i)
            tqdm_batch.update()
        
        score_ood = torch.cat(score_ood, 0)     # test_samples x 1

        results = {
            'score': score.detach().cpu().numpy(),
            'score_ood': score_ood.detach().cpu().numpy(),
        }  
        name = 'ood_{}_{}'.format(args.dataset_ood, args.split)
        np.save(ckpt_path.split('checkpoints', 1)[0] + name, results)

        plt.hist(results['score'], color='tab:blue', bins=50, alpha=0.4, label=args.dataset, density=True)
        plt.hist(results['score_ood'], color='tab:red', bins=50, alpha=0.4, label='{} (OoD)'.format(args.dataset_ood), density=True)
        plt.legend(loc='best', fontsize=16)

        if not os.path.isdir('{}experiments/figs/ood/'.format(LOGDIR)):
            os.mkdir('{}experiments/figs/ood/'.format(LOGDIR))

        plt.savefig('{}/experiments/figs/ood/{}_split_{}.pdf'.format(LOGDIR, args.model, args.split))
        
        # ===== Normalize score ===== #
        scores = torch.cat([score, score_ood])
        #scores = torch.sigmoid(scores).cpu().numpy()
        scores = scores.cpu().numpy()
        """scores = np.concatenate((results['score'], results['score_ood']))
        # make data positive
        scores = scores - scores.min()
        # put in [0, 1]
        scores = scores / scores.max()"""
        
        labels = np.concatenate((np.ones(len(results['score'])), np.zeros(len(results['score_ood'])) ))
        avpr = avpr(labels, scores)
        print('AVPR: {:f}'.format(avpr))
        