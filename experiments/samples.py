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

parser = argparse.ArgumentParser(description='Sampling images')

parser.add_argument('--models', nargs='+', type=str, default=('VAE', 'HVAE', 'HMCVAE', 'HHVAE'),
                    help='model to use (VAE, HVAE, HMCVAE, HHVAE, VAEM, HVAEM, HMCVAEM, HHVAEM)')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='dataset to train (boston, mnist, ...)')
parser.add_argument('--split', type=int, default=0,
                    help='train/test split index to use (default splits: 0, ..., 9)')
parser.add_argument('--version', type=str, default='version_0', 
                    help='name for the log in Tensorboard (defaul None for "version_0")')
parser.add_argument('--method', type=str, default="mi",
                    help='method to use for Information Reward (mi or kl)')  
parser.add_argument('--set', type=str, default='test',
                    help='train or test set')
parser.add_argument('--gpu', type=int, default=1,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()

args.cuda = args.gpu and torch.cuda.is_available()
args.cuda = args.cuda == True and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if str(device) == "cuda":
    print('cuda activated')

args.dataset = clean_dataset(args.dataset)

if args.dataset == 'celeba':
    channels = 3
    size = 64
    wind = 16
elif args.dataset == 'mnist':
    channels = 1
    size = 28
    wind = 7

samples = 100

start = time.time()
if __name__ == '__main__':

    with torch.no_grad():

        for m, model in enumerate(args.models):
            results = []
            ckpt_path = find_path(args.dataset, model, args.split, args.version)
            model = clean_model(model)
            pl_module = load_model(model, ckpt_path, device).eval()
            
            mixed = None
            if model.__contains__('VAEM'):
                mixed = True

            dataloader = get_dataset_loader(pl_module.dataset, args.set, path=pl_module.data_path, batch_size=pl_module.batch_size, split_idx=pl_module.split_idx)
            
            batch_idx = 4
            if m==0:
                iterator = iter(dataloader)
                for b in range(batch_idx):
                    batch = next(iterator)
                batch = [b[:10].to(pl_module.device) for b in batch]

                # Missing mask
                mask = torch.zeros_like(batch[0])
                center = np.random.randint(wind, size-wind, 2)
                mask = mask.reshape(-1, channels, size, size)
                mask[:, :, center[0]-wind:center[0]+wind, center[1]-wind:center[1]+wind] = 1
                mask = mask.reshape(-1, channels*size*size)

                observed_x = torch.logical_not(mask)

                # Get data
                x, _, y, observed_y = batch
                # put sunglasses
                #attr_idx = 16
                #y[:, attr_idx] = 1
                #observed_y[:, attr_idx] = 1

                original = x.clone().detach()

                batch = (x, observed_x, y, observed_y)
            
            xn = pl_module.normalize_x(x)
            xt, yt, xy, observed = pl_module.preprocess_batch(batch) 
            # xt is the preprocessed input (xt=x if no preprocessing)
            # observed is observed_x OR observed_y (for not using kl if no observed data)
            mu_z, logvar_z = pl_module.encoder(xy)

            z = pl_module.sample_z(mu_z, logvar_z, samples=samples)
            
            if isinstance(z, list):
                z = z[0]
            
            reconst = torch.sigmoid(pl_module.decoder(z)).reshape(10, samples, -1)
            reconst = reconst.mean(1)

            pl_module.train()
        
            images = x*observed_x
                
            original = original[:20].reshape(-1, channels, size, size)
            images = images[:20].reshape(-1, channels, size, size)
            reconst = reconst[:20].reshape(-1, channels, size, size)

            inpainted = images.clone().detach()
            mask = mask[:10].reshape_as(inpainted)
            inpainted[mask.bool()] = reconst[mask.bool()]

            if m==0:
                all = torch.cat([original, images, inpainted])
            else:
                all = torch.cat([all, inpainted])

        #all = torch.cat([images, reconst, inpainted])

    
    grid = torchvision.utils.make_grid(
            tensor=all,
            nrow=10,
            padding=2,
            normalize=False,
            range=None,
            scale_each=False,
            pad_value=0,
        )

    str_title = f"{pl_module.__class__.__name__}_inpainting"
    torchvision.utils.save_image(grid, 'experiments/figs/inpainting.png')
    
