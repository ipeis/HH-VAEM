# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Note: there is an argument in train.py for doing this test at the end

from src import *
#from training import *
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

# ============= ARGS ============= #

parser = argparse.ArgumentParser(description='Test the HH-VAEM model and baselines')

parser.add_argument('--model', type=str, default='HHVAEM',
                    help='model to use (VAE, HVAE, HMCVAE, HHVAE, VAEM, HVAEM, HMCVAEM, HHVAEM)')
parser.add_argument('--dataset', type=str, default='boston', 
                    help='dataset to train (boston, mnist, ...)')
parser.add_argument('--split', type=int, default=0, metavar='N',
                    help='train/test split index to use (default splits: 0, ..., 9)')
parser.add_argument('--version', type=str, default='version_0', 
                    help='name for the log in Tensorboard (defaul None for "version_0")')
parser.add_argument('--gpu', type=int, default=1,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()

# ============= Activate CUDA ============= #
args.cuda = args.gpu and torch.cuda.is_available()
args.cuda = args.cuda == True and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if str(device) == "cuda":
    print('cuda activated')
    
    
if __name__ == '__main__':
    
    with torch.no_grad():
        
        # ============= TEST ============= #
        print('Testing a {:s} on split {:d} of {:s}'.format(args.model, args.split, args.dataset))
        ckpt_path = find_path(args.dataset, args.model,args.split, args.version)

        model = load_model(args.model, ckpt_path, device)
        model_name = model.dataset + '/' + args.model + '/' + 'split_' + str(args.split)

        trainer = pl.Trainer(
            gpus=args.gpu,
            default_root_dir='{}/logs/'.format(LOGDIR),
            logger=TensorBoardLogger(name=model_name, save_dir='{}/logs/'.format(LOGDIR), version=args.version),
        )

        trainer.test(model)
