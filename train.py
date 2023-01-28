# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from src import *
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

from src.callbacks import ImageCondSampler

# ============= ARGS ============= #

parser = argparse.ArgumentParser(description='Train the HH-VAEM model and baselines')

parser.add_argument('--model', type=str, default='HHVAEM',
                    help='model to use (VAE, MIWAE, HVAE, HMCVAE, HHVAE, VAEM, MIWAEM, HVAEM, HMCVAEM, HHVAEM)')
parser.add_argument('--dataset', type=str, default='boston', 
                    help='dataset to train (boston, mnist, ...)')
parser.add_argument('--split', type=int, default=0,
                    help='train/test split index to use (default splits: 0, ..., 9)')
parser.add_argument('--version', type=str, default=None, 
                    help='name for the log in Tensorboard (defaul None for "version_0")')
parser.add_argument('--test', type=int, default=1, 
                    help='testing at training end (1) or not (0)')   
parser.add_argument('--gpu', type=int, default=1,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()


# ============= Activate CUDA ============= #
args.cuda = int(args.gpu>0) and torch.cuda.is_available()
args.cuda = args.cuda == True and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if str(device) == "cuda":
    print('cuda activated')

model_name = clean_dataset(args.dataset) + '/' + args.model + '/' + 'split_' + str(args.split)
args.model = clean_model(args.model)
config = get_config(args.model, args.dataset)
args.dataset = clean_dataset(args.dataset)      # for extracting 'fashion_mnist' from 'fashion_mnist_cnn'

config['split_idx'] = args.split
epochs = config['epochs']
config.pop('epochs')
config['data_path'] = '{}/data/'.format(LOGDIR)

if __name__ == '__main__':

    model = create_model(args.model, config)

    # ============= TRAIN ============= #
    print('Training a {:s} on split {:d} of {:s}'.format(args.model, args.split, args.dataset))
    trainer = pl.Trainer(
        max_epochs=epochs,
        #callbacks=[
        #    ImageSampler(channels=3, size=64),
        #    ImageReconstruction(channels=3, size=64),
            #Inpainting(),
            #ImageCondSampler()
        #    ],
        gpus=args.gpu,
        default_root_dir='{}/logs/'.format(LOGDIR),
        logger=TensorBoardLogger(name=model_name, save_dir='{}/logs/'.format(LOGDIR), version=args.version),
        num_sanity_val_steps=0
    )
    trainer.fit(model)

    # ============= TEST ============= #
    if args.test==1:
        with torch.no_grad():
            trainer.test()