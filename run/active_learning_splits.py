# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir("../experiments/")
from src import *
import argparse

"""
This script performs the SAIA experiment on all the splits, sequentially using the same 
GPU or CPU. It is preferrable to parallelize computations using your distributed system 
and calling the script active_learning.py for a single split.
"""
parser = argparse.ArgumentParser(description='Performs the SAIA experiment on all the test splits')
parser.add_argument('--model', type=str, default='HHVAEM',
                    help='model to use (VAE, HVAE, HMCVAE, HHVAE, VAEM, HVAEM, HMCVAEM, HHVAEM)')
parser.add_argument('--dataset', type=str, default='boston',
                    help='dataset to train (boston, mnist, ...)')
parser.add_argument('--version', type=str, default=None,
                    help='name for the log in Tensorboard (defaul None for "version_0")')
parser.add_argument('--method', type=str, default="mi",
                    help='method to use for Information Reward (mi or kl)')     
parser.add_argument('--set', type=str, default="test",
                    help='train or test set')
parser.add_argument('--step', type=int, default=1,
                    help='variables to add within each step')                  
parser.add_argument('--gpu', type=int, default=1,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()

ckpt_paths = find_splits_models(args.dataset, args.model, args.version)

config = configs_active_learning[args.dataset]

for s, ckpt_path in enumerate(ckpt_paths):
    print('Split {:d}:'.format(s))
    os.system("python active_learning.py --dataset {:s} --model {:s} --method {:s} --split {:d} --set {:s} --gpu {:d}".format(
        args.dataset, args.model, args.method, s, args.set, config['samples'],  config['bins'], config['step'], args.gpu))

