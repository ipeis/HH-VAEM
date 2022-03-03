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
parser = argparse.ArgumentParser(description='Performs the SAIA experiment on all the splits')
parser.add_argument('--dataset', type=str, default="boston",
                    help='dataset')
parser.add_argument('--model', type=str, default="VAEM",
                    help='name of the model (VAE, HVAE or SHVAE)')
parser.add_argument('--version', type=str, default="version_0",
                    help='name of the version')
parser.add_argument('--method', type=str, default="mi", metavar='N',
                    help='method to use for Information Reward (mi or kl)')     
parser.add_argument('--set', type=str, default="test",
                    help='train or test set')
parser.add_argument('--step', type=int, default=1, metavar='N',
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

