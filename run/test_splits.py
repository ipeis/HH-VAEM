import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir("..")
from src import *
import argparse

"""
When load is 0, this script performs the test evaluations on all the splits, sequentially, 
using the same GPU or CPU. It is preferrable to parallelize computations using your 
distributed system and calling the script test.py for a single split.
"""

parser = argparse.ArgumentParser(description='Average test evaluation on all the test splits')
parser.add_argument('--dataset', type=str, default="boston", metavar='N',
                    help='dataset')
parser.add_argument('--model', type=str, default="VAEM", metavar='N',
                    help='name of the model (VAE, HVAE or SHVAE)')
parser.add_argument('--version', type=str, default="version_0", metavar='N',
                    help='name of the model (VAE, HVAE or SHVAE)')
parser.add_argument('--load', type=int, default=1,
                    help='For loading pre computed split results (1) or computing first (0)')
parser.add_argument('--gpu', type=int, default=1,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()

ckpt_paths = find_splits_models(args.dataset, args.model, args.version)
  
if __name__ == '__main__':

    if args.load == 0:
        for s, ckpt_path in enumerate(ckpt_paths):
            print('Testing split {}...'.format(s))
            os.system("python test.py --dataset {:s} --model {:s} --version {:s} --split {:d} --gpu {:d}".format(args.dataset, args.model, args.version, s, args.gpu))

    metrics_splits = [np.load(ckpt_path.split('checkpoints', 1)[0] + 'test_metrics.npy', allow_pickle=True).tolist() for s, ckpt_path in enumerate(ckpt_paths)]

    metrics = {
            'll_y': [],
            'll_xu': [],
            'metric': [],
        }

    for key in metrics.keys():
        metrics[key] = np.concatenate([m[key][np.newaxis] for m in metrics_splits], axis=0)
    
    metrics_final = {
        'mean_ll_y': metrics['ll_y'].mean(axis=0),
        'std_ll_y': metrics['ll_y'].std(axis=0),
        'mean_ll_xu': metrics['ll_xu'].mean(axis=0),
        'std_ll_xu': metrics['ll_xu'].std(axis=0),
        'mean_metric': metrics['metric'].mean(axis=0),
        'std_metric': metrics['metric'].std(axis=0)
    }

    print(metrics_final)
    np.save(ckpt_paths[0].split('split_0')[0] + 'test_metrics_' + args.version, metrics_final)
    





