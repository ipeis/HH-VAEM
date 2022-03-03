

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir("../experiments/")
from src import *
import argparse
from sklearn.metrics import average_precision_score as avpr
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt

"""
This script performs the OoD experiment on all the splits, sequentially using the same 
GPU or CPU. It is preferrable to parallelize computations using your distributed system 
and calling the script ood.py for a single split.
"""
parser = argparse.ArgumentParser(description='OoD experiment')
parser.add_argument('--model', type=str, default="VAE", metavar='N',
                    help='name of the model')
parser.add_argument('--version', type=str, default="version_0",
                    help='name of the version')
parser.add_argument('--dataset', type=str, default="mnist", metavar='N',
                    help='dataset ON distribution')
parser.add_argument('--dataset_ood', type=str, default="fashion_mnist", metavar='N',
                    help='dataset OUT OF distribution')                   
parser.add_argument('--samples', type=int, default=100, metavar='N',
                    help='repetitions for each sample')
parser.add_argument('--load', type=int, default=1,
                    help='For loading pre computed split results (1) or computing first (0)')
parser.add_argument('--gpu', type=int, default=1,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()

ckpt_paths = find_splits_models(args.dataset, args.model, args.version)
    
if __name__ == '__main__':

    if args.load == 0:

        for s, ckpt_path in enumerate(ckpt_paths):
            print('OoD experiment, split {}...'.format(s))
            os.system("python ood.py --model {:s} --dataset {:s} --dataset_ood {:s} --samples {:d} --split {:d} --gpu {:d}".format(args.model, args.dataset, 
                args.dataset_ood, args.samples, s, args.gpu))

    results_list = [np.load(ckpt_path.split('checkpoints', 1)[0] + 'ood_{}_{}.npy'.format(args.dataset_ood, s), allow_pickle=True).tolist() for s, ckpt_path in enumerate(ckpt_paths)]

    avprs = []
    for split, results in enumerate(results_list):
        
        plt.hist(results['score'], color='tab:blue', bins=50, alpha=0.4, label=args.dataset, density=True)
        plt.hist(results['score_ood'], color='tab:red', bins=50, alpha=0.4, label='{} (OoD)'.format(args.dataset_ood), density=True)
        plt.legend(loc='best', fontsize=16)

        if not os.path.isdir('{}experiments/figs/ood/'.format(LOGDIR)):
            os.mkdir('{}experiments/figs/ood/'.format(LOGDIR))

        plt.savefig('{}/experiments/figs/ood/{}_split_{}.pdf'.format(LOGDIR, args.model, split))

        plt.clf()
        
        # ===== Normalize score ===== #
        scores = np.concatenate((results['score'], results['score_ood']))
        #scores = torch.Tensor(scores).to(device)
        #scores = torch.sigmoid(scores).cpu().numpy()

        """# make data positive
        scores = scores - scores.min()
        # put in [0, 1]
        scores = scores / scores.max()
        """
        labels = np.concatenate((np.ones(len(results['score'])), np.zeros(len(results['score_ood'])) ))

        avprs.append(avpr(labels, scores))

    ood_results = {
        'mean_avpr': np.mean(avprs),
        'std_avpr': np.std(avprs),
    }
    print('avpr: {:f} +- {:f}'.format(ood_results['mean_avpr'], ood_results['std_avpr']))
    np.save(ckpt_paths[0].split('split_0')[0] + 'ood_results_{}'.format(args.dataset_ood), ood_results)
    





