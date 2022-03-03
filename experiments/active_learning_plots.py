
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
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
colors = list(mcolors.TABLEAU_COLORS)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 20
plt.tight_layout()

parser = argparse.ArgumentParser(description='Plot SAIA curves. Plots the VAEM + KL method and the input models + MI method')
parser.add_argument('--dataset', type=str, default='boston', metavar='N',
                    help='Dataset name')
parser.add_argument('--models',  nargs='+', type=str, default=('VAEM',),
                    help='models to plot')
parser.add_argument('--set', type=str, default="test",
                    help='train or test set')         

args = parser.parse_args()

dataset = args.dataset



models_list = args.models
metric_name = configs_active_learning[args.dataset]['metric']
metric_name = 'rmse' 
edit_names = {
    'VAEM': 'VAEM + ours',
    'VAEM_KL': 'VAEM + KL',
    'HHVAEM': 'HH-VAEM + ours'
}

models = [
    {
            'name': 'VAEM + KL',
            'paths': [
                '{logdir:s}/logs/{dataset:s}/VAEM/split_{split:d}/version_0/active_learning_kl_{set:s}.npy'.format(
                    logdir=LOGDIR,
                    dataset=dataset, split=s,
                    set=args.set) for s in range(4) 
            ]
        },
]
for model in models_list:
    models.append(
        {
        'name': edit_names[model],
        'paths': [
            '{logdir:s}/logs/{dataset:s}/{model_name:s}/split_{split:d}/{version:s}/active_learning_mi_{set:s}.npy'.format(
                logdir=LOGDIR,
                model_name=model, dataset=dataset, split=s, version="version_0", 
                set=args.set) for s in range(4) 
        ]
    },
        )

dim_x = configs[args.dataset]['VAEM']['dim_x']

f1, ax1 = plt.subplots(figsize=(6, 5))
f2, ax2 = plt.subplots(figsize=(6, 5))
for m, model in enumerate(models):
    metric_list = []
    rand_metric_list = []
    ll_list = []
    rand_ll_list = []
    for path in model['paths']:
        results = np.load(path, allow_pickle=True).tolist()
        metric_list.append(results['metric'])
        rand_metric_list.append(results['rand_metric'])
        ll_list.append(results['ll'])
        rand_ll_list.append(results['rand_ll'])
    
    # splits, dims
    mean_metric = np.mean(metric_list, axis=0)       # dims
    std_metric = np.std(metric_list, axis=0)   # dims

    mean_ll = np.mean(ll_list, axis=0)       # dims
    std_ll = np.std(ll_list, axis=0)   # dims
    
    mean_rand_metric = np.mean(rand_metric_list, axis=0)    # dims
    std_rand_metric = np.std(rand_metric_list, axis=0) # dims

    mean_rand_ll = np.mean(rand_ll_list, axis=0)    # dims
    std_rand_ll = np.std(rand_ll_list, axis=0) # dims
    
    # For plotting the error
    if args.dataset=='wine':
        y = 1 - mean_metric
    else: y=mean_metric

    step=int(np.ceil(dim_x / len(y)))
    x = np.arange(dim_x+step, step=step)
    x[x>dim_x] = dim_x
    err=std_metric
    marker_flag = len(x) < 50
    ax1.errorbar(x.astype(int), y, yerr=err, linewidth=3, color=colors[m], linestyle="-", fmt=marker_flag*'o', markersize=6, label=model['name'])

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('{:s} {:s}'.format(args.set, metric_name))
    ax1.legend(loc='best')
    plt.grid()

    if not os.path.isdir('{}/experiments/figs/al/'.format(LOGDIR)):
        os.mkdir('{}/experiments/figs/al/'.format(LOGDIR))
    f1.savefig('{}/experiments/figs/al/comp_{}_{}_{}.pdf'.format(LOGDIR, dataset, args.set, metric_name), bbox_inches='tight')
    
    y=mean_ll
    x = np.arange(dim_x+step, step=step).astype(int)
    x[x>dim_x] = dim_x
    err=std_ll
    ax2.errorbar(x.astype(int), y, yerr=err, linewidth=2, color=colors[m], linestyle="-", fmt=marker_flag*'o', markersize=6, label=model['name'])
    
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('{:s} log-likelihood'.format(args.set))
    ax2.legend(loc='best')
    plt.grid()

    f2.savefig('{}/experiments/figs/al/comp_{}_{}_ll.pdf'.format(LOGDIR, dataset, args.set), bbox_inches='tight')

