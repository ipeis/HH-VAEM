

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir("..")
from src import *
import argparse

parser = argparse.ArgumentParser(description='Perform test evaluation on all the test splits')

parser.add_argument('--datasets',  nargs='+', type=str, default=('boston', 'energy', 'wine'),
                    help='models to plot')
parser.add_argument('--models',  nargs='+', type=str, default=('VAE', 'HVAE', 'SHVAE', 'VAEM', 'HVAEM', 'SHVAEM', 'HierVAE'),
                    help='models to plot')
args = parser.parse_args()



    
if __name__ == '__main__':

    table_ll_y = {}
    table_ll_xu = {}
    table_metric = {}
    for m in args.models:
        table_ll_y[m] = []
        table_ll_xu[m] = []
        table_metric[m] = []

    for dataset in args.datasets:
        for model in args.models:
            path = '{}/logs/{:s}/{:s}/'.format(LOGDIR, dataset, model)

            metrics = np.load(path + 'test_metrics_version_0.npy', allow_pickle=True).tolist()

            ll_y_str = '${:.4f} \\pm {:.4f}$'.format(metrics['mean_ll_y'], metrics['std_ll_y'])
            ll_xu_str = '${:.4f} \\pm {:.4f}$'.format(metrics['mean_ll_xu'], metrics['std_ll_xu'])
            metric_str = '${:.4f} \\pm {:.4f}$'.format(metrics['mean_metric'], metrics['std_metric'])

            table_ll_y[model].append(ll_y_str)
            table_ll_xu[model].append(ll_xu_str)
            table_metric[model].append(metric_str)

    table_ll_y_pd = pd.DataFrame(data=table_ll_y, index=args.datasets)
    table_ll_xu_pd = pd.DataFrame(data=table_ll_xu, index=args.datasets)
    table_metric_pd = pd.DataFrame(data=table_metric, index=args.datasets)

    table_ll_y_pd.to_csv('experiments/tables/table_ll_y.csv')
    table_ll_xu_pd.to_csv('experiments/tables/table_ll_xu.csv')
    table_metric_pd.to_csv('experiments/tables/table_metric.csv')

    print('\nlogp(y)')
    print(table_ll_y_pd)
    print('\nlogp(xu)')
    print(table_ll_xu_pd)
    print('\nMetric')
    print(table_metric_pd)



