# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from src import *
import argparse
import warnings
warnings.filterwarnings("ignore")
import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# ============= ARGS ============= #

parser = argparse.ArgumentParser(description='Train discriminative models from sklearn')

parser.add_argument('--dataset', type=str, default='boston', 
                    help='dataset to train (boston, mnist, ...)')
parser.add_argument('--split', type=int, default=0,
                    help='train/test split index to use (default splits: 0, ..., 9)')
args = parser.parse_args()

model_name = clean_dataset(args.dataset) + '/missForest/' + 'split_' + str(args.split)
args.dataset = clean_dataset(args.dataset)      # for extracting 'fashion_mnist' from 'fashion_mnist_cnn'


if __name__ == '__main__':


    data_path = '{}/data/'.format(LOGDIR)

    imputer = MissForest()

    # ============= Dataset ============= #
    loader = get_dataset_loader(args.dataset, split='train', path=data_path, split_idx=args.split)
    X_tr = loader.dataset.data
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)

    observed_train = (np.random.rand(X_tr.shape[0], X_tr.shape[1]) > 0.5).astype(int)
    X_tr[np.logical_not(observed_train)] = np.nan
    y_tr = loader.dataset.labels
    loader = get_dataset_loader(args.dataset, split='test', path=data_path, batch_size=100, split_idx=args.split)
    X_test = loader.dataset.data
    X_test = scaler.transform(X_test)
    observed_test = (np.random.rand(X_test.shape[0], X_test.shape[1]) > 0.5).astype(int)
    X_test_ground_truth = X_test.copy()
    X_test[np.logical_not(observed_test)] = np.nan
    observed_test = loader.dataset.observed
    y_test = loader.dataset.labels
    

    # ============= TRAIN ============= #
    print('Training a missForest on split {:d} of {:s}'.format(args.split, args.dataset))
    imputer.fit(X_tr)

    # ============= TEST ============= #
    X_pred = imputer.transform(X_test)
    
    imputed = X_pred[np.logical_not(observed_test)]
    ground_truth = X_test_ground_truth[np.logical_not(observed_test)]
    rmse = np.sqrt(mean_squared_error(ground_truth, imputed))

    print('Imputation RMSE: {}'.format(rmse))

    metrics_np = {
            'll_y': np.array([0.0]),
            'll_xu': np.array([0.0]),
            'll_xu_d': np.array([0.0]),
            'll_xo': np.array([0.0]),
            'll_xo_d': np.array([0.0]),
            'metric': np.array([0.0]),
            'metric': np.array([0.0]),
            'error_xu': rmse
            }
    log_path = "{}/logs/{}/missForest/split_{}".format(LOGDIR, args.dataset, args.split)
    if not os.path.isdir(log_path):
        os.makedirs(log_path+ '/checkpoints/')
    np.save('{}/test_metrics'.format(log_path), metrics_np)


