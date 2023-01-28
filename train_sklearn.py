# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from pydoc import doc
from src import *
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import warnings
warnings.filterwarnings("ignore")

# ============= ARGS ============= #

parser = argparse.ArgumentParser(description='Train discriminative models from sklearn')

parser.add_argument('--model', type=str, default='MLPClassifier',
                    help='model to use (KNeighborsRegressor, KNeighborsClassifier, RandomForestRegressor, RandomForestClassifier, SVR, SVC, MLPRegressor, MLPClassifier)')
parser.add_argument('--dataset', type=str, default='bank', 
                    help='dataset to train (boston, mnist, ...)')
parser.add_argument('--imputation_method', type=str, default='mean', 
                    help='imputation method for missing data (zi for Zero Imputation or mean)')
parser.add_argument('--split', type=int, default=0,
                    help='train/test split index to use (default splits: 0, ..., 9)')
parser.add_argument('--test', type=int, default=1, 
                    help='testing at training end (1) or not (0)')   
args = parser.parse_args()

model_name = clean_dataset(args.dataset) + '/' + args.model + '/' + 'split_' + str(args.split)
args.model = clean_model(args.model)
config = get_config(args.model, args.dataset)
args.dataset = clean_dataset(args.dataset)      # for extracting 'fashion_mnist' from 'fashion_mnist_cnn'


def impute(data, missing, method):
    if method == 'zi':
        return data * np.logical_not(missing)
    elif method == 'mean':
        observed = np.logical_not(missing)
        for var in range(data.shape[-1]):
            mean = data[observed[:, var], var].mean()
            data[missing[:, var], var] = mean
            return data

if __name__ == '__main__':

    config.pop('dataset')
    metric_name = config['metric']
    config.pop('metric')
    model = create_model(args.model, config)
    data_path = '{}/data/'.format(LOGDIR)

    # ============= Dataset ============= #
    loader = get_dataset_loader(args.dataset, split='train', path=data_path, batch_size=100, split_idx=args.split)
    X_tr = loader.dataset.data
    observed_train = (np.random.rand(X_tr.shape[0], X_tr.shape[1]) > 0.5).astype(int)
    X_tr = impute(X_tr, np.logical_not(observed_train), args.imputation_method)
    y_tr = loader.dataset.labels
    loader = get_dataset_loader(args.dataset, split='test', path=data_path, batch_size=100, split_idx=args.split)
    X_test = loader.dataset.data
    observed_test = loader.dataset.observed
    X_test= impute(X_test, np.logical_not(observed_test), args.imputation_method)
    y_test = loader.dataset.labels
    

    # ============= TRAIN ============= #
    print('Training a {:s} on split {:d} of {:s}'.format(args.model, args.split, args.dataset))
    model.fit(X_tr, y_tr)

    # ============= TEST ============= #
    if args.test==1:
        with torch.no_grad():
            if metric_name == 'rmse':
                metric = np.sqrt(np.mean((y_test - model.predict(X_test))**2) )
            else:
                metric = model.score(X_test, y_test)

            print('{}: {}'.format(metric_name, metric))

            metrics_np = {
                'll_y': np.array([0.0]),
                'll_xu': np.array([0.0]),
                'metric': metric
            }
            log_path = "{}/logs/{}/{}/split_{}/{}".format(LOGDIR, args.dataset, args.model, args.split, args.imputation_method)
            if not os.path.isdir(log_path):
                os.makedirs(log_path+ '/checkpoints/')
            np.save('{}/test_metrics'.format(log_path), metrics_np)




for i in boston energy wine concrete yatch;
do
for j in MIWAEM;
do
python test_splits.py --dataset $i --model $j
done
done





for i in boston energy wine concrete diabetes yatch naval avocado insurance bank;
do
python test_splits_xo_d.py --dataset $i
done








boston wine concrete yatch;