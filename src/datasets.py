# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from abc import abstractmethod
import h5py
import torch
import torch.utils.data as data
from torchvision import transforms
import os
import numpy as np
import urllib.request
from sklearn.preprocessing import LabelEncoder
import torchvision
from sklearn.datasets import load_boston, load_wine, load_diabetes
from torch import nn
import pandas as pd
from sklearn.model_selection import KFold
import zipfile

# Number of splits per dataset
global SPLITS 
SPLITS = 10


# ============= Base Dataset Class ============= #
class BaseDataset(data.Dataset):
    """
    Base Dataset class
    """
    def __init__(self, root='data/', name='dataset', train=True, transform=None, 
                download=False, test_missing_rate=0.5, split_idx=0, dim:int=None, mixed=False, dim_y=1):
        """
        Initialization of the Dataset

        Args:
            root (str, optional): Root directory. Defaults to 'data/'.
            name (str, optional): Name of the dataset. Defaults to 'dataset'.
            train (bool, optional): Train (True) or Test (False) set. Defaults to True.
            transform (_type_, optional): Transformation to be applied. Defaults to None.
            download (bool, optional): Download (True) if necessary. Defaults to False.
            test_missing_rate (float, optional): rate for fixed test missing data. Defaults to 0.5.
            split_idx (int, optional): split index (can be 0, ..., SPLITS). Defaults to 0.
            dim (_type_, int): dimension (column) to extract. Defaults to None (full matrix).
            mixed (bool, optional): preprocessing as mixed-type (True) or Gaussian (False) data. Defaults to False.
            dim_y (int, optional): dimension of the target. Defaults to 1.
        """

        self.root = os.path.expanduser(root + '/{:s}'.format(name))
        self.train = train 
        self.dim_y = dim_y
        self.test_missing_rate = test_missing_rate
        self.split_idx = split_idx
        self.dim = dim    
        self.mixed = mixed  

        if download and not self._check_exists():
            self.download()
            self.split_dataset()

        self.data, self.labels = self._get_data(train=train)

    
    def __getitem__(self, index: int) -> tuple:
        """
        Select a sample from the dataset

        Args:
            index (int): row for selecting one sample

        Returns:
            x: sample data
            observed_x: observed mask for data
            y: target
            observed_y: observed mask for target
        """
        x = self.data[index]
        y = self.labels[index]

        if self.train:
            missing_rate = np.random.rand(1) * 0.99
            observed_y = (np.random.rand(y.shape[0]) > missing_rate)
            observed_x = (np.random.rand(x.shape[0]) > missing_rate)
        else:
            observed_y = y * 0 # in test we do not observe any label
            observed_x = self.observed[index]

        if self.dim != None:
            x = x[self.dim].reshape(-1)
            observed_x = observed_x[self.dim].reshape(-1)

        return x, observed_x, y, observed_y

    def __len__(self) -> int:
        """
        Length N of the dataset 

        Returns:
            (int): Number of samples of the dataset
        """
        return len(self.data)

    def _get_data(self, train=True) -> tuple:
        """
        Load split data from root folder (previously downloaded and saved). Preprocess if mixed-type data

        Args:
            train (bool, optional): Train (True) or Test (False) splits. Defaults to True.

        Returns:
            (float32): dataset  (N  x   D)
            (float32): target     (N  x   P)
        """
        with h5py.File(os.path.join(self.root, 'data{:d}.h5'.format(self.split_idx)), 'r') as hf:
            data = hf.get('train' if train else 'test')
            data = np.array(data)

        labels = data[:, -self.dim_y:]
        data = data[:, :-self.dim_y]

        observed_file = 'test_observed{}.npy'.format(self.split_idx)
        if os.path.exists(os.path.join(self.root, observed_file)):
            observed = np.load(os.path.join(self.root, observed_file))
            self.observed = observed.astype(np.float32)    

        # If mixed-type data, encode categorical columns
        if self.mixed:

            # Load all data
            with h5py.File(os.path.join(self.root, 'data.h5'.format(self.split_idx)), 'r') as hf:
                dataset = hf.get('data')
                dataset = np.array(dataset)

            dataset = dataset
                
            encoded  = []
            data_types = []
            categories = []
            for variable in range(data.shape[-1]):
                column = dataset[:, variable]
                column_split = data[:, variable]
                # Less than 10 values --> Discrete data
                if len(np.unique(column)) < 10:
                    le = LabelEncoder()
                    encoded_var_all = le.fit_transform(column)
                    encoded_var = le.transform(column_split)[:, np.newaxis]
                    encoded.append(encoded_var)
                    ncats = len(np.unique(encoded_var_all))
                    
                    if ncats==2:
                        data_types.append('bernoulli')
                        categories.append(1)
                    else:
                        data_types.append('categorical')
                        categories.append(ncats)
                # Real positive
                elif np.sum(column < 0) == 0:
                    encoded.append(column_split[:, np.newaxis])
                    categories.append(1)
                    data_types.append('loggaussian')
                # Real
                else:
                    encoded.append(column_split[:, np.newaxis])
                    categories.append(1)
                    data_types.append('gaussian')

            data = np.concatenate(encoded, -1)

        return data.astype(np.float32), labels.astype(np.float32)

    @abstractmethod
    def download(self):
        """
        This function downloads the dataset and stores in the root folder
        """
        pass

    def split_dataset(self, nsplits:int=SPLITS):
        """
        Partinionate the dataset in SPLITS number of splits, save splits in the root folder, create artificial observation mask

        Args:
            nsplits (_type_, optional): number of splits. Defaults to SPLITS.
        """
        with h5py.File(os.path.join(self.root, 'data.h5'), 'r') as hf:
            X = hf.get('data')
            X = np.array(X)
            y = hf.get('target')
            y = np.array(y)
        
        kf = KFold(n_splits=nsplits, shuffle=True)
        s=0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


            data = {
                'train': np.concatenate((X_train, y_train), axis=-1),
                'test': np.concatenate((X_test, y_test), axis=-1),
            }

            with h5py.File(os.path.join(self.root, 'data{}.h5'.format(s)), 'w') as hf:
                hf.create_dataset('train', data=data['train'].reshape(-1, data['train'].shape[-1]))
                hf.create_dataset('test', data=data['test'].reshape(-1, data['train'].shape[-1]))

            observed_file = 'test_observed{}.npy'.format(s)
            observed = (np.random.rand(X_test.shape[0], X_test.shape[1]) > self.test_missing_rate).astype(int)
            local_filename = os.path.join(self.root, observed_file)
            np.save(local_filename, observed)
            s+=1
        print('Done!')

    def _check_exists(self) -> bool:
        """
        Check if dataset exists

        Returns:
            (bool): True if the dataset is already in the root folder
        """
        return os.path.exists(os.path.join(self.root, 'data.h5'))




# ============= UCI datasets ============= #
class BostonHousing(BaseDataset):

    def __init__(self, root='data/boston', train=True, transform=None, download=False, test_missing_rate=0.5, split_idx=0, dim=None, mixed=False):
        super(BostonHousing, self).__init__(root=root, train=train, transform=transform, 
            download=download, test_missing_rate=test_missing_rate, 
            split_idx=split_idx, dim=dim, mixed=mixed)


    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading boston...')
        X, y = load_boston(return_X_y=True)

        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 13))
            hf.create_dataset('target', data=y.reshape(-1, 1))


class Energy(BaseDataset):
   
    def __init__(self, root='data/energy', train=True, transform=None, download=False, test_missing_rate=0.5, split_idx=0, dim=None, mixed=False):
        super(Energy, self).__init__(root=root, train=train, transform=transform, 
            download=download, test_missing_rate=test_missing_rate, 
            split_idx=split_idx, dim=dim, mixed=mixed, dim_y=2)


    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading energy...')

        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        file_name = url.split('/')[-1]
        if not os.path.exists(self.root + file_name):

            urllib.request.urlretrieve(
                url, self.root + file_name)
        data_pd = pd.read_excel(self.root + 'ENB2012_data.xlsx',
                             header=0).values
        data_pd = data_pd[np.random.permutation(np.arange(len(data_pd)))]

        X = data_pd[:, :-2]
        y = data_pd[:, -2:]

        print('Done!')

        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 8))
            hf.create_dataset('target', data=y.reshape(-1, 2))


class Wine(BaseDataset):

    def __init__(self, root='data/wine', train=True, transform=None, download=False, test_missing_rate=0.5, split_idx=0, dim=None, mixed=False):
        super(Wine, self).__init__(root=root, train=train, transform=transform, 
            download=download, test_missing_rate=test_missing_rate, 
            split_idx=split_idx, dim=dim, mixed=mixed)


    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading wine...')
        X, y = load_wine(return_X_y=True)
        print('Done!')
        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 13))
            hf.create_dataset('target', data=y.reshape(-1, 1))


class Diabetes(BaseDataset):

    def __init__(self, root='data/diabetes', train=True, transform=None, download=False, test_missing_rate=0.5, split_idx=0, dim=None, mixed=False):
        super(Diabetes, self).__init__(root=root, train=train, transform=transform, 
            download=download, test_missing_rate=test_missing_rate, 
            split_idx=split_idx, dim=dim, mixed=mixed)

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading diabetes...')
        X, y = load_diabetes(return_X_y=True)

        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 10))
            hf.create_dataset('target', data=y.reshape(-1, 1))


class Concrete(BaseDataset):

    def __init__(self, root='data/concrete', train=True, transform=None, download=False, test_missing_rate=0.5, split_idx=0, dim=None, mixed=False):
        super(Concrete, self).__init__(root=root, train=train, transform=transform, 
            download=download, test_missing_rate=test_missing_rate, 
            split_idx=split_idx, dim=dim, mixed=mixed)


    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading concrete...')

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
        file_name = url.split('/')[-1]
        if not os.path.exists(self.root + file_name):
            urllib.request.urlretrieve(
                url, self.root + file_name)
        data_pd = pd.read_excel(self.root + 'Concrete_Data.xls',
                             header=0).values
        data_pd = data_pd[np.random.permutation(np.arange(len(data_pd)))]

        X = data_pd[:, :-1]
        y = data_pd[:, -1]

        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 8))
            hf.create_dataset('target', data=y.reshape(-1, 1))


class Naval(BaseDataset):

    def __init__(self, root='data/naval', train=True, transform=None, download=False, test_missing_rate=0.5, split_idx=0, dim=None, mixed=False):
        super(Naval, self).__init__(root=root, train=train, transform=transform, 
            download=download, test_missing_rate=test_missing_rate, 
            split_idx=split_idx, dim=dim, mixed=mixed, dim_y=2)

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading naval...')

        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip'
        file_name = url.split('/')[-1]
        print('Downloading from {}...'.format(url))
        local_filename = os.path.join(self.root, file_name)
        if not os.path.exists(self.root + file_name):
            urllib.request.urlretrieve(url, local_filename)
        with zipfile.ZipFile(os.path.join(self.root, file_name), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.root, file_name.split('.zip')[0]))

        data_pd = pd.read_csv(self.root + '/' + file_name.split('.zip')[0] + '/UCI CBM Dataset/data.txt', header=None, delimiter='\s+').values
        data_pd = data_pd[np.random.permutation(np.arange(len(data_pd)))]

        # column 8 is constant
        data_pd = np.delete(data_pd, [8, 11], 1)
        X = data_pd[:, :-2]
        y = data_pd[:, -2:]

        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 14))
            hf.create_dataset('target', data=y.reshape(-1, 2))


class Yatch(BaseDataset):

    def __init__(self, root='data/yatch', train=True, transform=None, download=False, test_missing_rate=0.5, split_idx=0, dim=None, mixed=False):
        super(Yatch, self).__init__(root=root, train=train, transform=transform, 
            download=download, test_missing_rate=test_missing_rate, 
            split_idx=split_idx, dim=dim, mixed=mixed, dim_y=1)

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading yatch...')

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
        file_name = url.split('/')[-1]
        if not os.path.exists(self.root + file_name):
            urllib.request.urlretrieve(
                url, self.root + file_name)
        data_pd = pd.read_csv(self.root + 'yacht_hydrodynamics.data', sep='\s+').values
        data_pd = data_pd[np.random.permutation(np.arange(len(data_pd)))]

        X = data_pd[:, :-1]
        y = data_pd[:, -1]

        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 6))
            hf.create_dataset('target', data=y.reshape(-1, 1))


class Avocado(BaseDataset):

    def __init__(self, root='data/avocado', train=True, transform=None, download=False, test_missing_rate=0.5, split_idx=0, dim=None, mixed=False):
        super(Avocado, self).__init__(root=root, train=train, transform=transform, 
            download=download, test_missing_rate=test_missing_rate, 
            split_idx=split_idx, dim=dim, mixed=mixed, dim_y=1)


    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading avocado...')
        url = "https://raw.githubusercontent.com/MayankkYogi/Avocado/master/avocado.csv"

        data = pd.read_csv(url, index_col=0)

        # encode text variables
        data["region"] = data["region"].astype('category')
        data["region"] = data["region"].cat.codes
        data["Date"] = data["Date"].astype('category')
        data["Date"] = data["Date"].cat.codes
        data["type"] = data["type"].astype('category')
        data["type"] = data["type"].cat.codes

        y = data['AveragePrice'].values
        X = data.drop('AveragePrice', axis=1).values

        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 12))
            hf.create_dataset('target', data=y.reshape(-1, 1))


class Insurance(BaseDataset):

    def __init__(self, root='data/insurance', train=True, transform=None, download=False, test_missing_rate=0.5, split_idx=0, dim=None, mixed=False):
        super(Insurance, self).__init__(root=root, train=train, transform=transform, 
            download=download, test_missing_rate=test_missing_rate, 
            split_idx=split_idx, dim=dim, mixed=mixed, dim_y=1)

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading insurance...')
        # Train set
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt"
        file_name = url.split('/')[-1]
        if not os.path.exists(self.root + file_name):
            urllib.request.urlretrieve(
                url, self.root + file_name)
        data_pd = pd.read_csv(self.root + file_name, sep='\s+').values
        data_pd = data_pd[np.random.permutation(np.arange(len(data_pd)))]

        X = data_pd[:, :-1]
        y = data_pd[:, -1]

        # Test set
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt"
        file_name = url.split('/')[-1]
        if not os.path.exists(self.root + file_name):
            urllib.request.urlretrieve(
                url, self.root + file_name)
        data_test_pd = pd.read_csv(self.root + file_name, sep='\s+').values
        X_test = data_test_pd

        # Test labels
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tictgts2000.txt"
        file_name = url.split('/')[-1]
        if not os.path.exists(self.root + file_name):
            urllib.request.urlretrieve(
                url, self.root + file_name)
        labels_test_pd = pd.read_csv(self.root + file_name, sep='\s+').values
        y_test = labels_test_pd.squeeze()

        # We mix the original train+test sets
        X = np.concatenate((X, X_test), axis=0)
        y = np.concatenate((y, y_test), axis=0)
        
        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 85))
            hf.create_dataset('target', data=y.reshape(-1, 1))


class Bank(BaseDataset):

    def __init__(self, root='data/bank', train=True, transform=None, download=False, test_missing_rate=0.5, split_idx=0, dim=None, mixed=False):
        super(Bank, self).__init__(root=root, train=train, transform=transform, 
            download=download, test_missing_rate=test_missing_rate, 
            split_idx=split_idx, dim=dim, mixed=mixed, dim_y=1)


    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading bank...')

        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
        file_name = url.split('/')[-1]
        print('Downloading from {}...'.format(url))
        local_filename = os.path.join(self.root, file_name)
        if not os.path.exists(self.root + file_name):
            urllib.request.urlretrieve(url, local_filename)
        with zipfile.ZipFile(os.path.join(self.root, file_name), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.root, file_name.split('.zip')[0]))

        data = pd.read_csv(self.root + '/' + file_name.split('.zip')[0] + '/bank-additional/bank-additional-full.csv', header=0, sep=';')
        for column in data.keys():
            if data[column].dtype.name == 'object':
                data[column] = data[column].astype('category').cat.codes

        X = data.values[:, :-1]
        y = data.values[:, -1]

        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 20))
            hf.create_dataset('target', data=y.reshape(-1, 1))



# ============= MNIST datasets ============= #
class MNIST(BaseDataset):

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading MNIST...')
        print('Downloading MNIST with fixed binarization...')
        data = {}
        labels = {}
        for split in ['train', 'valid', 'test']:
            dataset = torchvision.datasets.MNIST(train=split=='train', download=True, root=self.root)
            data[split] = dataset.data.reshape(-1, 28**2)
            labels[split] = dataset.targets.reshape(-1, 1)
        train_data = np.concatenate((data['train'], labels['train']), -1)
        test_data = np.concatenate((data['test'], labels['test']), -1)


        X = train_data[:, :-1]
        X = self.normalize(X)
        X = self.binarize(X)
        y = train_data[:, -1]
        
        # Test set
        X_test = test_data[:, :-1]
        X_test = self.normalize(X_test)
        X_test = self.binarize(X_test)
        y_test = test_data[:, -1]

        print('Done!')

        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 28**2))
            hf.create_dataset('target', data=y.reshape(-1, 1))

        with h5py.File(os.path.join(self.root, 'data_test.h5'), 'w') as hf:
            hf.create_dataset('data', data=X_test.reshape(-1, 28**2))
            hf.create_dataset('target', data=y_test.reshape(-1, 1))


    def binarize(self, data):
        return (data>0.5).astype(float)

    def normalize(self, data):
        return data/256


class FashionMNIST(BaseDataset):

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading Fashion MNIST...')
        data = {}
        labels = {}
        for split in ['train', 'valid', 'test']:
            dataset = torchvision.datasets.FashionMNIST(train=split=='train', download=True, root=self.root)
            data[split] = dataset.data.reshape(-1, 28**2)
            labels[split] = dataset.targets.reshape(-1, 1)
        train_data = np.concatenate((data['train'], labels['train']), -1)
        test_data = np.concatenate((data['test'], labels['test']), -1)


        X = train_data[:, :-1]
        X = self.normalize(X)
        X = self.binarize(X)
        y = train_data[:, -1]
        
        # Test set
        X_test = test_data[:, :-1]
        X_test = self.normalize(X_test)
        X_test = self.binarize(X_test)
        y_test = test_data[:, -1]

        print('Done!')

        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('data', data=X.reshape(-1, 28**2))
            hf.create_dataset('target', data=y.reshape(-1, 1))

        with h5py.File(os.path.join(self.root, 'data_test.h5'), 'w') as hf:
            hf.create_dataset('data', data=X_test.reshape(-1, 28**2))
            hf.create_dataset('target', data=y_test.reshape(-1, 1))


    def binarize(self, data):
        return (data>0.5).astype(float)

    def normalize(self, data):
        return data/256






# ============= Aux ============= #
class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """
    Loader for avoiding queues
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class View(nn.Module):
    """ For reshaping tensors inside Sequential objects"""
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshape tensor to a given size

        Args:
            tensor (torch.Tensor): original tensor

        Returns:
            torch.Tensor: reshaped tensor
        """
        return tensor.view(self.size)




# ============= Dataloaders ============= #
def get_dataset_loader(dataset: str, split='train', path='../data/', batch_size=128, num_workers=4, **kwargs) -> MultiEpochsDataLoader:

    """
    Function that maps datasets into loaders

    Args:
        dataset (str): name of the dataset
        split (str, optional): split of the dataset ('train' or 'test'). Defaults to 'train'.
        path (str, optional): root folder for store/load data. Defaults to '../data/'.
        batch_size (int, optional): batch size. Defaults to 128.
        num_workers (int, optional): _description_. Defaults to 4.

    Returns:
        (MultiEpochsDataLoader): train/test dataloader
    """

    dim = kwargs['dim'] if 'dim' in kwargs else None
    mixed = kwargs['mixed'] if 'mixed' in kwargs else None

    if dataset == 'boston':
        if split=='train':
            data = BostonHousing(root=path + 'boston/', download=True, train=True,
                                         transform=transforms.ToTensor(), 
                                         split_idx=kwargs['split_idx'], dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        elif split=='test':
            data = BostonHousing(root=path + 'boston/', download=True, train=False,
                                  transform=transforms.ToTensor(),
                                  split_idx=kwargs['split_idx'], dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return loader

    if dataset == 'energy':
        if split=='train':
            data = Energy(root=path + 'energy/', download=True, train=True,
                            transform=transforms.ToTensor(),
                            split_idx=kwargs['split_idx'], dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        elif split=='test':
            data = Energy(root=path + 'energy/', download=True, train=False,
                            transform=transforms.ToTensor(),
                            split_idx=kwargs['split_idx'], dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return loader

    if dataset == 'wine':
        if split=='train':
            data = Wine(root=path + 'wine/', download=True, train=True,
                            transform=transforms.ToTensor(),
                            split_idx=kwargs['split_idx'], dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        elif split=='test':
            data = Wine(root=path + 'wine/', download=True, train=False,
                            transform=transforms.ToTensor(),
                            split_idx=kwargs['split_idx'], dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False, )
        return loader

    if dataset == 'diabetes':
        if split=='train':
            data = Diabetes(root=path + 'diabetes/', download=True, train=True,
                                        transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                                        dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        elif split=='test':
            data = Diabetes(root=path + 'diabetes/', download=True, train=False,
                                transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                                dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return loader
        
    if dataset == 'avocado':
        if split=='train':
            data = Avocado(root=path + 'avocado/', download=True, train=True,
                                        transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                                        dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        elif split=='test':
            data = Avocado(root=path + 'avocado/', download=True, train=False,
                                transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                                dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return loader

    if dataset == 'concrete':
        if split=='train':
            data = Concrete(root=path + 'concrete/', download=True, train=True,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        elif split=='test':
            data = Concrete(root=path + 'concrete/', download=True, train=False,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False, )
        return loader

    if dataset == 'naval':
        if split=='train':
            data = Naval(root=path + 'naval/', download=True, train=True,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
        elif split=='test':
            data = Naval(root=path + 'naval/', download=True, train=False,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        return loader

    if dataset == 'yatch':
        if split=='train':
            data = Yatch(root=path + 'yatch/', download=True, train=True,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        elif split=='test':
            data = Yatch(root=path + 'yatch/', download=True, train=False,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False, )
        return loader
        
    if dataset == 'insurance':
        if split=='train':
            data = Insurance(root=path + 'insurance/', download=True, train=True,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        elif split=='test':
            data = Insurance(root=path + 'insurance/', download=True, train=False,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False, )
        return loader

    if dataset == 'bank':
        if split=='train':
            data = Bank(root=path + 'bank/', download=True, train=True,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        elif split=='test':
            data = Bank(root=path + 'bank/', download=True, train=False,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False, )
        return loader

    if dataset == 'mnist':
        if split=='train':
            data = MNIST(root=path + 'mnist/', download=True, train=True,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,)
        elif split=='test':
            data = MNIST(root=path + 'mnist/', download=True, train=False,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False,)
        return loader

    if dataset == 'fashion_mnist':
        if split=='train':
            data = FashionMNIST(root=path + 'fashion_mnist/', download=True, train=True,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,)
        elif split=='test':
            data = FashionMNIST(root=path + 'fashion_mnist/', download=True, train=False,
                            transform=transforms.ToTensor(), split_idx=kwargs['split_idx'],
                            dim=dim, mixed=mixed)
            loader = MultiEpochsDataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=False,)
        return loader



def clean_dataset(dataset: str):
    """Returns the clean name of a dataset.
    Given fashion_mnist_cnn, it returns fashion_mnist

    Args:
        dataset (str): name with '_' as separator

    Returns:
        str: clean name
    """
    sep = dataset.split('_')
    # If the dataset name itsel contains '_'
    if len(sep)>1 and dataset != 'fashion_mnist':
        clean = '_'.join(dataset.split('_')[:-1])  # mnist_cnn -> mnist
    elif len(sep)>2 and dataset == 'fashion_mnist':
        clean = 'fashion_mnist'
    else:
        clean = dataset
    return clean