# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from typing import Dict
from src.models.base import *
from src.models.miwae import *
from src.models.hmc_vae import *
from src.models.h_vae import *
from src.models.hh_vae import *
from src.models.vaem import *
from src.models.miwaem import *
from src.models.hmc_vaem import *
from src.models.h_vaem import *
from src.models.hh_vaem import *
from src.configs import *

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier



def create_model(model: str, config: Dict) -> object:
    """
    Creates a model with a given configuration

    Args:
        model (str): name of the model
        config (dict): dictionary of parameters

    Returns:
        (object): created model (HHVAEM, HMCVAEM, ...)
    """
    if model=='VAE':
        model = BaseVAE(**config)
    elif model=='MIWAE':
        model = MIWAE(**config) 
    elif model=='HMCVAE':
        model = HMCVAE(**config)
    elif model=='HVAE':
        model = HVAE(**config)
    elif model=='HHVAE':
        model = HHVAE(**config) 
    elif model=='VAEM':
        model = VAEM(**config)
    elif model=='MIWAEM':
        model = MIWAEM(**config) 
    elif model=='HMCVAEM':
        model = HMCVAEM(**config)
    elif model=='HVAEM':
        model = HVAEM(**config)
    elif model=='HHVAEM':
        model = HHVAEM(**config)

    # ============= SKLEARN ============= #

    elif model=='KNeighborsRegressor':
        pipe = Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor())])
        params_pipe_knn = config
        model = GridSearchCV(pipe, param_grid=params_pipe_knn, cv=3)
    elif model=='KNeighborsClassifier':
        pipe = Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier())])
        params_pipe_knn = config
        model = GridSearchCV(pipe, param_grid=params_pipe_knn, cv=3)
    elif model=='RandomForestRegressor':
        pipe = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor())])
        params_pipe_knn = config
        model = GridSearchCV(pipe, param_grid=params_pipe_knn, cv=3)
    elif model=='RandomForestClassifier':
        pipe = Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier())])
        params_pipe_knn = config
        model = GridSearchCV(pipe, param_grid=params_pipe_knn, cv=3)
    elif model=='SVR':
        pipe = Pipeline([('scaler', StandardScaler()), ('model', SVR())])
        params_pipe_knn = config
        model = GridSearchCV(pipe, param_grid=params_pipe_knn, cv=3)
    elif model=='SVC':
        pipe = Pipeline([('scaler', StandardScaler()), ('model', SVC())])
        params_pipe_knn = config
        model = GridSearchCV(pipe, param_grid=params_pipe_knn, cv=3)
    elif model=='MLPRegressor':
        pipe = Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor())])
        params_pipe_knn = config
        model = GridSearchCV(pipe, param_grid=params_pipe_knn, cv=3)
    elif model=='MLPClassifier':
        pipe = Pipeline([('scaler', StandardScaler()), ('model', MLPClassifier())])
        params_pipe_knn = config
        model = GridSearchCV(pipe, param_grid=params_pipe_knn, cv=3)

    return model

def load_model(model: str, path: str, device: str) -> object:
    """
    Load model into device from a given path

    Args:
        model (str): name of the model
        path (str): path to the model checkpoint
        device (str): 'cpu' or 'cuda'

    Returns:
        object: loaded model
    """

    if model=='VAE':
        model = BaseVAE.load_from_checkpoint(path, device=device).eval().to(device)
    elif model=='MIWAE':
        model = MIWAE.load_from_checkpoint(path, device=device).eval().to(device)
    elif model=='HMCVAE':
        model = HMCVAE.load_from_checkpoint(path, device=device).eval().to(device)
    elif model=='HVAE':
        model = HVAE.load_from_checkpoint(path, device=device).eval().to(device)
    elif model=='HHVAE':
        model = HHVAE.load_from_checkpoint(path, device=device).eval().to(device)
    elif model=='VAEM':
        model = VAEM.load_from_checkpoint(path, device=device).eval().to(device)
    elif model=='MIWAEM':
        model = MIWAEM.load_from_checkpoint(path, device=device).eval().to(device)
    elif model=='HMCVAEM':
        model = HMCVAEM.load_from_checkpoint(path, device=device).eval().to(device)
    elif model=='HVAEM':
        model = HVAEM.load_from_checkpoint(path, device=device).eval().to(device)
    elif model=='HHVAEM':
        model = HHVAEM.load_from_checkpoint(path, device=device).eval().to(device)
    return model

def find_path(dataset: str, model: str, split: int, version="version_0") -> str:
    """
    Finds the path to the last checkpoint for a given model, dataset and split

    Args:
        dataset (str): name of the dataset
        model (str): name of the model
        split (int): index of the split
        version (str, optional): name of the version. Defaults to "version_0".

    Returns:
        str: path to the last checkpoint
    """

    path = '{:s}/logs/{:s}/{:s}/split_{:d}/{:s}/checkpoints'.format(LOGDIR, dataset, model, split, version)
    ckpts = os.listdir(path)
    ckpts = [ckpt for ckpt in ckpts  if not ckpt.__contains__('test')]
    
    return os.path.join(path, ckpts[0])

def find_splits_models(dataset: str, model: str, version="version_0") -> list:
    """
    Returns a list with the path to the checkpoint of each split

    Args:
        dataset (str): name of the dataset
        model (str): name of the model
        version (str, optional): version of the model. Defaults to "version_0".

    Returns:
        list: list containing strings with the checkpoint paths
    """
    models = []
    for split in range(SPLITS):
        path = '{:s}/logs/{:s}/{:s}/split_{:d}/{:s}/checkpoints'.format(LOGDIR, dataset, model, split, version)
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts  if not ckpt.__contains__('test')]
        models.append(os.path.join(path, ckpts[0]))

    return models

def clean_model(model: str):
    """Returns the clean name of a model.
    Given HHVAE_cnn, it returns HHVAE

    Args:
        dataset (str): name with '_' as separator

    Returns:
        str: clean name
    """
    sep = model.split('_')
    # If the dataset name itsel contains '_'
    if len(sep)>1:
        clean = '_'.join(model.split('_')[:-1])  # fashion_mnist_cnn -> fashion_mnist
    else:
        clean = model.split('_')[0]
    return clean