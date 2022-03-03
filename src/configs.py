# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import os

########################################
# Environment variables
########################################
# global logdir = '../' # uncomment this line 
global LOGDIR

# Change your logdir to a local path to avoid problems in network filesystems
# Modify the bashrc file to include a logdir for storing data/ and logs/
# $ export LOGDIR="your_log_dir/"
"""LOGDIR = os.getenv('LOGDIR')
if LOGDIR==None: # you didnt set another logdir
    LOGDIR='./'"""

# Or run it locally 
LOGDIR = '{}/'.format( os.getcwd()[:os.getcwd().find('HH-VAEM') + 7] )

# Parameter configuration: {'dataset': {'model': {params}}}
configs = {
    'boston': {
        'VAE': {
            'dim_x': 13,
            'dim_y': 1, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 4000  # for 4000*5=20000 steps
        },
        'HMCVAE': {
            'dim_x': 13,
            'dim_y': 1,
            'latent_dim': 10,
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 4000,  # for 4000*5=20000 steps
            'sksd': 1,
            'pre_steps': 18e3,
            'T': 10,
        },
        'VAEM': {
            'dim_x': 13,
            'dim_y': 1, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_y': 'loggaussian', 
            'likelihoods_x': ['loggaussian', 'loggaussian', 'loggaussian', 'bernoulli', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian'],
            'categories_x': [1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
            'prediction_metric': 'rmse',
            'batch_size': 100,
            'marg_epochs': 200, # for 200*5=1k steps
            'epochs': 4000  # for 4000*5=20000 steps
        },
        'HMCVAEM': {
            'dim_x': 13,
            'dim_y': 1, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian', 'loggaussian', 'loggaussian', 'bernoulli', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian'],
            'categories_x': [1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
            'prediction_metric': 'rmse',
            'batch_size': 100,
            'marg_epochs': 200, # for 200*5=1k steps
            'epochs': 4000,  # for 4000*5=20000 steps
            'T': 10,
            'pre_steps': 18e3
        },
        'HVAE': {
            'dim_x': 13,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 4000  # for 4000*5=20000 steps
        },
        'HHVAE': {
            'dim_x': 13,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 4000,  # for 4000*5=20000 steps
            'sksd': 1,
            'pre_steps': 18e3,
            'T': 15,
        },
        'HVAEM': {
            'dim_x': 13,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_y': 'loggaussian', 
            'likelihoods_x': ['loggaussian', 'loggaussian', 'loggaussian', 'bernoulli', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian'],
            'categories_x': [1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 4000,  # for 4000*5=20000 steps,
            'marg_epochs': 200, # for 200*5=1k steps
        },
        'HHVAEM': {
            'dim_x': 13,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_y': 'loggaussian', 
            'likelihoods_x': ['loggaussian', 'loggaussian', 'loggaussian', 'bernoulli', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian'],
            'categories_x': [1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 4000,  # for 4000*5=20000 steps,
            'marg_epochs': 200, # for 200*5=1k steps
            'pre_steps': 18e3,
            'T': 10,
        },
    },
    'energy': {
        'VAE': {
            'dim_x': 8,
            'dim_y': 2, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 2858  # for 2858*7=20000 steps
        },
        'HMCVAE': {
            'dim_x': 8,
            'dim_y': 2, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 2858,  # for 2858*7=20000 steps
            'pre_steps': 18e3,
            'T': 10,
        },
        'VAEM': {
            'dim_x': 8,
            'dim_y': 2, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian', 'loggaussian', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 'categorical'],
            'categories_x': [1, 1, 7, 4, 1, 4, 4, 6], 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'marg_epochs': 10, # for 143*7=1k steps
            'epochs': 2858  # for 2858*7=20000 steps
        },
        'HMCVAEM': {
            'dim_x': 8,
            'dim_y': 2, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian', 'loggaussian', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 'categorical'],
            'categories_x': [1, 1, 7, 4, 1, 4, 4, 6], 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'marg_epochs': 143, # for 143*7=1k steps
            'epochs': 2858,  # for 2858*7=20000 steps
            'pre_steps': 18e3,
            'T': 10,
        },
        'HVAE': {
            'dim_x': 8,
            'dim_y': 2, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 2858  # for 2858*7=20000 steps,
        },
        'HHVAE': {
            'dim_x': 8,
            'dim_y': 2, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 2858,  # for 2858*7=20000 steps,
            'pre_steps': 18e3,
            'T': 15,
        },
        'HVAEM': {
            'dim_x': 8,
            'dim_y': 2, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian', 'loggaussian', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 'categorical'],
            'categories_x': [1, 1, 7, 4, 1, 4, 4, 6], 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 2858,  # for 2858*7=20000 steps,
            'marg_epochs': 143, # for 143*7=1k steps
        },
        'HHVAEM': {
            'dim_x': 8,
            'dim_y': 2, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian', 'loggaussian', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 'categorical'],
            'categories_x': [1, 1, 7, 4, 1, 4, 4, 6], 
            'prediction_metric':'rmse',
            'batch_size': 100,
            'epochs': 2858,  # for 2858*7=20000 steps,
            'marg_epochs': 143, # for 143*7=1k steps
            'pre_steps': 18e3,
            'T': 10,
        },
    },
    'wine': {
        'VAE': {
            'latent_dim': 10,
            'dim_x': 13,
            'dim_y': 1,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'categorical',
            'categories_y': 3,
            'prediction_metric':'accuracy',
            'batch_size': 24,
            'epochs': 2858,  # for 2858*7=20000 steps
        },
        'HMCVAE': {
            'latent_dim': 10,
            'dim_x': 13,
            'dim_y': 1,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'categorical',
            'categories_y': 3,
            'prediction_metric':'accuracy',
            'batch_size': 24,
            'epochs': 2858,  # for 2858*7=20000 steps
            'pre_steps': 18e3,
            'T': 10,
        },
        'VAEM': {
            'latent_dim': 10,
            'dim_x': 13,
            'dim_y': 1,
            'arch': 'base',
            'likelihoods_x': ['loggaussian'] * 13,
            'categories_x': [1] * 13,
            'likelihood_y': 'categorical',
            'categories_y': 3,
            'prediction_metric':'accuracy',
            'batch_size': 24,
            'epochs': 2858,  # for 2858*7=20000 steps
            'marg_epochs': 143  # for 143*7=1k steps
        },
        'HMCVAEM': {
            'latent_dim': 10,
            'dim_x': 13,
            'dim_y': 1,
            'arch': 'base',
            'likelihoods_x': ['loggaussian'] * 13,
            'categories_x': [1] * 13,
            'likelihood_y': 'categorical',
            'categories_y': 3,
            'prediction_metric':'accuracy',
            'batch_size': 24,
            'epochs': 2858,  # for 2858*7=20000 steps
            'marg_epochs': 143,  # for 143*7=1k steps
            'pre_steps': 18e3,
            'T': 10,
        },
        'HVAE': {
            'dim_x': 13,
            'dim_y': 1,
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'categorical',
            'categories_y': 3,
            'prediction_metric':'accuracy',
            'batch_size': 24,
            'epochs': 2858,  # for 2858*7=20000 steps
        },
        'HHVAE': {
            'dim_x': 13,
            'dim_y': 1,
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'categorical',
            'categories_y': 3,
            'prediction_metric':'accuracy',
            'batch_size': 24,
            'epochs': 2858,  # for 2858*7=20000 steps
            'pre_steps': 18e3,
            'T': 15,
        },
        'HVAEM': {
            'dim_x': 13,
            'dim_y': 1,
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihoods_x': ['loggaussian'] * 13,
            'categories_x': [1] * 13,
            'likelihood_y': 'categorical',
            'categories_y': 3,
            'prediction_metric':'accuracy',
            'batch_size': 24,
            'epochs': 2858,  # for 2858*7=20000 steps
            'marg_epochs': 143  # for 143*7=1k steps
        },
        'HHVAEM': {
            'dim_x': 13,
            'dim_y': 1,
            'latent_dims': [10, 5],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihoods_x': ['loggaussian'] * 13,
            'categories_x': [1] * 13,
            'likelihood_y': 'categorical',
            'categories_y': 3,
            'prediction_metric':'accuracy',
            'batch_size': 24,
            'epochs': 2858,  # for 2858*7=20000 steps
            'marg_epochs': 143,  # for 143*7=1k steps
            'pre_steps': 18e3,
            'T': 10,
        },
    },
    'diabetes': {
        'VAE': {
            'dim_x': 10,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 5000, # 5000*4=20k steps
            'batch_size': 100,
        },
        'HMCVAE': {
            'dim_x': 10,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 5000, # 5000*4=20k steps
            'batch_size': 100,
            'pre_steps': 18e3,
            'T': 5,
        },
        'VAEM': {
            'dim_x': 10,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 5000, # 5000*4=20k steps
            'batch_size': 100,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['gaussian', 'bernoulli'] + ['gaussian'] * 8,
            'categories_x': [1] * 10,
            'marg_epochs': 250  # for 250*4 =1000 steps
        },
        'HMCVAEM': {
            'dim_x': 10,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 5000, # 5000*4=20k steps
            'batch_size': 100,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['gaussian', 'bernoulli'] + ['gaussian'] * 8,
            'categories_x': [1] * 10,
            'marg_epochs': 250,  # for 250*4 =1000 steps
            'pre_steps': 18e3,
            'T': 5,
        },
        'HVAE': {
            'dim_x': 10,
            'dim_y': 1,
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'epochs': 5000, # 5000*4=20k steps
            'batch_size': 100,
        },
        'HHVAE': {
            'dim_x': 10,
            'dim_y': 1,
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'epochs': 5000, # 5000*4=20k steps
            'batch_size': 100,
            'pre_steps': 18e3,
            'T': 10,
        },
        'HVAEM': {
            'dim_x': 10,
            'dim_y': 1,
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['gaussian', 'bernoulli'] + ['gaussian'] * 8,
            'categories_x': [1] * 10,
            'prediction_metric':'rmse',
            'epochs': 5000, # 5000*4=20k steps
            'batch_size': 100,
            'marg_epochs': 250  # for 250*4 =1000 steps
        },
        'HHVAEM': {
            'dim_x': 10,
            'dim_y': 1,
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['gaussian', 'bernoulli'] + ['gaussian'] * 8,
            'categories_x': [1] * 10,
            'prediction_metric':'rmse',
            'epochs': 5000, # 5000*4=20k steps
            'marg_epochs': 250,  # for 250*4 =1000 steps
            'batch_size': 100,
            'pre_steps': 18e3,
            'T': 5,
        },
    },
    'concrete': {
        'VAE': {
            'dim_x': 8,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 2000, # 2000*10=20k steps
            'batch_size': 100,
        },
        'HMCVAE': {
            'dim_x': 8,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 2000, # 2000*10=20k steps
            'batch_size': 100,
            'T': 5,
            'pre_steps': 18e3,
        },
        'VAEM': {
            'dim_x': 8,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 2000, # 2000*10=20k steps
            'batch_size': 100,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian'] * 8,
            'categories_x': [1] * 8,
            'marg_epochs': 100  # for 100*10 =1000 steps
        },
        'HMCVAEM': {
            'dim_x': 8,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 2000, # 2000*10=20k steps
            'batch_size': 100,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian'] * 8,
            'categories_x': [1] * 8,
            'marg_epochs': 100,  # for 100*10 =1000 steps
            'T': 5,
            'pre_steps': 18e3,
        },
        'HVAE': {
            'dim_x': 8,
            'dim_y': 1,
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'epochs': 2000, # 2000*10=20k steps
            'batch_size': 100,
        },
        'HHVAE': {
            'dim_x': 8,
            'dim_y': 1,
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'epochs': 2000, # 2000*10=20k steps
            'batch_size': 100,
            'pre_steps': 18e3,
            'T': 10,
        },
        'HVAEM': {
            'dim_x': 8,
            'dim_y': 1,
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian'] * 8,
            'categories_x': [1] * 8,
            'prediction_metric':'rmse',
            'epochs': 2000, # 2000*10=20k steps
            'batch_size': 100,
            'marg_epochs': 100  # for 100*10 =1000 steps
        },
        'HHVAEM': {
            'dim_x': 8,
            'dim_y': 1,
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'arch': 'base',
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian'] * 8,
            'categories_x': [1] * 8,
            'prediction_metric':'rmse',
            'epochs': 2000, # 2000*10=20k steps
            'marg_epochs': 100,  # for 100*10 =1000 steps
            'batch_size': 100,
            'T': 5,
            'pre_steps': 18e3,
        },
    },
    'naval':{
        'VAE': {
            'dim_x': 14,
            'dim_y': 2,
            'latent_dim': 10,
            'epochs': 463, # for 463*108=50k steps
            'batch_size': 100,
        },
        'HMCVAE': {
            'dim_x': 14,
            'dim_y': 2,
            'latent_dim': 10,
            'epochs': 463, # for 463*108=50k steps
            'batch_size': 100,
            'T': 10,
            'pre_steps': 45e3
        },
        'VAEM': {
            'dim_x': 14,
            'dim_y': 2,
            'latent_dim': 10,
            'epochs': 463, # for 463*108=50k steps
            'batch_size': 100,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['categorical'] * 2 + ['loggaussian'] * 12,
            'categories_x': [9] * 2 + [1] * 12,
            'marg_epochs': 10,  # for 10*108=1000 steps
        },
        'HMCVAEM': {
            'dim_x': 14,
            'dim_y': 2,
            'latent_dim': 10,
            'epochs': 463, # for 463*108=50k steps
            'batch_size': 100,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['categorical'] * 2 + ['loggaussian'] * 12,
            'categories_x': [9] * 2 + [1] * 12,
            'marg_epochs': 10,  # for 10*108=1000 steps
            'T': 10,
            'pre_steps': 45e3
        },
        'HVAE': {
            'dim_x': 14,
            'dim_y': 2,
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'epochs': 463, # for 463*108=50k steps
            'batch_size': 100,
        },
        'HHVAE': {
            'dim_x': 14,
            'dim_y': 2,
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'gaussian', 
            'prediction_metric':'rmse',
            'epochs': 463, # for 463*108=50k steps
            'batch_size': 100,
            'pre_steps': 45e3,
            'T': 15,
        },
        'HVAEM': {
            'dim_x': 14,
            'dim_y': 2,
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['categorical'] * 2 + ['loggaussian'] * 12,
            'categories_x': [9] * 2 + [1] * 12,
            'prediction_metric':'rmse',
            'epochs': 463, # for 463*108=50k steps
            'batch_size': 100,
            'marg_epochs': 10,  # for 10*108=1000 steps
        },
        'HHVAEM': {
            'dim_x': 14,
            'dim_y': 2,
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['categorical'] * 2 + ['loggaussian'] * 12,
            'categories_x': [9] * 2 + [1] * 12,
            'prediction_metric':'rmse',
            'epochs': 463, # for 463*108=50k steps
            'marg_epochs': 10,  # for 10*108=1000 steps
            'batch_size': 100,
            'pre_steps': 45e3,
            'T': 10,
        },
    },
    'yatch':{
        'VAE': {
            'dim_x': 6,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 1429, # for 1429*14=20k steps
            'batch_size': 20,
        },
        'HMCVAE': {
            'dim_x': 6,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 1429, # for 1429*14=20k steps
            'batch_size': 20,
            'T': 5,
            'pre_steps': 18e3,
        },
        'VAEM': {
            'dim_x': 6,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 1429, # for 1429*14=20k steps
            'batch_size': 20,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['categorical', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'loggaussian'],
            'categories_x': [5, 1, 8, 1, 1, 1],
            'marg_epochs': 72,  # for 72*14=1000 steps
        },
        'HMCVAEM': {
            'dim_x': 6,
            'dim_y': 1,
            'latent_dim': 4,
            'epochs': 1429, # for 1429*14=20k steps
            'batch_size': 20,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['categorical', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'loggaussian'],
            'categories_x': [5, 1, 8, 1, 1, 1],
            'marg_epochs': 72,  # for 72*14=1000 steps
            'T': 5,
            'pre_steps': 18e3,
        },
        'HVAE': {
            'dim_x': 6,
            'dim_y': 1,
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'epochs': 1429, # for 1429*14=20k steps
            'batch_size': 20,
        },
        'HHVAE': {
            'dim_x': 6,
            'dim_y': 1,
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'epochs': 1429, # for 1429*14=20k steps
            'batch_size': 20,
            'pre_steps': 18e3,
            'T': 10,
        },
        'HVAEM': {
            'dim_x': 6,
            'dim_y': 1,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['categorical', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'loggaussian'],
            'categories_x': [5, 1, 8, 1, 1, 1],
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'epochs': 1429, # for 1429*14=20k steps
            'batch_size': 20,
            'marg_epochs': 72,  # for 72*14=1000 steps
        },
        'HHVAEM': {
            'dim_x': 6,
            'dim_y': 1,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['categorical', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'loggaussian'],
            'categories_x': [5, 1, 8, 1, 1, 1],
            'latent_dims': [4, 2],
            'balance_kl_steps': 1e3, 
            'anneal_kl_steps': 1e3,
            'epochs': 1429, # for 1429*14=20k steps
            'batch_size': 20,
            'marg_epochs': 72,  # for 72*14=1000 steps
            'pre_steps': 18e3,
            'T': 5,
        },
    },
    'avocado':{
        'VAE': {
            'dim_x': 12,
            'dim_y': 1,
            'latent_dim': 10,
            'epochs': 304, # for 304*165=50k steps
            'batch_size': 100,
        },
        'HMCVAE': {
            'dim_x': 12,
            'dim_y': 1,
            'latent_dim': 10,
            'epochs': 304, # for 304*165=50k steps
            'batch_size': 100,
            'T': 10,
            'pre_steps': 45e3
        },
        'VAEM': {
            'dim_x': 12,
            'dim_y': 1,
            'latent_dim': 10,
            'epochs': 304, # for 304*165=50k steps
            'batch_size': 100,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'bernoulli', 'categorical', 'loggaussian'],
            'categories_x': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1],
            'marg_epochs': 10,  # for 10*165=1650 steps
        },
        'HMCVAEM': {
            'dim_x': 12,
            'dim_y': 1,
            'latent_dim': 10,
            'epochs': 304, # for 304*165=50k steps
            'batch_size': 100,
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'bernoulli', 'categorical', 'loggaussian'],
            'categories_x': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1],
            'marg_epochs': 10,  # for 10*165=1650 steps
            'T': 10,
            'pre_steps': 45e3
        },
        'HVAE': {
            'dim_x': 12,
            'dim_y': 1,
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'epochs': 304, # for 304*165=50k steps
            'batch_size': 100,
        },
        'HHVAE': {
            'dim_x': 12,
            'dim_y': 1,
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'epochs': 304, # for 304*165=50k steps
            'batch_size': 100,
            'pre_steps': 45e3,
            'T': 15,
        },
        'HVAEM': {
            'dim_x': 12,
            'dim_y': 1,
            'latent_dims': [10, 5],
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'bernoulli', 'categorical', 'loggaussian'],
            'categories_x': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'epochs': 304, # for 304*165=50k steps
            'batch_size': 100,
            'marg_epochs': 10,  # for 10*165=1650 steps
        },
        'HHVAEM': {
            'dim_x': 12,
            'dim_y': 1,
            'latent_dims': [10, 5],
            'likelihood_y': 'loggaussian',
            'likelihoods_x': ['loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'bernoulli', 'categorical', 'loggaussian'],
            'categories_x': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'epochs': 304, # for 304*165=50k steps
            'marg_epochs': 10,  # for 10*165=1650 steps
            'batch_size': 100,
            'pre_steps': 45e3,
            'T': 10,
        },
    },
    'bank': {
        'VAE': {
            'dim_x': 20,
            'dim_y': 1, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 135  # for 135*371=50e3 steps
        },
        'HMCVAE': {
            'dim_x': 20,
            'dim_y': 1, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'bernoulli',
            'imbalanced_y': True, 
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 135,  # for 135*371=50e3 steps
            'T': 10,
            'pre_steps': 45e3,
        },
        'VAEM': {
            'dim_x': 20,
            'dim_y': 1, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'likelihoods_x': ['loggaussian', 'loggaussian', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'bernoulli', 'loggaussian', 
                'categorical', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'categorical', 'gaussian', 'loggaussian', 'gaussian', 'loggaussian', 
                'loggaussian'],
            'categories_x': [1, 1, 4, 8, 3, 3, 3, 1, 1, 5, 1, 1, 1, 8, 3, 1, 1, 1, 1, 1],
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 135,  # for 135*371=50e3 steps
            'marg_epochs': 10, # for 10*371=3710 steps
        },
        'HMCVAEM': {
            'dim_x': 20,
            'dim_y': 1, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 135,  # for 135*371=50e3 steps
            'likelihoods_x': ['loggaussian', 'loggaussian', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'bernoulli', 'loggaussian', 
                'categorical', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'categorical', 'gaussian', 'loggaussian', 'gaussian', 'loggaussian', 
                'loggaussian'],
            'categories_x': [1, 1, 4, 8, 3, 3, 3, 1, 1, 5, 1, 1, 1, 8, 3, 1, 1, 1, 1, 1],
            'marg_epochs': 10, # for 10*371=3710 steps
            'T': 10,
            'pre_steps': 45e3,
        },
        'HVAE': {
            'dim_x': 20,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 135,  # for 135*371=50e3 steps
        },
        'HHVAE': {
            'dim_x': 20,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 135,  # for 135*371=50e3 steps
            'T': 15,
            'pre_steps': 45e3,
        },
        'HVAEM': {
            'dim_x': 20,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'likelihoods_x': ['loggaussian', 'loggaussian', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'bernoulli', 'loggaussian', 
                'categorical', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'categorical', 'gaussian', 'loggaussian', 'gaussian', 'loggaussian', 
                'loggaussian'],
            'categories_x': [1, 1, 4, 8, 3, 3, 3, 1, 1, 5, 1, 1, 1, 8, 3, 1, 1, 1, 1, 1],
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 135,  # for 135*371=50e3 steps
            'marg_epochs': 10, # for 10*371=3710 steps
        },
        'HHVAEM': {
            'dim_x': 20,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'likelihoods_x': ['loggaussian', 'loggaussian', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'bernoulli', 'loggaussian', 
                'categorical', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'categorical', 'gaussian', 'loggaussian', 'gaussian', 'loggaussian', 
                'loggaussian'],
            'categories_x': [1, 1, 4, 8, 3, 3, 3, 1, 1, 5, 1, 1, 1, 8, 3, 1, 1, 1, 1, 1],
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 135,  # for 135*371=50e3 steps
            'marg_epochs': 10, # for 10*371=3710 steps
            'T': 10,
            'pre_steps': 45e3,
        },
    },
    'insurance': {
        'VAE': {
            'dim_x': 85,
            'dim_y': 1, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 562,  # for 562*89=50e3 steps
        },
        'HMCVAE': {
            'dim_x': 85,
            'dim_y': 1, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 562,  # for 562*89=50e3 steps
            'T': 10,
            'pre_steps': 45e3,
        },
        'VAEM': {
            'dim_x': 85,
            'dim_y': 1, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihoods_x': ['loggaussian', 'categorical', 'categorical', 'categorical', 'loggaussian', 'loggaussian', 
                'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'loggaussian', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'loggaussian', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'bernoulli', 
                'categorical', 'categorical', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 
                'bernoulli', 'bernoulli', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 
                'categorical', 'categorical'],
            'categories_x': [1, 9, 6, 6, 1, 1, 1, 6, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 9, 1, 8, 4, 7, 5, 7, 4, 6, 5, 6, 6, 
                6, 6, 1, 7, 3, 5, 9, 4, 7, 1, 7, 5, 3, 3, 1, 9, 6, 5, 5, 4, 7, 6, 4, 7, 1, 1, 3, 8, 1, 3, 5, 3, 3],
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 562,  # for 562*89=50e3 steps
            'marg_epochs': 12, # for 12*89=1e3 steps
        },
        'HMCVAEM': {
            'dim_x': 85,
            'dim_y': 1, 
            'latent_dim': 10, 
            'arch': 'base',
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 562,  # for 562*89=50e3 steps
            'likelihoods_x': ['loggaussian', 'categorical', 'categorical', 'categorical', 'loggaussian', 'loggaussian', 
                'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'loggaussian', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'loggaussian', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'bernoulli', 
                'categorical', 'categorical', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 
                'bernoulli', 'bernoulli', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 
                'categorical', 'categorical'],
            'categories_x': [1, 9, 6, 6, 1, 1, 1, 6, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 9, 1, 8, 4, 7, 5, 7, 4, 6, 5, 6, 6, 
                6, 6, 1, 7, 3, 5, 9, 4, 7, 1, 7, 5, 3, 3, 1, 9, 6, 5, 5, 4, 7, 6, 4, 7, 1, 1, 3, 8, 1, 3, 5, 3, 3],
            'marg_epochs': 12, # for 12*89=1e3 steps
            'T': 10,
            'pre_steps': 45e3,
        },
        'HVAE': {
            'dim_x': 85,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'bernoulli',
            'imbalanced_y': True, 
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 562,  # for 562*89=50e3 steps
        },
        'HHVAE': {
            'dim_x': 85,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_x': 'gaussian', 
            'likelihood_y': 'bernoulli',
            'imbalanced_y': True, 
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 562,  # for 562*89=50e3 steps
            'T': 10,
            'pre_steps': 45e3,
        },
        'HVAEM': {
            'dim_x': 85,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihoods_x': ['loggaussian', 'categorical', 'categorical', 'categorical', 'loggaussian', 'loggaussian', 
                'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'loggaussian', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'loggaussian', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'bernoulli', 
                'categorical', 'categorical', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 
                'bernoulli', 'bernoulli', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 
                'categorical', 'categorical'],
            'categories_x': [1, 9, 6, 6, 1, 1, 1, 6, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 9, 1, 8, 4, 7, 5, 7, 4, 6, 5, 6, 6, 
                6, 6, 1, 7, 3, 5, 9, 4, 7, 1, 7, 5, 3, 3, 1, 9, 6, 5, 5, 4, 7, 6, 4, 7, 1, 1, 3, 8, 1, 3, 5, 3, 3],
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 562,  # for 562*89=50e3 steps
            'marg_epochs': 12, # for 12*89=1e3 steps
        },
        'HHVAEM': {
            'dim_x': 85,
            'dim_y': 1, 
            'latent_dims': [10, 5],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihoods_x': ['loggaussian', 'categorical', 'categorical', 'categorical', 'loggaussian', 'loggaussian', 
                'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 'loggaussian', 
                'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'loggaussian', 'categorical', 
                'loggaussian', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'loggaussian', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'bernoulli', 
                'categorical', 'categorical', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 
                'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 
                'bernoulli', 'bernoulli', 'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical', 
                'categorical', 'categorical'],
            'categories_x': [1, 9, 6, 6, 1, 1, 1, 6, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 9, 1, 8, 4, 7, 5, 7, 4, 6, 5, 6, 6, 
                6, 6, 1, 7, 3, 5, 9, 4, 7, 1, 7, 5, 3, 3, 1, 9, 6, 5, 5, 4, 7, 6, 4, 7, 1, 1, 3, 8, 1, 3, 5, 3, 3],
            'likelihood_y': 'bernoulli', 
            'imbalanced_y': True,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 562,  # for 562*89=50e3 steps
            'marg_epochs': 12, # for 12*89=1e3 steps
            'T': 10,
            'pre_steps': 45e3,
        },
    },
    'mnist': {
        'VAE': {
            'latent_dim': 20,
            'dim_x': 28**2,
            'dim_y': 1,
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical',
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 93,  # for 93*540=50e3 steps
        },
        'HVAE': {
            'latent_dim': 30,
            'dim_x': 28**2,
            'dim_y': 1,
            'arch': 'base',
            'dim_h': 256,
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical',
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 223,  # for 223*450=100e3 steps
            'sksd': 0,
            'pre_steps': 100e3,
            'T': 30,
        },
        'HMCVAE': {
            'latent_dim': 30,
            'dim_x': 28**2,
            'dim_y': 1,
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical',
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 93,  # for 93*540=50e3 steps
            'pre_steps': 45e3,
            'T': 30,
        },

        'HVAE': {
            'dim_x': 28**2,
            'dim_y': 1, 
            'latent_dims': [20, 10],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical', 
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 93,  # for 93*540=50e3 steps
        },
        'HierarchicalHVAE': {
            'dim_x': 28**2,
            'dim_y': 1,
            'arch': 'base',
            'dim_h': 256,
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical',
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 223,  # for 223*450=100e3 steps
            'latent_dims': [20, 10],
            'dims_r': [20, 10],
            'sksd': 0,
            'pre_steps': 100e3,
            'T': 30,
        },
        'HHVAE': {
            'dim_x': 28**2,
            'dim_y': 1, 
            'latent_dims': [20, 10],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical', 
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 93,  # for 93*540=50e3 steps
            'pre_steps': 45e3,
            'T': 30,
        },
    },
    'fashion_mnist': {
        'VAE': {
            'latent_dim': 20,
            'dim_x': 28**2,
            'dim_y': 1,
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical',
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 93,  # for 93*540=50e3 steps
        },
        'HVAE': {
            'latent_dim': 30,
            'dim_x': 28**2,
            'dim_y': 1,
            'arch': 'base',
            'dim_h': 256,
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical',
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 223,  # for 223*450=100e3 steps
            'sksd': 0,
            'pre_steps': 100e3,
            'T': 30,
        },
        'HMCVAE': {
            'latent_dim': 20,
            'dim_x': 28**2,
            'dim_y': 1,
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical',
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 93,  # for 93*540=50e3 steps
            'pre_steps': 45e3,
            'T': 30,
        },

        'HVAE': {
            'dim_x': 28**2,
            'dim_y': 1, 
            'latent_dims': [20, 10],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'arch': 'base',
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical', 
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 93,  # for 93*540=50e3 steps
        },
        'HierarchicalHVAE': {
            'dim_x': 28**2,
            'dim_y': 1,
            'arch': 'base',
            'dim_h': 256,
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical',
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 223,  # for 223*450=100e3 steps
            'latent_dims': [20, 10],
            'dims_r': [20, 10],
            'sksd': 0,
            'pre_steps': 100e3,
            'T': 30,
        },
        'HHVAE': {
            'dim_x': 28**2,
            'dim_y': 1, 
            'latent_dims': [20, 10],
            'balance_kl_steps': 5e3, 
            'anneal_kl_steps': 5e3,
            'likelihood_x': 'bernoulli', 
            'likelihood_y': 'categorical', 
            'categories_y': 10,
            'prediction_metric':'accuracy',
            'batch_size': 100,
            'epochs': 93,  # for 93*540=50e3 steps
            'pre_steps': 45e3,
            'T': 30,
        },
    }
}

# Parameter configuration for AL experiment: {'dataset': {params}}
configs_active_learning = {

    'boston': {
        'bins': 7,
        'samples': 200,
        'step': 1,
        'batch_size': configs['boston']['VAE']['batch_size'],
        'metric': 'rmse'
    },
    'energy': {
        'bins': 5,
        'samples': 100,
        'step': 1,
        'batch_size': configs['energy']['VAE']['batch_size'],
        'metric': 'rmse'
    },
    'wine': {
        'bins': 5,
        'samples': 100,
        'step': 1,
        'batch_size': 100,
        'metric': 'accuracy'
    },
    'diabetes': {
        'bins': 5,
        'samples': 100,
        'step': 1,
        'batch_size': configs['diabetes']['VAE']['batch_size'],
        'metric': 'rmse'
    },
    'concrete': {
        'bins': 5,
        'samples': 100,
        'step': 1,
        
        'batch_size': configs['concrete']['VAE']['batch_size'],
        'metric': 'rmse'
    },
    'naval': {
        'bins': 5,
        'samples': 100,
        'step': 1,
        
        'batch_size': configs['naval']['VAE']['batch_size'],
        'metric': 'rmse'
    },
    'yatch': {
        'bins': 5,
        'samples': 100,
        'step': 1,
        
        'batch_size': configs['yatch']['VAE']['batch_size'],
        'metric': 'rmse'
    },
    'avocado': {
        'bins': 5,
        'samples': 100,
        'step': 1,
        
        'batch_size': configs['avocado']['VAE']['batch_size'],
        'metric': 'rmse'
    },
    'insurance': {
        'bins': 5,
        'samples': 100,
        'step': 1,
        
        'batch_size': configs['insurance']['VAE']['batch_size'],
        'metric': 'auroc'
    },
    'bank': {
        'bins': 5,
        'samples': 100,
        'step': 1,
        
        'batch_size': configs['bank']['VAE']['batch_size'],
        'metric': 'error_rate'
    },
    'mnist': {
        'bins': 5,
        'samples': 100,
        'step': 10,
        'batch_size': configs['mnist']['VAE']['batch_size'],
        'metric': 'auroc'
    },
}


def get_config(model: str, dataset: str) -> dict:
    """
    Get configuration for given model and dataset

    Args:
        model (str): name of the model
        dataset (str): name of the dataset

    Returns:
        (dict): dictionary containing configuration parameters
    """
    conf = configs[dataset][model]
    conf['dataset'] = dataset
    return conf
