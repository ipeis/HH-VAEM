# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from src.models.vaem import *


# ============= MIWAEM ============= #

class MIWAEM(VAEM):
    """
    Implements the MIWAEM (MIWAE https://arxiv.org/pdf/1812.02633.pdf for mixed-type data)

    """
    def __init__(self, 
            dataset: str, dim_x: int, dim_y: int, latent_dim = 10, arch='base', dim_h=256,
            likelihood_x = 'gaussian', likelihood_y = 'gaussian', variance=0.1, imbalanced_y = False,
            categories_y = 1, prediction_metric='rmse',
            batch_size=128, lr=1e-3, samples_MC = 50, data_path='../data/', split_idx=0,
            
            likelihoods_x: list = None, categories_x: list = None,
            marg_epochs=1000, lr_marg=1e-3
        ):
        """
        MIWAEM initialization

        Args:
            dataset (str): name of the dataset (boston, mnist, ...)
            dim_x (int): input data dimension
            dim_y (int): target data dimension
            latent_dim (int, optional): dimension of the latent space. Defaults to 10.
            arch (str, optional): name of the architecture for encoder/decoder from the 'archs' file. Defaults to 'base'.
            dim_h (int, optional): dimension of the hidden vectors. Defaults to 256.
            likelihood_x (str, optional): input data likelihood type. Defaults to 'gaussian'.
            likelihood_y (str, optional): target data likelihood type. Defaults to 'gaussian'.
            variance (float, optional): fixed variance for Gaussian likelihoods. Defaults to 0.1.
            imbalanced_y (bool, optional): True for compensating imbalanced classification. Defaults to False.
            categories_y (int, optional): number of categories when the target is categorical. Defaults to 1.
            prediction_metric (str, optional): name of the prediction metric for validation ('rmse', 'accuracy'). Defaults to 'rmse'.
            batch_size (int, optional): batch size. Defaults to 128.
            lr (_type_, optional): learning rate for the parameter optimization. Defaults to 1e-3.
            samples_MC (int, optional): number of MC samples for computing the ELBO. Defaults to 1.
            data_path (str, optional): path to load/save the data. Defaults to '../data/'.
            split_idx (int, optional): idx of the training split. Defaults to 0.
            likelihoods_x (list, optional): list of likelihoods per variable. Defaults to None.
            categories_x (list, optional): list of number of categories per variable. Defaults to None.
            marg_epochs (int, optional): epochs for pretraining marginal VAEs. Defaults to 1000.
            lr_marg (float, optional): learning rate for pretraining marginal VAEs. Defaults to 1e-3.
        """

        super(MIWAEM, self).__init__(dataset=dataset, dim_x=dim_x, dim_y=dim_y, 
            latent_dim = latent_dim, arch=arch, dim_h=dim_h, likelihood_x = likelihood_x, likelihoods_x = likelihoods_x, likelihood_y = likelihood_y, 
            variance=variance, imbalanced_y = imbalanced_y, categories_x=categories_x,
            categories_y=categories_y, prediction_metric=prediction_metric, batch_size=batch_size, lr=lr, 
            samples_MC = samples_MC, data_path=data_path, split_idx=split_idx)
        
        self.likelihoods_x = likelihoods_x
        self.categories_x = categories_x
        self.marg_epochs = marg_epochs
        self.lr_marg = lr_marg

        # To pretrain marginal VAEs
        self.stage_margVAE = True
        margVAEs_config = {
            'dim_x': 1,
            'variance': variance,
            'mixed_data': True,
            'dataset': dataset,
            'latent_dim': 1,
            'batch_size': batch_size,
            'lr': lr_marg,
            'samples_MC': samples_MC,
            'split_idx': split_idx,
            'data_path': data_path
        }
        margVAEs = []
        for i in range(dim_x):
            margVAEs_config['dim'] = i,
            margVAEs_config['likelihood_x'] = likelihoods_x[i]
            margVAEs_config['categories_x'] = categories_x[i]
            margVAEs.append(margVAE(**margVAEs_config).to(self.device))
        self.margVAEs = nn.ModuleList(margVAEs)

        self.save_hyperparameters('likelihoods_x', 'categories_x', 'marg_epochs', 'lr_marg')
    
    def forward(self, batch: tuple, samples: int) -> tuple:
        """
        Computes the MIWAE ELBO for a given batch

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            samples (int): number of MC samples for computing the ELBO

        Returns:
            torch.Tensor: mean loss (negative ELBO)                          
            torch.Tensor: Reconstruction term for x logp(x|z)           
            torch.Tensor: Reconstruction term for y logp(y|z,x)    
            torch.Tensor: KL term     

        """
        # Get data
        x, observed_x, y, observed_y = batch
        xn = self.normalize_x(x)
        xt, yt, xy, observed = self.preprocess_batch(batch) 
        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)
        mu_z, logvar_z = self.encoder(xy)

        z = self.sample_z(mu_z, logvar_z, samples=samples)
        theta_x = self.decoder(z)
        x_hat = self.build_x_hat(xn, observed_x, theta_x)

        zx = torch.cat([z,x_hat],dim=-1)
        
         # MIWAE ELBO 
        log_p = self.logp(xt, observed_x, yt, observed_y, z, xn=xn)
        log_q = self.logq(z, xt, observed_x, yt, observed_y)
        w = log_p - log_q               # (batch_size, samples_MC)

        elbo_miwae = torch.logsumexp(w, -1)
        elbo_miwae = elbo_miwae * observed[:,0]
        #elbo_iwae_m = torch.mean(elbo_iwae)  # batch_size

       
        rec_x = self.decoder.logp(xt, observed_x, z=z, theta=theta_x).sum(-1)
        rec_y = self.predictor.logp(yt, observed_y, z=zx).sum(-1)
        kls = self.encoder.regularizer(mu_z, logvar_z, observed)
        
        #elbo = rec_x + rec_y - kls.sum(0).unsqueeze(-1)
        elbo = elbo_miwae
        
        elbo = elbo[elbo!=0].mean()
        rec_x = rec_x[rec_x!=0].mean()
        rec_y = rec_y[rec_y!=0].mean()

        kl_mean = torch.zeros(len(kls)).type_as(rec_x)
        for l, kl in enumerate(kls):
            kl_mean[l]= kl[kl!=0].mean()

        return -elbo, rec_x, rec_y, kl_mean
     