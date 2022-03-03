
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from src.models.base import *
from src.models.hmc import *


# ============= HMCVAE ============= #

class HMCVAE(BaseVAE):
    """
    Implements a Hamiltonian VAE (HMC-VAE) as described in https://arxiv.org/abs/2202.04599

    """

    def __init__(self, 
            dataset: str, dim_x: int, dim_y: int, latent_dim = 10, arch='base', dim_h=256,
            likelihood_x = 'gaussian', likelihood_y = 'gaussian', variance=0.1, imbalanced_y = False,
            categories_y = 1, prediction_metric='rmse',
            batch_size=128, lr=1e-3, samples_MC = 1, data_path='../data/', split_idx=0,
        
            L=5, T=10, chains=1, chains_sksd=30, sksd=1, pre_steps=2e3, lr_pre=1e-3, 
            lr_encoder=1e-3, lr_decoder=1e-3, lr_predictor=1e-3, lr_hmc=1e-3, lr_scale = 1e-2,
            update_s_each=10
        ):
        """
        HMCVAE Initialization

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
            lr (float, optional): learning rate for the parameter optimization. Defaults to 1e-3.
            samples_MC (int, optional): number of MC samples for computing the ELBO. Defaults to 1.
            data_path (str, optional): path to load/save the data. Defaults to '../data/'.
            split_idx (int, optional): idx of the training split. Defaults to 0.
            L (int, optional): number of Leapfrog steps. Defaults to 5.
            T (int, optional): length of the HMC chains. Defaults to 10.
            chains (int, optional): number of parallel HMC chains. Defaults to 1.
            chains_sksd (int, optional): number of parallel HMC chains for computing the SKSD. Defaults to 30.
            sksd (int, optional): learn a scale factor for q(eps|zy) using the SKSD regularizer (1) or not (0). Defaults to 1.
            pre_steps (float, optional): number of standard VI training steps (before using HMC). Defaults to 18e3.
            lr_pre (float, optional): learning reate for all the parameters during the VI training stage. Defaults to 1e-3.
            lr_encoder (float, optional): Learning rate for the encoder parameters. Defaults to 1e-3.
            lr_decoder (float, optional): Learning rate for the decoder (p(x|z1)). Defaults to 1e-3.
            lr_predictor (float, optional): Learning rate for the predictor. Defaults to 1e-3.
            lr_hmc (float, optional): Learning rate for the HMC hyperparameters (matrix of step sizes). Defaults to 1e-3.
            lr_scale (_type_, optional): Learning rate for the scale (inflation) factor  Defaults to 1e-2.
            update_s_each (int, optional): Interval of steps for optimizing the scale factor. Defaults to 10.
        """

        super(HMCVAE, self).__init__(dataset=dataset, dim_x=dim_x, dim_y=dim_y, 
            latent_dim = latent_dim, arch=arch, dim_h=dim_h, likelihood_x = likelihood_x, likelihood_y = likelihood_y, 
            variance=variance, imbalanced_y=imbalanced_y,
            categories_y=categories_y, prediction_metric=prediction_metric, batch_size=batch_size, lr=lr, 
            samples_MC = samples_MC, data_path=data_path, split_idx=split_idx)

        self.HMC = HMC(dim=latent_dim, L=L, T=T, chains=chains, chains_sksd=chains_sksd, logp=None)

        self.automatic_optimization=False
        self.L = L
        self.T = T
        self.chains = chains
        self.chains_sksd = chains_sksd
        self.sksd = sksd
        self.pre_steps = pre_steps
        self.lr_pre = lr_pre
        self.lr_encoder = lr_encoder
        self.lr_decoder = lr_decoder
        self.lr_predictor = lr_predictor
        self.lr_hmc = lr_hmc
        self.lr_scale = lr_scale
        self.update_s_each = update_s_each
        self.hmc=True

        self.save_hyperparameters('L', 'T', 'chains', 'sksd', 'pre_steps', 
            'lr_pre', 'lr_encoder', 'lr_decoder', 'lr_predictor', 'lr_hmc',
            'lr_scale', 'update_s_each')

        self.step_idx=0 # training step index
    
    # ============= Modified base functions ============= #
    def forward(self, batch: tuple, hmc=True, samples=1) -> tuple:
        """
        Forward data through the model. For the pretraining stage, use the ELBO. For the rest, use HMC

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            hmc (bool): sample posterior using HMC (True). Defaults to True
            samples (int): number of MC samples for computing the ELBO

        Returns:
            If hmc=False, returns:
                loss_VI, rec_x, rec_y, kl
            If hmc=True, returns:
                loss_VI, loss_HMC, loss_SKSD, rec_x, rec_y, kl 

        """
        if hmc==True:
            # Activate only encoder
            activate(self.encoder)
            deactivate(self.decoder)
            deactivate(self.predictor)
            self.HMC.log_eps.requires_grad = False
            self.HMC.log_inflation.requires_grad = False

        # Get data
        x, observed_x, y, observed_y = batch
        xn = self.normalize_x(x)
        xt, yt, xy, observed = self.preprocess_batch(batch) 
        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)
        mu_z, logvar_z = self.encoder(xy)

        z = self.sample_z(mu_z, logvar_z, samples=samples, hmc=False)
        theta_x = self.decoder(z)
        x_hat = self.build_x_hat(xn, observed_x, theta_x)

        zx = torch.cat([z,x_hat],dim=-1)

        rec_x = self.decoder.logp(xt, observed_x, z=z, theta=theta_x).sum(-1)
        rec_y = self.predictor.logp(yt, observed_y, z=zx).sum(-1)
        kls = self.encoder.regularizer(mu_z, logvar_z, observed)
        
        elbo = rec_x + rec_y - kls.sum(0).unsqueeze(-1)

        elbo = elbo[elbo!=0].mean()
        rec_x = rec_x[rec_x!=0].mean()
        rec_y = rec_y[rec_y!=0].mean()

        kl_mean = torch.zeros(len(kls)).to(self.device)
        for l, kl in enumerate(kls):
            kl_mean[l]= kl[kl!=0].mean()
        
        loss_3 = -elbo
        
        if hmc==False: # returns elbo
            return loss_3, rec_x, rec_y, kl_mean
        
        else: # returns elbo, logp and sksd
            
            # Activate decoder, predictor and hmc
            activate(self.decoder)
            activate(self.predictor)
            self.HMC.log_eps.requires_grad = True
            deactivate(self.encoder)
            self.HMC.log_inflation.requires_grad = False
            
            # Encoder again for not sharing gradients
            mu_z, logvar_z = self.encoder(xy)
            zT = self.sample_z(mu_z, logvar_z, samples=samples)
            loss_1 = -self.HMC.logp(zT)
            loss_1 = loss_1[loss_1!=0].mean()

            if self.sksd==1:
                # Deactivate everything except scale
                self.HMC.log_inflation.requires_grad = True
                deactivate(self.encoder)
                deactivate(self.decoder)
                deactivate(self.predictor)
                self.HMC.log_eps.requires_grad = False
                loss_2 = self.HMC.evaluate_sksd(mu_z, torch.exp(logvar_z))
            else:
                loss_2 = None

            return loss_3, loss_1, loss_2
    
    def training_step(self, batch: tuple, batch_idx: int, logging: bool=True):
        """
        Perform a traning step following https://arxiv.org/abs/2202.04599
            - For the first pre_steps, optimize parameters by maximizing the ELBO
            - For the rest, optimize encoder using ELBO, and the rest using HMC objective and SKSD

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            batch_idx (int): batch index from the training set
            logging (bool): log metrics into Tensorboard (True). Default True                            

        """
        (opt_vae, opt_decoder, opt_predictor, opt_encoder, opt_hmc, opt_scale) = self.optimizers(use_pl_optimizer=True)
    
        if self.step_idx < self.pre_steps:
            self.hmc=False
            loss_3, rec_x, rec_y, kls = self.forward(batch, hmc=False, samples=self.samples_MC)

            opt_vae.zero_grad()
            self.manual_backward(loss_3)
            opt_vae.step()

            self.log('ELBO', -loss_3, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if logging:
                self.log('-rec_x', -rec_x, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log('-rec_y', -rec_y, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                for l, kl in enumerate(kls):
                    self.log('kl_{:d}'.format(l), kl, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        else:
            self.hmc=True
            loss_3, loss_1, loss_2 = self.forward(batch, samples=self.chains)

            ##### Optimization
            # Optimize psi (encoder)
            activate(self.encoder)
            deactivate(self.decoder)
            deactivate(self.predictor)
            self.HMC.log_eps.requires_grad = False
            self.HMC.log_inflation.requires_grad = False
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            opt_predictor.zero_grad()
            opt_hmc.zero_grad()
            opt_scale.zero_grad()
            self.manual_backward(loss_3)
            opt_encoder.step()

            # Optimize theta_x, theta_y and phi (decoders and HMC)
            activate(self.decoder)
            activate(self.predictor)
            self.HMC.log_eps.requires_grad = True
            deactivate(self.encoder)
            self.HMC.log_inflation.requires_grad = False
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            opt_predictor.zero_grad()
            opt_hmc.zero_grad()
            opt_scale.zero_grad()
            self.manual_backward(loss_1)#, [opt_decoder, opt_predictor, opt_hmc])
            opt_decoder.step()
            opt_predictor.step()
            opt_hmc.step()

            if self.sksd and self.step_idx % self.update_s_each == 0:
                self.HMC.log_inflation.requires_grad = True
                deactivate(self.encoder)
                deactivate(self.decoder)
                deactivate(self.predictor)
                self.HMC.log_eps.requires_grad = False
                opt_encoder.zero_grad()
                opt_decoder.zero_grad()
                opt_predictor.zero_grad()
                opt_hmc.zero_grad()
                opt_scale.zero_grad()
                self.manual_backward(loss_2)#, opt_scale)
                opt_scale.step()

                scale = torch.exp(self.HMC.log_inflation)
                self.log('scale', scale, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                if logging:
                    self.log('SKSD', loss_2, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            self.log('HMC_objective', -loss_1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
        self.step_idx += 1

    def preprocess_batch(self, batch: tuple):
        """
        Preprocessing operations for the batch (overrides the base class function) for defining the HMC objective p(epsilon)(x, y)

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)

        Returns:
            tuple: preprocessed batch, contains (data, observed_data, target, observed_target)
        """

        batch = [b.to(self.device) for b in batch]
        
        x, observed_x, y, observed_y = batch

        x = x.view(-1, self.dim_x)
        # Normalize the data 
        xn = self.normalize_x(x)

        xo = xn * observed_x
        x_tilde = torch.cat([xo, observed_x], axis=1)
        y = y.view(-1, self.dim_y)
        observed_y = observed_y.view(-1, self.dim_y)

        # Normalize the target 
        yn = self.normalize_y(y)
        yon = yn * observed_y

        y_tilde = torch.cat([yon, observed_y], axis=1)

        xy = torch.cat([x_tilde, y_tilde], axis=1)

        observed = torch.logical_or(observed_x.sum(-1, keepdim=True)>0, observed_y.sum(-1, keepdim=True)>0)

        # Define the HMC objective
        self.HMC.logp = self.logp_func(xo, observed_x, yon, observed_y)  
        
        return xn, yn, xy, observed

    def sample_z(self, mu: torch.Tensor, logvar: torch.Tensor, samples=1, hmc=True) -> torch.Tensor:
        """
        Draw latent samples from a given approx posterior parameterized by mu and logvar

        Args:
            mu (torch.Tensor): tensor with the means                            (batch_size, latent_dim)
            logvar (torch.Tensor): tensor with the log variances                (batch_size, latent_dim)
            samples (int, optional): number of samples. Defaults to 1.     
            hmc (bool, optional): draw hmc samples or Gaussian samples from the proposal. Defaults to True.    

        Returns:
            torch.Tensor: latent samples
        """
        if hmc==False or self.validation and self.global_step < self.pre_steps:
            # Repeat samples_MC times for Monte Carlo
            mu = mu.repeat(samples, 1, 1).transpose(0, 1)
            logvar = logvar.repeat(samples, 1, 1).transpose(0, 1)
            # Reparametrization
            z = reparameterize(mu, torch.exp(logvar))
        else: # sample from the true posterior
            z, _ = self.HMC.generate_samples_HMC(mu, torch.exp(logvar), chains=samples)
        return z

    # ============= Modified PL functions ============= #
    def configure_optimizers(self):
        opt_vae = torch.optim.Adam(list(self.decoder.parameters()) + list(self.predictor.parameters())
                                   + list(self.encoder.parameters()), lr=self.lr_pre, weight_decay=0.01)
        opt_decoder = torch.optim.Adam(list(self.decoder.parameters()), lr=self.lr_decoder, weight_decay=0.01)
        opt_predictor = torch.optim.Adam(list(self.predictor.parameters()), lr=self.lr_predictor, weight_decay=0.01)
        opt_encoder = torch.optim.Adam(list(self.encoder.parameters()), lr=self.lr_encoder, weight_decay=0.01)
        opt_hmc = torch.optim.Adam([self.HMC.log_eps], lr=self.lr_hmc)
        opt_scale = torch.optim.Adam([self.HMC.log_inflation], lr=self.lr_scale)


        return [opt_vae, opt_decoder, opt_predictor, opt_encoder, opt_hmc, opt_scale]

