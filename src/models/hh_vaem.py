
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from src.models.hh_vae import *
from src.models.vaem import *


# ============= HHVAEM ============= #

class HHVAEM(HHVAE):
    """
    Implements a Hierarchical Hamiltonian VAEM (HH-VAEM) as described in https://arxiv.org/abs/2202.04599

    """
    def __init__(self, 
            dataset: str, dim_x: int, dim_y: int, arch='base', dim_h=256,
            likelihood_x = 'gaussian', likelihood_y = 'gaussian', variance=0.1, imbalanced_y = False,
            categories_y = 1, prediction_metric='rmse',
            batch_size=128, lr=1e-3, samples_MC = 1, data_path='../data/', split_idx=0,

            latent_dims: list=[10, 5], sr_coef=0.001, balance_kl_steps=15e3, anneal_kl_steps=1e3,
            update_prior = False,

            L=5, T=10, chains=1, chains_sksd=30, sksd=1,
            pre_steps=18e3,
            lr_pre=1e-3, lr_encoder=1e-3, lr_decoder=1e-3, lr_prior = 1e-3, lr_predictor=1e-3, lr_hmc=1e-3, lr_scale = 1e-2,
            update_s_each=10,
        
            likelihoods_x: list = None, categories_x: list = None,
            marg_epochs=1000, lr_marg=1e-3, 
        
        ):
        """
        HHVAEM initialization

        Args:
            dataset (str): name of the dataset (boston, mnist, ...)
            dim_x (int): input data dimension
            dim_y (int): target data dimension
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
            latent_dims (list): list of ints containing the latent dimension at each layer. First element corresponds to the shallowest layer, connected to the data
            sr_coef (float, optional): coefficient for spectral normalization of parameters (Non-used). Defaults to 0.001.
            balance_kl_steps (float, optional): number of steps for balancing the KL terms of the different layers. Defaults to 2e3.
            anneal_kl_steps (float, optional): number of steps for annealing the KL. Defaults to 1e3.
            update_prior (bool, optional): update the prior variance via ML and VI at the end of each epoch (True). Defaults to False.
            L (int, optional): number of Leapfrog steps. Defaults to 5.
            T (int, optional): length of the HMC chains. Defaults to 10.
            chains (int, optional): number of parallel HMC chains. Defaults to 1.
            chains_sksd (int, optional): number of parallel HMC chains for computing the SKSD. Defaults to 30.
            sksd (int, optional): learn a scale factor for q(eps|zy) using the SKSD regularizer (1) or not (0). Defaults to 1.
            pre_steps (float, optional): number of standard VI training steps (before using HMC). Defaults to 18e3.
            lr_pre (float, optional): learning reate for all the parameters during the VI training stage. Defaults to 1e-3.
            lr_encoder (float, optional): Learning rate for the encoder parameters. Defaults to 1e-3.
            lr_decoder (float, optional): Learning rate for the decoder (p(x|z1)). Defaults to 1e-3.
            lr_prior (float, optional): Learning rate for the hierarchical transformations (f(zl|zl+1)). Defaults to 1e-3.
            lr_predictor (float, optional): Learning rate for the predictor. Defaults to 1e-3.
            lr_hmc (float, optional): Learning rate for the HMC hyperparameters (matrix of step sizes). Defaults to 1e-3.
            lr_scale (_type_, optional): Learning rate for the scale (inflation) factor  Defaults to 1e-2.
            update_s_each (int, optional): Interval of steps for optimizing the scale factor. Defaults to 10.
            likelihoods_x (list, optional): list of likelihoods per variable. Defaults to None.
            categories_x (list, optional): list of number of categories per variable. Defaults to None.
            marg_epochs (int, optional): epochs for pretraining marginal VAEs. Defaults to 1000.
            lr_marg (float, optional): learning rate for pretraining marginal VAEs. Defaults to 1e-3.
        """

        super(HHVAEM, self).__init__(dataset=dataset, dim_x=dim_x, dim_y=dim_y,
            arch=arch, dim_h=dim_h, likelihood_x = likelihood_x, likelihood_y = likelihood_y, 
            variance=variance, imbalanced_y = imbalanced_y,
            categories_y=categories_y,
            prediction_metric=prediction_metric, batch_size=batch_size, lr=lr, samples_MC = samples_MC, 
            data_path=data_path, split_idx=split_idx,
            latent_dims=latent_dims, sr_coef=sr_coef,
            balance_kl_steps=balance_kl_steps, anneal_kl_steps=anneal_kl_steps, update_prior=update_prior,

            L=L, T=T, chains=chains, chains_sksd=chains_sksd, sksd=sksd,
            pre_steps=pre_steps, lr_pre=lr_pre, lr_encoder=lr_encoder, lr_decoder=lr_decoder,
            lr_predictor=lr_predictor, lr_prior=lr_prior, lr_hmc=lr_hmc, lr_scale=lr_scale,
            update_s_each=update_s_each, 
        )
    
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

    # ============= Modified HHVAE functions ============= #
    def preproc_x(self, x: torch.Tensor, observed_x: torch.Tensor, normalise=True):
        """
        Preprocessing tensor x with marginal VAEs. Convers xd into a sample from q(zd|xd)

        Args:
            x (torch.Tensor): input data                                        (batch_size, dim_x)
            observed_x (torch.Tensor): observation mask                         (batch_size, dim_x)
            normalise (bool, optional): normalize x first. Defaults to True.       

        Returns:
            torch.tensor:                                                       (batch_size, dim_x)
        """
        # Preprocess with marginal VAEs
        # x is normalised inside each margVAE
        xt = []
        for d, margVAE in enumerate(self.margVAEs):
            batch_d= [x[:, d].unsqueeze(-1), observed_x[:, d].unsqueeze(-1), torch.Tensor([]), torch.Tensor([])]   # select one variable
            # Preprocess 1d-data with margVAE
            xn, x_tilde = margVAE.preprocess_batch(batch_d, normalise)
            mu_z, logvar_z = margVAE.encoder(x_tilde) 

            xt.append(reparameterize(mu_z, torch.exp(logvar_z)))
            #xt.append(mu_z)
        xt = torch.cat(xt, dim=-1)

        return xt    
    
    def preprocess_batch(self, batch: tuple, normalise=True) -> tuple:
        """
        Preprocessing operations for the batch (overrides the base class function) for defining the HMC objective p(epsilon)(x, y)

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)

        Returns:
            tuple: preprocessed batch, contains (data, observed_data, target, observed_target)
        """
        x, observed_x, y, observed_y = batch
        x = x.view(-1, self.dim_x)
        xt = self.preproc_x(x, observed_x, normalise)
        xn = self.normalize_x(x)
        xo = xt * observed_x
        x_tilde = torch.cat([xo, observed_x], axis=1)

        y = y.view(-1, self.dim_y)
        observed_y = observed_y.view(-1, self.dim_y)

        # Normalize the target 
        if normalise:
            yn = self.normalize_y(y)
            yon = yn * observed_y

        y_tilde = torch.cat([yon, observed_y], axis=1)
        xy = torch.cat([x_tilde, y_tilde], axis=1)
        observed = torch.logical_or(observed_x.sum(-1, keepdim=True)>0, observed_y.sum(-1, keepdim=True)>0)

        # Define the HMC objective
        self.HMC.logp = self.logp_func(xo, observed_x, yon, observed_y, xn)  
        
        return xt, yn, xy, observed
    
    def get_var_z(self):
        """Get prior variances 

        Returns:
            torch.tensor: prior variance
        """
        return torch.clamp(self.variance, 1e-5, 0.5)

    def train_margVAEs(self):
        """
        Performs the first training stage: Pretraining of the marginal VAEs

        """
        for i, margVAE in enumerate(self.margVAEs):
            print('\n\nTraining margVAE for dimension {}'.format(i))
            trainer = pl.Trainer(
            max_epochs=self.marg_epochs,
            gpus=int(self.device.type=='cuda'),
            #default_root_dir='{}/logs/'.format(LOGDIR),,    # Uncomment if you want to save apart the marginal models
            logger=False, 
            checkpoint_callback=False,
            #save_dir='{}/logs/{}/margVAE/split_{}/'.format(LOGDIR, self.dataset, self.split_idx, i)), # Uncomment if you want to save apart the marginal models
            )
            trainer.fit(margVAE.train())
            margVAE = margVAE.to(self.device)
     
    def on_train_start(self) -> None:
        """
        Pre-training stage: Trains the marginal VAEs
        """
        self.train_margVAEs()
        deactivate(self.margVAEs)
        [margVAE.to(self.device) for margVAE in self.margVAEs]
        self.print('\n Training VAE')
        return super().on_train_start()
    
    # Uncomment for validating after each epoch
    '''def validation_step(self, batch, batch_idx, samples=1000, *args, **kwargs) -> dict:
        """
        Compute validation metrics (overrides the base class function)

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            batch_idx (int): batch index from the validation set
            samples (int): number of samples from the latent for MC. Defaults to 100

        Returns:
            dict: containing approx log likelihoods logp(x) and logp(y), the chosen metric and the ELBO                             

        """
        batch_ = [b.clone() for b in batch]
        y_true = batch_[2].clone().to(self.device)

        # We do not observe the target
        batch_[2] = torch.zeros(batch[0].shape[0], self.dim_y).to(self.device)
        batch_[3] = torch.zeros_like(batch_[2]).to(self.device)
        batch_ = [b.to(self.device) for b in batch_]
        
        # Get data
        x, observed_x, y, observed_y = batch_
        #xn = self.normalize_x(x)

        # Get data
        xn = self.normalize_x(x)

        z, yt, zy, observed = self.preprocess_batch(batch_) 
        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)
        mu_h, logvar_h = self.encoder(zy)

        h = self.sample_z(mu_h, logvar_h, samples=samples)
        theta_z = self.decoder(h)

        zd = theta_z
        #zd = reparameterize(theta_z, torch.ones_like(theta_z)*self.get_var_z())
        
        x_hat = self.build_x_hat(xn, observed_x, theta_z, sample_z=False)

        hx = torch.cat([h,x_hat],dim=-1)

        theta_y = self.predictor(hx)
        
        rec_z = self.decoder.logp(z, observed_x, z=h, theta=theta_z).sum(-1)
        rec_y = self.predictor.logp(yt, observed_y, z=hx).sum(-1)
        kls = self.encoder.regularizer(mu_h, logvar_h, observed)
        
        elbo = rec_z + rec_y - kls.sum(0).unsqueeze(-1)
        elbo = elbo[elbo!=0].mean()

        # Log-likelihood of the normalised target
        yn_true = self.normalize_y(y_true)
        rec = self.predictor.logp(yn_true, torch.ones_like(y_true), theta=theta_y)
        # mean per dimension (y is all observed in test)
        rec = rec.sum(-1, keepdim=True)

        ll_y = torch.logsumexp(rec, dim=-2) - np.log(samples)
        # mean per sample
        ll_y = ll_y.mean()

        theta_y = self.denormalize_y(theta_y)

        if self.likelihood_y == 'categorical':
            theta_y = torch.exp(theta_y)
        elif self.likelihood_y == 'bernoulli':
            theta_y = torch.sigmoid(theta_y)

        # Prediction metric
        metric = self.prediction_metric(theta_y.mean(-2), y_true)

        # Log-likelihood of the unobserved variables
        rec = []
        for d, margVAE in enumerate(self.margVAEs):
            rec.append(margVAE.decoder.logp(xn[:, d].unsqueeze(-1), 
                torch.logical_not(observed_x)[:, d].unsqueeze(-1), z=zd[..., d].unsqueeze(-1)))
        rec = torch.cat(rec, dim=-1)
        # mean per dimension
        rec = rec.sum(-1, keepdim=True) / torch.logical_not(observed_x).sum(-1, keepdim=True).unsqueeze(-2)
        ll_xu = torch.logsumexp(rec, dim=-2) - np.log(samples)

        # mean per sample
        ll_xu = ll_xu[torch.isfinite(ll_xu)].mean()

        self.log('{}_test'.format(self.prediction_metric_name), metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('ll_y_test', ll_y, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('ll_xu_test', ll_xu, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"ll_y_test": ll_y, "ll_x_test": ll_xu, "metric_test": metric, "elbo_test": elbo}'''
    
    def test_step(self, batch: tuple, batch_idx: int, samples=1000, *args, **kwargs) -> dict:
        """
        Compute test metrics (overrides the base class function)

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            batch_idx (int): batch index from the validation set
            samples (int): number of samples from the latent for MC. Defaults to 1000

        Returns:
            dict: containing approx log likelihoods logp(x) and logp(y), the chosen metric and the ELBO                             

        """
        
        batch_ = [b.clone() for b in batch]
        y_true = batch_[2].clone().to(self.device)

        # We do not observe the target
        batch_[2] = torch.zeros(batch[0].shape[0], self.dim_y).to(self.device)
        batch_[3] = torch.zeros_like(batch_[2]).to(self.device)
        batch_ = [b.to(self.device) for b in batch_]
        
        # Get data
        x, observed_x, y, observed_y = batch_

        # Get data
        xn = self.normalize_x(x)

        z, yt, zy, observed = self.preprocess_batch(batch_) 
        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)
        mu_h, logvar_h = self.encoder(zy)

        h = self.sample_z(mu_h, logvar_h, samples=samples)
        theta_z = self.decoder(h)

        zd = theta_z
        #zd = reparameterize(theta_z, torch.ones_like(theta_z)*self.get_var_z())
        
        x_hat = self.build_x_hat(xn, observed_x, theta_z, sample_z=False)

        hx = torch.cat([h,x_hat],dim=-1)

        theta_y = self.predictor(hx)
        
        # Log-likelihood of the normalised target
        yn_true = self.normalize_y(y_true)
        rec = self.predictor.logp(yn_true, torch.ones_like(y_true), theta=theta_y)
        # mean per dimension (y is all observed in test)
        rec = rec.sum(-1, keepdim=True)

        ll_y = torch.logsumexp(rec, dim=-2) - np.log(samples)
        # mean per sample
        ll_y = ll_y.mean()

        theta_y = self.denormalize_y(theta_y)

        if self.likelihood_y == 'categorical':
            theta_y = torch.exp(theta_y)
        elif self.likelihood_y == 'bernoulli':
            theta_y = torch.sigmoid(theta_y)

        # Prediction metric
        metric = self.prediction_metric(theta_y.mean(-2), y_true)

        # Log-likelihood of the unobserved variables
        rec = []
        for d, margVAE in enumerate(self.margVAEs):
            rec.append(margVAE.decoder.logp(xn[:, d].unsqueeze(-1), 
                torch.logical_not(observed_x)[:, d].unsqueeze(-1), z=zd[..., d].unsqueeze(-1)))
        rec = torch.cat(rec, dim=-1)
        # mean per dimension
        rec = rec.sum(-1, keepdim=True) / torch.logical_not(observed_x).sum(-1, keepdim=True).unsqueeze(-2)
        ll_xu = torch.logsumexp(rec, dim=-2) - np.log(samples)

        # mean per sample
        ll_xu = ll_xu[torch.isfinite(ll_xu)].mean()

        return {'ll_y': ll_y, 'metric': metric, 'll_xu': ll_xu}   
    
    def build_x_hat(self, x: torch.Tensor, observed_x: torch.Tensor, theta_z: torch.Tensor, sample_z = True) -> torch.Tensor:
        """
        Builds the imputation vector by:
            - Decoding each xd from each zd
            - Replacing the missing variables by their reconstructed mean from p(xd|z)
            - Concatenating the missing mask

        Args:
            x (torch.Tensor): input data containing zeros in the missing variables      (batch_size, dim_x)
            observed_x (torch.Tensor): observation mask                                 (batch_size, dim_x)
            theta_z (torch.Tensor): parameters of the posterior distribution            (batch_size, latent_samples, dim_x)
            sample_z (bool): sample from the posterior (True) or use the mean. Defaults to True.          

        Returns:
            torch.Tensor: resulting imputation vector                                   (batch_size, latent_samples, 2*dim_x)
        """
        if sample_z:
            z = reparameterize(theta_z, torch.ones_like(theta_z) * self.variance)
        else:
            z = theta_z

        xu = self.map_x(z)

        # reshape to fit MC repetitions
        x = x*observed_x
        x = x.repeat(xu.shape[1], 1, 1).transpose(0, 1)
        observed_x = observed_x.repeat(xu.shape[1], 1, 1).transpose(0, 1)

        # fill observed data with imputed xu
        x = x + xu*torch.logical_not(observed_x)
        indicator = torch.logical_not(observed_x)
        x_hat = torch.cat([x, indicator], dim=-1)

        return x_hat

    def map_x(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes each dimension of z into the corresponding data variable

        Args:
            z (torch.Tensor): latent code                               (batch_size, latent_samples, dim_x)

        Returns:
            torch.Tensor: decoded data x                                (batch_size, latent_samples, dim_x) 
        """
        # Preprocess with marginal VAEs
        x = []
        for d, margVAE in enumerate(self.margVAEs):
            theta_x = margVAE.to(self.device).decoder(z[..., d].unsqueeze(-1))
            if margVAE.likelihood_x=='categorical':
                xd = torch.argmax(torch.exp(theta_x), dim=-1, keepdim=True)
            elif margVAE.likelihood_x=='bernoulli':
                xd = torch.round(torch.sigmoid(theta_x))
            else:
                xd = theta_x
            x.append(xd)

        x = torch.cat(x, -1)
        return x
    
    def psample_x(self, theta: torch.Tensor, samples: int) -> torch.Tensor:
        """
        Sample x from the likelihood distributions (overrides the base class function)

        Args:
            theta (torch.Tensor): parameters of the likelihood            (batch_size, ..., dim_x)  
            samples (int): number of samples

        Returns:
            torch.Tensor: samples                                         (batch_size, ..., dim_x)  
        """
        
        extra_dims = len(theta.shape) - 1 # batch_size x ...

        if extra_dims == 1:
            theta = theta.repeat(samples, 1, 1).transpose(0, 1)
        elif extra_dims == 2:
            theta = theta.repeat(samples, 1, 1, 1).transpose(0, 1)

        # Preprocess with marginal VAEs
        x = []
        for d, margVAE in enumerate(self.margVAEs):
            theta_x = margVAE.to(self.device).decoder(theta[..., d].unsqueeze(-1))
            xd = margVAE.psample_x(theta_x, samples=1)
            x.append(xd)

        x = torch.cat(x, -1)
            
        return x
    
    def psample_y(self, theta: torch.Tensor, samples: int) -> torch.Tensor:
        """
        Sample y from the likelihood distribution (overrides the base class function)

        Args:
            theta (torch.Tensor): parameters of the likelihood            (batch_size, ..., dim_y)  
            samples (int): number of samples

        Returns:
            torch.Tensor: samples                                         (batch_size, ..., dim_y)  
        """
        extra_dims = len(theta.shape) - 1 # batch_size x ...

        if extra_dims == 1:
            theta = theta.repeat(samples, 1, 1).transpose(0, 1)
        elif extra_dims == 2:
            theta = theta.repeat(samples, 1, 1, 1).transpose(0, 1)

        if self.likelihood_y=='categorical':
            dist = torch.distributions.categorical.Categorical(logits=theta)
            y = dist.sample().unsqueeze(-1)
        elif self.likelihood_y=='bernoulli':
            dist = torch.distributions.bernoulli.Bernoulli(logits=theta)
            y = dist.sample()
        elif self.likelihood_y in ['loggaussian', 'gaussian']:
            y = reparameterize(theta, torch.ones_like(theta) * self.variance)

        return y

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale Gaussian data with precomputed mean and std

        Args:
            x (torch.Tensor): input unnormalized data      (batch_size, ..., dim_x)

        Returns:
            torch.Tensor: normalized data      (batch_size, ..., dim_x)
        """
        xn = []
        for d, margVAE in enumerate(self.margVAEs):
            if margVAE.likelihood_x in ['gaussian', 'loggaussian']:
                xn.append(margVAE.normalize_x(x[:, d].unsqueeze(-1)))
            else:
                xn.append(x[:, d].unsqueeze(-1))
        xn = torch.cat(xn, -1)
        return xn

    def denormalize_x(self, xn: torch.Tensor) -> torch.Tensor:
        """
        Denormalize Gaussian data with precomputed mean and std

        Args:
            xn (torch.Tensor): input normalized data      (batch_size, ..., dim_x)

        Returns:
            torch.Tensor: denormalized data      (batch_size, ..., dim_x)
        """
        x = []
        for d, margVAE in enumerate(self.margVAEs):
            if margVAE.likelihood_x in ['gaussian', 'loggaussian']:
                x.append(margVAE.denormalize_x(xn[:, d].unsqueeze(-1)))
            else:
                x.append(xn[:, d].unsqueeze(-1))
        x = torch.cat(x, -1)
        return x

    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        """
        Scale target with precomputed mean and std

        Args:
            y (torch.Tensor): input unnormalized target      (batch_size, ..., dim_y)

        Returns:
            torch.Tensor: normalized target      (batch_size, ..., dim_y)
        """
        if self.likelihood_y == 'gaussian':
            expand_dims = len(y.shape)-2
            mean_y = self.mean_y
            std_y = self.std_y
            for i in range(expand_dims):
                mean_y = mean_y.unsqueeze(0)
                std_y = std_y.unsqueeze(0)

            yn = (y-self.mean_y) / self.std_y
       
        elif self.likelihood_y == 'loggaussian':
            y_log = torch.log(1 + y)
            expand_dims = len(y_log.shape)-2
            mean_y = self.mean_y
            std_y = self.std_y
            for i in range(expand_dims):
                mean_y = mean_y.unsqueeze(0)
                std_y = std_y.unsqueeze(0)

            yn = (y_log-self.mean_y) / self.std_y
        else: yn = y
        return yn

    def denormalize_y(self, yn: torch.Tensor) -> torch.Tensor:
        """
        Denormalize target with precomputed mean and std

        Args:
            yn (torch.Tensor): input normalized target      (batch_size, ..., dim_y)

        Returns:
            torch.Tensor: denormalized target      (batch_size, ..., dim_y)
        """
        if self.likelihood_y == 'gaussian':
            expand_dims = len(yn.shape)-2
            mean_y = self.mean_y
            std_y = self.std_y
            for i in range(expand_dims):
                mean_y = mean_y.unsqueeze(0)
                std_y = std_y.unsqueeze(0)

            y = yn*self.std_y + self.mean_y

        elif self.likelihood_y == 'loggaussian':
            expand_dims = len(yn.shape)-2
            mean_y = self.mean_y
            std_y = self.std_y
            for i in range(expand_dims):
                mean_y = mean_y.unsqueeze(0)
                std_y = std_y.unsqueeze(0)

            logy = yn*self.std_y + self.mean_y
            y = torch.exp(logy) - 1
        else: y = yn
        return y



    # ============= Modified PL functions ============= #
    def train_dataloader(self):
        """
        Modified dataloader for getting mixed-type data

        """
        loader = get_dataset_loader(self.dataset, split='train', path=self.data_path, 
            batch_size=self.batch_size, split_idx=self.split_idx, mixed=True)

        labels = loader.dataset.labels
        if self.likelihood_y == 'loggaussian':
                labels = np.log(1 + labels)
        self.register_buffer('mean_y', torch.Tensor(labels.mean(0, keepdims=True)).to(self.device))
        self.register_buffer('std_y', torch.Tensor(labels.std(0, keepdims=True)).to(self.device))
        self.register_buffer('mean_x', torch.Tensor(loader.dataset.data.mean(0, keepdims=True)).to(self.device))
        self.register_buffer('std_x', torch.Tensor(loader.dataset.data.std(0, keepdims=True)).to(self.device))
        
        if self.likelihood_y=='bernoulli':
            if self.imbalanced_y:
                pos_class = loader.dataset.labels.sum()
                neg_class = len(loader.dataset.labels) - loader.dataset.labels.sum()
                pos_weight = [neg_class * 1.0 / pos_class]
                self.predictor.likelihood.register_buffer('pos_weight', 
                    torch.Tensor(pos_weight).to(self.device))
            else:
                self.predictor.likelihood.register_buffer('pos_weight', 
                    torch.Tensor([1]).to(self.device))
        if self.likelihood_x=='bernoulli':
            self.decoder.likelihood.register_buffer('pos_weight', 
                    torch.Tensor([1]).to(self.device))

        return loader

    # Uncomment for validating after each epoch
    '''def val_dataloader(self):
        """
        Modified dataloader for getting mixed-type data

        """
        return get_dataset_loader(self.dataset, split='test', path=self.data_path, 
        batch_size=self.batch_size, split_idx=self.split_idx, mixed=True)
    '''

    def test_dataloader(self):
        """
        Modified dataloader for getting mixed-type data

        """
        return get_dataset_loader(self.dataset, split='test', path=self.data_path, 
        batch_size=self.batch_size, split_idx=self.split_idx, mixed=True)


    
