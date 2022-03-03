# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from src.models.base import *


# ============= VAEM ============= #

class VAEM(BaseVAE):
    """
    Implements a VAEM (VAE for mixed-type data) as described in https://arxiv.org/abs/2202.04599

    """
    def __init__(self, 
            dataset: str, dim_x: int, dim_y: int, latent_dim = 10, arch='base', dim_h=256,
            likelihood_x = 'gaussian', likelihood_y = 'gaussian', variance=0.1, imbalanced_y = False,
            categories_y = 1, prediction_metric='rmse',
            batch_size=128, lr=1e-3, samples_MC = 1, data_path='../data/', split_idx=0,
            
            likelihoods_x: list = None, categories_x: list = None,
            marg_epochs=1000, lr_marg=1e-3
        ):
        """
        VAEM initialization

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

        super(VAEM, self).__init__(dataset=dataset, dim_x=dim_x, dim_y=dim_y, 
            latent_dim = latent_dim, arch=arch, dim_h=dim_h, likelihood_x = likelihood_x, likelihood_y = likelihood_y, 
            variance=variance, imbalanced_y = imbalanced_y,
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
    
    # ============= Modified BaseVAE functions ============= #
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
        Preprocessing operations for the batch (overrides the base class function)

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)

        Returns:
            tuple: preprocessed batch, contains (data, observed_data, target, observed_target)
        """
        x, observed_x, y, observed_y = batch
        x = x.view(-1, self.dim_x)
        xt = self.preproc_x(x, observed_x, normalise)        
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
        
        return xt, yn, xy, observed
    
    def get_var_z(self):
        """Get prior variance 

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
    '''def validation_step(self, batch: tuple, batch_idx: int, samples=100, *args, **kwargs) -> dict:
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

        #theta_x = self.invert_preproc_theta(theta_x)
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

    """def val_dataloader(self):
        return get_dataset_loader(self.dataset, split='test', path=self.data_path, 
        batch_size=self.batch_size, split_idx=self.split_idx, mixed=True)
    """
    
    def test_dataloader(self):    
        return get_dataset_loader(self.dataset, split='test', path=self.data_path, 
        batch_size=self.batch_size, split_idx=self.split_idx, mixed=True)


# ============= Marginal VAE ============= #

class EncoderX(nn.Module):
    """
    Implements an encoder for only input data x

    """
    def __init__(self, network: nn.Module=None, anneal_steps=0):
        """
        EncoderX Initialization

        Args:
            network (nn.Module, optional): module for computing approx. posterior parameters. Defaults to None.
            anneal_steps (int, optional): annealing steps (0 for not doing annealing). Defaults to 0.
        """
        super(EncoderX, self).__init__()

        self.encoder = network
        self.anneal_steps = anneal_steps
        self.global_step = 0

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Encode data into the parameters of the approximated posterior
        Args:
            x (torch.Tensor): input data            (batch_size, dim_x)

        Returns:
            (mu_z, logvar_z)    tensors with shape (batch_size, latent_dim)
        """
        phi = self.encoder(x.float())
        mu, logvar = torch.chunk(phi, 2, -1)
        return mu, logvar

    def regularizer(self, mu: torch.Tensor, logvar: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL regularizer

        Args:
            mu (torch.Tensor): mean of the Gaussian approx posterior
            logvar (torch.Tensor): logvar of the Gaussian approx posterior
            observed (torch.Tensor): observation mask

        Returns:
            torch.Tensor: KL result
        """

        kl = -0.5 * torch.sum(1. + logvar - mu ** 2 - torch.exp(logvar), dim=-1, keepdim=True)
        kl = kl * observed

        if self.global_step < self.anneal_steps:
            kl = kl * np.minimum( (self.global_step+1) / self.anneal_steps, 1)

        self.global_step += 1
        return kl.T


class margVAE(pl.LightningModule):
    """
    Implements a VAE for marginally encoding data dimensions

    """
    def __init__(self, 
            dataset: str, dim_x: int, dim:int=None, mixed_data: bool=None, latent_dim = 1, arch='base',
            likelihood_x = 'gaussian', categories_x = 1, imbalanced_x = True, variance=0.1, dim_h = 32,
            anneal_steps = 0,
            batch_size=128, lr=1e-2, samples_MC = 1, data_path='../data/', split_idx=0):
        """
        margVAE Initialization

        Args:
            dataset (str): name of the dataset (boston, mnist, ...)
            dim_x (int): input data dimension
            dim (int, optional): index of the dimension (column) to encode. Defaults to None.
            mixed_data (bool, optional): indicator for loading mixed-type data (True) or not (False). Defaults to None.
            latent_dim (int, optional): dimension of the latent space. Defaults to 10.
            arch (str, optional): name of the architecture for encoder/decoder from the 'archs' file. Defaults to 'base'.
            likelihood_x (str, optional): input data likelihood type. Defaults to 'gaussian'.
            categories_x (list, optional): list of number of categories per variable. Defaults to None.
            imbalanced_y (bool, optional): True for compensating imbalanced classification. Defaults to False.
            variance (float, optional): fixed variance for Gaussian likelihoods. Defaults to 0.1.
            dim_h (int, optional): dimension of the hidden vectors. Defaults to 256.
            anneal_steps (int, optional): annealing steps (0 for not doing annealing). Defaults to 0.
            batch_size (int, optional): _description_. Defaults to 128.
            lr (float, optional): learning rate for the parameter optimization. Defaults to 1e-3.
            samples_MC (int, optional): number of MC samples for computing the ELBO. Defaults to 1.
            data_path (str, optional): path to load/save the data. Defaults to '../data/'.
            split_idx (int, optional): idx of the training split. Defaults to 0.
        """

        super(margVAE, self).__init__()

        encoder = nn.Sequential(nn.Linear(categories_x+1, dim_h), nn.ReLU(), nn.Linear(dim_h, 2 * latent_dim))
        decoder = nn.Sequential(nn.Linear(latent_dim, dim_h), nn.ReLU(), nn.Linear(dim_h, categories_x))

        self.encoder = EncoderX(encoder, anneal_steps=anneal_steps)
        self.decoder = Decoder(likelihood=likelihood_x, network=decoder, variance=variance)
        self.prior = Prior(type='standard')

        self.dataset = dataset
        self.dim_x = dim_x
        self.categories_x = categories_x
        self.imbalanced_x = imbalanced_x
        self.latent_dim = latent_dim
        self.arch = arch
        self.likelihood_x = likelihood_x
        self.variance=variance
        self.dim_h = dim_h
        self.dim = dim # for modeling one dimension (marginal VAE)
        self.mixed_data = mixed_data
        self.anneal_steps = anneal_steps
        
        self.batch_size = batch_size
        self.lr = lr
        self.samples_MC = samples_MC
        self.data_path = data_path
        self.split_idx = split_idx
        self.dataset = dataset

        # call train_dataloader to store the means and stds for scaling
        self.train_dataloader()
        
        self.save_hyperparameters('dim_x', 'likelihood_x', 'variance', 'dim_h', 'anneal_steps', 'dim', 'mixed_data', 'categories_x', 'latent_dim', 'arch', 'batch_size', 'lr',  'samples_MC', 'dataset')
 
    def forward(self, batch: tuple, samples: int) -> tuple:
        """
        Computes the mean ELBO for a given batch

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
        x, observed_x, _, _ = batch
        xt, x_tilde = self.preprocess_batch(batch) 
        # xt is the preprocessed input (xt=x if no preprocessing)
        mu_z, logvar_z = self.encoder(x_tilde)

        z = self.sample_z(mu_z, logvar_z, samples=samples)
        theta_x = self.decoder(z)

        rec_x = self.decoder.logp(xt, observed_x, theta=theta_x).sum(-1)
        kls = self.encoder.regularizer(mu_z, logvar_z, observed_x.sum(-1, keepdim=True))

        if self.likelihood_x in ['categorical', 'bernoulli']:
            #beta = 0.25
            beta = 0.2
        else: beta = 1.0

        elbo = rec_x - beta * kls.sum(0).unsqueeze(-1)

        elbo = elbo[elbo!=0].mean()
        rec_x = rec_x[rec_x!=0].mean()

        kl_mean = torch.zeros(len(kls)).to(self.device)
        for l, kl in enumerate(kls):
            kl_mean[l]= kl[kl!=0].mean()

        return -elbo, rec_x, kl_mean
        
    def training_step(self, batch: tuple, batch_idx: int, logging: bool=True) -> torch.Tensor:
        """
        Returns the loss (negative ELBO) for the minimization

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            batch_idx (int): batch index from the training set
            logging (bool): log metrics into Tensorboard (True). Default True

        Returns:
            torch.Tensor: mean loss (negative ELBO)                                  

        """
        loss, rec_x, kls = self.forward(batch, samples=self.samples_MC)

        self.log('ELBO', -loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.log('rec_x', rec_x, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for l, kl in enumerate(kls):
            self.log('kl_{:d}'.format(l), kl, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    # Uncomment for validating after each epoch
    '''def validation_step(self, batch, batch_idx, samples=100, *args, **kwargs) -> dict:
        """
        Compute validation metrics

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            batch_idx (int): batch index from the validation set
            samples (int): number of samples from the latent for MC. Defaults to 100

        Returns:
            dict: containing approx log likelihoods logp(x) and logp(y), and the chosen metric                          

        """
        loss, rec_x, kls = self.forward(batch, samples=self.samples_MC)

        self.log('ELBO', -loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.log('rec_x_test', rec_x, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for l, kl in enumerate(kls):
            self.log('kl_{:d}_test'.format(l), kl, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss"""
    
    
    """def test_step(self, batch, batch_idx, samples=10000, *args, **kwargs):
        
        y_true = batch[2].to(self.device)
        batch_ = [b.clone().to(self.device) for b in batch]
         
        # We do not observe the target
        batch_[2] = torch.zeros(batch[0].shape[0], self.dim_y).to(self.device)
        batch_[3] = torch.zeros_like(batch_[2]).to(self.device)
        
        # Get data
        x, observed_x, y, observed_y = batch_
        xn = self.normalize_x(x)

        xt, yt, xy, observed = self.preprocess_batch(batch_) 

        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)

        mu_z, logvar_z = self.encoder(xy)
        z = self.sample_z(mu_z, logvar_z, samples=samples)
        theta_x = self.decoder(z)
        x_hat = self.build_x_hat(xt, observed_x, theta_x)
        zx = torch.cat([z,x_hat],dim=-1)

        theta_y = self.predictor(zx)

        # Log-likelihood of the normalised target
        yn_true = self.normalize_y(y_true)
        rec = self.predictor.logp(yn_true, torch.ones_like(y_true), theta=theta_y)
        # mean per dimension (y is all observed in test)
        rec = rec.sum(-1, keepdim=True)

        ll_y = torch.logsumexp(rec, dim=-2) - np.log(samples)
        # mean per sample
        ll_y = ll_y.mean()

        theta_y = self.denormalize_y(theta_y)

        # Prediction metric
        metric = self.prediction_metric(theta_y.mean(-2), y_true)

        #theta_x = self.invert_preproc_theta(theta_x)
        # Log-likelihood of the unobserved variables
        rec = []
        for d, margVAE in enumerate(self.margVAEs):
            rec.append(margVAE.decoder.logp(xn[:, d].unsqueeze(-1), 
                torch.logical_not(observed_x)[:, d].unsqueeze(-1), z=theta_x[:,:,d].unsqueeze(-1)))
        rec = torch.cat(rec, dim=-1)
        # mean per dimension
        rec = rec.sum(-1, keepdim=True) / torch.logical_not(observed_x).sum(-1, keepdim=True).unsqueeze(-2)
        ll_xu = torch.logsumexp(rec, dim=-2) - np.log(samples)

        # mean per sample
        ll_xu = ll_xu[torch.isfinite(ll_xu)].mean()

        return {'ll_y': ll_y, 'metric': metric, 'll_xu': ll_xu}'''
    
    def preprocess_batch(self, batch: tuple, normalise=True) -> tuple:
        """
        Preprocessing operations for the batch

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            normalise (bool): for normalising the data (True) or not (False)

        Returns:
            tuple: preprocessed batch, contains (data, observed_data, target, observed_target)
        """
        batch = [b.to(self.device) for b in batch]
        x, observed_x, _, _ = batch
        x = x.view(-1, self.dim_x)

        if normalise:
            # Normalize the data 
            xn = self.normalize_x(x)
        else: xn = x

        xo = xn * observed_x
        x_tilde = torch.cat([xo, observed_x], axis=1)

        return xn, x_tilde

    def sample_z(self, mu: torch.Tensor, logvar: torch.Tensor, samples=1) -> torch.Tensor:
        """
        Draw latent samples from a given approx posterior parameterized by mu and logvar

        Args:
            mu (torch.Tensor): tensor with the means                            (batch_size, latent_dim)
            logvar (torch.Tensor): tensor with the log variances                (batch_size, latent_dim)
            samples (int, optional): number of samples. Defaults to 1.          

        Returns:
            torch.Tensor: latent samples
        """
        # Repeat samples_MC times for Monte Carlo
        mu = mu.repeat(samples, 1, 1).transpose(0, 1)
        logvar = logvar.repeat(samples, 1, 1).transpose(0, 1)
        # Reparametrization
        z = reparameterize(mu, torch.exp(logvar))
        return z

    def psample_x(self, theta: torch.Tensor, samples: int) -> torch.Tensor:
        """
        Sample x from the likelihood distribution

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

        if self.likelihood_x=='categorical':
            dist = torch.distributions.categorical.Categorical(logits=theta)
            x = dist.sample().unsqueeze(-1)
        elif self.likelihood_x=='bernoulli':
            dist = torch.distributions.bernoulli.Bernoulli(logits=theta)
            x = dist.sample()
        elif self.likelihood_x in ['loggaussian', 'gaussian']:
            x = reparameterize(theta, torch.ones_like(theta) * self.variance)

        return x

    def log_likelihood_xu(self, batch: tuple, samples=100):
        """
        Computes the approximated log likelihood of the data p(x)

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            samples (int): number of samples from the latent for MC. Defaults to 100.

        Returns:
            torch.Tensor: mean log likelihood of the data
        """
        batch_ = [b.clone() for b in batch]
         
        # We do not observe the target
        batch_[2] = torch.zeros(batch[0].shape[0], self.dim_y).to(self.device)
        batch_[3] = torch.zeros_like(batch_[2]).to(self.device)
        

        # Get data
        x, observed_x, y, observed_y = batch_
        xt, yt, xy, observed = self.preprocess_batch(batch_) 
        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)

        mu_z, logvar_z = self.encoder(xy)
        z = self.sample_z(mu_z, logvar_z, samples=samples)
        theta_x = self.decoder(z)

        # Log-likelihood of the unobserved variables
        rec = self.decoder.logp(xt, torch.logical_not(observed_x), theta=theta_x)
        ll_xu = torch.logsumexp(rec, dim=-2) - torch.log(torch.Tensor([samples])).to(self.device)

        # divide each xu by the number of variables:
        # mean per sample
        ll_xu = ll_xu[torch.isfinite(ll_xu)].mean()
        
        # When all the data is observed, logp will be nan
        return ll_xu

    def elbo(self, batch: tuple, samples=1000) -> torch.Tensor:
        """
        Computes the mean ELBO of a batch

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            samples (int): number of samples from the latent for MC. Defaults to 1000.

        Returns:
            torch.Tensor: mean elbo
        """
        # Get data
        x, observed_x, _, _ = batch
        xt, x_tilde = self.preprocess_batch(batch) 
        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)
        mu_z, logvar_z = self.encoder(x_tilde)
        z = self.sample_z(mu_z, logvar_z, samples=samples)
        theta_x = self.decoder(z)

        rec_x = self.decoder.logp(xt, observed_x, z=z, theta=theta_x).sum(-1)
        kls = self.encoder.regularizer(mu_z, logvar_z, observed_x.sum(-1, keepdim=True))
        elbo = rec_x - kls.sum(0).unsqueeze(-1)
        elbo = elbo[elbo!=0].mean()

        return elbo

    def elbo_iwae(self, batch: tuple, samples=1000):
        """
        Computes the ELBO (following the Importance Weighted Autoencoder, IWAE) of each datapoint in a batch

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            samples (int): number of samples from the latent for MC. Defaults to 1000.

        Returns:
            torch.Tensor: iwae elbo per datapoint
        """
        # Get data
        x, observed_x, _, _ = batch
        xt, x_tilde = self.preprocess_batch(batch) 
        mu_z, logvar_z = self.encoder(x_tilde)

        z = self.sample_z(mu_z, logvar_z, samples=samples)


        logp = self.logp(xt, observed_x, z)
        logq = self.encoder.logq(z, xt, observed_x)

        log_w = logp - logq


        elbo_iwae = logmeanexp(log_w, -1)  # batch_size, R

        return elbo_iwae.mean()

    def logp(self, xt, observed_x, z):
        
        theta_x = self.decoder(z)
        x_hat = self.build_x_hat(xt, observed_x, theta_x)

        logpx_z = self.decoder.logp(xt, observed_x, theta=theta_x).sum(-1)

        logpz = self.prior(z, observed_x.sum(-1, keepdim=True))

        logp = logpx_z + logpz

        return logp

    def logp_func(self, xt: torch.Tensor, observed_x: torch.Tensor, yt: torch.Tensor, observed_y: torch.Tensor):
        """
        Returns a function for computing logp(x, z) with fixed x (only depending on z). This function is used as HMC objective.

        Args:
            xt (torch.Tensor): normalized and preprocessed data                                                                             (batch_size, dim_x)
            observed_x (torch.Tensor): observation mask of the data                                                                         (batch_size, dim_x)
            yt (torch.Tensor): normalized target                                                                                            (batch_size, dim_y)
            observed_y (torch.Tensor): observation mask of the target                                                                       (batch_size, dim_y)
            xn (torch.Tensor, optional): normalized data when xt is a transformation (preprocessing or embedding). Defaults to None.        (batch_size, dim_x)

        Returns:
            function depending on z ( logp(z, x) for fixed x )
        """
        def logp(z):
            theta_x = self.decoder(z)
            x_hat = self.build_x_hat(xt, observed_x, theta_x)

            logpx_z = self.decoder.logp(xt, observed_x, theta=theta_x).sum(-1)

            logpz = self.prior(z, observed_x.sum(-1, keepdim=True))

            logp = logpx_z + logpz
            return logp
        return logp

    def logq(self, z: torch.Tensor, xt: torch.Tensor, observed_x: torch.Tensor, yt: torch.Tensor, observed_y: torch.Tensor) ->  torch.Tensor:
        """
        Computes the log prob of a latent sample under approximated Gaussian posterior given by the encoder

        Args:
            z (torch.Tensor): latent samples                                                 (batch_size, latent_samples, latent_dim)
            xt (torch.Tensor): normalized and preprocessed data                              (batch_size, dim_x)
            observed_x (torch.Tensor): observation mask of the data                          (batch_size, dim_x)
            yt (torch.Tensor): normalized target                                             (batch_size, dim_y)
            observed_y (torch.Tensor): observation mask of the target                        (batch_size, dim_y)

        Returns:
            torch.Tensor: log probs                                                          (batch_size, 1) 
        """
        return self.encoder.logq(z, xt, observed_x)  

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale data with precomputed mean and std

        Args:
            x (torch.Tensor): input unnormalized data      (batch_size, ..., dim_x)

        Returns:
            torch.Tensor: normalized data      (batch_size, ..., dim_x)
        """
        if self.likelihood_x == 'gaussian':
            expand_dims = len(x.shape)-2
            mean_x = self.mean_x
            std_x = self.std_x
            for i in range(expand_dims):
                mean_x = mean_x.unsqueeze(0)
                std_x = std_x.unsqueeze(0)

            xn = (x-self.mean_x) / self.std_x
        elif self.likelihood_x == 'categorical':
            xn = F.one_hot(x.reshape(-1).long(), num_classes=self.categories_x)
        elif self.likelihood_x == 'loggaussian':
            x_log = torch.log(1 + x)
            expand_dims = len(x_log.shape)-2
            mean_x = self.mean_x
            std_x = self.std_x
            for i in range(expand_dims):
                mean_x = mean_x.unsqueeze(0)
                std_x = std_x.unsqueeze(0)

            xn = (x_log-self.mean_x) / self.std_x
        else: xn = x
        return xn

    def denormalize_x(self, xn: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data with precomputed mean and std

        Args:
            xn (torch.Tensor): input normalized data      (batch_size, ..., dim_x)

        Returns:
            torch.Tensor: denormalized data      (batch_size, ..., dim_x)
        """
        if self.likelihood_x == 'gaussian':
            expand_dims = len(xn.shape)-2
            mean_x = self.mean_x
            std_x = self.std_x
            for i in range(expand_dims):
                mean_x = mean_x.unsqueeze(0)
                std_x = std_x.unsqueeze(0)

            x = xn*self.std_x + self.mean_x
        elif self.likelihood_x == 'loggaussian':
            expand_dims = len(xn.shape)-2
            mean_x = self.mean_x
            std_x = self.std_x
            for i in range(expand_dims):
                mean_x = mean_x.unsqueeze(0)
                std_x = std_x.unsqueeze(0)

            logx = xn*self.std_x + self.mean_x
            x = torch.exp(logx) - 1
            x[x<0] = 0
        else: x = xn
        return x

    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr, weight_decay=0.01)
        
        return [opt]

    def train_dataloader(self):
        loader = get_dataset_loader(self.dataset, split='train', path=self.data_path, 
            batch_size=self.batch_size, split_idx=self.split_idx, dim=self.dim, mixed=self.mixed_data, num_workers=0)

        data = torch.Tensor(loader.dataset.data)[:, self.dim].to(self.device)

        if  self.likelihood_x in ['gaussian', 'loggaussian']:  
            if self.likelihood_x == 'loggaussian':
                data = torch.log(1 + data)
            mean_x = data.mean().reshape(1, 1)
            std_x = data.std().reshape(1, 1)
            self.register_buffer('mean_x', mean_x)
            self.register_buffer('std_x', std_x)

        if self.likelihood_x=='bernoulli':
            if self.imbalanced_x:
                pos_class = data.sum()
                neg_class = len(data) - pos_class
                pos_weight = [neg_class * 1.0 / pos_class]
                self.decoder.likelihood.register_buffer('pos_weight', 
                    torch.Tensor(pos_weight).to(self.device))
            else:
                self.decoder.likelihood.register_buffer('pos_weight', 
                    torch.Tensor([1]).to(self.device))
        
        return loader

    """def val_dataloader(self):
        return get_dataset_loader(self.dataset, split='val', path=self.data_path, 
        batch_size=self.batch_size, split_idx=self.split_idx, dim=self.dim, mixed=self.mixed_data, num_workers=0)
    """

    def test_dataloader(self):
        return get_dataset_loader(self.dataset, split='test', path=self.data_path, 
        batch_size=self.batch_size, split_idx=self.split_idx, dim=self.dim, mixed=self.mixed_data, num_workers=0)
             

    