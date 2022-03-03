from torch.nn.modules.loss import BCEWithLogitsLoss
import torch
import numpy as np
from src.datasets import *
from src.models.archs import *
from src.mutual_information import *
import pytorch_lightning as pl
from torchmetrics import AUROC
from torch.nn import functional as F
import time

# ============= VAE submodules ============= #

class Likelihood(nn.Module):
    """
    Implements the likelihood functions
    """
    def __init__(self, type='gaussian', variance=0.1):
        """
        Likelihood initialization

        Args:
            type (str, optional): likelihood type ('gaussian', 'categorical', 'loggaussian' or 'bernoulli'). Defaults to 'gaussian'.
            variance (float, optional): fixed variance for gaussian/loggaussian variables. Defaults to 0.1.
        """
        super(Likelihood, self).__init__()
        self.type=type
        self.variance=variance # only for Gaussian
    
    def forward(self, theta: torch.Tensor, data: torch.Tensor, observed: torch.Tensor, variance=None) -> torch.Tensor:
        """
        Computes the log probability of a given data under parameters theta

        Args:
            theta (torch.Tensor): tensor with params                (batch_size, latent_samples, dim_data)
            data (torch.Tensor): tensor with data                   (batch_size, dim_data)
            observed (torch.Tensor): tensor with observation mask   (batch_size, dim_data)
            variance (float, optional): Gaussian fixed variance (None for using the predefined). Defaults to None.

        Returns:
            torch.Tensor: tensor with the log probs                 (batch_size, latent_samples, dim_data)
        """
        
        # ============= Gaussian ============= #
        if self.type in ['gaussian', 'loggaussian']:
            # If variance is not specified we use the predefined
            if variance==None:
                variance = torch.ones_like(theta) * self.variance

            # Add dimension for latent samples
            data = data.unsqueeze(1)
            observed = observed.unsqueeze(1)
            # Make sure we have 0s in the missing positions
            data = data*observed
            theta = theta*observed
            # Compute logp
            mu = theta
            logvar = torch.log(variance) * observed
            cnt = (np.log(2 * np.pi) + logvar) * observed
            logp = -0.5 * (cnt + (data - mu) * torch.exp(-logvar) * (data - mu))

        # ============= Categorical ============= #
        elif self.type=='categorical':
            if data.shape[-1] > 1:
                # data is one-hot encoded
                data = torch.where(data==1)[1].unsqueeze(-1)

            data = data.repeat(theta.shape[-2], 1, 1).permute(1, 0, 2)[:, :, 0]
            logits = theta.permute(0, 2, 1)
            logp=-nn.NLLLoss(reduction='none')(input=logits, target=data.long()).unsqueeze(-1)
            logp = logp * observed.unsqueeze(-1)

        # ============= Bernoulli ============= #
        elif self.type=='bernoulli':
            data = data.repeat(theta.shape[-2], 1, 1).permute(1, 0, 2)
            logp = -BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)(theta, data)
            logp = logp * observed.unsqueeze(-2)
        
        return logp

    def sample(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Returns samples or modes from the likelihood distributions

        Args:
            theta (torch.Tensor): tensor with the probs             (B, latent_samples, dim_data)

        Returns:
            torch.Tensor: samples                                   (B, latent_samples, dim_data) for Gaussian
                          or modes                                  (B, latent_samples, 1)        for Categorical
        """
        # ============= Gaussian ============= #
        if self.type in ['gaussian', 'loggaussian']:
            var = torch.ones_like(theta) * self.variance
            x = reparameterize(theta, var)
        
        # ============= Categorical ============= #
        if self.type =='categorical':
            x = torch.argmax(theta.exp(), -1, keepdim=True)

        # ============= Bernoulli ============= #
        if self.type =='bernoulli':
            x = torch.round(torch.sigmoid(theta))

        return x

class Prior(nn.Module):
    """
    Implements a Prior distribution
    """
    def __init__(self, type='standard'):
        """
        Prior initialization

        Args:
            type (str, optional): prior type. Defaults to 'standard'.
        """
        super(Prior, self).__init__()
        self.type=type
    
    def forward(self, z: torch.Tensor, observed: torch.Tensor=None) -> torch.Tensor:
        """
        Computes the log prob of given latent z under the prior

        Args:
            z (torch.Tensor): latent samples                    (B, latent_samples, latent_dim)
            observed (torch.Tensor, optional): optional observed mask. Defaults to None.  

        Returns:
            torch.Tensor: tensor with the log probs             (B, latent_samples)
        """
        if self.type=='standard':
            cnt = z.shape[-1] * np.log(2 * np.pi)
            logp = -0.5 * (cnt + torch.sum((z) ** 2, dim=-1))
            if observed!=None:
                logp = logp * observed
        return logp

class Decoder(nn.Module):
    """
    Implements a decoder

    """
    def __init__(self, likelihood = 'gaussian', network: nn.Module=None, variance=0.1):
        """
        Initialization of the decoder

        Args:
            likelihood (str, optional): likelihood type ('gaussian', 'loggaussian', 'categorical', 'bernoulli'). Defaults to 'gaussian'.
            network (nn.Module, optional): module for computing likelihood parameters. Defaults to None.
            variance (float, optional): Gaussian fixed variance. Defaults to 0.1.
        """
        super(Decoder, self).__init__()

        self.decoder = network
        self.likelihood = Likelihood(likelihood, variance=variance)
        self.distribution = likelihood
        self.variance = variance

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the likelihood parameters or logits for the discrete distributions

        Args:
            z (torch.Tensor): tensor with latent samples            (batch_size, latent_samples, latent_dim)

        Returns:
            torch.Tensor: tensor with likelihood parameters         (batch_size, latent_samples, data_dim)
        """

        theta = self.decoder(z)
        if self.distribution=='bernoulli':
            theta = theta
        elif self.distribution=='categorical':
            theta = torch.nn.LogSoftmax(dim=-1)(theta)
        return theta
    
    def logp(self, x: torch.Tensor, observed: torch.Tensor, z: torch.Tensor=None, theta: torch.Tensor=None, variance: float=None) -> torch.Tensor:
        """
        Computes the log probability of given x under likelihood parameterized by theta

        Args:
            x (torch.Tensor): data tensor                                                                                   (batch_size, x_dim)
            observed (torch.Tensor): observation mask                                                                       (batch_size, x_dim)
            z (torch.Tensor, optional): latent samples (None if theta is given). Defaults to None.                          (batch_size, latent_samples, latent_dim)
            theta (torch.Tensor, optional): likelihood params (None to compute from given z). Defaults to None.             (batch_size, latent_samples, x_dim)
            variance (float, optional): Gaussian variance (None for using the predefined fixed variance). Defaults to None.            

        Returns:
            torch.Tensor: tensor with log probs
        """
        if theta==None:
            # if theta==None z is needed
            theta = self.forward(z)
        logp = self.likelihood(theta, x, observed, variance)
        return logp

class Encoder(nn.Module):
    """
    Implements an encoder

    """
    def __init__(self, network: nn.Module=None):
        """
        Encoder initialization

        Args:
            network (nn.Module, optional): module for computing approx. posterior parameters. Defaults to None.
        """
        super(Encoder, self).__init__()

        self.encoder = network

    def forward(self, x):
        phi = self.encoder(x)
        mu, logvar = torch.chunk(phi, 2, -1)
        return mu, logvar

    def regularizer(self, mu, logvar, observed):
        kl = -0.5 * torch.sum(1. + logvar - mu ** 2 - torch.exp(logvar), dim=-1, keepdim=True)
        kl = kl * observed
        return kl.T
    
    def logq(self, z, xt, observed_x, yt, observed_y):
        # xt and yt must be normalised
        xo = xt * observed_x
        x_tilde = torch.cat([xo, observed_x], axis=1)
        # Normalize the target 
        yo = yt * observed_y

        y_tilde = torch.cat([yo, observed_y], axis=1)

        xy = torch.cat([x_tilde, y_tilde], axis=1)

        mu_z, logvar_z = self.forward(xy)
        mu_z = mu_z.unsqueeze(-2)
        logvar_z = logvar_z.unsqueeze(-2)
        cnt = mu_z.shape[-1] * np.log(2 * np.pi) + torch.sum(logvar_z, dim=-1)
        logqz_xy = -0.5 * (cnt + torch.sum((z - mu_z)**2 * torch.exp(-logvar_z), dim=-1))

        observed = torch.logical_or(observed_x.sum(-1, keepdim=True)>0, observed_y.sum(-1, keepdim=True)>0)
        logqz_xy = logqz_xy * observed

        return logqz_xy


# ============= Vanilla VAE ============= #

class BaseVAE(pl.LightningModule):
    """
    Implements the structure of a vanilla VAE https://arxiv.org/abs/1312.6114

    """
    def __init__(self, 
        dataset: str, dim_x: int, dim_y: int, latent_dim = 10, arch='base', dim_h=256,
        likelihood_x = 'gaussian', likelihood_y = 'gaussian', variance=0.1, imbalanced_y = False,
        categories_y = 1, prediction_metric='rmse',
        batch_size=128, lr=1e-3, samples_MC = 1, data_path='../data/', split_idx=0):
        """
        VAE initialization

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
        """

        super(BaseVAE, self).__init__()

        encoder, decoder, predictor = get_arch(dim_x, dim_y, latent_dim, arch, categories_y=categories_y, dim_h=dim_h)

        self.encoder = Encoder(encoder)
        self.decoder = Decoder(likelihood=likelihood_x, network=decoder, variance=variance)
        self.predictor = Decoder(likelihood=likelihood_y, network=predictor, variance=variance)
        self.prior = Prior(type='standard')
        self.dataset = dataset
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.latent_dim = latent_dim
        self.arch = arch
        self.dim_h = dim_h
        self.likelihood_x = likelihood_x
        self.likelihood_y = likelihood_y
        self.variance = variance
        self.imbalanced_y = imbalanced_y
        self.categories_y = categories_y
        self.prediction_metric_name = prediction_metric
        self.batch_size = batch_size
        self.lr = lr
        self.samples_MC = samples_MC
        self.data_path = data_path
        self.split_idx = split_idx
        self.dataset = dataset
        self.validation=False

        self.save_hyperparameters('dim_x', 'dim_y', 'likelihood_x', 'likelihood_y', 'variance', 'imbalanced_y', 'categories_y',
             'latent_dim', 'arch', 'batch_size', 'lr', 'dim_h', 'samples_MC', 'dataset', 'data_path')

        # call train_dataloader to store the means and stds for scaling
        self.train_dataloader()
    
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

        rec_x = self.decoder.logp(xt, observed_x, z=z, theta=theta_x).sum(-1)
        rec_y = self.predictor.logp(yt, observed_y, z=zx).sum(-1)
        kls = self.encoder.regularizer(mu_z, logvar_z, observed)
        
        elbo = rec_x + rec_y - kls.sum(0).unsqueeze(-1)

        elbo = elbo[elbo!=0].mean()
        rec_x = rec_x[rec_x!=0].mean()
        rec_y = rec_y[rec_y!=0].mean()

        kl_mean = torch.zeros(len(kls)).type_as(rec_x)
        for l, kl in enumerate(kls):
            kl_mean[l]= kl[kl!=0].mean()

        return -elbo, rec_x, rec_y, kl_mean
        
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
        loss, rec_x, rec_y, kls = self.forward(batch, samples=self.samples_MC)

        self.log('ELBO', -loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if logging:
            self.log('-rec_x', -rec_x, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('-rec_y', -rec_y, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            for l, kl in enumerate(kls):
                self.log('kl_{:d}'.format(l), kl, on_step=False, on_epoch=True, prog_bar=False, logger=True)

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
            dict: containing approx log likelihoods logp(x) and logp(y), the chosen metric and the ELBO                             

        """

        self.validation=True

        y_true = batch[2]
        batch_ = [b.clone() for b in batch]
         
        # We do not observe the target in test
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

        rec_x = self.decoder.logp(xt, observed_x, z=z, theta=theta_x).sum(-1)
        rec_y = self.predictor.logp(yt, observed_y, z=zx).sum(-1)
        kls = self.encoder.regularizer(mu_z, logvar_z, observed)
        
        elbo = rec_x + rec_y - kls.sum(0).unsqueeze(-1)

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

        # Prediction metric
        metric = self.prediction_metric(theta_y.mean(-2), y_true)

        theta_x = self.invert_preproc(theta_x)
        # Log-likelihood of the unobserved variables
        rec = self.decoder.logp(xn, torch.logical_not(observed_x), theta=theta_x)
        # mean per dimension
        rec = rec.sum(-1, keepdim=True) / torch.logical_not(observed_x).sum(-1, keepdim=True).unsqueeze(-2)
        ll_xu = torch.logsumexp(rec, dim=-2) - np.log(samples)
        
        # mean per sample
        ll_xu = ll_xu[torch.isfinite(ll_xu)].mean()

        self.validation=False
        
        return {"ll_y_test": ll_y, "ll_x_test": ll_xu, "metric_test": metric, "elbo_test": elbo}
 
    def validation_epoch_end(self, outputs: list):
        """
        Compute mean epoch validation metrics from the batches

        Args:
            outputs (list): containing the dict with metrics from each batch (output of the validation_step)
        """
        ll_y = torch.stack(
            [x["ll_y_test"] for x in outputs]).mean()
        ll_xu = torch.stack(
            [x["ll_x_test"] for x in outputs]).mean()
        metric = torch.stack(
            [x["metric_test"] for x in outputs]).mean()
        elbo = torch.stack(
            [x["elbo_test"] for x in outputs]).mean()
        self.log('{}_test'.format(self.prediction_metric_name), metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('ll_y_test', ll_y, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('ll_xu_test', ll_xu, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('elbo_test', elbo, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    '''

    def test_step(self, batch: tuple, batch_idx: int, samples=1000, *args, **kwargs):
        """
        Compute test metrics

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            batch_idx (int): batch index from the validation set
            samples (int): number of samples from the latent for MC. Defaults to 1000

        Returns:
            dict: containing approx log likelihoods logp(x) and logp(y), the chosen metric and the ELBO                             

        """
                
        y_true = batch[2].to(self.device)
        batch_ = [b.clone() for b in batch]
        batch_ = [b.to(self.device) for b in batch_]
         
        # We do not observe the target in test
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
        if self.likelihood_y == 'categorical':
            theta_y = torch.exp(theta_y)
        elif self.likelihood_y == 'bernoulli':
            theta_y = torch.sigmoid(theta_y)

        # Prediction metric
        metric = self.prediction_metric(theta_y.mean(-2), y_true)

        # Log-likelihood of the unobserved variables
        rec = self.decoder.logp(xt, torch.logical_not(observed_x), theta=theta_x)
        # mean per dimension
        rec = rec.sum(-1, keepdim=True) / torch.logical_not(observed_x).sum(-1, keepdim=True).unsqueeze(-2)
        ll_xu = torch.logsumexp(rec, dim=-2) - torch.log(torch.Tensor([samples])).to(self.device)

        # mean per sample
        ll_xu = ll_xu[torch.isfinite(ll_xu)].mean()

        
        return {'ll_y': ll_y, 'metric': metric, 'll_xu': ll_xu}    
    
    def test_epoch_end(self, outputs: list, save=True) -> None:
        """
        Compute mean epoch test metrics from the batches

        Args:
            outputs (list): containing the dict with metrics from each batch (output of the validation_step)
        """

        metric_mean =  torch.mean(torch.stack([o['metric'] for o in outputs]))
        ll_y_mean =  torch.mean(torch.stack([o['ll_y'] for o in outputs]))
        ll_xu_mean =  torch.mean(torch.stack([o['ll_xu'] for o in outputs]))

        metrics = {'ll_y_mean': ll_y_mean, 'metric_mean': metric_mean, 'll_xu_mean': ll_xu_mean}
        
        if save:
            metrics_np = {
                    'll_y': metrics['ll_y_mean'].cpu().detach().numpy(),
                    'll_xu': metrics['ll_xu_mean'].cpu().detach().numpy(),
                    'metric': metrics['metric_mean'].cpu().detach().numpy(),
                }
            np.save("{}/test_metrics".format(self.logger.log_dir), metrics_np)

            self.log("final_test_ll_y", ll_y_mean)
            self.log("final_test_ll_xu", ll_xu_mean)
            self.log("final_test_metric", metric_mean)

        return metrics
    
    def preprocess_batch(self, batch: tuple) -> tuple:
        """
        Preprocessing operations for the batch

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)

        Returns:
            tuple: preprocessed batch, contains (data, observed_data, target, observed_target)
        """
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
        
        return xn, yn, xy, observed

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
        # Repeat samples times for Monte Carlo
        mu = mu.repeat(samples, 1, 1).transpose(0, 1)
        logvar = logvar.repeat(samples, 1, 1).transpose(0, 1)
        # Reparametrization
        z = reparameterize(mu, torch.exp(logvar))
        return z

    def build_x_hat(self, x: torch.Tensor, observed_x: torch.Tensor, theta_x: torch.Tensor) -> torch.Tensor:
        """
        Builds the imputation vector by:
            - Replacing the missing variables by their reconstructed mean from p(xu|z)
            - Concatenating the missing mask

        Args:
            x (torch.Tensor): input data containing zeros in the missing variables      (batch_size, dim_x)
            observed_x (torch.Tensor): observation mask                                 (batch_size, dim_x)
            theta_x (torch.Tensor): parameters of the likelihood distribution           (batch_size, latent_samples, dim_x)

        Returns:
            torch.Tensor: resulting imputation vector                                   (batch_size, latent_samples, 2*dim_x)
        """
        
        # reshape to fit MC repetitions
        x = x*observed_x
        x = x.repeat(theta_x.shape[1], 1, 1).transpose(0, 1)
        observed_x = observed_x.repeat(theta_x.shape[1], 1, 1).transpose(0, 1)

        if self.likelihood_x=='categorical':
            xu = torch.argmax(theta_x, dim=-1, keepdim=True)
        elif self.likelihood_x=='bernoulli': # MNIST
            theta_x = torch.clamp(theta_x, -1e10, 1e10)
            xu = torch.sigmoid(theta_x)
            xu = torch.round(xu)
        else: 
            xu = theta_x

        # fill observed data with imputed xu
        x = x + xu*torch.logical_not(observed_x)
        # concatenate missing indicator
        indicator = torch.logical_not(observed_x)
        x_hat = torch.cat([x, indicator], dim=-1)

        return x_hat
  
    def predict(self, batch: torch.Tensor, samples=10) -> torch.Tensor:
        """
        Prediction the target y

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            samples (int): number of samples from the latent for MC. Defaults to 10.

        Returns:
            torch.Tensor: predicted target in the original target scale
        """
        
        batch_ = [b.clone() for b in batch]
         
        # We do not observe the target when predicting
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
        x_hat = self.build_x_hat(xt, observed_x, theta_x)
        zx = torch.cat([z,x_hat],dim=-1)

        theta_y = self.predictor(zx).mean(-2)
        y_pred = self.denormalize_y(theta_y)

        return y_pred

    def prediction_metric(self, theta_y: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the chosen prediction metric 

        Args:
            theta_y (torch.Tensor): parameters of the likelihood distribution           (batch_size, latent_samples, dim_y)
            y (torch.Tensor): true target                                               (batch_size, dim_y)

        Returns:
            torch.Tensor: mean metric
        """

        if self.likelihood_y=='bernoulli' and self.prediction_metric_name in ['accuracy', 'error_rate']:
                y_pred = torch.round(theta_y)
        elif self.likelihood_y=='categorical' and self.prediction_metric_name in ['accuracy', 'error_rate']:
                y_pred = torch.argmax(theta_y, dim=-1, keepdim=True)
        else:
            y_pred = theta_y

        if self.prediction_metric_name=='rmse':
            metric = torch.sqrt(torch.nn.MSELoss()(theta_y, y))
        elif self.prediction_metric_name=='auroc':
            if y_pred.shape[-1] == 1:
                # binary case
                y_pred = y_pred.reshape(-1)
            metric = AUROC(num_classes=self.categories_y)(y_pred, y[:, 0].long().squeeze())
        elif self.prediction_metric_name=='accuracy':
            metric = (y_pred==y).float().mean()
        elif self.prediction_metric_name=='error_rate':
            metric = (y_pred!=y).float().mean()

        return metric

    def log_likelihood_y(self, batch: tuple, samples=100) -> torch.Tensor:
        """
        Computes the approximated log likelihood of the target p(y)

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            samples (int): number of samples from the latent for MC. Defaults to 100.

        Returns:
            torch.Tensor: mean log likelihood of the target
        """
        
        y_true = batch[2]
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
        x_hat = self.build_x_hat(xt, observed_x, theta_x)
        zx = torch.cat([z,x_hat], dim=-1)

        theta_y = self.predictor(zx)

        # Log-likelihood of the normalised target
        yn_true = self.normalize_y(y_true)
        rec = self.predictor.logp(yn_true, torch.ones_like(y_true), theta=theta_y)
        # mean per dimension (y is all observed in test)
        rec = rec.sum(-1, keepdim=True)

        ll_y = torch.logsumexp(rec, dim=-2) - np.log(samples)
        # mean per sample
        ll_y = ll_y.mean()

        return ll_y

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

        xn = self.normalize_x(x)
        
        xt, yt, xy, observed = self.preprocess_batch(batch_) 
        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)

        mu_z, logvar_z = self.encoder(xy)
        z = self.sample_z(mu_z, logvar_z, samples=samples)
        theta_x = self.decoder(z)

        theta_x = self.invert_preproc(theta_x)

        # Log-likelihood of the unobserved variables
        rec = self.decoder.logp(xn, torch.logical_not(observed_x), theta=theta_x)
        ll_xu = torch.logsumexp(rec, dim=-2) - np.log(samples)

        # divide each xu by the number of variables:
        # mean per sample
        ll_xu = ll_xu[torch.isfinite(ll_xu)].mean()
        
        # When all the data is observed, logp will be nan
        return ll_xu

    def active_learning(self, batch: tuple, bins=5, samples=1000, step=1) -> tuple:
        """
        Returns the metrics obtained at each step of the SAIA experiment

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            bins (int): number of intervals for the histograms. Defaults to 5.
            samples (int): number of samples from the latent for MC. Defaults to 1000.
            step (int, optional): number of variables to add within each step. Defaults to 1.

        Returns:
            tuple: 
                - (torch.Tensor): mean metric at each step              (dim_x + 1)
                - (torch.Tensor): mean log likelihood at each step      (dim_x + 1)
                - (np.array): mean elapsed time at each step            (dim_x + 1)
        """
        
        batch = [b.to(self.device) for b in batch]
        observations = batch[0]
        y_true = batch[2].clone().to(self.device)
        y = torch.zeros_like(y_true).to(self.device)
        observed_x = torch.zeros_like(observations).to(self.device)
        xo = torch.zeros_like(observations).to(self.device)
        observed_y = torch.zeros_like(y)

        xoi = xo.clone()

        tqdm_step = tqdm(total=self.dim_x, desc='Step (IR)', position=2, leave=False)
        metric=[]
        times = []
        ll=[]
        start = time.time()
        for d in range(int(np.ceil(self.dim_x / step))+1):
            
            if d==0:
                times.append(0.0)
            else:
                times.append(time.time() - start)

            # Get data
            batch_ = [xoi, observed_x, y, observed_y]
            x, observed_x, y, observed_y = batch_
            xn = self.normalize_x(x)
            xt, yt, xy, observed = self.preprocess_batch(batch_) 
            # xt is the preprocessed input (xt=x if no preprocessing)
            # observed is observed_x OR observed_y (for not using kl if no observed data)

            mu_z, logvar_z = self.encoder(xy)
            z = self.sample_z(mu_z, logvar_z, samples=samples)
            theta_x = self.decoder(z)
            x_hat = self.build_x_hat(xn, observed_x, theta_x)
            zx = torch.cat([z,x_hat],dim=-1)

            theta_y = self.predictor(zx)
            y_pred = self.denormalize_y(theta_y.mean(-2))
            if self.likelihood_y == 'categorical':
                y_pred = torch.exp(y_pred)
            elif self.likelihood_y == 'bernoulli':
                y_pred = torch.sigmoid(y_pred)
            
            metric.append(self.prediction_metric(y_pred, y_true))
            
            # Log-likelihood of the normalised target
            yn_true = self.normalize_y(y_true)
            rec = self.predictor.logp(yn_true, torch.ones_like(y_true), theta=theta_y)
            # mean per dimension (y is all observed in test)
            rec = rec.sum(-1, keepdim=True)
            ll_y = torch.logsumexp(rec, dim=-2) - np.log(samples)
            # mean per sample
            ll_y = ll_y.mean()

            ll.append(ll_y)

            tqdm_step.set_postfix({self.prediction_metric_name: metric[-1].detach().cpu().numpy(), "ll": ll[-1].detach().cpu().numpy()})
            
            if d < int(np.ceil(self.dim_x / step)):
                theta_y = self.predictor(zx)
                theta_x = self.invert_preproc(theta_x)

                xi = self.psample_x(theta_x, samples=1).reshape(theta_x.shape[0], theta_x.shape[1], theta_x.shape[2])

                next_variables = torch.stack([torch.where(row == 0)[0] for row in observed_x])
                # for the rest of missing variables
                xi_all = []
                for i in range(self.dim_x-d*step):
                    next = next_variables[:, i]
                    xi_ = torch.stack([xi[n,:, next[n]] for n in range(len(xi))])
                    xi_all.append(xi_)

                xi_all = torch.stack(xi_all,-1)
                xi_all = xi_all.permute(0, 2, 1).unsqueeze(-1)
                y_all = self.psample_y(theta_y, samples=1).reshape(theta_y.shape[0], theta_y.shape[1], self.dim_y)
                y_all = self.denormalize_y(y_all)
                y_all = y_all.repeat(xi_all.shape[1], 1, 1, 1).permute(1, 0, 2, 3)
                
                R = mutual_information(xi_all, y_all, bins=bins, device=self.device)

                if (observed_x[0,:]==0).sum() > step:
                    selected = torch.argsort(R, dim=-1, descending=True)[:, :step]
                else:
                    selected = np.arange(R.shape[1])[:, np.newaxis].repeat(R.shape[0],1).T
                selected = torch.stack([next_variables[n, selected[n]] for n in range(len(selected))])

                for n, s in enumerate(selected):
                    xoi[n, s] = observations[n, s]
                    observed_x[n, s] = 1
                
                tqdm_step.update(step)

        return torch.stack(metric), torch.stack(ll), np.array(times)
    
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
        x, observed_x, y, observed_y = batch
        xt, yt, xy, observed = self.preprocess_batch(batch) 
        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)
        mu_z, logvar_z = self.encoder(xy)

        z = self.sample_z(mu_z, logvar_z, samples=samples)
        theta_x = self.decoder(z)
        x_hat = self.build_x_hat(xt, observed_x, theta_x)
        zx = torch.cat([z,x_hat],dim=-1)

        rec_x = self.decoder.logp(xt, observed_x, z=z, theta=theta_x).sum(-1)
        rec_y = self.predictor.logp(yt, observed_y, z=zx).sum(-1)
        kls = self.encoder.regularizer(mu_z, logvar_z, observed)
        
        elbo = rec_x + rec_y - kls.sum(0).unsqueeze(-1)
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
        x, observed_x, y, observed_y = batch

        xt, yt, xy, observed = self.preprocess_batch(batch) 
        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)
        mu_z, logvar_z = self.encoder(xy)

        z = self.sample_z(mu_z, logvar_z, samples=samples)


        logp = self.logp(xt, observed_x, yt, observed_y, z)
        logq = self.encoder.logq(z, xt, observed_x, yt, observed_y)

        log_w = logp - logq

        elbo_iwae = logmeanexp(log_w, -1)

        return elbo_iwae

    def logp(self, xt: torch.Tensor, observed_x: torch.Tensor, yt: torch.Tensor, observed_y: torch.Tensor, z: torch.Tensor, xn: torch.Tensor=None) -> torch.Tensor:
        """
        Returns the log joint logp(x, y, z) of the model

        Args:
            xt (torch.Tensor): normalized and preprocessed data                                                                             (batch_size, dim_x)
            observed_x (torch.Tensor): observation mask of the data                                                                         (batch_size, dim_x)
            yt (torch.Tensor): normalized target                                                                                            (batch_size, dim_y)
            observed_y (torch.Tensor): observation mask of the target                                                                       (batch_size, dim_y)
            z (torch.Tensor): latent samples                                                                                                (batch_size, latent_samples, latent_dim)
            xn (torch.Tensor, optional): normalized data when xt is a transformation (preprocessing or embedding). Defaults to None.        (batch_size, dim_x)

        Returns:    
            torch.Tensor: log probs                                                                                                         (batch_size, 1)                        
        """
        # When xt is a transformation of x, feed the predictor with xn
        if xn==None:
            xn = xt

        theta_x = self.decoder(z)
        x_hat = self.build_x_hat(xn, observed_x, theta_x)
        zx = torch.cat([z,x_hat],dim=-1)

        logpx_z = self.decoder.logp(xt, observed_x, theta=theta_x).sum(-1)
        logpy_z = self.predictor.logp(yt, observed_y, z=zx).sum(-1)

        observed = torch.logical_or(observed_x.sum(-1, keepdim=True)>0, observed_y.sum(-1, keepdim=True)>0)
        logpz = self.prior(z, observed)

        logp = logpx_z + logpy_z + logpz

        return logp

    def logp_func(self, xt: torch.Tensor, observed_x: torch.Tensor, yt: torch.Tensor, observed_y: torch.Tensor, xn: torch.Tensor=None):
        """
        Returns a function for computing logp(x, y, z) with fixed x, y (only depending on z). This function is used as HMC objective.

        Args:
            xt (torch.Tensor): normalized and preprocessed data                                                                             (batch_size, dim_x)
            observed_x (torch.Tensor): observation mask of the data                                                                         (batch_size, dim_x)
            yt (torch.Tensor): normalized target                                                                                            (batch_size, dim_y)
            observed_y (torch.Tensor): observation mask of the target                                                                       (batch_size, dim_y)
            xn (torch.Tensor, optional): normalized data when xt is a transformation (preprocessing or embedding). Defaults to None.        (batch_size, dim_x)

        Returns:
            function depending on z ( logp(z, x, y) for fixed x and y )
        """
        def logp(z):
            return self.logp(xt, observed_x, yt, observed_y, z, xn)

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
        return self.encoder.logq(z, xt, observed_x, yt, observed_y)

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

        x = self.decoder.likelihood.sample(theta)

        return x

    def psample_y(self, theta: torch.Tensor, samples: int) -> torch.Tensor:
        """
        Sample y from the likelihood distribution

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

        y = self.predictor.likelihood.sample(theta)

        return y

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
        else: x = xn
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
        else: y = yn
        return y
    
    def preproc_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply some transformation to the input data. Used in more complex models

        Args:
            x (torch.Tensor): input data        (batch_size, dim_x)

        Returns:
            torch.Tensor: transformed data      (batch_size, dim_x)
        """
        return x
        
    def invert_preproc(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Reverts transformed data to the original domain

        Args:
            xt (torch.Tensor): transformed data 

        Returns:
            torch.Tensor: data in original domain
        """
        return xt

    # ============= Modified PL functions ============= #
    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()) +
            list(self.predictor.parameters()), lr=self.lr, weight_decay=0.01)  
        return [opt]

    def train_dataloader(self):

        loader = get_dataset_loader(self.dataset, split='train', path=self.data_path, batch_size=self.batch_size, split_idx=self.split_idx)
        self.register_buffer('mean_y', torch.Tensor(loader.dataset.labels.mean(0, keepdims=True)).to(self.device))
        self.register_buffer('std_y', torch.Tensor(loader.dataset.labels.std(0, keepdims=True)).to(self.device))
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
    """def val_dataloader(self):
        return get_dataset_loader(self.dataset, split='test', path=self.data_path, batch_size=self.batch_size, split_idx=self.split_idx)
    """
    
    def test_dataloader(self):
        return get_dataset_loader(self.dataset, split='test', path=self.data_path, batch_size=self.batch_size, split_idx=self.split_idx)
    

# ============= Extra functions ============= #

def reparameterize(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    Reparameterized samples from a Gaussian distribution

    Args:
        mu (torch.Tensor): means                    (batch_size, ..., dim)
        var (torch.Tensor): variances               (batch_size, ..., dim)

    Returns:
        torch.Tensor: samples                       (batch_size, ..., dim)
    """
    std = var**0.5
    eps = torch.randn_like(std)
    return mu + eps*std

def logmeanexp(inputs: torch.Tensor, dim=1) -> torch.Tensor:
    """
    Apply the logmeanexp trick to a given input

    Args:
        inputs (torch.Tensor): input tensor
        dim (int, optional): dimension to apply the mean. Defaults to 1.

    Returns:
        torch.Tensor: resulting logmeanexp
    """
    if inputs.size(dim) == 1:
        return inputs
    else:
        input_max = inputs.max(dim, keepdim=True)[0]
        return (inputs - input_max).exp().mean(dim).log() + input_max.squeeze()

def deactivate(model: nn.Module):
    """
    Freeze or deactivate gradients of all the parameters in a module

    Args:
        model (nn.Module): module to deactivate
    """
    for param in model.parameters():
        param.requires_grad = False

def activate(model):
    """
    Activate gradients of all the parameters in a module

    Args:
        model (nn.Module): module to activate
    """
    for param in model.parameters():
        param.requires_grad = True

def print_parameters(model: nn.Module):
    """
    Print all the parameters in a module

    Args:
        model (nn.Module): module
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print((name, param.data))

class View(nn.Module):
    """
    Reshape tensor inside Sequential objects. Use as: nn.Sequential(...,  View(shape), ...)
    """
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)