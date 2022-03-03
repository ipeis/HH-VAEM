# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from src.models.base import *
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


# ============= HVAE submodules ============= #

class HierarchicalPrior(nn.Module):
    """
    Implements a Hierarchical Generative model with reparameterized layers

    """
    
    def __init__(self, latent_dims: list, dims_h: int=256):
        """
        Hierarchical Prior initialization

        Args:
            latent_dims (list): list of ints containing the latent dimension at each layer. First element corresponds to the shallowest layer, connected to the data
            dims_h (int, optional): number of units for hidden vectors. Defaults to 256.

        """
        
        
        super(HierarchicalPrior, self).__init__()

        self.layers = len(latent_dims)
        self.latent_dims = latent_dims
    
        # NNs for computing AR latent prior 
        self.NNs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dims[l+1], dims_h), nn.ReLU(), 
            nn.Linear(dims_h, 2*latent_dims[l])) for l in range(len(latent_dims)-1)])
        
        self.logvar = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(l)) for l in latent_dims])

    def forward(self, E: list, observed: torch.Tensor=None) -> torch.Tensor:
        """
        Computes the log probs of each layer 

        Args:
            E (list): contains the latent samples epsilon at each layer
            observed (torch.Tensor, optional): observed mask. If given, true values when both data and target are unobserved. Defaults to None.

        Returns:
            torch.Tensor: tensor with log probs at each layer   (nlayers, latent_samples, 1)
        """
        
        logp = []
        logvar = self.get_logvar()
        for l, e in enumerate(E):
            cnt = e.size(-1) * np.log(2 * np.pi) + torch.sum(logvar[l], dim=-1)
            logp_l = -0.5 * (cnt + torch.sum(e**2 * torch.exp(-logvar[l]), dim=-1))
            logp.append(logp_l * observed)
        return torch.stack(logp, 0)

    def update_prior(self, E: list):
        """
        Updates the prior variance sing VI and ML after each epoch.

        Args:
            E (list): concatenated list of latent samples from the posterior for each datapoint in the dataset
        """
        logvar = torch.log(1/E.shape[0] * (E**2).sum(0))
        inds = [0] + np.cumsum(self.latent_dims).tolist()
        self.logvar = torch.nn.ParameterList([torch.nn.Parameter(logvar[inds[i]:inds[i+1]]) for i in range(self.layers)])

    def get_logvar(self):
        """
        Get the prior log variance of each layer

        Returns:
            object: logvars 
        """
        logvar = self.logvar
        deactivate(logvar)
        return logvar

class HierarchicalEncoder(nn.Module):
    """
    Implements a Hierarchical Encoder
    """
    def __init__(self, dim_x: int, dim_y: int,  latent_dims: list, dims_h: int=256, balance_kl_steps: float=2e3, anneal_kl_steps: float=1e3):
        """
        Encoder initialization
        
        Args:
            dim_x (int): dimension of the data.
            dim_y (int): dimension of the target.
            latent_dims (list): list of ints containing the latent dimension at each layer. First element corresponds to the shallowest layer, connected to the data.
            dims_h (int, optional): number of units for hidden vectors. Defaults to 256.
            balance_kl_steps (float, optional): number of steps for balancing the KL terms of the different layers. Defaults to 2e3.
            anneal_kl_steps (float, optional): number of steps for annealing the KL. Defaults to 1e3.
        """
        
        super(HierarchicalEncoder, self).__init__()

        self.layers = len(latent_dims)
        self.latent_dims = latent_dims
        self.dims_h = dims_h
        self.balance_kl_steps=balance_kl_steps
        self.anneal_kl_steps=anneal_kl_steps
        self.dims_r = [2*dim_x + 2*dim_y] + [dims_h] * len(latent_dims)
        
        # NNs for computing deltas on the z parameters
        self.encoders = nn.ModuleList([
            nn.Linear(self.dims_r[l+1], 2 * latent_dims[l]) for l in range(len(latent_dims))])

        self.NNs_r = nn.ModuleList([
            nn.Sequential(nn.Linear(self.dims_r[l], self.dims_r[l+1]), 
            nn.ReLU()) 
            for l in range(len(latent_dims))]
            )

        self.logvar = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(l)) for l in latent_dims])
        self.global_step=1
        
    def _encode(self, R: list) -> tuple:
        """
        Obtain the approx posterior parameters from the deterministic bottom-up path

        Args:
            R (list): list with deterministic hidden vectors containing the bottom up path. Element 0 is r1. 

        Returns:
            - list with the means for each layer
            - list with the logvars for each layer
        """
        mu = []
        logvar = []
        for l in range(self.layers):
            mu_l, logvar_l = torch.chunk(self.encoders[l](R[l]), 2, -1)
            mu.append(mu_l)
            logvar.append(logvar_l)

        return mu, logvar
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Encodes data to obtain the parameters of the approx posterior

        Args:
            x (torch.Tensor): input data

        Returns:
            - list with the means for each layer
            - list with the logvars for each layer
        """
        R = self._deterministic_path(x)
        mu, logvar = self._encode(R)
        return mu, logvar

    def regularizer(self, mus: list, logvars: list, observed) -> torch.Tensor:
        """
        Computes the list of KLs for each latent layer

        Args:
            mus (list): means of the approx posterior 
            logvars (list): logvars of the approx posterior
            observed (torch.Tensor): observation mask (used when both data and target are unobserved)

        Returns:
            torch.Tensor: tensor containing the KLs for each layer      (nlayers, batch_size)
        """
        
        KLs = []
        logvar = self.get_logvar()
        for l in range(self.layers):
            logvar_p = logvar[l].squeeze(0)
            # KL for diagonal Gaussians with 0-mean prior
            KL = -0.5 * torch.sum(1. + logvars[l] - logvar_p - mus[l]**2 * torch.exp(-logvar_p) - torch.exp(logvars[l]) * torch.exp(-logvar_p), dim=-1, keepdim=True)
            KL = KL * observed
            KLs.append(KL)

        KLs = torch.stack(KLs)
            
        if self.global_step < self.balance_kl_steps:
            # Balancing KLs
            KLs_l = (KLs.mean(1) * torch.Tensor(self.latent_dims).type_as(KLs).unsqueeze(-1) )
            KLsum = KLs_l.sum(0)
            weights = KLs_l / KLsum
            weights = weights# + 1e-5
        else:
            weights = torch.ones(self.layers, 1, 1).to(KLs.device)

        if self.global_step < self.anneal_kl_steps:
            weights = weights * np.minimum(self.global_step / self.anneal_kl_steps, 1)
        
        KLs = KLs * weights.unsqueeze(-2)
        KLs = KLs.mean(-1)
        
        return KLs

    def _deterministic_path(self, input: torch.Tensor) -> list:
        """
        Returns the deterministic hidden vectors of the bottom-up path [r1, ..., rL]

        Args:
            input (torch.Tensor): input data

        Returns:
            list: list with [r1, ..., rL]
        """
        R = []
        r=input
        for l in range(self.layers):
            r = self.NNs_r[l](r)
            R.append(r)
        return R
    
    def get_logvar(self):
        """
        Gets the logvar at each layer. Thsi function is used when updating prior variance after each epoch.

        Returns:
            object: logvars
        """
        logvar = self.logvar
        deactivate(logvar)
        return logvar


# ============= HVAE ============= #

class HVAE(BaseVAE):
    """
    Implements a Hierarchical VAE (H-VAE) as described in https://arxiv.org/abs/2202.04599

    """
    def __init__(self, 
            dataset: str, dim_x: int, dim_y: int, arch='base', dim_h=256,
            likelihood_x = 'gaussian', likelihood_y = 'gaussian', variance=0.1, imbalanced_y = False,
            categories_y = 1, prediction_metric='rmse',
            batch_size=128, lr=1e-3, samples_MC = 1, data_path='../data/', split_idx=0,
            
            latent_dims: list=[10, 5], sr_coef=0.001, balance_kl_steps=15e3, anneal_kl_steps=1e3,
            update_prior = False
            ):
        """
        HVAE initialization

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
        """
        super(HVAE, self).__init__(dataset=dataset, dim_x=dim_x, dim_y=dim_y, 
            latent_dim = latent_dims[0], arch=arch, dim_h=dim_h, likelihood_x = likelihood_x, likelihood_y = likelihood_y, 
            variance=variance, imbalanced_y=imbalanced_y,
            categories_y=categories_y,
            prediction_metric=prediction_metric, batch_size=batch_size, lr=lr, samples_MC = samples_MC, 
            data_path=data_path, split_idx=split_idx,
        
        )

        self.latent_dims = latent_dims
        self.layers = len(latent_dims)        

        self.prior = HierarchicalPrior(latent_dims, dim_h)
        self.encoder = HierarchicalEncoder(dim_x, dim_y, latent_dims, dim_h, 
            balance_kl_steps=balance_kl_steps, anneal_kl_steps=anneal_kl_steps)

        # collect all norm params in Linear layers
        self.all_layers = [module for module in self.modules() if isinstance(module, nn.Linear)]
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4
        self.sr_coef = sr_coef
        self.balance_kl_steps = balance_kl_steps
        self.anneal_kl_steps = anneal_kl_steps
        self.update_prior = update_prior
        self.E_epoch = []

        self.save_hyperparameters('latent_dims', 'sr_coef', 'balance_kl_steps', 'anneal_kl_steps')

    # ============= Modified base functions ============= #

    def sample_z(self, mus: list, logvars: list, samples=1, all_layers=False):
        """
        Draw latent reparameterized samples Z from a given approx hierarchical posterior of Epsilon, parameterized by mu and logvar

        Args:
            mu (list): list with the means for each layer                           L x (batch_size, latent_dim)
            logvar (torch.Tensor): tensor with the log variances                    L x (batch_size, latent_dim)
            samples (int, optional): number of samples. Defaults to 1.      

            all_layers (bool, optional): return samples from all layers (True) or the shallowest (only for decoding). Defaults to False.

        Returns:
            samples from the shallow layer (default) or list with samples from all layers (from z1 to zL)
        """
        mus = [m.repeat(samples, 1, 1).permute(1, 0, 2) for m in mus]
        logvars = [l.repeat(samples, 1, 1).permute(1, 0, 2) for l in logvars]

        # Top layer
        Z = [reparameterize(mus[-1], torch.exp(logvars[-1]))]
        E = [Z[-1].clone()]
        for l in np.arange(self.layers-2, -1, -1):
            mu_z, logvar_z = torch.chunk(self.prior.NNs[l](Z[-1]), 2, dim=-1)
            e = reparameterize(mus[l], torch.exp(logvars[l])) # sampled from the approx posterior
            E.append(e)
            z = mu_z + torch.sqrt(torch.exp(logvar_z)) * e
            Z.append(z)

        # Flip Z to follow the notation of the model (element 0 -> z1)
        Z = Z[::-1]
        E = E[::-1]
        E = torch.cat(E, -1)

        if self.training and self.validation==False:
            self.E_epoch.append(E.reshape(E.shape[0], np.sum(self.latent_dims)))

        #Z = torch.cat(Z, -1)

        if all_layers: 
            return Z
        else: return Z[0]

    def training_step(self, batch: tuple, batch_idx: int, logging: bool=True):
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
        self.encoder.global_step += 1

        #loss_sr = self.sr_coef * self.spectral_norm_parallel()
        #loss = loss + loss_sr

        self.log('ELBO', -loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        #self.log('loss_sr', loss_sr, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if logging:
            self.log('-rec_x', -rec_x, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('-rec_y', -rec_y, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            for l, kl in enumerate(kls):
                self.log('kl_{:d}'.format(l), kl, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """
        Updates prior variance at the end of each epoch, by ML and VI

        """

        if self.update_prior:
            E = torch.cat(self.E_epoch)
            self.prior.update_prior(E)
            self.encoder.logvar = self.prior.get_logvar()
            self.E = []
            i=0
            for l in self.prior.get_logvar():
                for d in l:
                    self.log('var_{:d}'.format(i), torch.exp(d))
                    i+=1
        return super().on_epoch_end()
  
    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all layers in parallel. This method should be called
         after calling the forward method of all the conv layers in each iteration. """

        weights = {}   # a dictionary indexed by the shape of weights
        for l in self.all_layers:
            weight = l.weight
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
                                               dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
                                               dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss

    def logp(self, xt: torch.Tensor, observed_x: torch.Tensor, yt: torch.Tensor, observed_y: torch.Tensor, e: list, xn: torch.Tensor=None) -> torch.Tensor:
        """
        Returns the log joint logp(x, y, epsilon) of the model

        Args:
            xt (torch.Tensor): normalized and preprocessed data                                                                             (batch_size, dim_x)
            observed_x (torch.Tensor): observation mask of the data                                                                         (batch_size, dim_x)
            yt (torch.Tensor): normalized target                                                                                            (batch_size, dim_y)
            observed_y (torch.Tensor): observation mask of the target                                                                       (batch_size, dim_y)
            e (list or torch.Tensor): latent samples epsilon from each layer                                                                L x (batch_size, latent_samples, latent_dim)
            xn (torch.Tensor, optional): normalized data when xt is a transformation (preprocessing or embedding). Defaults to None.        (batch_size, dim_x)

        Returns:
            torch.Tensor: log probs                                                                                                         (batch_size, 1)
        """
        
        # When xt is a transformation of x, feed the predictor with xn
        if xn==None:
            xn = xt

        # If z is from the latent joint, we separate each layer
        if type(e)!= list:
            E = self.make_tensor_list(e)

        else: E = e

        # Compute zs
        # Top layer
        Z = [E[-1].clone()]
        for l in np.arange(self.layers-2, -1, -1):
            mu_z, logvar_z = torch.chunk(self.prior.NNs[l](Z[-1]), 2, dim=-1)
            z = mu_z + torch.sqrt(torch.exp(logvar_z)) * E[l]
            Z.append(z)
        Z = Z[::-1]
            
        theta_x = self.decoder(Z[0])

        x_hat = self.build_x_hat(xn, observed_x, theta_x)
        zx = torch.cat([Z[0],x_hat],dim=-1)

        logpx_z = self.decoder.logp(xt, observed_x, theta=theta_x).sum(-1)
        logpy_z = self.predictor.logp(yt, observed_y, z=zx).sum(-1)

        observed = torch.logical_or(observed_x.sum(-1, keepdim=True)>0, observed_y.sum(-1, keepdim=True)>0)
        logpz = self.prior(E, observed)

        logp = logpx_z + logpy_z + logpz.sum(0)

        return logp

    def logq(self, e: list, xt: torch.Tensor, observed_x: torch.Tensor, yt: torch.Tensor, observed_y: torch.Tensor) -> torch.Tensor:
        """
        Computes the log prob of a latent sample under approximated Gaussian posterior given by the encoder

        Args:
            e (torch.Tensor): latent samples                                                 (batch_size, latent_samples, latent_dim)
            xt (torch.Tensor): normalized and preprocessed data                              (batch_size, dim_x)
            observed_x (torch.Tensor): observation mask of the data                          (batch_size, dim_x)
            yt (torch.Tensor): normalized target                                             (batch_size, dim_y)
            observed_y (torch.Tensor): observation mask of the target                        (batch_size, dim_y)

        Returns:                                                                                                                          
            torch.Tensor: log probs                                                          (batch_size, 1) 
        """

        if type(e)!= list:
            E = self.make_tensor_list(e)

        else:
            E = e
        # xt and yt must be normalised
        xo = xt * observed_x
        x_tilde = torch.cat([xo, observed_x], axis=1)

        # Normalize the target 
        yo = yt * observed_y

        y_tilde = torch.cat([yo, observed_y], axis=1)
        xy = torch.cat([x_tilde, y_tilde], axis=1)
        mus, logvars = self.encoder(xy)
        
        samples = E[0].shape[-2]
        mus = [d.repeat(samples, 1, 1).permute(1, 0, 2) for d in mus]
        logvars = [d.repeat(samples, 1, 1).permute(1, 0, 2) for d in logvars]

        observed = torch.logical_or(observed_x.sum(-1, keepdim=True)>0, observed_y.sum(-1, keepdim=True)>0)
        logq_xy = []
        for (e_l, mu_l, logvar_l) in zip(E, mus, logvars):
            cnt = mu_l.shape[-1] * np.log(2 * np.pi) + torch.sum(logvar_l, dim=-1)
            logq_l = -0.5 * (cnt + torch.sum((e_l - mu_l)**2 * torch.exp(-logvar_l), dim=-1))
            logq_xy.append(logq_l * observed)

        logq_xy = torch.stack(logq_xy).sum(0)

        return logq_xy

    def make_tensor_list(self, z: torch.Tensor) -> list:
        """
        Convers a tensor (L, batch_size, ...) into a list with L x (batch_size, ...)

        Args:
            z (torch.Tensor): tensor with shape (L, batch_size, ...)

        Returns:
            list: list with L elements of shape (batch_size, ...)
        """
        ind = np.concatenate(([0], np.cumsum(self.latent_dims)))
        Z = [z[:, :, ind[i]:ind[i+1]] for i in range(len(self.latent_dims))]
        return Z
    
    def elbo(self, batch, samples=1000):
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

        z = self.sample_z(mu_z, logvar_z, samples=samples, only_first_layer=False)
        theta_x = self.decoder(z[0])
        x_hat = self.build_x_hat(xt, observed_x, theta_x)
        zx = torch.cat([z[0],x_hat],dim=-1)

        rec_x = self.decoder.logp(xt, observed_x, z=z, theta=theta_x).sum(-1)
        rec_y = self.predictor.logp(yt, observed_y, z=zx).sum(-1)
        kls = self.encoder.regularizer(mu_z, logvar_z, observed)
        
        elbo = rec_x + rec_y - kls.sum(0).unsqueeze(-1)
        elbo = elbo[elbo!=0].mean()

        return elbo

    def elbo_iwae(self, batch, samples=1000):
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
        z = self.sample_z(mu_z, logvar_z, samples=samples, all_layers=True)

        logp = self.logp(xt, observed_x, yt, observed_y, z)
        logq = self.logq(z, xt, observed_x, yt, observed_y)
        log_w = logp - logq

        elbo_iwae = logmeanexp(log_w, -1)

        return elbo_iwae

    # ============= Modified PL functions ============= #
    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.decoder.parameters()) + list(self.predictor.parameters()) + list(self.prior.parameters()) +
                                   list(self.encoder.parameters()), lr=self.lr, weight_decay=0.01)    
        return [opt]
