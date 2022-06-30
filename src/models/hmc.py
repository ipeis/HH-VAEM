# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from torch import nn
import numpy as np
from typing import Callable
from tqdm import trange


# ============= HMC module ============= #
class HMC(nn.Module):
    """
    Implements an HMC sampler with trainable hyperparameters
    """
    def __init__(self, dim: int, logp: Callable, T: int=30,  L: int=5, chains: int=1, chains_sksd: int=30, scale_per_layer=None):
        """
        HMC initialization

        Args:
            dim (int): dimension of the sampling space
            logp (Callable): objective log p(z) (can be unnormalized)
            T (int, optional): length of the HMC chains. Defaults to 10.
            L (int, optional): number of Leapfrog steps. Defaults to 5.
            chains (int, optional): number of parallel HMC chains. Defaults to 1.
            chains_sksd (int, optional): number of parallel HMC chains for computing the SKSD. Defaults to 30.
            scale_per_layer (bool, optional): learn a scale parameter per layer in the Hierarchy (input a list with the sizes) or a scalar value (None). Defaults to None.
        """
        super().__init__()

        self.dim=dim    # Dimension of the parameter space
        self.L = L      # Leapfrog steps
        self.T = T      # Length of the chain
        self.chains = chains      # Number of parallel chains
        self.chains_sksd = chains_sksd # Number of parallel chains for computing sksd
        self.scale_per_layer = scale_per_layer
        self.init_random_params(T, dim)
        self.logp = logp    # Function that computes objective logp

    def init_random_params(self, T: int, dim: int):
        """
        Initilize HMC hyperparameters

        Args:
            T (int, optional): length of the HMC chains. Defaults to 10.
            L (int, optional): number of Leapfrog steps. Defaults to 5.
        """
        # Matrix with step sizes
        eps = torch.Tensor(np.random.uniform(0.02, 0.05, size=(T, dim)))
        #eps = 0.01 + torch.rand(T, dim) * 0.015

        # Must be positive (optimize log parameter)
        self.log_eps = torch.nn.Parameter(torch.log(eps))
        #self.log_v_r = torch.nn.Parameter(torch.zeros([L, dim]))

        self.log_v_r = torch.zeros([T, dim])
        
        # Learn a scale per hierarchical layer
        if self.scale_per_layer != None:
            self.log_inflation = torch.nn.Parameter(torch.zeros(len(self.scale_per_layer)))
        # Learn a scale as a global factor
        else: 
            self.log_inflation = torch.nn.Parameter(torch.zeros(1))
        self.g = torch.nn.Parameter(torch.eye(self.dim))

    def generate_samples_HMC(self, mu0: torch.Tensor, var0: torch.Tensor, chains: int=None):
        """
        Sample from p(z) with HMC, given a Gaussian proposal (which is inflated by the scale parameter).
        By using this function you keep activated the gradients of the step sizes hyperparameter

        Args:
            mu0 (torch.Tensor): mean of the Gaussian proposal               (batch_size, dim)
            var0 (torch.Tensor): variance of the Gaussian proposal          (batch_size, dim)
            chains (int, optional): Number of parallel chains (if None, takes the predefined). Defaults to None.       

        Returns:
            torch.Tensor: samples                                           (batch_size, chains, dim)
            torch.Tensor: chains                                            (chains, batch_size, chains, dim)
        """

        log_eps = torch.log(torch.exp(self.log_eps))
        log_v_r = self.log_v_r

        sigma0 = torch.sqrt(var0)
        # Learn a scale per hierarchical layer
        if self.scale_per_layer != None:
            log_inflation = torch.cat([torch.zeros([1, s]).to(sigma0.device) + self.log_inflation[l] for l, s in enumerate(self.scale_per_layer)], -1).data.detach()
        else:
            log_inflation = self.log_inflation.data.detach() # same scale per all dimension
        inflation = torch.exp(log_inflation).data.detach()
        # Learn a scale as a global factor
        if chains==None:
            chains = self.chains
        
        # Repeat for parallel chains
        mu0 = mu0.repeat(chains, 1, 1).transpose(0, 1).data.detach()
        sigma0 = sigma0.repeat(chains, 1, 1).transpose(0, 1).data.detach()
        z = torch.randn_like(mu0) * (sigma0 * inflation) + mu0

        z_list = [z]    # to store the whole chains
        for t in range(self.T):
            r = torch.randn_like(z) * torch.exp(0.5 * log_v_r[t, :]).type_as(z)
            z_new, r_new = self.leapfrog(z, r, torch.exp(log_eps[t, :]), log_v_r[t, :], self.dlogp)
            pot_init = -self.logp(z)
            kin = torch.sum(0.5 * r ** 2, dim=-1)
            pot_end = -self.logp(z_new)
            kin_end = torch.sum(0.5 * r_new ** 2, dim=-1)
            dH = pot_init + kin - (pot_end + kin_end)
            exp_dH = torch.exp(dH).type_as(z)

            # correct if integrator diverges:
            exp_dH[exp_dH.isinf()] = 0
            exp_dH[exp_dH.isnan()] = 0
            p_acceptance = torch.min(torch.Tensor([1.0]).type_as(z), exp_dH)
            accepted = torch.rand(z.shape[0], chains).type_as(z) < p_acceptance
            accepted = accepted.repeat([self.dim, 1, 1]).transpose(1, 2).T.int()
            z = z_new * accepted + (1 - accepted) * z
            z_list.append(z)
        return z, torch.stack(z_list)

    def generate_samples_KSD(self, mu0: torch.Tensor, var0: torch.Tensor, chains: int=None):
        """
        Sample from p(z) with HMC, given a Gaussian proposal (which is inflated by the scale parameter). 
        By using this function you keep activated the gradients of the scale hyperparameter

        Args:
            mu0 (torch.Tensor): mean of the Gaussian proposal               (batch_size, dim)
            var0 (torch.Tensor): variance of the Gaussian proposal          (batch_size, dim)
            chains (int, optional): Number of parallel chains (if None, takes the predefined). Defaults to None.       

        Returns:
            torch.Tensor: samples                                           (batch_size, chains, dim)
            torch.Tensor: chains                                            (chains, batch_size, chains, dim)
        """
        log_eps = torch.log(torch.exp(self.log_eps) )  # add a min step size
        #log_v_r = self.log_v_r.data
        log_v_r = self.log_v_r
        sigma0 = torch.sqrt(var0)

        # Learn a scale per hierarchical layer
        if self.scale_per_layer != None:
            log_inflation = torch.cat([torch.zeros([1, s]).to(sigma0.device) + self.log_inflation[l] for l, s in enumerate(self.scale_per_layer)], -1)
        # Learn a scale as a global factor
        else:
            log_inflation = self.log_inflation # same scale per all dimension
        inflation = torch.exp(log_inflation)

        if chains==None:
            chains = self.chains

        # Repeat for parallel chains
        mu0 = mu0.repeat(chains, 1, 1).transpose(0, 1)
        sigma0 = sigma0.repeat(chains, 1, 1).transpose(0, 1)

        z = torch.randn_like(mu0) * (sigma0 * inflation) + mu0
        for t in range(self.T):
            r = torch.randn_like(z) * torch.exp(0.5 * log_v_r[t, :]).type_as(z)
            z_new, r_new = self.leapfrog(z, r, torch.exp(log_eps[t, :]), log_v_r[t, :], self.dlogp)
            pot_init = -self.logp(z)
            kin = torch.sum(0.5 * r ** 2, dim=-1)
            pot_end = -self.logp(z_new)
            kin_end = torch.sum(0.5 * r_new ** 2, dim=-1)
            dH = pot_init + kin - (pot_end + kin_end)
            exp_dH = torch.exp(dH).type_as(z)

            # correct if integrator diverges:
            exp_dH[exp_dH.isinf()] = 0
            exp_dH[exp_dH.isnan()] = 0
            p_acceptance = torch.min(torch.Tensor([1.0]).type_as(z), exp_dH.type_as(z))
            accepted = torch.rand(z.shape[0], chains).type_as(z) < p_acceptance
            accepted = accepted.repeat([self.dim, 1, 1]).transpose(1, 2).T.int()
            z = z_new * accepted + (1 - accepted) * z
        return z

    def leapfrog(self, z: torch.Tensor, r: torch.Tensor, eps: torch.Tensor, log_v_r: torch.Tensor, dlogp: Callable) -> tuple:
        """
        Performs a Leapfrog integration

        Args:
            z (torch.Tensor): Current state sample                                  (batch_size, chains, dim)
            r (torch.Tensor): Momentum sample                                       (batch_size, chains, dim)
            eps (torch.Tensor): step sizes                                          (T, dim)
            log_v_r (torch.Tensor): log of the momentum variance                    (T, dim)
            dlogp (Callable): function that returns the Jacobian of logp() 

        Returns:
            tuple: updated state, updated momentum
        """
        z_init = z.clone()
        r_init = r.clone()

        grads = -dlogp(z).detach()
        ill = torch.logical_or(grads.isnan(), grads.isinf())
        grads[ill] = 0
        ill = ill.int()

        r = r - 0.5 * eps * grads
        for i in range(1, self.L + 1):
            z = z + eps * r / torch.exp(log_v_r).type_as(z)
            if i < self.L:
                grads = -dlogp(z).detach()
                ill_i = torch.logical_or(grads.isnan(), grads.isinf())
                grads[ill_i] = 0    
                ill += ill_i.int()
                r = r - eps * grads

        grads = -dlogp(z).detach()
        ill_i = torch.logical_or(grads.isnan(), grads.isinf())
        grads[ill_i] = 0    
        ill += ill_i.int()
        r = r - 0.5 * eps * grads

        ill = ill.bool()
        z[ill] = z_init[ill]
        r[ill] = r_init[ill]
        return z, r

    def evaluate_sksd(self, mu0: torch.Tensor, var0: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the SKSD discrepancy for a given proposal q(mu0, var0)     https://arxiv.org/abs/2006.16531

        Args:
            mu0 (torch.Tensor): mean of the Gaussian proposal               (batch_size, dim)
            var0 (torch.Tensor): variance of the Gaussian proposal          (batch_size, dim)

        Returns:
            torch.Tensor: SKSD discrepancy
        """
        samples1 = self.generate_samples_KSD(mu0, var0, chains=self.chains_sksd)       # input_batch * sample_size * latent_dim
        samples2 = samples1.clone()                                 # input_batch * sample_size * latent_dim

        score1 = self.logp(samples1)
        gradients1 = self.dlogp(samples1).data.detach() # input_batch * sample_size * latent_dim

        #score2 = self.logp(samples2)
        gradients2 = self.dlogp(samples2).data.detach()  # input_batch * sample_size * latent_dim

        # g normalized
        g_direction = self.g / torch.sqrt(torch.sum(self.g ** 2, dim=-1, keepdims=True))

        sksd = torch.stack([self.compute_max_SKSD(z1, z2, grad1, grad2, g_direction)
                           for z1, z2, grad1, grad2 in zip(samples1, samples2, gradients1, gradients2)])
        
        observed = score1[:,0]!=0
        sksd = sksd * observed
        sksd = sksd[sksd!=0].mean()

        return sksd

    def dlogp(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the Jacobian of the objective function logp(z)

        Args:
            z (torch.Tensor): sample z

        Returns:
            torch.Tensor: Jacobian of the objective logp(z), evaluated on z
        """
        with torch.set_grad_enabled(True):  # this line makes it work for evaluation mode
            def sum_logp(z):
                logp = self.logp(z).sum()
                #if logp.sum().isinf():
                #    print('stop')
                return logp
            grads = torch.autograd.functional.jacobian(sum_logp, z, create_graph=False)
        return grads
    
    def fit(self, epochs=20, *args, **kwargs):
        """
        Fit the HMC hyperparameters using the given objective

        Args:
            epochs (int, optional): Number of epochs. Defaults to 20.
        """
        objective=[]
        elbo=[]
        ksd=[]
        t = trange(epochs, desc='Loss')
        for e in t:
            self.optimizer.zero_grad()
            _loss, _elbo, _ksd = self.evaluate_objective()
            t.set_description('HMC (objective=%g)' % -_loss)
            #print('Epoch: {} \t loss: {}'.format(e, loss))
            _loss.backward()
            self.optimizer.step()
            objective.append(-_loss)
            elbo.append(-_elbo)
            ksd.append(-_ksd)
            
    def evaluate_ksd(self, mu0: torch.Tensor, var0: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the KSD discrepancy for a given proposal q(mu0, var0)      https://arxiv.org/abs/1602.03253 

        Args:
            mu0 (torch.Tensor): mean of the Gaussian proposal               (batch_size, dim)
            var0 (torch.Tensor): variance of the Gaussian proposal          (batch_size, dim)

        Returns:
            torch.Tensor: KSD discrepancy
        """

        samples_KSD = self.generate_samples_KSD(mu0, var0)
        Sqx = self.dlogp(samples_KSD)
        ksd = torch.stack([self.KSD_no_second_gradient(samples_KSD_k, Sqx_k)
                           for samples_KSD_k, Sqx_k in zip(samples_KSD, Sqx)]).mean()

        return ksd

    def KSD_no_second_gradient(self, z, Sqx):
        # adapted from https://github.com/YingzhenLi/SteinGrad/blob/master/hamiltonian/ksd.py

        # compute the rbf kernel
        K, dimZ = z.shape
        pdist_square = torch.sum((z[None, :] - z[:, None]) ** 2, -1)

        xx = torch.triu(pdist_square, diagonal=1).flatten()
        xx = xx[xx > 0]

        # use median
        M = xx.shape[0]
        if M % 2 == 1:
            med_ind = torch.argsort(xx)[int((M - 1) / 2)]
            h_square = xx[med_ind]
        else:
            med_ind1 = torch.argsort(xx)[int(M / 2)]
            med_ind2 = torch.argsort(xx)[int(M / 2 - 1)]
            h_square = 0.5 * (xx[med_ind1] + xx[med_ind2])

        Kxy = torch.exp(- pdist_square / h_square / 2.0)

        # now compute KSD
        Sqxdy = torch.mm(Sqx.data, z.T) - \
                (torch.sum(Sqx.data * z, 1, keepdims=True)).repeat(1, K)
        Sqxdy = -Sqxdy / h_square

        dxSqy = Sqxdy.T
        dxdy = -pdist_square / (h_square ** 2) + dimZ / h_square
        # M is a (K, K) tensor
        M = (torch.mm(Sqx.data, Sqx.data.T) + Sqxdy + dxSqy + dxdy) * Kxy

        # the following for U-statistic
        M2 = M - torch.diag(torch.diag(M))
        return torch.sum(M2) / (K * (K - 1))

    def median_heruistic_proj(self, sample1, sample2):
        '''
        Median Heuristic for projected samples
        '''
        # samples 1 is * x g x N x 1
        # samples 2 is * x g x N x 1

        G = torch.sum(sample1 * sample1, dim=-1)  # * x num_g x N or r x g x N
        G_exp = G.unsqueeze(-2)  # * x num_g x 1 x N or * x r x g x 1 x N

        H = torch.sum(sample2 * sample2, dim=-1)  # * x num_g x N or * x r x g x N
        H_exp = H.unsqueeze(-1)  # * x numb_g x N x 1 or * x r x g x N x 1

        dist = G_exp + H_exp - 2 * torch.matmul(sample2, sample1.permute(0, 2, 1))  # * x G x N x N

        dist_triu = torch.triu(dist, diagonal=0)

        def get_median(v):  # v is g * N * N

            length_triu = 0.5 * (v.shape[1] + 1) * v.shape[1]

            mid = int(0.5 * (length_triu) + 1)

            return torch.topk(v.reshape(v.shape[0], -1), mid)[0].data[:, -1]

        return get_median(dist_triu).data.detach()

    def SE_kernel(self, sample1, sample2, **kwargs):
        '''
        Compute the square exponential kernel
        :param sample1: x
        :param sample2: y
        :param kwargs: kernel hyper-parameter: bandwidth
        :return:
        '''

        bandwidth = kwargs['kernel_hyper']['bandwidth_array']  # g or * x g

        bandwidth_exp = bandwidth.unsqueeze(-1).unsqueeze(-1)  # g x 1 x 1
        K = torch.exp(-(sample1 - sample2) ** 2 / (bandwidth_exp ** 2 + 1e-9))  # g x sam1 x sam2
        return K

    def d_SE_kernel(self, sample1, sample2, **kwargs):
        'The gradient of RBF kernel'
        K = kwargs['K']  # * x g x sam1 x sam2

        bandwidth = kwargs['kernel_hyper']['bandwidth_array']  # g or r x g or * x g

        bandwidth_exp = bandwidth.unsqueeze(-1).unsqueeze(-1)  # g x 1 x 1
        d_K = K * (-1 / (bandwidth_exp ** 2 + 1e-9) * 2 * (sample1 - sample2))  # g x sam1 x sam2

        return d_K

    def dd_SE_kernel(self, sample1, sample2, **kwargs):
        K = kwargs['K']  # * x g x sam1 x sam2

        bandwidth = kwargs['kernel_hyper']['bandwidth_array']  # g or r x g or * x g

        bandwidth_exp = bandwidth.unsqueeze(-1).unsqueeze(-1)  # g x 1 x 1
        dd_K = K * (2 / (bandwidth_exp ** 2 + 1e-9) - 4 / (bandwidth_exp ** 4 + 1e-9) * (sample1 - sample2) ** 2)

        return dd_K  # g x N x N

    def compute_max_SKSD(self, samples1, samples2, score1, score2, g, bandwidth_scale=1.0):
        '''
        tensorflow version of maxSKSD with median heuristics
        :param samples1: samples from q with shape: N x dim
        :param samples2: samples from q with shape: N x dim
        :param score1: score of p for samples 1 with shape N x dim
        :param score2: score of p for samples 2 with shape N x dim
        :param g: sliced direction with shape dim x dim
        :param bandwidth_scale: coefficient for bandwidth (default:1)
        :return: KDSSD: discrepancy value; divergence:each component for KDSSD (used for debug or GOF Test)
        '''

        kernel = self.SE_kernel
        d_kernel = self.d_SE_kernel
        dd_kernel = self.dd_SE_kernel

        dim = samples1.data.shape[-1]
        r = torch.eye(dim).type_as(samples1)

        kernel_hyper = {}
        ##### Compute the median for each slice direction g
        if samples1.shape[0] > 500:  # To reduce the sample number for median computation
            idx_crop = 500
        else:
            idx_crop = samples1.shape[0]

        g_cp_exp = g.unsqueeze(1)  # g x 1 x dim
        samples1_exp = samples1[0:idx_crop, :].unsqueeze(0)     # 1 x N x dim
        samples2_exp = samples2[0:idx_crop, :].unsqueeze(0)     # 1 x N x dim
        proj_samples1 = torch.sum(samples1_exp * g_cp_exp, dim=-1, keepdim=True)  # g x N x 1
        proj_samples2 = torch.sum(samples2_exp * g_cp_exp, dim=-1, keepdim=True)  # g x N x 1
        median_dist = self.median_heruistic_proj(proj_samples1, proj_samples2)  # g
        bandwidth_array = bandwidth_scale * 2 * torch.sqrt(0.5 * median_dist)
        kernel_hyper['bandwidth_array'] = bandwidth_array

        ##### Now compute the SKSD with slice direction g for each dimension
        # Compute Term1

        g_exp = g.reshape(g.shape[0], 1, g.shape[-1])  # g x 1 x D
        samples1_crop_exp = samples1.unsqueeze(0)  # 1 x N x D
        samples2_crop_exp = samples2.unsqueeze(0)  # 1 x N x D
        proj_samples1_crop_exp = torch.sum(samples1_crop_exp * g_exp, dim=-1)  # g x sam1
        proj_samples2_crop_exp = torch.sum(samples2_crop_exp * g_exp, dim=-1)  # g x sam2

        r_exp = r.unsqueeze(1)  # r x 1 x dim
        proj_score1 = torch.sum(r_exp * score1.data.detach().unsqueeze(0).type_as(samples1),
                                dim=-1, keepdim=True)
        proj_score2 = torch.sum(r_exp * score2.data.detach().unsqueeze(0).type_as(samples2), dim=-1)

        proj_score1_exp = proj_score1  # r x sam1 x 1
        proj_score2_exp = proj_score2.reshape(proj_score2.shape[0], 1, proj_score2.shape[-1])  # r x 1 x sam2


        K = kernel(sample1=proj_samples1_crop_exp.unsqueeze(-1), sample2=proj_samples2_crop_exp.unsqueeze(-2),
                   kernel_hyper=kernel_hyper)  # g x sam1 x sam 2


        Term1 = proj_score1_exp * K * proj_score2_exp  # g x sam1 x sam2

        # Compute Term2
        r_exp_exp = r_exp.unsqueeze(1)  # r x 1 x 1 x dim
        rg = torch.sum(r_exp_exp * g_exp.unsqueeze(-2), dim=-1)  # r x 1 x 1

        grad_2_K = -d_kernel(proj_samples1_crop_exp.unsqueeze(-1),
                             proj_samples2_crop_exp.unsqueeze(-2), kernel_hyper=kernel_hyper,
                             K=K)  # g x N x N

        Term2 = rg * proj_score1_exp * grad_2_K  # g x sam1 x sam2

        # Compute Term3

        grad_1_K = d_kernel(proj_samples1_crop_exp.unsqueeze(-1),
                            proj_samples2_crop_exp.unsqueeze(-2), kernel_hyper=kernel_hyper,
                            K=K)  # g x N x N
        Term3 = rg * proj_score2_exp * grad_1_K

        # Compute Term4

        grad_21_K = dd_kernel(proj_samples1_crop_exp.unsqueeze(-1),
                              proj_samples2_crop_exp.unsqueeze(-2), kernel_hyper=kernel_hyper,
                              K=K)  # g x N x N
        Term4 = (rg ** 2) * grad_21_K  # g x N x N

        divergence = Term1 + Term2 + Term3 + Term4  # g x sam1  x sam2

        KDSSD = divergence.sum() / (samples1.shape[0] * samples2.shape[0])

        return KDSSD