import numpy as np
import torch
from tqdm import tqdm


def mutual_information(x: torch.Tensor, y: torch.Tensor, bins: int=5, device: str='cuda') -> torch.Tensor:
    """
    Approximates the mutual information between tensors x and y in their last dimension by
    computing their histograms

    Args:
        x (torch.Tensor): Tensor with dimensions        (B  x  D  x  samples  x  1)
        y (torch.Tensor): Tensor with dimensions        (B  x  D  x  samples  x  P)
        bins (int, optional): Number of bins for the histograms. Defaults to 5.
        device (str, optional): Device for the computation. Defaults to 'cuda'.

    Returns:
        torch.Tensor: Mutual Information 
    """
 
    y = y[:, 0, :, :].unsqueeze(1) # note that y does not change wrt to xi

    mi_dims = []
    # shape of x: batch x dims x samples x 1
    for d in range(x.shape[1]):
        xi = x[:, d, :, :].unsqueeze(1)
        
        # If categorical, bins=categories
        if len(xi.unique()) < bins:
            bins_x = len(xi.unique())
        else: bins_x = bins

        if len(y.unique()) < bins:
            bins_y = len(y.unique())
        else: bins_y = bins

        dim_y = y.shape[-1]
        vars = 1
        samples = x.shape[-2]  # number of samples p(x,y) for each observation
        batch_size = x.shape[0]
        
        
        if dim_y==1:
            Nxy = torch.zeros(batch_size, vars, bins_x, bins_y).to(device)
            Nx = torch.zeros(batch_size, vars, bins_x, 1).to(device)
            Ny = torch.zeros(batch_size, vars, bins_y, 1).to(device)
            
            Qx = quantize(xi, bins_x)
            Qy = quantize(y, bins_y)

            for j in range(bins_y):
                Ny[:, :, j] = (Qy == j).sum(-2)
            for i in range(bins_x):
                Nx[:, :, i] = (Qx == i).sum(-2)
                for j in range(bins_y):
                    Nxy[:, :, i, j] = torch.logical_and(Qx == i, Qy == j).sum(-2).reshape(batch_size, vars)

            Px = Nx / samples
            Py = Ny / samples
            Py = Py.permute(0, 1, 3, 2)
            Pxy = Nxy / samples
        
            aux = torch.matmul(Px, Py).unsqueeze(-1)
            mi = Pxy * (torch.log(Pxy) - torch.log(torch.matmul(Px, Py)) )
            
            mi[torch.isfinite(mi)==False] = 0
            mi = mi.sum(-1).sum(-1)
                

        elif dim_y == 2:
            
            Nxy = torch.zeros(batch_size, vars, bins_x, bins_y).to(device)
            Nx = torch.zeros(batch_size, vars, bins_x, 1).to(device)
            Ny = torch.zeros(batch_size, vars, bins_y, 1).to(device)
            
            Qx = quantize(xi, bins_x)
            Qy = quantize(y, bins_y)

            j=0
            for j1 in range(int(np.sqrt(bins_y))):
                for j2 in range(int(np.sqrt(bins_y))):
                    Ny[:, :, j] = torch.logical_and(Qy[:, :, :, 0] == j1, Qy[:, :, :, 1]==j2).sum(-1).unsqueeze(-1)
                    j+=1

            for i in range(bins_x):
                Nx[:, :, i] = (Qx == i).sum(-2)
                j=0
                for j1 in range(int(np.sqrt(bins_y))):
                    for j2 in range(int(np.sqrt(bins_y))):
                        aux = torch.logical_and(Qy[:, :, :, 0] == j1, Qy[:, :, :, 1]==j2).unsqueeze(-1)
                        Nxy[:, :, i, j]= torch.logical_and(Qx == i, aux).sum(-2).reshape(batch_size, vars)
                        j+=1

            Px = Nx / samples
            Py = Ny / samples
            Py = Py.permute(0, 1, 3, 2)
            Pxy = Nxy / samples
        
            aux = torch.matmul(Px, Py)

            mi = Pxy * (torch.log(Pxy) - torch.log(aux))

            mi[torch.isfinite(mi)==False] = 0
            mi = mi.sum(-1).sum(-1)

        mi_dims.append(mi)

    mi = torch.cat(mi_dims, -1)
    return mi


def quantize(batch: torch.Tensor, bins: int) -> torch.Tensor:
    """
    Returns the quantized version of the input batch given a number of bins

    Args:
        batch (torch.Tensor): batch containing data (..., dim_data)
        bins (int): number of intervals for the quantization

    Returns:
        torch.Tensor: quantized tensor containing values from 0 to bins-1
    """

    mins_x = batch.min(dim=-2)[0].unsqueeze(-2)
    maxs_x = batch.max(dim=-2)[0].unsqueeze(-2)

    q = (maxs_x - mins_x) / bins

    index = torch.floor((batch-mins_x) / q)
    index[index==bins] = bins - 1

    return index


def random_learning(batch, model, K_metric=1000, step=1):

    batch = [b.to(model.device) for b in batch]
    observations = batch[0]
    
    y_true = batch[2].clone().to(model.device)
    y = torch.zeros_like(y_true).to(model.device)
    observed_x = torch.zeros_like(observations).to(model.device)
    xo = torch.zeros_like(observations).to(model.device)
    observed_y = torch.zeros_like(y)

    xoi = xo.clone()
    metric = []
    ll = []
    #print("\t\t\t\t No observed variables , MSE: {mse:f}".format(mse=mse[-1]))
    tqdm_step = tqdm(total=model.dim_x, desc='Step (rand)', position=4, leave=False)
    for d in range(int(np.ceil(model.dim_x / step))+1):
        next_variables = torch.stack([torch.where(row == 0)[0] for row in observed_x])

        # Get data
        batch_ = [xoi, observed_x, y, observed_y]
        x, observed_x, y, observed_y = batch_
        xn = model.normalize_x(xoi)
        xt, yt, xy, observed = model.preprocess_batch(batch_) 
        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)

        mu_z, logvar_z = model.encoder(xy)
        z = model.sample_z(mu_z, logvar_z, samples=K_metric)
        theta_x = model.decoder(z)
        x_hat = model.build_x_hat(xn, observed_x, theta_x)
        zx = torch.cat([z,x_hat],dim=-1)

        theta_y = model.predictor(zx)
        y_pred = model.denormalize_y(theta_y.mean(-2))
        if model.likelihood_y == 'categorical':
                y_pred = torch.exp(y_pred)
        elif model.likelihood_y == 'bernoulli':
            y_pred = torch.sigmoid(y_pred)
        metric.append(model.prediction_metric(y_pred, y_true))
        
        # Log-likelihood of the normalised target
        yn_true = model.normalize_y(y_true)
        rec = model.predictor.logp(yn_true, torch.ones_like(y_true), theta=theta_y)
        # mean per dimension (y is all observed in test)
        rec = rec.sum(-1, keepdim=True)
        ll_y = torch.logsumexp(rec, dim=-2) - np.log(K_metric)
        # mean per sample
        ll_y = ll_y.mean()

        ll.append(ll_y)

        tqdm_step.set_postfix({model.prediction_metric_name: metric[-1].detach().cpu().numpy(), "ll": ll[-1].detach().cpu().numpy()})
        
        if d < int(np.ceil(model.dim_x / step)):
            if (observed_x[0,:]==0).sum() > step:
                selected_idx = np.stack([ np.random.permutation(model.dim_x-(d*step))[:step] for n in range(xoi.shape[0])])
            else:
                selected_idx = np.stack([ np.random.permutation(next_variables.shape[-1]) for n in range(xoi.shape[0])])

            for n, s in enumerate(selected_idx):
                selected = next_variables[n, s]
                xoi[n, selected] = observations[n, selected]
                observed_x[n, selected] = 1

            tqdm_step.update(step)

        #print("\t\t\t\t Observed variable {selected:d} (iteration {d:d}), MSE: {mse:f}".format(selected=selected, d=d, mse=mse_d))

    return torch.stack(metric), torch.stack(ll)


def active_learning_kl(model: object, batch: tuple, samples=1000, x_samples=1, step=1) -> tuple:
    """
    Returns the metric and log-likelihood curves of the SAIA experiment using the 
    KL method from https://arxiv.org/abs/2006.11941 

    Args:
        model (object): model with the active_learning method defined
        batch (tuple): batch containing x, observed_x, y, observed_y
        samples (int, optional): number of samples from the latent space. Defaults to 1000.
        x_samples (int, optional): number of samples from the predictive distribution. Defaults to 1.
        step (int, optional): number of variables to add within each step. Defaults to 1.

    Returns:
        tuple: 
            - (torch.Tensor): mean metric at each step              (dim_x + 1)
            - (torch.Tensor): mean log likelihood at each step      (dim_x + 1)
            - (np.array): mean elapsed time at each step        (dim_x + 1)
    """
    import time

    batch = [b.to(model.device) for b in batch]
    observations = batch[0]
    y_true = batch[2].clone().to(model.device)

    y = torch.zeros_like(y_true).to(model.device)
    observed_x = torch.zeros_like(observations).to(model.device)
    xo = torch.zeros_like(observations).to(model.device)
    observed_y = torch.zeros_like(y)

    xoi = xo.clone()

    batch_size = len(xoi)
    metric = []
    ll = []
    times = []


    tqdm_step = tqdm(total=model.dim_x, desc='Step (IR)', position=2, leave=False)
    start = time.time()
    for d in range(int(np.ceil(model.dim_x / step))+1):

        if d==0:
                times.append(0.0)
        else:
            times.append(time.time() - start)

        batch_ = [xoi, observed_x, y, observed_y]

        # Get data
        x, _, y, _ = batch_
        xn = model.normalize_x(x)
        xt, yt, xy, observed = model.preprocess_batch(batch_) 
        xt = xt * batch_[1]
        # xt is the preprocessed input (xt=x if no preprocessing)
        # observed is observed_x OR observed_y (for not using kl if no observed data)

        mu_z, logvar_z = model.encoder(xy)
        z = model.sample_z(mu_z, logvar_z, samples=samples)
        theta_x = model.decoder(z)
        x_hat = model.build_x_hat(xn, observed_x, theta_x)
        zx = torch.cat([z,x_hat],dim=-1)

        theta_y = model.predictor(zx)
        y_pred = model.denormalize_y(theta_y.mean(-2))
        y_pred = model.denormalize_y(theta_y.mean(-2))
        if model.likelihood_y == 'categorical':
            y_pred = torch.exp(y_pred)
        elif model.likelihood_y == 'bernoulli':
            y_pred = torch.sigmoid(y_pred)
        
        metric.append(model.prediction_metric(y_pred, y_true))
        
        # Log-likelihood of the normalised target
        yn_true = model.normalize_y(y_true)
        rec = model.predictor.logp(yn_true, torch.ones_like(y_true), theta=theta_y)
        # mean per dimension (y is all observed in test)
        rec = rec.sum(-1, keepdim=True)
        ll_y = torch.logsumexp(rec, dim=-2) - np.log(samples)
        # mean per sample
        ll_y = ll_y.mean()
        ll.append(ll_y)
        
        xi = model.psample_x(theta_x, samples=x_samples).reshape(-1, theta_x.shape[-1])

        xi = model.denormalize_x(xi)
        xi = model.preproc_x(xi, torch.ones_like(xi)).reshape(theta_x.shape[0], x_samples, samples, xi.shape[-1])
        
        ys = model.psample_y(theta_y, samples=x_samples)
        

        tqdm_step.set_postfix({model.prediction_metric_name: metric[-1].detach().cpu().numpy(), "ll": ll[-1].detach().cpu().numpy()})

        if d < int(np.ceil(model.dim_x / step)):

            #theta_x = model.invert_preproc(theta_x)

            next_variables = torch.stack([torch.where(row == 0)[0] for row in observed_x])
            
            # for the rest of missing variables
            # repeat for MC
            xoi_MC = xt.repeat(samples, 1, 1).transpose(0, 1)
            xoi_MC = xoi_MC.repeat(x_samples, 1, 1, 1).transpose(0, 1)
            observed_x_MC = observed_x.repeat(samples, 1, 1).transpose(0, 1)
            observed_x_MC = observed_x_MC.repeat(x_samples, 1, 1, 1).transpose(0, 1)
            yo_MC = torch.zeros([batch_size, x_samples, samples, model.dim_y], device=model.device)
            observed_y_MC = torch.zeros_like(yo_MC)

            mu_zo = mu_z.repeat(samples, 1, 1).transpose(0, 1)
            mu_zo = mu_zo.repeat(x_samples, 1, 1, 1).transpose(0, 1)
            logvar_zo = logvar_z.repeat(samples, 1, 1).transpose(0, 1)
            logvar_zo = logvar_zo.repeat(x_samples, 1, 1, 1).transpose(0, 1)
            R = torch.zeros(batch_size, model.dim_x-d*step)
            for i in range(model.dim_x-(d*step)):

                next = next_variables[:, i]
                mask = torch.zeros_like(xi)
                for n in range(len(mask)):
                    mask[n, :, :, next[n]] = 1

                # obtain q(z | xi, xo)
                xoi_ = xoi_MC + xi * mask   # this includes next variable xi sampled from q
                observed_xoi_ = observed_x_MC + mask

                xy_io = torch.cat([xoi_, observed_xoi_, yo_MC, observed_y_MC], axis=-1)
                xy_io_flat = xy_io.reshape(-1, xy_io.shape[-1])
                mu_zio, logvar_zio = model.encoder(xy_io_flat)
                mu_zio = mu_zio.reshape(batch_size, x_samples, samples, model.latent_dim)
                logvar_zio = logvar_zio.reshape(batch_size, x_samples, samples, model.latent_dim)
                
                # we already have q(z | xo) in mu_zo, logvar_zo

                # obtain q(z | y, xi, xo)
                xy_ioy = torch.cat([xoi_, observed_xoi_, ys, torch.ones_like(ys)], axis=-1)
                xy_ioy_flat = xy_ioy.reshape(-1, xy_ioy.shape[-1])
                mu_zioy, logvar_zioy = model.encoder(xy_ioy_flat)
                mu_zioy = mu_zioy.reshape(batch_size, x_samples, samples, model.latent_dim)
                logvar_zioy = logvar_zioy.reshape(batch_size, x_samples, samples, model.latent_dim)

                # obtain q(z | y, xo)
                xy_oy = torch.cat([xoi_MC, observed_x_MC, ys, torch.ones_like(ys)], axis=-1)
                xy_oy_flat = xy_oy.reshape(-1, xy_oy.shape[-1])
                mu_zoy, logvar_zoy = model.encoder(xy_oy_flat)
                mu_zoy = mu_zoy.reshape(batch_size, x_samples, samples,  model.latent_dim)
                logvar_zoy = logvar_zoy.reshape(batch_size, x_samples, samples, model.latent_dim)

                R_next = information_reward( mu_zio, logvar_zio, mu_zo, logvar_zo, mu_zioy, logvar_zioy, mu_zoy, logvar_zoy )
                #R[n] = R_next
                R[:, i] = R_next

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


def information_reward(mu_zio: torch.Tensor, logvar_zio: torch.Tensor, mu_zo: torch.Tensor, logvar_zo: torch.Tensor,
                    mu_zioy: torch.Tensor, logvar_zioy: torch.Tensor, mu_zoy: torch.Tensor, logvar_zoy: torch.Tensor) -> torch.Tensor:
    """
    Implements the information reward approximation from https://arxiv.org/abs/2006.11941 

    Args:
        mu_zio (torch.Tensor):                          mu (z | xi, xo )        
        logvar_zio (torch.Tensor):                      logvar (z | xi, xo)
        mu_zo (torch.Tensor): _description_             mu (z | xo)
        logvar_zo (torch.Tensor): _description_         logvar (z | xo)
        mu_zioy (torch.Tensor): _description_           mu (z | xi, xo, y)
        logvar_zioy (torch.Tensor): _description_       logvar (z | xi, xo, y)
        mu_zoy (torch.Tensor): _description_            mu (z | xo, y)
        logvar_zoy (torch.Tensor): _description_        logvar (z | xo, y)

    Returns:
        torch.Tensor: information reward
    """
    var_zio = torch.exp(logvar_zio)
    var_zo = torch.exp(logvar_zo)
    var_zioy = torch.exp(logvar_zioy)
    var_zoy = torch.exp(logvar_zoy)

    kl_x = 0.5 * torch.sum(
            (mu_zio - mu_zo)**2 / var_zo + var_zio / var_zo - 1. - logvar_zio + logvar_zo,
            axis=-1)
    kl_y = 0.5 * torch.sum(
            (mu_zioy - mu_zoy)**2 / var_zoy + var_zioy / var_zoy - 1. - logvar_zioy + logvar_zoy,
            axis=-1)

    Ri = kl_x - kl_y
    return Ri.mean(-1).mean(-1)