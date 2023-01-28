# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
if _TORCHVISION_AVAILABLE:
    import torchvision
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
import numpy as np
from scipy.stats import multivariate_normal as mvn
import os
from src import *
from src.models.hmc_vae import HMCVAE
from src.models.hh_vae import HHVAE
from src.models.h_vae import HVAE
from src.models.h_vae_no_reparam import HVAENoReparam
from src.models.base import reparameterize

# ============= Callbacks for PL Trainer ============= #

class logWeights(Callback):
    """
    Callback for logging model weights in Tensorboard 

    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # iterating through all parameters
        str_title = f"{pl_module.__class__.__name__}_images"
        for name, params in pl_module.named_parameters():
            pl_module.logger.experiment.add_histogram(name, params, pl_module.current_epoch)

class ImageSampler(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the ``forward`` function for generation
    Requirements::
        # model must have img_dim arg
        model.img_dim = (1, 28, 28)
        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)
    Example::
        from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
    """

    def __init__(
        self,
        num_samples: int = 20,
        nrow: int = 10,
        padding: int = 2,
        channels: int = 1,
        size: int = 28,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.channels = channels
        self.size = size
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        
        if hasattr(pl_module, 'latent_dims'):
            if isinstance(pl_module, HVAENoReparam):
                z = [torch.normal(mean=0.0, std=1.0, 
                        size=(self.num_samples, 1,  pl_module.hparams.latent_dims[-1]), 
                        device=pl_module.device) ]
                for l in np.arange(pl_module.layers-2, -1, -1):
                    mu_l, logvar_l = torch.chunk(pl_module.prior.NNs[l](z[-1]), 2, dim=-1)
                    z.append(reparameterize(mu_l, torch.exp(logvar_l)))
                z = z[::-1]
            else:
                Epsilon = [torch.normal(mean=0.0, std=1.0, 
                        size=(self.num_samples, 1, dim), 
                        device=pl_module.device) for dim in pl_module.latent_dims]
                z = pl_module.prior.generative_path(Epsilon)
        else:
            z = torch.normal(mean=0.0, std=1.0, 
                        size=(self.num_samples, 1,  pl_module.hparams.latent_dim), 
                        device=pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            if isinstance(z, list):
                z = z[0]
            images = torch.sigmoid(pl_module.decoder(z))
            pl_module.train()

        images = images.reshape(-1, self.channels, self.size, self.size)

        grid = torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        str_title = f"{pl_module.__class__.__name__}_samples"
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

class ImageReconstruction(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the ``forward`` function for generation
    Requirements::
        # model must have img_dim arg
        model.img_dim = (1, 28, 28)
        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)
    Example::
        from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
    """

    def __init__(
        self,
        num_images: int = 10,
        padding: int = 2,
        channels: int = 1,
        size: int = 28,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.num_images = num_images
        self.padding = padding
        self.channels = channels
        self.size = size
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # generate images
        with torch.no_grad():
            pl_module.eval()
            
            batch = next(iter(pl_module.test_dataloader()))
            
            batch = [b[:self.num_images].to(pl_module.device) for b in batch]
            # Get data
            x, observed_x, y, observed_y = batch
            
            xn = pl_module.normalize_x(x)
            xt, yt, xy, observed = pl_module.preprocess_batch(batch) 
            # xt is the preprocessed input (xt=x if no preprocessing)
            # observed is observed_x OR observed_y (for not using kl if no observed data)
            mu_z, logvar_z = pl_module.encoder(xy)
            
            if isinstance(pl_module, HVAE) or isinstance(pl_module, HHVAE):  
                mu_z = pl_module.prior.generative_path(mu_z)

            if isinstance(pl_module, HMCVAE) or isinstance(pl_module, HHVAE):
                z = pl_module.sample_z(mu_z, logvar_z, samples=100, hmc=pl_module.hmc)
            elif isinstance(pl_module, HVAENoReparam):
                z, _, _ =  pl_module.sample_z(mu_z, logvar_z, samples=100)
            else:
                z = pl_module.sample_z(mu_z, logvar_z, samples=100)
            
            if isinstance(z, list):
                z = z[0]
            
            reconst = torch.sigmoid(pl_module.decoder(z)).reshape(self.num_images, 100, -1)
            reconst = reconst.mean(1)

            pl_module.train()
        
        images = x*observed_x

        images = images[:self.num_images].reshape(-1, self.channels, self.size, self.size)
        reconst = reconst[:self.num_images].reshape(-1, self.channels, self.size, self.size)

        all = torch.cat([images, reconst])

        grid = torchvision.utils.make_grid(
            tensor=all,
            nrow=self.num_images,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        str_title = f"{pl_module.__class__.__name__}_reconstructions"
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

class Inpainting(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the ``forward`` function for generation
    Requirements::
        # model must have img_dim arg
        model.img_dim = (1, 28, 28)
        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)
    Example::
        from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
    """

    def __init__(
        self,
        num_images: int = 10,
        padding: int = 2,
        channels: int = 1,
        size: int = 28,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.num_images = num_images
        self.padding = padding
        self.channels = channels
        self.size = size
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # generate images
        with torch.no_grad():
            pl_module.eval()
            
            batch = next(iter(pl_module.test_dataloader()))
            batch = [b[:self.num_images].to(pl_module.device) for b in batch]

            # Missing mask
            mask = torch.zeros_like(batch[0])
            center = np.random.randint(16, 48, 2)
            mask = mask.reshape(-1, self.channels, self.size, self.size)
            mask[:, :, center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 1
            mask = mask.reshape(-1, self.channels * self.size * self.size)

            observed_x = torch.logical_not(mask)

            # Get data
            x, _, y, observed_y = batch

            batch = (x, observed_x, y, observed_y)
            
            xn = pl_module.normalize_x(x)
            xt, yt, xy, observed = pl_module.preprocess_batch(batch) 
            # xt is the preprocessed input (xt=x if no preprocessing)
            # observed is observed_x OR observed_y (for not using kl if no observed data)
            mu_z, logvar_z = pl_module.encoder(xy)
            
            if isinstance(pl_module, HMCVAE) or isinstance(pl_module, HHVAE):
                z = pl_module.sample_z(mu_z, logvar_z, samples=100, hmc=pl_module.hmc)
            else:
                z = pl_module.sample_z(mu_z, logvar_z, samples=100)
            
            if isinstance(z, list):
                z = z[0]
            
            reconst = torch.sigmoid(pl_module.decoder(z)).reshape(self.num_images, 100, -1)
            reconst = reconst.mean(1)

            pl_module.train()
        
        images = x*observed_x

        images = images[:self.num_images].reshape(-1, self.channels, self.size, self.size)
        reconst = reconst[:self.num_images].reshape(-1, self.channels, self.size, self.size)

        inpainted = images.clone().detach()
        mask = mask[:self.num_images].reshape_as(inpainted)
        inpainted[mask.bool()] = reconst[mask.bool()]

        all = torch.cat([images, reconst, inpainted])

        grid = torchvision.utils.make_grid(
            tensor=all,
            nrow=self.num_images,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        str_title = f"{pl_module.__class__.__name__}_inpainting"
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

class ImageCondSampler(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the ``forward`` function for generation
    Requirements::
        # model must have img_dim arg
        model.img_dim = (1, 28, 28)
        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)
    Example::
        from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
    """

    def __init__(
        self,
        num_images: int = 20,
        nrow: int = 10,
        padding: int = 2,
        channels: int = 1,
        size: int = 28,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.num_images = num_images
        self.nrow = nrow
        self.padding = padding
        self.channels = channels
        self.size = size
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # generate images
        with torch.no_grad():
            pl_module.eval()
            
            batch = next(iter(pl_module.test_dataloader()))
            
            batch = [b[:1,:].to(pl_module.device) for b in batch]

            attr = np.random.randint(0,40)
            attr_name = pl_module.test_dataloader().dataset.attr_names[attr]

            # Get data
            x, observed_x, y, observed_y = batch
            y = torch.zeros_like(y)
            y[:,attr] = 1
            observed_y = torch.zeros_like(observed_y)
            observed_y[:,attr] = 1

            x = torch.zeros_like(x)
            observed_x = torch.zeros_like(observed_x)
            
            batch = [x, observed_x, y, observed_y]
            
            xn = pl_module.normalize_x(x)
            xt, yt, xy, observed = pl_module.preprocess_batch(batch) 
            # xt is the preprocessed input (xt=x if no preprocessing)
            # observed is observed_x OR observed_y (for not using kl if no observed data)
            mu_z, logvar_z = pl_module.encoder(xy)

            if isinstance(pl_module, HMCVAE) or isinstance(pl_module, HHVAE):
                z = pl_module.sample_z(mu_z, logvar_z, samples=self.num_images, hmc=pl_module.hmc)
            else:
                z = pl_module.sample_z(mu_z, logvar_z, samples=self.num_images)
            
            if hasattr(pl_module, 'latent_dims'):
                z = z[0]
            
            reconst = torch.sigmoid(pl_module.decoder(z).reshape(self.num_images, -1))

            pl_module.train()
        
        #images = x*observed_x

        #images = images[:self.num_images].reshape(-1, self.channels, self.size, self.size)
        reconst = reconst[:self.num_images].reshape(-1, self.channels, self.size, self.size)

        #all = torch.cat([images, reconst])

        grid = torchvision.utils.make_grid(
            tensor=reconst,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        str_title = f"{pl_module.__class__.__name__}_{attr_name}_samples"
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)


class plotHMCsteps(Callback):
    """
    Callback for logging images with the HMC step sizes in Tensorboard 

    """

    def __init__(
        self,
        log_steps=1
    ) -> None:
        """
        Args:
            log_steps: interval of steps for logging
        """

        super().__init__()
        self.log_steps = log_steps

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        if pl_module.step_idx % self.log_steps == 0:
            f = plt.figure(figsize=(4, 4))
            eps = torch.exp(pl_module.HMC.log_eps) + 0.01
            plt.imshow(eps.cpu().detach().numpy(), cmap="Blues")
            plt.xlabel(r'$d$'); plt.ylabel(r'$t$')
            plt.colorbar()

            plt.gcf().subplots_adjust(bottom=0.15)

            str_title = f"{pl_module.__class__.__name__}_HMC_steps"
            save_path = '{}/logs/'.format(LOGDIR) + trainer.logger.name + '/version_' + str(
                trainer.logger.version) + '/_HMC_steps.png'
            plt.savefig(save_path, dpi=150)

            im = Image.open(save_path)
            im = transforms.ToTensor()(im)

            trainer.logger.experiment.add_image(str_title, im, global_step=trainer.global_step)

            plt.close(f)

class CheckpointEveryNSteps(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=True,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix

    def on_batch_end(self, trainer: Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            filename = f"epoch={epoch}-step={global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

class plot2DEncodingsPointPred(Callback):
    """
    Plots an approximation of the true posterior (green contour), the Gaussian proposal given
    by the encoder (blue contour) and samples from HMC (orange stars).
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        
        if pl_module.global_step>10e3:#pl_module.pre_steps:
            if pl_module.current_epoch % 1 == 0:
                #print('Plotting latent space and reconstructions...')
                f, ax = plt.subplots(figsize=(4, 4))
                plt.subplots_adjust(wspace=0.3)

                #ind = torch.randint(size=(1,), high=len(pl_module.test_dataloader().dataset)).cpu().numpy()
                ind = np.random.randint(0, len(pl_module.test_dataloader().dataset))
                batch = pl_module.test_dataloader().dataset.__getitem__(ind)
                batch = [torch.Tensor(b).to(pl_module.device).reshape(-1, b.shape[-1]) for b in batch]

                x, observed_x, y, observed_y = batch

                xn = pl_module.normalize_x(x)
                yn = pl_module.normalize_y(y)
                xo = xn * observed_x
                yo = yn*observed_y

                # Get data
                xt, yt, xy, observed = pl_module.preprocess_batch(batch) 

                # Encode q(z | x_tilde, y_tilde)
                muz, logvarz = pl_module.encoder(xy)

                covz = torch.exp(logvarz)
                sigma_z = torch.sqrt(covz)*torch.exp(pl_module.HMC.log_inflation)
                covz = sigma_z**2
                z = torch.distributions.multivariate_normal.MultivariateNormal(muz, torch.diag(covz.squeeze())).sample(
                    [10000]).squeeze()
                K=100
                zT = pl_module.sample_z(muz, torch.exp(logvarz), samples=K)

                # approximate normalization constant with IW
                logp = pl_module.elbo_iwae(batch, samples=1000).mean().cpu().detach().numpy()


                muz = muz.cpu().detach().squeeze().numpy()
                sigma_z = sigma_z.detach().cpu().numpy()
                covz = np.diag(covz.detach().squeeze().cpu().numpy())
                zT = zT.detach().cpu().numpy()
                #x_dec = x_dec.detach().cpu().numpy()
                z = z.detach().cpu().numpy()

                intervals = 200
                span=0.8
                x0min = muz[0] - span
                x0max = muz[0] + span
                x1min = muz[1] - span
                x1max = muz[1] + span
                x0 = np.linspace(x0min, x0max, intervals)
                x1 = np.linspace(x1min, x1max, intervals)
                X0, X1 = np.meshgrid(x0, x1)
                xs = np.vstack([X0.ravel(), X1.ravel()]).T

                zs = torch.from_numpy(xs.copy()).type(torch.float32).to(pl_module.device).unsqueeze(0)
                Y = np.exp(pl_module.logp(xo, observed_x, yo, observed_y, zs).cpu().detach().numpy() - logp)
                Y = Y.reshape([intervals, intervals])
                cont1 = ax.contour(X0, X1, Y, 15, cmap='Greens')

                # 2 Plot q0(z)
                Y = mvn(muz, covz).pdf(xs)
                Y = Y.reshape([intervals, intervals])
                cont2 = ax.contour(X0, X1, Y, 15, cmap='Blues')

                h1, _ = cont1.legend_elements()
                h2, _ = cont2.legend_elements()

                # 3 Plot samples from zT
                ax.plot(zT[0, :, 0], zT[0, :, 1], linestyle='', marker='*', color='orange', markersize=8, alpha=0.4)
                plt.axis('off')
                ax.set(xlabel=r'$z_0$', ylabel=r'$z_1$', xlim=[x0min, x0max], ylim=[x1min, x1max])

                plt.gcf().subplots_adjust(bottom=0.15)
                save_path = LOGDIR + '/logs/' + trainer.logger.name + '/version_' + str(
                    trainer.logger.version) + '/' + str(ind) + '_posterior.pdf'
                plt.savefig(save_path)

                """im = Image.open(save_path)
                im = transforms.ToTensor()(im)
                str_title = f"{pl_module.__class__.__name__}_encodings_point_pred"
                trainer.logger.experiment.add_image(str_title, im, global_step=trainer.global_step)
                """
                plt.close(f)