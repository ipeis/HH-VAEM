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