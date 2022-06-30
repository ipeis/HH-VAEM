
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from torch import nn
import numpy as np
import torch

def get_arch(dim_x: int, dim_y: int, latent_dim: int, arch_name='base', categories_x=1, categories_y=1, dim_h=256):
    """
    Get NNs for the params of q(z|xy), p(x|z) and p(y|z,x) 

    Args:
        dim_x (int): dimension of the input data
        dim_y (int): dimension of the target
        latent_dim (int): dimension of the latent space
        arch_name (str, optional): name of the architecture. Defaults to 'base'.
        categories_x (int, optional): number of input categories when categorical. Defaults to 1.
        categories_y (int, optional): number of target categories when categorical. Defaults to 1.
        dim_h (int, optional): dimension of the hidden units. Defaults to 256.

    Returns:
        torch.nn.Sequential: encoder    q(z|xy)
        torch.nn.Sequential: decoder    p(x|z)
        torch.nn.Sequential: predictor  p(y|z,x) 
    """
    if arch_name=='base':
        encoder = nn.Sequential(nn.Linear(2*dim_x + 2*dim_y, dim_h), nn.ReLU(), nn.Linear(dim_h, 2 * latent_dim))
        decoder = nn.Sequential(nn.Linear(latent_dim, dim_h), nn.ReLU(), nn.Linear(dim_h, dim_x*categories_x))
        predictor = nn.Sequential(nn.Linear(latent_dim + 2*dim_x*categories_x, dim_h), nn.ReLU(), nn.Linear(dim_h, dim_y*categories_y))

    # You can add more complex architectures here
    if arch_name=='mnist_cnn':
        # Convolutional Architecture for MNIST
        encoder_net = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
            ) 
        # Output channels of the convolution
        dim_hx = 512
        encoder = ConvEncoderXY(dim_x, dim_hx, dim_y, encoder_net, dim_h, latent_dim)
        decoder_net = nn.Sequential(
            nn.Linear(latent_dim, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, 512),
            nn.ReLU(),
            View((-1, 32, 4, 4)),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
            View((-1, dim_x)),
        )
        decoder = ConvDecoder(decoder_net)
        predictor = nn.Sequential(nn.Linear(latent_dim + 2*dim_x*categories_x, dim_h), nn.ReLU(), nn.Linear(dim_h, dim_y*categories_y))

    return encoder, decoder, predictor


class ConvEncoderXY(nn.Module):
    """
    Convolutional encoder for input X and MLP for the concatenation of
    the transformed X and the target Y.
    """
    def __init__(self, dim_x, dim_hx, dim_y, conv_net, dim_h, latent_dim, **args):
        """
        Inialization

        Args:
            dim_x (_type_): _description_
            dim_hx (_type_): _description_
            dim_y (_type_): _description_
            conv_net (_type_): _description_
            dim_h (_type_): _description_
            latent_dim (_type_): _description_
        """
        super().__init__()
        self.dim_x = dim_x
        self.dim_hx = dim_hx
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.latent_dim = latent_dim
        self.conv_net = conv_net
        self.encoderXY = nn.Sequential(
            nn.Sequential(nn.Linear(dim_hx + dim_x + 2*dim_y, dim_h), nn.ReLU(), nn.Linear(dim_h, 2 * latent_dim))
        )

    def forward(self, xy):
        x = xy[..., :self.dim_x]
        obs_x = xy[..., self.dim_x:2*self.dim_x]
        y_and_mask = xy[..., -2*self.dim_y:]
        x_image = x.reshape(x.shape[0], 1, int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1]))) # Assume square images (batch_size, 1, HW, HW)
        hx = self.conv_net(x_image)
        hx = hx.reshape(hx.shape[0], self.dim_hx)
        hxy = torch.cat([hx, obs_x, y_and_mask], -1)
        theta_z = self.encoderXY(hxy)

        return theta_z
        

class ConvDecoder(nn.Module):
    """
    Implements a Convolutional decoder
    """
    def __init__(self, network):
        """Network for the decoder

        Args:
            network (_type_): _description_
        """
        super().__init__()
        self.network = network

    def forward(self, z: torch.Tensor):
        """Performs the decoding and flattens the output

        Args:
            z (torch.Tensor): latent code           (batch_size, samples, latent_dim)

        Returns:
            torch.Tensor: decoded distribution      (batch_size, samples, dim_x**2)
        """
        batch_size, samples = z.shape[:2]
        theta_x = self.network(z)
        return theta_x.reshape(batch_size, samples, theta_x.shape[-1])

class View(nn.Module):
    """ For reshaping tensors inside Sequential objects"""
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Flatten(nn.Module):
    """ For flattening image tensors to match HxW as the last dimension"""
    def __init__(self):
        super(View, self).__init__()

    def forward(self, image):
        return torch.flatten(image, -2, -1)