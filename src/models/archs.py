
from torch import nn

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

    return encoder, decoder, predictor

