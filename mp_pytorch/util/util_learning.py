"""
    Utilities of learning operation
"""
from typing import Union
import numpy as np
import torch
import mp_pytorch.util as util


def joint_to_conditional(joint_mean: Union[np.ndarray, torch.Tensor],
                         joint_L: Union[np.ndarray, torch.Tensor],
                         sample_x: Union[np.ndarray, torch.Tensor]) -> \
        [Union[np.ndarray, torch.Tensor]]:
    """
    Given joint distribution p(x,y), and a sample of x, do:
    Compute conditional distribution p(y|x)
    Args:
        joint_mean: mean of joint distribution
        joint_L: cholesky distribution of joint distribution
        sample_x: samples of x

    Returns:
        conditional mean and L
    """

    # Shape of joint_mean:
    # [*add_dim, dim_x + dim_y]
    #
    # Shape of joint_L:
    # [*add_dim, dim_x + dim_y, dim_x + dim_y]
    #
    # Shape of sample_x:
    # [*add_dim, dim_x]
    #
    # Shape of conditional_mean:
    # [*add_dim, dim_y]
    #
    # Shape of conditional_cov:
    # [*add_dim, dim_y, dim_y]

    # Check dimension
    dim_x = sample_x.shape[-1]
    # dim_y = joint_mean.shape[-1] - dim_x

    # Decompose joint distribution parameters
    mu_x = joint_mean[..., :dim_x]
    mu_y = joint_mean[..., dim_x:]

    L_x = joint_L[..., :dim_x, :dim_x]
    L_y = joint_L[..., dim_x:, dim_x:]
    L_x_y = joint_L[..., dim_x:, :dim_x]

    if util.is_ts(joint_mean):
        cond_mean = mu_y + \
                    torch.einsum('...ik,...lk,...lm,...m->...i', L_x_y, L_x,
                                 torch.cholesky_inverse(L_x), sample_x - mu_x)
    elif util.is_np(joint_mean):
        # Scipy cho_solve does not support batch operation
        cond_mean = mu_y + \
                    np.einsum('...ik,...lk,...lm,...m->...i', L_x_y, L_x,
                                 torch.cholesky_inverse(torch.from_numpy(
                                     L_x)).numpy(),
                              sample_x - mu_x)
    else:
        raise NotImplementedError

    cond_L = L_y

    return cond_mean, cond_L
