"""
    Utilities of matrix operation
"""
from typing import Union, Optional
import numpy as np
import torch


def build_lower_matrix(param_diag: torch.Tensor,
                       param_off_diag: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Compose the lower triangular matrix L from diag and off-diag elements
    It seems like faster than using the cholesky transformation from PyTorch
    Args:
        param_diag: diagonal parameters
        param_off_diag: off-diagonal parameters

    Returns:
        Lower triangular matrix L

    """
    dim_pred = param_diag.shape[-1]
    # Fill diagonal terms
    L = param_diag.diag_embed()
    if param_off_diag is not None:
        # Fill off-diagonal terms
        [row, col] = torch.tril_indices(dim_pred, dim_pred, -1)
        L[..., row, col] = param_off_diag[..., :]

    return L


def transform_to_cholesky(mat: torch.Tensor) -> torch.Tensor:
    """
    Transform an unconstrained matrix to cholesky, will abandon half of the data
    Args:
        mat: an unconstrained square matrix

    Returns:
        lower triangle matrix as Cholesky
    """
    lct = torch.distributions.transforms.LowerCholeskyTransform(cache_size=0)
    return lct(mat)


def add_expand_dim(data: Union[torch.Tensor, np.ndarray],
                   add_dim_indices: [int],
                   add_dim_sizes: [int]) -> Union[torch.Tensor, np.ndarray]:
    """
    Add additional dimensions to tensor and expand accordingly
    Args:
        data: tensor to be operated. Torch.Tensor or numpy.ndarray
        add_dim_indices: the indices of added dimensions in the result tensor
        add_dim_sizes: the expanding size of the additional dimensions

    Returns:
        result: result tensor after adding and expanding
    """
    num_data_dim = data.ndim
    num_dim_to_add = len(add_dim_indices)

    add_dim_reverse_indices = [num_data_dim + num_dim_to_add + idx
                               for idx in add_dim_indices]

    str_add_dim = ""
    str_expand = ""
    add_dim_index = 0
    for dim in range(num_data_dim + num_dim_to_add):
        if dim in add_dim_indices or dim in add_dim_reverse_indices:
            str_add_dim += "None, "
            str_expand += str(add_dim_sizes[add_dim_index]) + ", "
            add_dim_index += 1
        else:
            str_add_dim += ":, "
            if type(data) == torch.Tensor:
                str_expand += "-1, "
            elif type(data) == np.ndarray:
                str_expand += "1, "
            else:
                raise NotImplementedError

    str_add_dime_eval = "data[" + str_add_dim + "]"
    if type(data) == torch.Tensor:
        return eval("eval(str_add_dime_eval).expand(" + str_expand + ")")
    else:
        return eval("np.tile(eval(str_add_dime_eval),[" + str_expand + "])")
