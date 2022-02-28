"""
    Utilities of numerical computation
"""
from typing import Union, Optional
import torch
import numpy as np
import mp_pytorch.util as util


def to_log_space(data: Union[np.ndarray, torch.Tensor],
                 lower_bound: Optional[float]) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    project data to log space

    Args:
        data: original data
        lower_bound: customized lower bound in runtime, will override the
                     default value

    Returns: log(data + lower_bound)

    """
    # Determine lower bound, runtime? config? default?
    actual_lower_bound = util.decide_hyperparameter(to_log_space, lower_bound,
                                            "log_lower_bound", 1e-8)
    # Compute
    assert data.min() >= 0
    if type(data) == np.ndarray:
        log_data = np.log(data + actual_lower_bound)
    elif type(data) == torch.Tensor:
        log_data = torch.log(data + actual_lower_bound)
    else:
        raise NotImplementedError
    return log_data


def to_softplus_space(data: Union[np.ndarray, torch.Tensor],
                      lower_bound: Optional[float]) -> \
        Union[np.ndarray, torch.Tensor]:
    """
    Project data to exp space

    Args:
        data: original data
        lower_bound: runtime lower bound of the result

    Returns: softplus(data) + lower_bound

    """
    # todo, should we use a fixed lower bound or adaptive to the values?
    # Determine lower bound, runtime? config? default?
    actual_lower_bound = \
        util.decide_hyperparameter(to_softplus_space, lower_bound,
                           "softplus_lower_bound", 1e-2)
    # Compute
    softplus = torch.nn.Softplus()
    sp_result = softplus(data) + actual_lower_bound
    return sp_result


def interpolate(x_ori: np.ndarray, y_ori: np.ndarray,
                num_tar: int) -> np.ndarray:
    """
    Interpolates trajectories to desired length and data density

    Args:
        x_ori: original data time, shape [num_x]
        y_ori: original data value, shape [num_x, dim_y]
        num_tar: number of target sequence points

    Returns:
        interpolated y data, [num_tar, dim_y]
    """

    # Setup interpolation scale
    start, stop = x_ori[0], x_ori[-1]
    x_tar = np.linspace(start, stop, num_tar)

    # check y dim
    if y_ori.ndim == 1:
        y_tar = np.interp(x_tar, x_ori, y_ori)
    else:
        # Initialize result array as shape
        y_tar = np.zeros((num_tar, y_ori.shape[1]))

        # Loop over y's dim
        for k in range(y_ori.shape[1]):
            y_tar[:, k] = np.interp(x_tar, x_ori, y_ori[:, k])

    return y_tar
