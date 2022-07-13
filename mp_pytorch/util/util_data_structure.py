"""
    Utilities of data type and structure
"""
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch


def make_iterable(data: any, default: str = 'tuple') \
        -> Union[Tuple, List]:
    """
    Make data a tuple or list, i.e. (data) or [data]
    Args:
        data: some data
        default: default type
    Returns:
        (data) if it is not a tuple
    """
    if isinstance(data, tuple):
        return data
    elif isinstance(data, list):
        return data
    else:
        if default == 'tuple':
            return (data,)  # Do not use tuple()
        elif default == 'list':
            return [data, ]
        else:
            raise NotImplementedError


def to_np(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Transfer any type and device of tensor to a numpy ndarray
    Args:
        tensor: np.ndarray, cpu tensor or gpu tensor

    Returns:
        tensor in np.ndarray
    """
    if is_np(tensor):
        return tensor
    elif is_ts(tensor):
        return tensor.detach().cpu().numpy()
    else:
        np.array(tensor)


def to_nps(*tensors: [Union[np.ndarray, torch.Tensor]]) -> [np.ndarray]:
    """
    transfer a list of any type of tensors to np.ndarray
    Args:
        tensors: a list of tensors

    Returns:
        a list of np.ndarray
    """
    return [to_np(tensor) for tensor in tensors]


def is_np(data: any) -> bool:
    """
    is data a numpy array?
    """
    return isinstance(data, np.ndarray)


def to_ts(data: Union[int, float, np.ndarray, torch.Tensor],
          dtype: torch.dtype = torch.float32,
          device: str = "cpu") -> torch.Tensor:
    """
    Transfer any numerical input to a torch tensor in default data type + device

    Args:
        device: device of the tensor, default: cpu
        dtype: data type of tensor, float 32 or float 64 (double)
        data: float, np.ndarray, torch.Tensor

    Returns:
        tensor in torch.Tensor
    """

    return torch.as_tensor(data, dtype=dtype, device=device)


def to_tss(*datas: [Union[int, float, np.ndarray, torch.Tensor]],
           dtype: torch.dtype = torch.float32,
           device: str = "cpu") \
        -> [torch.Tensor]:
    """
    transfer a list of any type of numerical input to a list of tensors in given
    data type and device

    Args:
        datas: a list of data
        dtype: data type of tensor, float 32 or float 64 (double)
        device: device of the tensor, default: cpu

    Returns:
        a list of np.ndarray
    """
    return [to_ts(data, dtype, device) for data in datas]


def is_ts(data: any) -> bool:
    """
    is data a torch Tensor?
    """
    return isinstance(data, torch.Tensor)
