"""
    Utilities of data type and structure
"""
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import numpy as np
import torch


def make_iterable(data: any, default: Literal['tuple', 'list'] = 'tuple') \
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
        if tensor.device.type == "cpu":
            return tensor.numpy()
        elif tensor.device.type == "cuda":
            return tensor.cpu().numpy()
    raise NotImplementedError


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


def is_ts(data: any) -> bool:
    """
    is data a torch Tensor?
    """
    return isinstance(data, torch.Tensor)
