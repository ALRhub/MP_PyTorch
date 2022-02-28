"""
    Utilities of data type and structure
"""
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import numpy as np
import torch

import mp_pytorch.util as util


def current_device():
    """
    Return current torch default device

    Returns: "cpu" or "gpu"

    """
    if not hasattr(current_device, "device"):
        return "cpu"
    else:
        return current_device.device


def use_cpu():
    """
    Switch to cpu tensor
    Returns:
        None
    """
    torch.set_default_tensor_type('torch.FloatTensor')
    current_device.device = "cpu"


def use_cuda() -> bool:
    """
    Check if GPU is available and set default torch datatype

    Returns:
        None
    """
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        current_device.device = "cuda"
        # torch.multiprocessing.set_start_method(method="spawn")

        return True
    else:
        current_device.device = "cpu"
        return False


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


def from_string_to_array(s: str) -> np.ndarray:
    """
    Convert string in Pandas DataFrame cell to numpy array
    Args:
        s: string, e.g. "[1.0   2.3   4.5 \n 5.3   5.6]"

    Returns:
        1D numpy array
    """
    return np.asarray(s[1:-1].split(),
                      dtype=np.float64)


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


def to_tensor_dict(np_dict: dict) -> dict:
    """
    Transform a nested dict of np.ndarray into a dict of torch tensor
    The default tensor device and type shall be used

    Args:
        np_dict: np dict

    Returns:
        ts_dict: torch dict
    """
    ts_dict = dict()

    for name, data in np_dict.items():
        if isinstance(data, dict):
            ts_dict[name] = to_tensor_dict(data)
        elif is_np(data) or isinstance(data, (list, tuple)):
            ts_dict[name] = torch.Tensor(data)
        else:
            raise NotImplementedError
    return ts_dict


def to_numpy_dict(ts_dict: dict) -> dict:
    """
    Transform a nested dict of torch tensor into a dict of np.ndarray

    Args:
        ts_dict: torch dict

    Returns:
        np_dict: np dict
    """
    np_dict = dict()

    for name, data in ts_dict.items():
        if isinstance(data, dict):
            np_dict[name] = to_numpy_dict(data)
        elif is_ts(data) or isinstance(data, (list, tuple)):
            np_dict[name] = util.to_np(torch.Tensor(data))
        else:
            raise NotImplementedError
    return np_dict


def conv2d_size_out(size: int, kernel_size: int = 5, stride=1) -> int:
    """
    Get output size of cnn

    Args:
        size: size of input image
        kernel_size: kernel size
        stride: stride

    Returns:
        output size
    """
    return (size - (kernel_size - 1) - 1) // stride + 1


def maxpool2d_size_out(size: int, kernel_size: int = 2, stride=None) -> int:
    """
    Get output size of max-pooling

    Args:
        size: size of input image
        kernel_size: kernel size
        stride: stride

    Returns:
        output size
    """
    if stride is None:
        stride = kernel_size
    return conv2d_size_out(size, kernel_size=kernel_size, stride=stride)


def image_output_size(size: int,
                      num_cnn: int,
                      cnn_kernel_size: int = 5,
                      cnn_stride: int = 1,
                      max_pool: bool = True,
                      maxpool_kernel_size: int = 2,
                      max_pool_stride: int = None):
    """
    Get output size of multiple cnn-maxpool layers
    Args:
        size:  size of input image
        num_cnn: number of cnns
        cnn_kernel_size
        cnn_stride
        max_pool
        maxpool_kernel_size
        max_pool_stride

    Returns:

    """
    for _ in range(num_cnn):
        size = conv2d_size_out(size, cnn_kernel_size, cnn_stride)
        if max_pool:
            size = maxpool2d_size_out(size, maxpool_kernel_size,
                                      max_pool_stride)

    return size
