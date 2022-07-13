"""
    Utilities for generating media stuff
"""

from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt

from mp_pytorch import util


def fill_between(x: Union[np.ndarray, torch.Tensor],
                 y_mean: Union[np.ndarray, torch.Tensor],
                 y_std: Union[np.ndarray, torch.Tensor],
                 axis=None, std_scale: int = 2, draw_mean: bool = False,
                 alpha=0.2, color='gray'):
    """
    Utilities to draw std plot
    Args:
        x: x value
        y_mean: y mean value
        y_std: standard deviation of y
        axis: figure axis to draw
        std_scale: filling range of [-scale * std, scale * std]
        draw_mean: plot mean curve as well
        alpha: transparency of std plot
        color: color to fill

    Returns:
        None
    """
    x, y_mean, y_std = util.to_nps(x, y_mean, y_std)
    if axis is None:
        axis = plt.gca()
    if draw_mean:
        axis.plot(x, y_mean)
    axis.fill_between(x=x,
                      y1=y_mean - std_scale * y_std,
                      y2=y_mean + std_scale * y_std,
                      alpha=alpha, color=color)
