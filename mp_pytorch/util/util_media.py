"""
    Utilities for generating media stuff
"""

from typing import Union, Literal, List

import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt
import mp_pytorch.util as util


def savefig(figs: Union[plt.Figure, List[plt.Figure]], media_name,
            fmt=Literal['pdf', 'png', 'jpeg'], dpi=200, overwrite=False):
    """

    Args:
        figs: figure object or a list of figures
        media_name: name of the media
        fmt: format of the figures
        dpi: resolution
        overwrite: if overwrite when old exists

    Returns:
        None

    """
    path = util.get_media_dir(media_name)
    util.mkdir(path, overwrite=overwrite)

    figs = util.make_iterable(figs)

    for i, fig in enumerate(figs):
        fig_path = util.join_path(path, str(i) + '.' + fmt)
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")


def from_figures_to_video(figure_list: [], video_name: str,
                          interval: int = 2000, overwrite=False) -> str:
    """
    Generate and save a video given a list of figures
    Args:
        figure_list: list of matplotlib figure objects
        video_name: name of video
        interval: interval between two figures in [ms]
        overwrite: if overwrite when old exists
    Returns:
        path to the saved video
    """
    figure, ax = plt.subplots()
    figure.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    frames = []

    video_path = util.get_media_dir(video_name)
    util.mkdir(video_path, overwrite)
    for i, fig in enumerate(figure_list):
        fig.savefig(util.join_path(video_path, "{}.png".format(i)), dpi=300,
                    bbox_inches="tight")

    for j in range(len(figure_list)):
        image = plt.imread(util.join_path(video_path, "{}.png".format(j)))
        img = plt.imshow(image, animated=True)
        plt.axis('off')
        plt.gca().set_axis_off()

        frames.append([img])

    ani = animation.ArtistAnimation(figure, frames, interval=interval,
                                    blit=True,
                                    repeat=False)
    save_path = util.join_path(video_path, video_name + '.mp4')
    ani.save(save_path, dpi=300)

    return save_path


def save_selected_subplots():
    # todo
    pass


def fill_between(x: Union[np.ndarray, torch.Tensor],
                 y_mean: Union[np.ndarray, torch.Tensor],
                 y_std: Union[np.ndarray, torch.Tensor],
                 std_scale: int = 2, draw_mean: bool = False,
                 alpha=0.2, color='gray'):
    """
    Utilities to draw std plot
    Args:
        x: x value
        y_mean: y mean value
        y_std: standard deviation of y
        std_scale: filling range of [-scale * std, scale * std]
        draw_mean: plot mean curve as well
        alpha: transparency of std plot
        color: color to fill

    Returns:
        None
    """
    x, y_mean, y_std = util.to_nps(x, y_mean, y_std)

    axis = plt.gca()
    if draw_mean:
        axis.plot(x, y_mean, color=color)
    axis.fill_between(x=x,
                      y1=y_mean - std_scale * y_std,
                      y2=y_mean + std_scale * y_std,
                      alpha=alpha, color=color)
