"""
    Utilities for debugging
"""

import time
from typing import Callable
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from mp_pytorch import util


def how_fast(repeat: int, func: Callable, *args, **kwargs):
    """
    Test how fast a given function call is
    Args:
        repeat: number of times to run the function
        func: function to be tested
        *args: list of arguments used in the function call
        **kwargs: dict of arguments used in the function call

    Returns:
        avg duration function call

    Raise:
        any type of exception when test the function call
    """
    run_time_test(lock=True)
    try:
        for i in range(repeat):
            func(*args, **kwargs)
        duration = run_time_test(lock=False)
        if duration is not None:
            print(f"total_time of {repeat} runs: {duration} s")
        print(f"avg_time of each run: {duration / repeat} s")
        return duration / repeat
    except RuntimeError:
        raise
    except Exception:
        raise


def run_time_test(lock: bool) -> Optional[float]:
    """
    A manual running time computing function. It will print the running time
    for every second call

    E.g.:
    run_time_test(lock=True)
    some_func1()
    some_func2()
    ...
    run_time_test(lock=False)

    Args:
        lock: flag indicating if time counter starts

    Returns:
        None (every first call) or duration (every second call)

    Raise:
        RuntimeError if is used in a wrong way
    """
    # Initialize function attribute
    if not hasattr(run_time_test, "lock_state"):
        run_time_test.lock_state = False
        run_time_test.last_run_time = time.time()
        run_time_test.duration_list = list()

    # Check correct usage
    if run_time_test.lock_state == lock:
        run_time_test.lock_state = False
        raise RuntimeError("run_time_test is wrongly used.")

    # Setup lock
    run_time_test.lock_state = lock

    # Update time
    if lock is False:
        duration = time.time() - run_time_test.last_run_time
        run_time_test.duration_list.append(duration)
        run_time_test.last_run_time = time.time()
        print("duration", duration)
        return duration
    else:
        run_time_test.last_run_time = time.time()
        return None


def debug_plot(x: Union[np.ndarray, torch.Tensor],
               y: [], labels: [] = None, title="debug_plot", grid=True) -> \
        plt.Figure:
    """
    One line to plot some variable for debugging, numpy + torch
    Args:
        x: data used for x-axis, can be None
        y: list of data used for y-axis
        labels: labels in plots
        title: title of current plot
        grid: show grid or not

    Returns:
        None
    """
    fig = plt.figure()
    y = util.make_iterable(y)
    if labels is not None:
        labels = util.make_iterable(labels)

    for i, yi in enumerate(y):
        yi = util.to_np(yi)
        label = labels[i] if labels is not None else None
        if x is not None:
            x = util.to_np(x)
            plt.plot(x, yi, label=label)
        else:
            plt.plot(yi, label=label)

    plt.title(title)
    if labels is not None:
        plt.legend()
    if grid:
        plt.grid(alpha=0.5)
    plt.show()
    return fig
