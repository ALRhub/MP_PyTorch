import numpy as np
from scipy.interpolate import make_interp_spline as spi_make_interp_spline

from mp_pytorch import PhaseGenerator


# TODO: Adjust to mp_pytorch lib
class SmoothPhaseGenerator(PhaseGenerator):

    def __init__(self, duration: float = 1):
        self.left = [(1, 0.0), (2, 0.0)]
        self.right = [(1, 0.0), (2, 0.0)]

    def phase(self, t: np.ndarray, duration: float):
        spline = spi_make_interp_spline([0, duration], [0, 1],
                                        bc_type=(self.left, self.right), k=5)
        return spline(t)
