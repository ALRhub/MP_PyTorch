# TODO: some things still missing
from typing import Tuple

import numpy as np

from mp_pytorch import BasisGenerator
from mp_pytorch import PhaseGenerator


class RhythmicBasisGenerator(BasisGenerator):

    def __init__(
            self, phase_generator: PhaseGenerator, n_basis: int = 5,
            duration: float = 1,
            basis_bandwidth_factor: float = 3
    ):
        BasisGenerator.__init__(self, phase_generator, n_basis)

        self.num_bandwidth_factor = basis_bandwidth_factor
        self.centers = np.linspace(0, 1, self.n_basis)

        tmp_bandwidth = np.hstack((self.centers[1:] - self.centers[0:-1],
                                   self.centers[-1] - self.centers[- 2]))

        # The Centers should not overlap too much (makes w almost random due to aliasing effect).Empirically chosen
        self.bandwidth = self.num_bandwidth_factor / (tmp_bandwidth ** 2)

    def basis_and_phase(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        phase = self.getInputTensorIndex(0)

        diff = phase - self.centers
        diff_cos = np.array([np.cos(diff * self.bandwidth * 2 * np.pi)])
        basis = np.exp(diff_cos)

        sum_b = np.sum(basis, axis=1)
        basis = [column / sum_b for column in basis.transpose()]
        return np.array(basis).transpose(), phase
