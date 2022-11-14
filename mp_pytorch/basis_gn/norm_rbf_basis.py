import torch

from mp_pytorch.phase_gn import PhaseGenerator
from .basis_generator import BasisGenerator


class NormalizedRBFBasisGenerator(BasisGenerator):

    def __init__(self,
                 phase_generator: PhaseGenerator,
                 num_basis: int = 10,
                 basis_bandwidth_factor: int = 3,
                 num_basis_outside: int = 0,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu'):
        """
        Constructor of class RBF

        Args:
            phase_generator: phase generator
            num_basis: number of basis function
            basis_bandwidth_factor: basis bandwidth factor
            num_basis_outside: basis function outside the duration
            dtype: torch data type
            device: torch device to run on
        """
        self.basis_bandwidth_factor = basis_bandwidth_factor
        self.num_basis_outside = num_basis_outside

        super(NormalizedRBFBasisGenerator, self).__init__(phase_generator,
                                                          num_basis,
                                                          dtype, device)

        # Compute centers and bandwidth
        # Distance between basis centers
        assert self.phase_generator.tau.nelement() == 1

        if self._num_basis > 1:
            basis_dist = self.phase_generator.tau / (self._num_basis - 2 *
                                                     self.num_basis_outside - 1)

            # RBF centers in time scope
            centers_t = torch.linspace(-self.num_basis_outside * basis_dist
                                       + self.phase_generator.delay,
                                       self.num_basis_outside * basis_dist
                                       + self.phase_generator.tau
                                       + self.phase_generator.delay,
                                       self._num_basis, dtype=self.dtype,
                                       device=self.device)
            # RBF centers in phase scope
            self.centers_p = self.phase_generator.unbound_phase(centers_t)

            tmp_bandwidth = \
                torch.cat((self.centers_p[1:] - self.centers_p[:-1],
                           self.centers_p[-1:] - self.centers_p[-2:-1]), dim=-1)

        elif self._num_basis == 1:
            # RBF centers in time scope
            centers_t = torch.tensor([self.phase_generator.delay
                                      + 0.5 * self.phase_generator.tau],
                                     dtype=self.dtype, device=self.device)
            # RBF centers in phase scope
            self.centers_p = self.phase_generator.unbound_phase(centers_t)
            tmp_bandwidth = torch.tensor([1], dtype=self.dtype,
                                         device=self.device)

        else:
            raise NotImplementedError

        # The Centers should not overlap too much (makes w almost random due
        # to aliasing effect).Empirically chosen
        self.bandwidth = self.basis_bandwidth_factor / (tmp_bandwidth ** 2)

    def basis(self, times: torch.Tensor) -> torch.Tensor:
        """
        Generate values of basis function at given time points
        Args:
            times: times in Tensor

        Returns:
            basis: basis functions in Tensor
        """
        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of basis:
        # [*add_dim, num_times, num_basis]

        # Extract dimension
        num_times = times.shape[-1]

        # Time to phase
        phase = self.phase_generator.phase(times)

        # Add one axis (basis centers) to phase and get shape:
        # [*add_dim, num_times, num_basis]
        phase = phase[..., None]
        phase = phase.expand([*phase.shape[:-1], self._num_basis])

        # Add one axis (times) to centers in phase scope and get shape:
        # [num_times, num_basis]
        centers = self.centers_p[None, :]
        centers = centers.expand([num_times, -1])

        # Basis
        tmp = torch.einsum('...ij,...j->...ij', (phase - centers) ** 2,
                           self.bandwidth)
        basis = torch.exp(-tmp / 2)

        # Normalization
        if self._num_basis > 1:
            sum_basis = torch.sum(basis, dim=-1, keepdim=True)
            basis = basis / sum_basis

        # Return
        return basis


class ZeroPaddingNormalizedRBFBasisGenerator(NormalizedRBFBasisGenerator):
    def __init__(self,
                 phase_generator: PhaseGenerator,
                 num_basis: int = 10,
                 num_basis_zero_start: int = 2,
                 num_basis_zero_goal: int = 0,
                 basis_bandwidth_factor: float = 3,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu'):
        """
        Constructor of class RBF with zero padding basis functions
        Args:
            phase_generator: phase generator
            num_basis: number of basis function
            num_basis_zero_start: number of basis padding in front
            num_basis_zero_goal: number of basis padding afterwards
            basis_bandwidth_factor: basis bandwidth factor
            dtype: data type
            device: device of the data
        """
        self.num_basis_zero_start = num_basis_zero_start
        self.num_basis_zero_goal = num_basis_zero_goal
        super().__init__(phase_generator=phase_generator,
                         num_basis=num_basis + num_basis_zero_start
                                   + num_basis_zero_goal,
                         basis_bandwidth_factor=basis_bandwidth_factor,
                         num_basis_outside=0,
                         dtype=dtype, device=device)

    @property
    def num_basis(self):
        return super().num_basis - self.num_basis_zero_start \
               - self.num_basis_zero_goal
