"""
@brief:     Basis generators in PyTorch
"""
from typing import Tuple

from mp_pytorch.phase_gn.phase_generator import *


class BasisGenerator(ABC):
    @abstractmethod
    def __init__(self,
                 phase_generator: PhaseGenerator,
                 num_basis: int = 10,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 ):
        """
        Constructor for basis class
        Args:
            phase_generator: phase generator
            num_basis: number of basis functions
            dtype: torch data type
            device: torch device to run on
        """
        self.dtype = dtype
        self.device = device

        # Internal number of basis
        self._num_basis = num_basis
        self.phase_generator = phase_generator

        # Flag of finalized basis generator
        self.is_finalized = False

    @property
    def num_basis(self) -> int:
        """
        Returns: the number of basis with learnable weights
        """
        return self._num_basis

    @property
    def _num_local_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        return 0

    @property
    def num_params(self) -> int:
        """
        Returns: number of parameters of current class plus parameters of all
        attributes
        """
        return self._num_local_params + self.phase_generator.num_params

    def set_params(self,
                   params: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Set parameters of current object and attributes
        Args:
            params: parameters to be set

        Returns:
            None
        """
        params = torch.as_tensor(params, dtype=self.dtype, device=self.device)
        remaining_params = self.phase_generator.set_params(params)
        self.finalize()
        return remaining_params

    def get_params(self) -> torch.Tensor:
        """
        Return all learnable parameters
        Returns:
            parameters
        """
        # Shape of params
        # [*add_dim, num_params]
        params = self.phase_generator.get_params()
        return params

    def get_params_bounds(self) -> torch.Tensor:
        """
        Return all learnable parameters' bounds
        Returns:
            parameters bounds
        """
        # Shape of params_bounds
        # [2, num_params]

        params_bounds = self.phase_generator.get_params_bounds()
        return params_bounds

    @abstractmethod
    def basis(self, times: torch.Tensor) -> torch.Tensor:
        """
        Interface to generate value of single basis function at given time
        points
        Args:
            times: times in Tensor

        Returns:
            basis functions in Tensor

        """
        pass

    def basis_multi_dofs(self,
                         times: torch.Tensor,
                         num_dof: int) -> torch.Tensor:
        """
        Interface to generate value of single basis function at given time
        points
        Args:
            times: times in Tensor
            num_dof: num of Degree of freedoms
        Returns:
            basis_multi_dofs: Multiple DoFs basis functions in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]
        #
        # Shape of basis_multi_dofs
        # [*add_dim, num_dof * num_times, num_dof * num_basis]

        # Extract additional dimensions
        add_dim = list(times.shape[:-1])

        # Get single basis, shape: [*add_dim, num_times, num_basis]
        basis_single_dof = self.basis(times)
        num_times = basis_single_dof.shape[-2]
        num_basis = basis_single_dof.shape[-1]

        # Multiple Dofs, shape:
        # [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dofs = torch.zeros(*add_dim, num_dof * num_times,
                                       num_dof * num_basis, dtype=self.dtype,
                                       device=self.device)
        # Assemble
        for i in range(num_dof):
            row_indices = slice(i * num_times, (i + 1) * num_times)
            col_indices = slice(i * num_basis, (i + 1) * num_basis)
            basis_multi_dofs[..., row_indices, col_indices] = basis_single_dof

        # Return
        return basis_multi_dofs

    def finalize(self):
        """
        Mark the basis generator as finalized so that the parameters cannot be
        updated any more
        Returns: None

        """
        self.is_finalized = True

    def reset(self):
        """
        Unmark the finalization
        Returns: None

        """
        self.phase_generator.reset()
        self.is_finalized = False

    def show_basis(self, plot=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute basis function values for debug usage
        The times are in the range of [delay - tau, delay + 2 * tau]

        Returns: basis function values

        """
        tau = self.phase_generator.tau
        delay = self.phase_generator.delay
        assert tau.ndim == 0 and delay.ndim == 0
        times = torch.linspace(delay - tau, delay + 2 * tau, steps=1000)
        basis_values = self.basis(times)
        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            for i in range(basis_values.shape[-1]):
                plt.plot(times, basis_values[:, i], label=f"basis_{i}")
            plt.grid()
            plt.legend()
            plt.axvline(x=delay, linestyle='--', color='k', alpha=0.3)
            plt.axvline(x=delay + tau, linestyle='--', color='k', alpha=0.3)
            plt.show()
        return times, basis_values
