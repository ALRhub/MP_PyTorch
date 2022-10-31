from typing import Tuple

import torch

from mp_pytorch import util
from mp_pytorch.phase_gn import ExpDecayPhaseGenerator
from .norm_rbf_basis import NormalizedRBFBasisGenerator


class ProDMPBasisGenerator(NormalizedRBFBasisGenerator):
    def __init__(self, phase_generator: ExpDecayPhaseGenerator,
                 num_basis: int = 10,
                 basis_bandwidth_factor: int = 3,
                 num_basis_outside: int = 0,
                 dt: float = 0.01,
                 alpha: float = 25,
                 pre_compute_length_factor=5,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu'):
        """

        Args:
            phase_generator: phase generator
            num_basis: number of basis function
            basis_bandwidth_factor: basis bandwidth factor
            num_basis_outside: basis function outside the duration
            dt: time step
            alpha: alpha value of DMP
            pre_compute_length_factor: (n x tau) time length in pre-computation
            dtype: data type
            device: device of the data
        """
        super(ProDMPBasisGenerator, self).__init__(phase_generator,
                                                   num_basis,
                                                   basis_bandwidth_factor,
                                                   num_basis_outside,
                                                   dtype, device)

        self.alpha = alpha
        self.scaled_dt = dt / self.phase_generator.tau

        assert pre_compute_length_factor <= 5, \
            "For numerical stability, please use a length factor <= 5."
        self.pre_compute_length_factor = pre_compute_length_factor

        self.y_1_value = None
        self.y_2_value = None
        self.dy_1_value = None
        self.dy_2_value = None
        self.pc_pos_basis = None
        self.pc_vel_basis = None

        self.num_basis_g = self.num_basis + 1
        self.auto_basis_scale_factors = None
        self.pre_compute()

    def pre_compute(self):
        """
        Precompute basis functions and other stuff

        Returns: None

        """

        # Shape of pc_scaled_time
        # [num_pc_times]

        # Shape of y_1_value, y_2_value, dy_1_value, dy_2_value:
        # [num_pc_times]
        #
        # Shape of q_1_value, q_2_value:
        # [num_pc_times]
        #
        # Shape of p_1_value, p_2_value:
        # [num_pc_times, num_basis]
        #
        # Shape of pos_basis, vel_basis:
        # [num_pc_times, num_basis_g]
        # Note: num_basis_g = num_basis + 1

        # Pre-compute scaled time steps in [0, 1]
        num_pre_compute = self.pre_compute_length_factor * \
                          torch.round(1 / self.scaled_dt).long().item() + 1
        pc_scaled_times = torch.linspace(0, self.pre_compute_length_factor,
                                         num_pre_compute, dtype=self.dtype,
                                         device=self.device)

        # y1 and y2
        self.y_1_value = torch.exp(-0.5 * self.alpha * pc_scaled_times)
        self.y_2_value = pc_scaled_times * self.y_1_value

        self.dy_1_value = -0.5 * self.alpha * self.y_1_value
        self.dy_2_value = -0.5 * self.alpha * self.y_2_value + self.y_1_value

        # q_1 and q_2
        q_1_value = \
            (0.5 * self.alpha * pc_scaled_times - 1) \
            * torch.exp(0.5 * self.alpha * pc_scaled_times) + 1
        q_2_value = \
            0.5 * self.alpha \
            * (torch.exp(0.5 * self.alpha * pc_scaled_times) - 1)

        # Get basis of one DOF, shape [num_pc_times, num_basis]
        pc_times = self.phase_generator.linear_phase_to_time(pc_scaled_times)

        basis_single_dof = super().basis(pc_times)
        assert list(basis_single_dof.shape) == [*pc_times.shape,
                                                self.num_basis]

        # Get canonical phase x, shape [num_pc_times]
        canonical_x = self.phase_generator.phase(pc_times)
        assert list(canonical_x.shape) == [*pc_times.shape]

        # p_1 and p_2
        dp_1_value = \
            torch.einsum('...i,...i,...ij->...ij',
                         pc_scaled_times
                         * torch.exp(self.alpha * pc_scaled_times / 2),
                         canonical_x,
                         basis_single_dof)
        dp_2_value = \
            torch.einsum('...i,...i,...ij->...ij',
                         torch.exp(self.alpha * pc_scaled_times / 2),
                         canonical_x,
                         basis_single_dof)

        p_1_value = torch.zeros(size=dp_1_value.shape, dtype=self.dtype,
                                device=self.device)
        p_2_value = torch.zeros(size=dp_2_value.shape, dtype=self.dtype,
                                device=self.device)

        for i in range(pc_scaled_times.shape[0]):
            p_1_value[i] = torch.trapz(dp_1_value[:i + 1],
                                       pc_scaled_times[:i + 1], dim=0)
            p_2_value[i] = torch.trapz(dp_2_value[:i + 1],
                                       pc_scaled_times[:i + 1], dim=0)

        # Compute integral form basis values
        pos_basis_w = p_2_value * self.y_2_value[:, None] \
                      - p_1_value * self.y_1_value[:, None]
        pos_basis_g = q_2_value * self.y_2_value \
                      - q_1_value * self.y_1_value
        vel_basis_w = p_2_value * self.dy_2_value[:, None] \
                      - p_1_value * self.dy_1_value[:, None]
        vel_basis_g = q_2_value * self.dy_2_value \
                      - q_1_value * self.dy_1_value

        # Pre-computed pos and vel basis
        self.pc_pos_basis = \
            torch.cat([pos_basis_w, pos_basis_g[:, None]], dim=-1)
        self.pc_vel_basis = \
            torch.cat([vel_basis_w, vel_basis_g[:, None]], dim=-1)

        self.auto_compute_basis_scale_factors()

    def auto_compute_basis_scale_factors(self):
        """
        Compute scale factors for each basis function
        :return: None
        """
        assert self.pc_pos_basis is not None, "Pos basis is not pre-computed."
        self.auto_basis_scale_factors = 1. / self.pc_pos_basis.max(axis=0).values

    def times_to_indices(self, times: torch.Tensor, round_int: bool = True):
        """
        Map time points to pre-compute indices
        Args:
            times: time points
            round_int: if indices should be rounded to the closest integer

        Returns:
            time indices
        """
        # times to scaled times

        scaled_times = self.phase_generator.left_bound_linear_phase(times)
        if scaled_times.max() > self.pre_compute_length_factor:
            raise RuntimeError("Time is beyond the pre-computation range. "
                               "Set larger pre-computation factor")
        indices = scaled_times / self.scaled_dt
        if round_int:
            indices = torch.round(indices).long()

        return indices

    def basis(self, times: torch.Tensor):
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
        # [*add_dim, num_times, num_basis_g]
        time_indices = self.times_to_indices(times, False)
        basis = util.indexing_interpolate(data=self.pc_pos_basis,
                                          indices=time_indices)
        return basis

    def vel_basis(self, times: torch.Tensor):
        """
        Generate values of velocity basis function at given time points
        Args:
            times: times in Tensor

        Returns:
            vel_basis: velocity basis functions in Tensor
        """
        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of vel_basis:
        # [*add_dim, num_times, num_basis_g]

        time_indices = self.times_to_indices(times, False)

        vel_basis = util.indexing_interpolate(data=self.pc_vel_basis,
                                              indices=time_indices)
        return vel_basis

    def basis_multi_dofs(self, times: torch.Tensor, num_dof: int):
        """
        Generate blocked-diagonal multiple dof basis matrix

        Args:
            times: time points
            num_dof: num of dof

        Returns:
            pos_basis_multi_dofs
        """
        # Shape of time
        # [*add_dim, num_times]
        #
        # Shape of pos_basis_multi_dofs
        # [*add_dim, num_dof * num_times, num_dof * num_basis_g]

        # Here the super class will take the last dimension of a single basis
        # matrix as num_basis, so no worries for the extra goal basis term
        pos_basis_multi_dofs = super().basis_multi_dofs(times, num_dof)
        return pos_basis_multi_dofs

    def vel_basis_multi_dofs(self, times: torch.Tensor, num_dof: int):
        """
        Generate blocked-diagonal multiple dof velocity basis matrix

        Args:
            times: times in Tensor
            num_dof: num of Degree of freedoms

        Returns:
            vel_basis_multi_dofs: Multiple DoFs velocity basis functions

        """
        # Shape of time
        # [*add_dim, num_times]
        #
        # Shape of vel_basis_multi_dofs
        # [*add_dim, num_dof * num_times, num_dof * num_basis_g]

        # Extract additional dimensions
        add_dim = list(times.shape[:-1])

        # Get single basis, shape: [*add_dim, num_times, num_basis_g]
        vel_basis_single_dof = self.vel_basis(times)
        num_times = vel_basis_single_dof.shape[-2]

        # Multiple Dofs, shape:
        # [*add_dim, num_times, num_dof, num_dof * num_basis]
        vel_basis_multi_dofs = torch.zeros(*add_dim,
                                           num_dof * num_times,
                                           num_dof * self.num_basis_g,
                                           dtype=self.dtype, device=self.device)
        # Assemble
        for i in range(num_dof):
            row_indices = slice(i * num_times,
                                (i + 1) * num_times)
            col_indices = slice(i * self.num_basis_g,
                                (i + 1) * self.num_basis_g)
            vel_basis_multi_dofs[..., row_indices, col_indices] = \
                vel_basis_single_dof

        # Return
        return vel_basis_multi_dofs

    def general_solution_values(self, times: torch.Tensor):
        """
        Get values of general solution functions and their derivatives

        Args:
            times: time points

        Returns:
            values of y1, y2, dy1, dy2 at given time steps
        """
        # Shape of times
        # [*add_dim, num_times]
        #
        # Shape of each return
        # [*add_dim, num_times]

        time_indices = self.times_to_indices(times, False)

        y_1_value = util.indexing_interpolate(data=self.y_1_value,
                                              indices=time_indices)
        y_2_value = util.indexing_interpolate(data=self.y_2_value,
                                              indices=time_indices)
        dy_1_value = util.indexing_interpolate(data=self.dy_1_value,
                                               indices=time_indices)
        dy_2_value = util.indexing_interpolate(data=self.dy_2_value,
                                               indices=time_indices)

        return y_1_value, y_2_value, dy_1_value, dy_2_value

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
            fig, axes = plt.subplots(1, 2, sharex=True, squeeze=False)
            for i in range(basis_values.shape[-1] - 1):
                axes[0, 0].plot(times, basis_values[:, i], label=f"w_basis_{i}")
            axes[0, 0].grid()
            axes[0, 0].legend()
            axes[0, 0].axvline(x=delay, linestyle='--', color='k', alpha=0.3)
            axes[0, 0].axvline(x=delay + tau, linestyle='--', color='k',
                               alpha=0.3)

            axes[0, 1].plot(times, basis_values[:, -1], label=f"goal_basis")
            axes[0, 1].grid()
            axes[0, 1].legend()
            axes[0, 1].axvline(x=delay, linestyle='--', color='k', alpha=0.3)
            axes[0, 1].axvline(x=delay + tau, linestyle='--', color='k',
                               alpha=0.3)

            plt.show()
        return times, basis_values

    def get_basis_scale_factors(self):
        """
        Compute the scale factors of all basis functions, so that their
        magnitudes are all equal to 1

        Returns:
            auto_basis_scale_factors: scale factors
        """
        assert self.auto_basis_scale_factors is not None, "Basis scale factors is not computed."
        return self.auto_basis_scale_factors
