"""
@brief:     Basis generators in PyTorch
"""

from mp_pytorch.phase_generator import *


class BasisGenerator(ABC):
    @abstractmethod
    def __init__(self,
                 phase_generator: PhaseGenerator,
                 num_basis: int = 10):
        """
        Constructor for basis class
        Args:
            phase_generator: phase generator
            num_basis: number of basis functions
        """
        self.num_basis = num_basis
        self.phase_generator = phase_generator

    @property
    def tau(self):
        """
        time scaling factor
        Returns:
            scaling factor
        """
        return self.phase_generator.tau

    @property
    def wait(self):
        """
        wait time factor
        Returns:
            waiting time
        """
        return self.phase_generator.wait

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

    def set_params(self, params: torch.Tensor) -> torch.Tensor:
        """
        Set parameters of current object and attributes
        Args:
            params: parameters to be set

        Returns:
            None
        """

        remaining_params = self.phase_generator.set_params(params)
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
        # [*add_dim, num_times, num_dof, num_dof * num_basis]
        basis_multi_dofs = torch.zeros(*add_dim,
                                       num_dof * num_times,
                                       num_dof * num_basis)
        # Assemble
        for i in range(num_dof):
            row_indices = slice(i * num_times,
                                (i + 1) * num_times)
            col_indices = slice(i * num_basis,
                                (i + 1) * num_basis)
            basis_multi_dofs[..., row_indices, col_indices] = basis_single_dof

        # Return
        return basis_multi_dofs


class NormalizedRBFBasisGenerator(BasisGenerator):

    def __init__(self,
                 phase_generator: PhaseGenerator,
                 num_basis: int = 10,
                 basis_bandwidth_factor: int = 3,
                 num_basis_outside: int = 0):
        """
        Constructor of class RBF

        Args:
            phase_generator: phase generator
            num_basis: number of basis function
            basis_bandwidth_factor: basis bandwidth factor
            num_basis_outside: basis function outside the duration
        """
        self.basis_bandwidth_factor = basis_bandwidth_factor
        self.num_basis_outside = num_basis_outside

        super(NormalizedRBFBasisGenerator, self).__init__(phase_generator,
                                                          num_basis)

        # Compute centers and bandwidth
        # Distance between basis centers
        assert self.tau.nelement() == 1
        basis_dist = \
            self.tau / (self.num_basis - 2 * self.num_basis_outside - 1)

        # RBF centers in time scope
        centers_t = \
            torch.linspace(-self.num_basis_outside * basis_dist,
                           self.tau + self.num_basis_outside * basis_dist,
                           self.num_basis)

        # RBF centers in phase scope
        self.centers_p = self.phase_generator.unbound_phase(centers_t)

        tmp_bandwidth = torch.cat((self.centers_p[1:] - self.centers_p[:-1],
                                   self.centers_p[-1:] - self.centers_p[-2:-1]),
                                  dim=-1)

        # The Centers should not overlap too much (makes w almost random due
        # to aliasing effect).Empirically chosen
        self.bandWidth = self.basis_bandwidth_factor / (tmp_bandwidth ** 2)

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
        phase = phase.expand([*phase.shape[:-1], self.num_basis])

        # Add one axis (times) to centers in phase scope and get shape:
        # [num_times, num_basis]
        centers = self.centers_p[None, :]
        centers = centers.expand([num_times, -1])

        # Basis
        tmp = torch.einsum('...ij,...j->...ij', (phase - centers) ** 2,
                           self.bandWidth)
        basis = torch.exp(-tmp / 2)

        # Normalization
        sum_basis = torch.sum(basis, dim=-1, keepdim=True)
        basis = basis / sum_basis

        # Return
        return basis


class DMPBasisGenerator(NormalizedRBFBasisGenerator):
    def __init__(self,
                 phase_generator: PhaseGenerator,
                 num_basis: int = 10,
                 basis_bandwidth_factor: int = 3,
                 num_basis_outside: int = 0):
        """
        Constructor of class DMPBasisGenerator

        Args:
            phase_generator: phase generator
            num_basis: number of basis function
            basis_bandwidth_factor: basis bandwidth factor
            num_basis_outside: basis function outside the duration
        """
        super(DMPBasisGenerator, self).__init__(phase_generator,
                                                num_basis,
                                                basis_bandwidth_factor,
                                                num_basis_outside)

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
        phase = self.phase_generator.phase(times)
        rbf_basis = super(DMPBasisGenerator, self).basis(times)

        # Einsum shape: [*add_dim, num_times]
        #               [*add_dim, num_times, num_basis]
        #            -> [*add_dim, num_times, num_basis]
        dmp_basis = torch.einsum('...i,...ij->...ij', phase, rbf_basis)
        return dmp_basis


class IDMPBasisGenerator(DMPBasisGenerator):
    def __init__(self, phase_generator: ExpDecayPhaseGenerator,
                 num_basis: int = 10,
                 basis_bandwidth_factor: int = 3,
                 num_basis_outside: int = 0,
                 dt: float = 0.01,
                 alpha: float = 25, ):
        """

        Args:
            phase_generator: phase generator
            num_basis: number of basis function
            basis_bandwidth_factor: basis bandwidth factor
            num_basis_outside: basis function outside the duration
            dt: time step
            alpha: alpha value of DMP
        """
        super(IDMPBasisGenerator, self).__init__(phase_generator,
                                                 num_basis,
                                                 basis_bandwidth_factor,
                                                 num_basis_outside)

        self.alpha = alpha
        self.scaled_dt = dt / self.phase_generator.tau
        self.y_1_value = None
        self.y_2_value = None
        self.dy_1_value = None
        self.dy_2_value = None
        self.pc_pos_basis = None
        self.pc_vel_basis = None

        self.num_basis_g = self.num_basis + 1
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
        num_pre_compute = torch.round(1 / self.scaled_dt).long().item() + 1
        pc_scaled_time = torch.linspace(0, 1, num_pre_compute)

        # y1 and y2
        self.y_1_value = torch.exp(-0.5 * self.alpha * pc_scaled_time)
        self.y_2_value = pc_scaled_time * self.y_1_value

        self.dy_1_value = -0.5 * self.alpha * self.y_1_value
        self.dy_2_value = -0.5 * self.alpha * self.y_2_value + self.y_1_value

        # q_1 and q_2
        q_1_value = \
            (0.5 * self.alpha * pc_scaled_time - 1) \
            * torch.exp(0.5 * self.alpha * pc_scaled_time) + 1
        q_2_value = \
            0.5 * self.alpha \
            * (torch.exp(0.5 * self.alpha * pc_scaled_time) - 1)

        # p_1 and p_2
        # Get basis of one DOF, shape [num_pc_times, num_basis]
        basis_single_dof = super().basis(pc_scaled_time)
        assert list(basis_single_dof.shape) == [*pc_scaled_time.shape,
                                                self.num_basis]

        dp_1_value = \
            torch.einsum('...i,...ij->...ij',
                         pc_scaled_time
                         * torch.exp(self.alpha * pc_scaled_time / 2),
                         basis_single_dof)
        dp_2_value = \
            torch.einsum('...i,...ij->...ij',
                         torch.exp(self.alpha * pc_scaled_time / 2),
                         basis_single_dof)

        p_1_value = torch.zeros(size=dp_1_value.shape)
        p_2_value = torch.zeros(size=dp_2_value.shape)

        for i in range(pc_scaled_time.shape[0]):
            p_1_value[i] = torch.trapz(dp_1_value[:i + 1],
                                       pc_scaled_time[:i + 1], dim=0)
            p_2_value[i] = torch.trapz(dp_2_value[:i + 1],
                                       pc_scaled_time[:i + 1], dim=0)

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

    def times_to_indices(self, times: torch.Tensor):
        """
        Map time points to pre-compute indices
        Args:
            times: time points

        Returns:
            time indices
        """
        # times to scaled times
        scaled_times = \
            super(ExpDecayPhaseGenerator, self.phase_generator).phase(times)
        return torch.round(scaled_times / self.scaled_dt).long()

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
        time_indices = self.times_to_indices(times)

        basis = self.pc_pos_basis[time_indices, :]
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
        time_indices = self.times_to_indices(times)

        vel_basis = self.pc_vel_basis[time_indices, :]
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
                                           num_dof * self.num_basis_g)
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
        # [*add_dim, num_times] or [*add_dim]
        #
        # Shape of each return
        # [*add_dim, num_times] or [*add_dim]

        time_indices = self.times_to_indices(times)

        # Shape [num_pc_times] -> [*add_dim]
        y_1_value = self.y_1_value[time_indices]
        y_2_value = self.y_2_value[time_indices]
        dy_1_value = self.dy_1_value[time_indices]
        dy_2_value = self.dy_2_value[time_indices]

        return y_1_value, y_2_value, dy_1_value, dy_2_value
