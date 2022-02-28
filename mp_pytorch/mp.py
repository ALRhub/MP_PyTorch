"""
@brief:     Movement Primitives in PyTorch
"""
from abc import ABC, abstractmethod
import torch
from torch.distributions import MultivariateNormal

import mp_pytorch.util as util


# Classes of Phase Generator
class PhaseGenerator(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Basis class constructor
        """
        pass

    @abstractmethod
    def phase(self, times: torch.Tensor) -> torch.Tensor:
        """
        Basis class phase interface
        Args:
            times: times in Tensor

        Returns: phases in Tensor

        """
        pass


class LinearPhaseGenerator(PhaseGenerator):
    def __init__(self,
                 phase_velocity=1.0):
        """
        Constructor for linear phase generator
        Args:
            phase_velocity: coefficient transfer time to phase
        """
        super(LinearPhaseGenerator, self).__init__()
        self.phase_velocity = phase_velocity

    def phase(self,
              times: torch.Tensor) -> torch.Tensor:
        """
        Compute phase
        Args:
            times: times in Tensor

        Returns:
            phase in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]

        phase = times * self.phase_velocity
        return phase


class ExpDecayPhaseGenerator(PhaseGenerator):
    def __init__(self,
                 tau,
                 alpha_phase=3.0):
        """
        Constructor for exponential decay phase generator
        Args:
            tau: time scale (normalization factor)
                 Use duration of the movement is a good choice
            alpha_phase: decaying factor: tau * dx/dt = -alpha_phase * x
        """
        super(ExpDecayPhaseGenerator, self).__init__()
        self.tau = tau
        self.alpha_phase = alpha_phase

    def phase(self, times):
        """
        Compute phase
        Args:
            times: times Tensor

        Returns:
            phase in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]

        phase = torch.exp(-self.alpha_phase * times / self.tau)
        return phase


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
                 duration: float = 1.,
                 basis_bandwidth_factor: int = 3,
                 num_basis_outside: int = 0):
        """
        Constructor of class RBF

        Args:
            phase_generator: phase generator
            num_basis: number of basis function
            duration: "Time Duration!" to be covered, not phase!
            basis_bandwidth_factor: ...
            num_basis_outside: basis function outside the duration
        """
        super(NormalizedRBFBasisGenerator, self).__init__(phase_generator,
                                                          num_basis)

        self.basis_bandwidth_factor = basis_bandwidth_factor
        self.num_basis_outside = num_basis_outside

        # Distance between basis centers
        basis_dist = \
            duration / (self.num_basis - 2 * self.num_basis_outside - 1)

        # RBF centers in time scope
        centers_t = \
            torch.linspace(-self.num_basis_outside * basis_dist,
                           duration + self.num_basis_outside * basis_dist,
                           self.num_basis)

        # RBF centers in phase scope
        self.centers_p = self.phase_generator.phase(centers_t)

        tmp_bandwidth = \
            torch.cat((self.centers_p[1:] - self.centers_p[:-1],
                       self.centers_p[-1:] - self.centers_p[-2:-1]), dim=-1)

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
                 duration: float = 1.,
                 basis_bandwidth_factor: int = 3,
                 num_basis_outside: int = 0):
        """
        Constructor of class DMPBasisGenerator

        Args:
            phase_generator: phase generator
            num_basis: number of basis function
            duration: "Time Duration!" to be covered, not phase!
            basis_bandwidth_factor: ...
            num_basis_outside: basis function outside the duration
        """
        super(DMPBasisGenerator, self).__init__(phase_generator,
                                                num_basis,
                                                duration,
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

        # Einsum shape: [*add_dim, num_times, num_basis]
        #               [*add_dim, num_basis]
        #            -> [*add_dim, num_times, num_basis]
        dmp_basis = torch.einsum('...i,...ij->...ij', phase, rbf_basis)
        return dmp_basis


class MPInterface(ABC):
    @abstractmethod
    def __init__(self, basis_gn: BasisGenerator, num_dof: int, **kwargs):
        """
        Constructor interface
        Args:
            basis_gn: basis generator
            num_dof: number of dof
            **kwargs: keyword arguments
        """
        # Additional batch dimension
        self.add_dim = list()

        # The basis and phase generators
        self.basis_gn = basis_gn
        self.phase_gn = basis_gn.phase_generator

        # Number of basis per DoF and number of DoFs
        self.num_basis = basis_gn.num_basis
        self.num_dof = num_dof

        # Compute values at these time points
        self.times = None

        # Learnable parameters
        self.params = None

        # Boundary conditions
        self.bc_time = None
        self.bc_pos = None
        self.bc_vel = None

        # Runtime computation results, shall be reset every time when
        # inputs are reset
        self.pos = None
        self.vel = None

    def clear_computation_result(self):
        """
        Clear runtime computation result

        Returns:
            None
        """

        self.pos = None
        self.vel = None

    def set_add_dim(self, add_dim):
        """
        Set additional batch dimension
        Args:
            add_dim: additional batch dimension

        Returns: None

        """
        self.add_dim = add_dim
        self.clear_computation_result()

    def set_mp_times(self, times):
        """
        Set MP time points
        Args:
            times: desired time points

        Returns:
            None
        """

        # Shape of times
        # [*add_dim, num_times]

        self.times = times
        self.clear_computation_result()

    def set_mp_params(self, params):
        """
        Set MP params
        Args:
            params: parameters of MP

        Returns: None

        """

        # Shape of params
        # [*add_dim, num_params]

        self.params = params
        self.clear_computation_result()

    def set_boundary_conditions(self, bc_time, bc_pos, bc_vel):
        """
        Set boundary conditions in a batched manner

        Args:
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity

        Returns:
            None
        """

        # Shape of bc_time:
        # [*add_dim]
        #
        # Shape of bc_pos:
        # [*add_dim, num_dof]
        #
        # Shape of bc_vel:
        # [*add_dim, num_dof]

        self.bc_time = bc_time
        self.bc_pos = bc_pos
        self.bc_vel = bc_vel
        self.clear_computation_result()

    def update_mp_inputs(self, times=None, params=None,
                         bc_time=None, bc_pos=None, bc_vel=None, **kwargs):
        """
        Update MP
        Args:
            times: desired time points
            params: parameters of MP
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            kwargs: other keyword arguments

        Returns: None

        """
        if times is not None:
            self.set_mp_times(times)
        if params is not None:
            self.set_mp_params(params)
        if all([data is not None for data in {bc_time, bc_pos, bc_vel}]):
            self.set_boundary_conditions(bc_time, bc_pos, bc_vel)
        self.clear_computation_result()

    def get_mp_trajs(self, get_pos: bool = True, get_vel: bool = True) -> dict:
        """
        Get movement primitives trajectories given flag
        Args:
            get_pos: True if pos shall be computed
            get_vel: True if vel shall be computed

        Returns:
            results in dictionary
        """

        # Initialize result dictionary
        result = dict()

        # Position
        if get_pos:
            result["pos"] = self.get_pos()
        else:
            result["pos"] = None

        # Velocity
        if get_vel:
            result["vel"] = self.get_vel()
        else:
            result["vel"] = None

        # Return
        return result

    @abstractmethod
    def get_pos(self, times=None, params=None,
                bc_time=None, bc_pos=None, bc_vel=None):
        """
        Get trajectory position
        Args:
            times: time points
            params: learnable parameters
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity

        Returns:
            pos
        """
        pass

    @abstractmethod
    def get_vel(self, times=None, params=None,
                bc_time=None, bc_pos=None, bc_vel=None):
        """
        Get trajectory velocity

        Args:
            times: time points
            params: learnable parameters
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity

        Returns: vel
        """
        pass

    @abstractmethod
    def learn_mp_params_from_trajs(self, times, trajs, reg=1e-9):
        """
        Learn params from trajectories

        Args:
            times: time points of the trajectories
            trajs: demonstration trajectories
            reg: regularization term of linear ridge regression

        Returns:
            learned parameters
        """
        pass


class ProbabilisticMPInterface(MPInterface):
    def __init__(self, basis_gn: BasisGenerator, num_dof: int, **kwargs):
        """
        Constructor interface
        Args:
            basis_gn: basis generator
            num_dof: number of dof
            **kwargs: keyword arguments
        """

        super().__init__(basis_gn, num_dof, **kwargs)

        # Learnable parameters variance
        self.params_L = None

        # Runtime computation results, shall be reset every time when
        # inputs are reset
        self.pos_cov = None
        self.pos_std = None
        self.vel_cov = None
        self.vel_std = None

    def clear_computation_result(self):
        """
        Clear runtime computation result

        Returns:
            None
        """
        super().clear_computation_result()
        self.pos_cov = None
        self.pos_std = None
        self.vel_cov = None
        self.vel_std = None

    def set_mp_params_variances(self, params_L):
        """
        Set variance of MP params
        Args:
            params_L: cholesky of covariance matrix of the MP parameters

        Returns: None

        """
        # Shape of params_L
        # [*add_dim, num_params, num_params]

        self.params_L = params_L
        self.clear_computation_result()

    def update_mp_inputs(self, times=None, params=None, params_L=None,
                         bc_time=None, bc_pos=None, bc_vel=None, **kwargs):
        """
        Set MP
        Args:
            times: desired time points
            params: parameters of MP
            params_L: cholesky of covariance matrix of the MP parameters
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            kwargs: other keyword arguments

        Returns: None

        """
        super().update_mp_inputs(times, params, bc_time, bc_pos, bc_vel)
        if params_L is not None:
            self.set_mp_params_variances(params_L)
        self.clear_computation_result()

    @property
    def params_cov(self):
        """
        Compute params covariance using params_L
        Returns:
            covariance matrix of parameters
        """
        assert self.params_L is not None
        params_cov = torch.einsum('...ij,...kj->...ik',
                                  self.params_L,
                                  self.params_L)
        return params_cov

    def get_mp_trajs(self, get_pos=True, get_pos_cov=True, get_pos_std=True,
                     get_vel=True, get_vel_cov=True, get_vel_std=True,
                     flat_shape=False, reg: float = 1e-4):
        """
        Get movement primitives trajectories given flag
        Args:
            get_pos: True if pos shall be computed
            get_vel: True if vel shall be computed
            get_pos_cov: True if pos_cov shall be computed
            get_pos_std: True if pos_std shall be computed
            get_vel_cov: True if vel_cov shall be computed
            get_vel_std: True if vel_std shall be computed
            flat_shape: if flatten the Dof and time dimensions
            reg: regularization term

        Returns:
            results in dictionary
        """
        # Initialize result dictionary
        result = dict()

        # pos
        if get_pos:
            result["pos"] = self.get_pos(flat_shape=flat_shape)
        else:
            result["pos"] = None

        # vel
        if get_vel:
            result["vel"] = self.get_vel(flat_shape=flat_shape)
        else:
            result["vel"] = None

        # pos_cov
        if get_pos_cov:
            result["pos_cov"] = self.get_pos_cov(reg=reg)
        else:
            result["pos_cov"] = None

        # pos_std
        if get_pos_std:
            result["pos_std"] = self.get_pos_std(flat_shape=flat_shape, reg=reg)
        else:
            result["pos_std"] = None

        # vel_cov
        if get_vel_cov:
            result["vel_cov"] = self.get_vel_cov(reg=reg)
        else:
            result["vel_cov"] = None

        # vel_std
        if get_vel_std:
            result["vel_std"] = self.get_vel_std(flat_shape=flat_shape, reg=reg)
        else:
            result["vel_std"] = None

        return result

    @abstractmethod
    def get_pos(self, times=None, params=None,
                bc_time=None, bc_pos=None, bc_vel=None,
                flat_shape=False):
        """
        Get trajectory position
        Args:
            times: time points
            params: learnable parameters
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions

        Returns:
            pos
        """
        pass

    @abstractmethod
    def get_pos_cov(self, times=None, params_L=None,
                    bc_time=None, bc_pos=None, bc_vel=None,
                    reg: float = 1e-4):
        """
        Get trajectory covariance
        Returns: cov

        Args:
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            reg: regularization term

        Returns:
            pos cov
        """
        pass

    @abstractmethod
    def get_pos_std(self, times=None, params_L=None,
                    bc_time=None, bc_pos=None, bc_vel=None,
                    flat_shape=False, reg: float = 1e-4):
        """
        Get trajectory standard deviation
        Args:
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions
            reg: regularization term

        Returns:
            pos std
        """
        pass

    @abstractmethod
    def get_vel(self, times=None, params=None,
                bc_time=None, bc_pos=None, bc_vel=None,
                flat_shape=False):
        """
        Get trajectory velocity
        Returns: vel

        Args:
            times: time points
            params: learnable parameters
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions

        Returns:
            vel
        """
        pass

    @abstractmethod
    def get_vel_cov(self, times=None, params_L=None,
                    bc_time=None, bc_pos=None, bc_vel=None,
                    reg: float = 1e-4):
        """
        Get trajectory covariance
        Args:
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            reg: regularization term

        Returns:
            vel cov
        """
        pass

    @abstractmethod
    def get_vel_std(self, times=None, params_L=None,
                    bc_time=None, bc_pos=None, bc_vel=None,
                    flat_shape=False, reg: float = 1e-4):
        """
        Get trajectory standard deviation
        Args:
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions
            reg: regularization term

        Returns:
            vel std
        """
        pass

    def sample_trajectories(self, times=None, params=None, params_L=None,
                            bc_time=None, bc_pos=None, bc_vel=None,
                            num_smp=1, flat_shape=False):
        """
        Sample trajectories from MP

        Args:
            times: time points
            params: learnable parameters
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            num_smp: num of trajectories to be sampled
            flat_shape: if flatten the Dof and time dimensions

        Returns:
            sampled trajectories
        """

        # Shape of pos_smp
        # [*add_dim, num_smp, num_times, num_dof]
        # or [*add_dim, num_smp, num_dof * num_times]

        if all([data is None for data in {times, params, params_L, bc_time,
                                          bc_pos, bc_vel}]):
            times = self.times
            params = self.params
            params_L = self.params_L
            bc_time = self.bc_time
            bc_pos = self.bc_pos
            bc_vel = self.bc_vel

        num_add_dim = params.ndim - 1

        # Add additional sample axis to time
        # Shape [*add_dim, num_smp, num_times]
        times = util.add_expand_dim(times, [num_add_dim], [num_smp])

        # Sample parameters, shape [num_smp, *add_dim, num_params]
        params_smp = MultivariateNormal(loc=params,
                                        scale_tril=params_L,
                                        validate_args=False).rsample([num_smp])

        # Switch axes to [*add_dim, num_smp, num_params]
        params_smp = torch.einsum('i...j->...ij', params_smp)

        # Add additional sample axis to boundary condition
        bc_time = util.add_expand_dim(bc_time, [num_add_dim], [num_smp])
        bc_pos = util.add_expand_dim(bc_pos, [num_add_dim], [num_smp])
        bc_vel = util.add_expand_dim(bc_vel, [num_add_dim], [num_smp])

        # Update inputs
        self.update_mp_inputs(times, params_smp, None, bc_time, bc_pos, bc_vel)

        # Get sample trajectories
        pos_smp = self.get_pos(flat_shape=flat_shape)

        return pos_smp


class ProMP(ProbabilisticMPInterface):
    """ProMP in PyTorch"""

    def __init__(self,
                 basis_gn: BasisGenerator,
                 num_dof: int,
                 **kwargs):
        """
        Constructor of ProMP
        Args:
            basis_gn: basis function value generator
            num_dof: number of Degrees of Freedoms
            kwargs: keyword arguments
        """

        super().__init__(basis_gn, num_dof, **kwargs)

        # Number of parameters
        self.num_params = basis_gn.num_basis * self.num_dof

        # Runtime variables
        self.basis_multi_dofs = None

    def set_mp_times(self, times):
        """
        Set MP time points
        Args:
            times: desired time points

        Returns:
            None
        """
        # Shape of times
        # [*add_dim, num_times]

        super().set_mp_times(times)

        # Shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        self.basis_multi_dofs = \
            self.basis_gn.basis_multi_dofs(times, self.num_dof)

    def set_mp_params(self, params):
        """
        Set MP params
        Args:
            params: parameters of MP

        Returns: None

        """
        # Shape of params:
        # [*add_dim, num_dof * num_basis]

        # Check number of params
        assert params.shape[-1] == self.num_params

        # Set additional batch size
        self.set_add_dim(list(params.shape[:-1]))

        # Set params
        super().set_mp_params(params)

    def set_mp_params_variances(self, params_L):
        """
        Set variance of MP params
        Args:
            params_L: cholesky of covariance matrix of the MP parameters

        Returns: None

        """
        # Shape of params_L:
        # [*add_dim, num_dof * num_basis, num_dof * num_basis]

        if params_L is not None:
            assert list(params_L.shape) == [*self.add_dim, self.num_params,
                                            self.num_params]
        super().set_mp_params_variances(params_L)

    def get_pos(self, times=None, params=None,
                bc_time=None, bc_pos=None, bc_vel=None,
                flat_shape=False):
        """
        Get trajectory position

        Refer setting functions for desired shape of inputs

        Args:

            times: time points
            params: learnable parameters
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions

        Returns:
            pos
        """

        # Shape of pos
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_mp_inputs(times, params, None, bc_time, bc_pos, bc_vel)

        # Reuse result if existing
        if self.pos is not None:
            return self.pos

        assert self.params is not None and self.basis_multi_dofs is not None

        # Get basis of all Dofs
        # Shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dof = self.basis_multi_dofs

        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis],
        #               [*add_dim, num_dof * num_basis]
        #            -> [*add_dim, num_dof * num_times]
        pos = torch.einsum('...ij,...j->...i', basis_multi_dof, self.params)

        if not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            pos = pos.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            pos = torch.einsum('...ji->...ij', pos)

        self.pos = pos

        return self.pos

    def get_pos_cov(self, times=None, params_L=None, bc_time=None, bc_pos=None,
                    bc_vel=None, reg: float = 1e-4):
        """
        Compute position covariance

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            reg: regularization term

        Returns:
            pos_cov
        """

        # Shape of pos_cov
        # [*add_dim, num_dof * num_times, num_dof * num_times]

        # Update inputs
        self.update_mp_inputs(times, None, params_L, bc_time, bc_pos, bc_vel)

        # Reuse result if existing
        if self.pos_cov is not None:
            return self.pos_cov

        # Otherwise recompute result
        if self.params_L is None:
            return None
        else:
            assert self.basis_multi_dofs is not None

        # Get basis of all Dofs
        # Shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dof = self.basis_multi_dofs

        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        #               [*add_dim, num_dof * num_basis, num_dof * num_basis]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis]
        #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
        pos_cov = torch.einsum('...ik,...kl,...jl->...ij',
                               basis_multi_dof,
                               self.params_cov,
                               basis_multi_dof)

        # Determine regularization term to make traj_cov positive definite
        traj_cov_reg = reg
        reg_term_pos = torch.max(torch.einsum('...ii->...i',
                                              pos_cov)).item() * traj_cov_reg

        # Add regularization term for numerical stability
        self.pos_cov = pos_cov + torch.eye(pos_cov.shape[-1]) * reg_term_pos
        return self.pos_cov

    def get_pos_std(self, times=None, params_L=None, bc_time=None, bc_pos=None,
                    bc_vel=None, flat_shape=False, reg: float = 1e-4):
        """
        Compute position standard deviation

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions
            reg: regularization term

        Returns:
            pos_std
        """

        # Shape of pos_std
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_mp_inputs(times, None, params_L, bc_time, bc_pos, bc_vel)

        # Reuse result if existing
        if self.pos_std is not None:
            return self.pos_std

        # Otherwise recompute
        if self.pos_cov is not None:
            pos_cov = self.pos_cov
        else:
            pos_cov = self.get_pos_cov()

        if pos_cov is None:
            pos_std = None
        else:
            # Shape [*add_dim, num_dof * num_times]
            pos_std = torch.sqrt(torch.einsum('...ii->...i', pos_cov))

            if not flat_shape:
                # Reshape to [*add_dim, num_dof, num_times]
                pos_std = pos_std.reshape(*self.add_dim, self.num_dof, -1)

                # Switch axes to [*add_dim, num_times, num_dof]
                pos_std = torch.einsum('...ji->...ij', pos_std)

        self.pos_std = pos_std
        return self.pos_std

    def get_vel(self, times=None, params=None,
                bc_time=None, bc_pos=None, bc_vel=None,
                flat_shape=False):
        """
        Get trajectory velocity

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params: learnable parameters
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions

        Returns:
            vel
        """
        # todo interpolation?
        self.vel = None
        return self.vel

    def get_vel_cov(self, times=None, params_L=None, bc_time=None, bc_pos=None,
                    bc_vel=None, reg: float = 1e-4):
        """
        Get velocity covariance

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            reg: regularization term

        Returns:
            vel_cov
        """
        self.vel_cov = None
        return self.vel_cov

    def get_vel_std(self, times=None, params_L=None, bc_time=None, bc_pos=None,
                    bc_vel=None, flat_shape=False, reg: float = 1e-4):
        """
        Get trajectory standard deviation

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions
            reg: regularization term

        Returns:
            vel_std
        """
        self.vel_std = None
        return self.vel_std

    def learn_mp_params_from_trajs(self, times, trajs, reg=1e-9) -> dict:
        """
        Learn ProMP weights from demonstration

        Args:
            times: trajectory time points
            trajs: trajectory from which weights should be learned
            reg: regularization term

        Returns:
            param_dict: dictionary of parameters containing
                - weights
        """
        # Shape of times
        # [*add_dim, num_times]
        #
        # Shape of trajs:
        # [*add_dim, num_times, num_dof]
        #
        # Shape of params:
        # [*add_dim, num_dof * num_basis]

        assert trajs.shape[:-1] == times.shape
        assert trajs.shape[-1] == self.num_dof

        # Get multiple dof basis function values
        # Tensor [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dofs = \
            self.basis_gn.basis_multi_dofs(times, self.num_dof)

        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis],
        #               [*add_dim, num_dof * num_times, num_dof * num_basis],
        #            -> [*add_dim, num_dof * num_basis, num_dof * num_basis]
        A = torch.einsum('...ki,...kj->...ij', basis_multi_dofs,
                         basis_multi_dofs)
        A += torch.eye(self.num_params) * reg

        # Reorder axis [*add_dim, num_times, num_dof]
        #           -> [*add_dim, num_dof, num_times]
        trajs = torch.Tensor(trajs)
        trajs = torch.einsum('...ij->...ji', trajs)

        # Reshape: [*add_dim, num_dof, num_times]
        #       -> [*add_dim, num_dof * num_times]
        add_dim = trajs.shape[:-2]
        trajs = trajs.reshape(*add_dim, -1)

        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis],
        #               [*add_dim, num_dof * num_times],
        #            -> [*add_dim, num_dof * num_basis]
        B = torch.einsum('...ki,...k->...i', basis_multi_dofs, trajs)

        # Solve for weights, shape [*add_dim, num_dof * num_basis]
        params = torch.linalg.solve(A, B)

        return {"params": params}


class DMP(MPInterface):
    """DMP in PyTorch"""

    def __init__(self,
                 basis_gn: BasisGenerator,
                 num_dof: int,
                 **kwargs):
        """
        Constructor of DMP
        Args:
            basis_gn: basis function value generator
            num_dof: number of Degrees of Freedoms
            kwargs: keyword arguments
        """

        super().__init__(basis_gn, num_dof, **kwargs)

        # Number of parameters
        self.num_basis_g = self.num_basis + 1
        self.num_params = self.num_basis_g * self.num_dof

        # Control parameters
        self.alpha = kwargs["alpha"]
        self.beta = self.alpha / 4

        # Time scaling parameter
        self.tau = kwargs["tau"]
        self.dt = kwargs["dt"]

    def set_mp_params(self, params):
        """
        Set MP params
        Args:
            params: parameters of MP

        Returns: None

        """
        # Shape of params:
        # [*add_dim, num_dof * num_basis_g]

        # Check number of w and g
        assert params.shape[-1] == self.num_params

        # Set additional batch size
        self.add_dim = list(params.shape[:-1])

        # Set params
        super().set_mp_params(params)

    def set_boundary_conditions(self, bc_time, bc_pos, bc_vel):
        """
        Set boundary conditions in a batched manner

        Args:
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity

        Returns:
            None
        """
        # Shape of bc_time:
        # [*add_dim]
        #
        # Shape of bc_pos:
        # [*add_dim, num_dof]
        #
        # Shape of bc_vel:
        # [*add_dim, num_dof]

        assert list(bc_time.shape) == [*self.add_dim]
        assert list(bc_pos.shape) == list(bc_vel.shape) \
               and list(bc_vel.shape) == [*self.add_dim, self.num_dof]
        super().set_boundary_conditions(bc_time, bc_pos, bc_vel)

    def get_pos(self, times=None, params=None,
                bc_time=None, bc_pos=None, bc_vel=None):
        """
        Compute trajectories at desired time points

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params: learnable parameters
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity

        Returns:
            pos
        """

        # Shape of pos
        # [*add_dim, num_times, num_dof]

        # Update inputs
        self.update_mp_inputs(times, params, bc_time, bc_pos, bc_vel)

        # Reuse result if existing
        if self.pos is not None:
            return self.pos

        # Split weights and goal
        # Shape of w:
        # [*add_dim, num_dof, num_basis]
        # Shape of g:
        # [*add_dim, num_dof, 1]
        w, g = self.split_weights_goal(self.params)

        # Get basis, shape [*add_dim, num_times, num_basis]
        basis = self.basis_gn.basis(self.times)

        # Get forcing function
        # Einsum shape: [*add_dim, num_times, num_basis]
        #               [*add_dim, num_dof, num_basis]
        #            -> [*add_dim, num_times, num_dof]
        f = torch.einsum('...ik,...jk->...ij', basis, w)

        # Initialize trajectory position, velocity
        pos = torch.zeros([*self.add_dim, self.times.shape[-1], self.num_dof])
        vel = torch.zeros([*self.add_dim, self.times.shape[-1], self.num_dof])

        # Check boundary condition, the desired times should start from
        # boundary condition time steps
        assert torch.all(torch.abs(self.bc_time - self.times[..., 0]) < 1e-8)
        pos[..., 0, :] = self.bc_pos
        vel[..., 0, :] = self.bc_vel

        # Apply Euler Integral
        for i in range(self.times.shape[-1] - 1):
            acc = (self.alpha * (self.beta * (g - pos[..., i, :])
                                 - self.tau * vel[..., i, :]) + f[..., i, :]) \
                  / self.tau ** 2
            vel[..., i + 1, :] = vel[..., i, :] + self.dt * acc
            pos[..., i + 1, :] = pos[..., i, :] + self.dt * vel[..., i + 1, :]

        # Store pos and vel
        self.pos = pos
        self.vel = vel

        return pos

    def get_vel(self, times=None, params=None,
                bc_time=None, bc_pos=None, bc_vel=None):
        """
        Get trajectory velocity

        Refer setting functions for desired shape of inputs

        Args:
            times: time points, can be None
            params: learnable parameters, can be None
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity

        Returns:
            vel
        """

        # Shape of vel
        # [*add_dim, num_times, num_dof]

        # Update inputs
        self.update_mp_inputs(times, params, bc_time, bc_pos, bc_vel)

        # Reuse result if existing
        if self.vel is not None:
            return self.vel

        # Recompute otherwise
        # Velocity is computed together with position in DMP
        self.get_pos()
        return self.vel

    def learn_mp_params_from_trajs(self, times, trajs, reg=1e-9):
        # todo
        raise NotImplementedError

    def split_weights_goal(self, wg):
        """
        Helper function to split weights and goal

        Args:
            wg: vector storing weights and goal

        Returns:
            w: weights
            g: goal

        """
        # Shape of wg:
        # [*add_dim, num_dof * num_basis_g]
        #
        # Shape of w:
        # [*add_dim, num_dof, num_basis]
        #
        # Shape of g:
        # [*add_dim, num_dof, 1]

        wg = wg.reshape([*wg.shape[:-1], self.num_dof, self.num_basis_g])
        w = wg[..., :-1]
        g = wg[..., -1]

        return w, g


class IDMP(ProMP):
    """Integral form of DMPs"""

    def __init__(self,
                 basis_gn: BasisGenerator,
                 num_dof: int,
                 **kwargs):
        """
        Constructor of IDMP
        Args:
            basis_gn: basis function value generator
            num_dof: number of Degrees of Freedoms
            kwargs: keyword arguments
        """

        super().__init__(basis_gn, num_dof, **kwargs)

        # Control parameters
        self.alpha = kwargs["alpha"]
        self.beta = self.alpha / 4

        # Time scaling parameter
        self.tau = kwargs["tau"]
        self.dt = kwargs["dt"]

        # Pre-compute times, shape: [num_times]
        self.num_pc_times = int(self.tau / self.dt) + 1
        self.pc_times = torch.linspace(0, self.tau, self.num_pc_times)

        # Number of parameters
        self.num_basis_g = self.num_basis + 1
        self.num_params = self.num_basis_g * self.num_dof

        # Pre-computed terms
        self.y_1_value = None
        self.y_2_value = None
        self.dy_1_value = None
        self.dy_2_value = None
        self.pos_basis = None
        self.vel_basis = None

        # Pre-compute
        self._pre_compute()

        # Computation results
        self.pos_cov = None
        self.vel_cov = None

        # Runtime intermediate variables shared by different getting functions
        self.pos_det = None
        self.vel_det = None
        self.pos_vary_ = None
        self.vel_vary_ = None

        self.xi_1 = None
        self.xi_2 = None
        self.xi_3 = None
        self.xi_4 = None
        self.dxi_1 = None
        self.dxi_2 = None
        self.dxi_3 = None
        self.dxi_4 = None
        self.pos_basis_bc_multi_dofs = None
        self.vel_basis_bc_multi_dofs = None
        self.pos_basis_multi_dofs = None
        self.vel_basis_multi_dofs = None

    def set_boundary_conditions(self, bc_time, bc_pos, bc_vel):
        """
        Set boundary conditions in a batched manner

        Args:
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity

        Returns:
            None
        """
        # Shape of bc_time:
        # [*add_dim]
        #
        # Shape of bc_pos:
        # [*add_dim, num_dof]
        #
        # Shape of bc_vel:
        # [*add_dim, num_dof]

        assert list(bc_time.shape) == [*self.add_dim]
        assert list(bc_pos.shape) == list(bc_vel.shape) \
               and list(bc_vel.shape) == [*self.add_dim, self.num_dof]
        super().set_boundary_conditions(bc_time, bc_pos, bc_vel)

        # Update shared intermediate variables
        self.compute_intermediate_shared_variables()

    def get_pos(self, times=None, params=None,
                bc_time=None, bc_pos=None, bc_vel=None,
                flat_shape=False):
        """
        Compute trajectory pos

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params: learnable parameters
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions

        Returns:
            pos
        """

        # Shape of pos
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_mp_inputs(times, params, None, bc_time, bc_pos, bc_vel)

        # Reuse result if existing
        if self.pos is not None:
            return self.pos

        # Recompute otherwise
        # Position and velocity variant (part 3)
        # Einsum shape: [*add_dim, num_dof * num_times, num_basis_g * num_dof],
        #               [*add_dim, num_basis_g * num_dof]
        #            -> [*add_dim, num_dof * num_times]
        pos_vary = torch.einsum('...ij,...j->...i', self.pos_vary_, self.params)

        self.pos = self.pos_det + pos_vary

        if not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            self.pos = self.pos.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            self.pos = torch.einsum('...ji->...ij', self.pos)

        return self.pos

    def get_pos_cov(self, times=None, params_L=None, bc_time=None, bc_pos=None,
                    bc_vel=None, reg: float = 1e-4):
        """
        Compute and return position covariance

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            reg: regularization term

        Returns:
            pos_cov
        """

        # Shape of pos_cov
        # [*add_dim, num_dof * num_times, num_dof * num_times]

        # Update inputs
        self.update_mp_inputs(times, None, params_L, bc_time, bc_pos, bc_vel)

        # Reuse result if existing
        if self.pos_cov is not None:
            return self.pos_cov

        # Recompute otherwise
        if self.params_L is None:
            return None

        # Uncertainty of position
        # Einsum shape: [*add_dim, num_dof * num_times, num_basis_g * num_dof],
        #               [*add_dim, num_basis_g * num_dof, num_basis_g * num_dof]
        #               [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
        pos_cov = torch.einsum('...ik,...kl,...jl->...ij',
                               self.pos_vary_, self.params_cov, self.pos_vary_)

        # Determine regularization term to make traj_cov positive definite
        traj_cov_reg = reg
        reg_term_pos = torch.max(torch.einsum('...ii->...i',
                                              pos_cov)).item() * traj_cov_reg

        # Add regularization term for numerical stability
        self.pos_cov = pos_cov + torch.eye(pos_cov.shape[-1]) * reg_term_pos

        return self.pos_cov

    def get_pos_std(self, times=None, params_L=None, bc_time=None, bc_pos=None,
                    bc_vel=None, flat_shape=False, reg: float = 1e-4):
        """
        Compute trajectory standard deviation

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions
            reg: regularization term

        Returns:
            pos_std
        """

        # Shape of pos_std
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_mp_inputs(times, None, params_L, bc_time, bc_pos, bc_vel)

        # Reuse result if existing
        if self.pos_std is not None:
            return self.pos_std

        # Recompute otherwise
        pos_cov = self.get_pos_cov(reg=reg)
        if pos_cov is None:
            pos_std = None
        else:
            # Shape [*add_dim, num_dof * num_times]
            pos_std = torch.sqrt(torch.einsum('...ii->...i', pos_cov))

            if not flat_shape:
                # Reshape to [*add_dim, num_dof, num_times]
                pos_std = pos_std.reshape(*self.add_dim, self.num_dof, -1)

                # Switch axes to [*add_dim, num_times, num_dof]
                pos_std = torch.einsum('...ji->...ij', pos_std)

        self.pos_std = pos_std
        return self.pos_std

    def get_vel(self, times=None, params=None,
                bc_time=None, bc_pos=None, bc_vel=None,
                flat_shape=False):
        """
        Compute trajectory velocity

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params: learnable parameters
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions

        Returns:
            vel
        """

        # Shape of vel
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        if times is not None:
            self.set_mp_times(times)
        if params is not None:
            self.set_mp_params(params)

        # Reuse result if existing
        if self.vel is not None:
            return self.vel

        # Recompute otherwise
        # Position and velocity variant (part 3)
        # Einsum shape: [*add_dim, num_dof * num_times, num_basis_g * num_dof],
        #               [*add_dim, num_basis_g * num_dof]
        #            -> [*add_dim, num_dof * num_times]
        vel_vary = torch.einsum('...ij,...j->...i', self.vel_vary_, self.params)

        self.vel = self.vel_det + vel_vary

        if not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            self.vel = self.vel.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            self.vel = torch.einsum('...ji->...ij', self.vel)

        return self.vel

    def get_vel_cov(self, times=None, params_L=None, bc_time=None, bc_pos=None,
                    bc_vel=None, reg: float = 1e-4):
        """
        Get trajectory velocity covariance

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            reg: regularization term

        Returns:
            vel_cov
        """

        # Shape of vel_cov
        # [*add_dim, num_dof * num_times, num_dof * num_times]

        # Update inputs
        self.update_mp_inputs(times, None, params_L, bc_time, bc_pos, bc_vel)

        # Reuse result if existing
        if self.vel_cov is not None:
            return self.vel_cov

        # Recompute otherwise
        if self.params_L is None:
            return None

        # Uncertainty of velocity
        # Einsum shape: [*add_dim, num_dof * num_times, num_basis_g * num_dof],
        #               [*add_dim, num_basis_g * num_dof, num_basis_g * num_dof]
        #               [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
        vel_cov = torch.einsum('...ik,...kl,...jl->...ij',
                               self.vel_vary_, self.params_cov, self.vel_vary_)

        # Determine regularization term to make traj_cov positive definite
        traj_cov_reg = reg
        reg_term_vel = torch.max(torch.einsum('...ii->...i',
                                              vel_cov)).item() * traj_cov_reg

        # Add regularization term for numerical stability
        self.vel_cov = vel_cov + torch.eye(vel_cov.shape[-1]) * reg_term_vel

        return self.vel_cov

    def get_vel_std(self, times=None, params_L=None, bc_time=None, bc_pos=None,
                    bc_vel=None, flat_shape=False, reg: float = 1e-4):
        """
        Compute trajectory velocity standard deviation

        Refer setting functions for desired shape of inputs

        Args:
            reg: regularization term
            times: time points
            params_L: learnable parameters' variance
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            flat_shape: if flatten the Dof and time dimensions

        Returns:
            vel_std

        """

        # Shape of vel_std
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_mp_inputs(times, None, params_L, bc_time, bc_pos, bc_vel)

        # Reuse result if existing
        if self.vel_std is not None:
            return self.vel_std

        # Recompute otherwise
        vel_cov = self.get_vel_cov(reg=reg)
        if vel_cov is None:
            vel_std = None
        else:
            # Shape [*add_dim, num_dof * num_times]
            vel_std = torch.sqrt(torch.einsum('...ii->...i', vel_cov))

            if not flat_shape:
                # Reshape to [*add_dim, num_dof, num_times]
                vel_std = vel_std.reshape(*self.add_dim, self.num_dof, -1)

                # Switch axes to [*add_dim, num_times, num_dof]
                vel_std = torch.einsum('...ji->...ij', vel_std)

        self.vel_std = vel_std
        return self.vel_std

    def learn_mp_params_from_trajs(self, times, trajs, reg=1e-9) -> dict:
        """
        Learn DMP weights and goal given trajectory position
        Use the initial position and velocity as boundary condition

        Args:
            times: trajectory time points
            trajs: trajectory position in batch
            reg: regularization term

        Returns:
            param_dict: dictionary of parameters containing
                - params (w + g)
                - bc_time
                - bc_pos
                - bc_vel
        """
        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of trajs:
        # [*add_dim, num_times, num_dof]

        # Assert trajs shape
        assert trajs.shape[:-1] == times.shape
        assert trajs.shape[-1] == self.num_dof

        trajs = torch.Tensor(trajs)

        # Get boundary conditions
        bc_time = times[..., 0]
        bc_pos = trajs[..., 0, :]
        bc_vel = torch.diff(trajs, dim=-2)[..., 0, :] / self.dt

        # Setup stuff
        self.set_add_dim(list(trajs.shape[:-2]))
        self.set_mp_times(times)
        self.set_boundary_conditions(bc_time, bc_pos, bc_vel)

        # Solve this: Aw = B -> w = A^{-1} B
        # Einsum_shape: [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        #               [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        #            -> [*add_dim, num_basis_g * num_dof, num_basis_g * num_dof]
        A = torch.einsum('...ki,...kj->...ij', self.pos_vary_, self.pos_vary_)
        A += torch.eye(self.num_params) * reg

        # Swap axis and reshape: [*add_dim, num_times, num_dof]
        #                     -> [*add_dim, num_dof, num_times]
        trajs = torch.einsum("...ij->...ji", trajs)

        # Reshape [*add_dim, num_dof, num_times]
        #      -> [*add_dim, num_dof * num_times]
        trajs = trajs.reshape([*self.add_dim, -1])

        # Position minus boundary condition terms,
        pos_wg = trajs - self.pos_det

        # Einsum_shape: [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        #               [*add_dim, num_dof * num_times]
        #            -> [*add_dim, num_basis_g * num_dof]
        B = torch.einsum('...ki,...k->...i', self.pos_vary_, pos_wg)

        # Shape of weights: [*add_dim, num_params=num_basis_g * num_dof]
        params = torch.linalg.solve(A, B)

        return {"params": params,
                "bc_time": bc_time,
                "bc_pos": bc_pos,
                "bc_vel": bc_vel}

    def times_to_indices(self, times):
        """
        Map time points to time indices
        Args:
            times: time points

        Returns:
            time indices
        """
        return torch.round(times / self.dt).long()

    def indices_to_times(self, time_indices):
        """
        Map time indices to time points
        Args:
            time_indices: time indices

        Returns:
            time points
        """
        return time_indices * self.dt

    def _pre_compute(self):
        """
        Pre-compute the integral form basis

        Returns:
            None

        """
        # Shape of pc_times
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

        # y_1 and y_2
        self.y_1_value = torch.exp(-0.5 * self.alpha / self.tau * self.pc_times)
        self.y_2_value = self.pc_times * self.y_1_value

        self.dy_1_value = -0.5 * self.alpha / self.tau * self.y_1_value
        self.dy_2_value = -0.5 * self.alpha / self.tau * self.y_2_value \
                          + self.y_1_value

        # q_1 and q_2
        q_1_value = \
            (0.5 * self.alpha / self.tau * self.pc_times - 1) \
            * torch.exp(0.5 * self.alpha / self.tau * self.pc_times) + 1
        q_2_value = \
            0.5 * self.alpha / self.tau \
            * (torch.exp(0.5 * self.alpha / self.tau * self.pc_times) - 1)

        # p_1 and p_2
        # Get basis of one DOF, shape [num_pc_times, num_basis]
        basis_single_dof = self.basis_gn.basis(self.pc_times)
        assert list(basis_single_dof.shape) == [*self.pc_times.shape,
                                                self.num_basis]

        dp_1_value = \
            torch.einsum('...i,...ij->...ij',
                         self.pc_times / self.tau ** 2
                         * torch.exp(self.alpha * self.pc_times / self.tau / 2),
                         basis_single_dof)
        dp_2_value = \
            torch.einsum('...i,...ij->...ij',
                         1 / self.tau ** 2
                         * torch.exp(self.alpha * self.pc_times / self.tau / 2),
                         basis_single_dof)

        p_1_value = torch.zeros(size=dp_1_value.shape)
        p_2_value = torch.zeros(size=dp_2_value.shape)

        for i in range(self.pc_times.shape[0]):
            p_1_value[i] = \
                torch.trapz(dp_1_value[:i + 1], self.pc_times[:i + 1], dim=0)
            p_2_value[i] = \
                torch.trapz(dp_2_value[:i + 1], self.pc_times[:i + 1], dim=0)

        # Compute integral form basis values
        pos_basis_w = p_2_value * self.y_2_value[:, None] \
                      - p_1_value * self.y_1_value[:, None]
        pos_basis_g = q_2_value * self.y_2_value \
                      - q_1_value * self.y_1_value
        vel_basis_w = p_2_value * self.dy_2_value[:, None] \
                      - p_1_value * self.dy_1_value[:, None]
        vel_basis_g = q_2_value * self.dy_2_value \
                      - q_1_value * self.dy_1_value
        self.pos_basis = torch.cat([pos_basis_w, pos_basis_g[:, None]], dim=-1)
        self.vel_basis = torch.cat([vel_basis_w, vel_basis_g[:, None]], dim=-1)

    def compute_intermediate_shared_variables(self):
        """
        Evaluate boundary condition values of the pre-computed terms, as well as
        the formed up position and velocity variables

        Returns:
            None
        """

        # Shape of bc_index: [*add_dim]
        bc_index = self.times_to_indices(self.bc_time)

        # Shape of time_indices: [*add_dim, num_times] or None
        if self.times is not None:
            time_indices = self.times_to_indices(self.times)
        else:
            time_indices = None

        # Given index, extract boundary condition values
        # Shape [num_pc_times] -> [*add_dim]
        y_1_bc = self.y_1_value[bc_index]
        y_2_bc = self.y_2_value[bc_index]
        dy_1_bc = self.dy_1_value[bc_index]
        dy_2_bc = self.dy_2_value[bc_index]

        # Shape [num_pc_times, num_basis_g] -> [*add_dim, num_basis_g]
        pos_basis_bc = self.pos_basis[bc_index, :]
        vel_basis_bc = self.vel_basis[bc_index, :]

        # Determinant of boundary condition,
        # Shape: [*add_dim]
        det = y_1_bc * dy_2_bc - y_2_bc * dy_1_bc

        if time_indices is not None:
            num_times = time_indices.shape[-1]
            time_indices = time_indices.long()

            # Get pre-computed values at desired time indices
            # Shape [num_pc_times] -> [*add_dim, num_times]
            y_1_value = self.y_1_value[time_indices]
            y_2_value = self.y_2_value[time_indices]
            dy_1_value = self.dy_1_value[time_indices]
            dy_2_value = self.dy_2_value[time_indices]

            # Shape [num_pc_times, num_basis_g]
            #    -> [*add_dim, num_times, num_basis_g]
            pos_basis = self.pos_basis[time_indices, :]
            vel_basis = self.vel_basis[time_indices, :]

            # Einstein summation convention string
            einsum_eq = "...,...i->...i"

        else:
            # Use all time indices
            num_times = self.num_pc_times
            y_1_value = self.y_1_value
            y_2_value = self.y_2_value
            dy_1_value = self.dy_1_value
            dy_2_value = self.dy_2_value
            pos_basis = self.pos_basis
            vel_basis = self.vel_basis

            # Einstein summation convention string
            einsum_eq = "...,i->...i"

        # Compute coefficients to form up traj position and velocity
        # If use all time indices:
        # Shape: [*add_dim], [num_times] -> [*add_dim, num_times]
        # Else:
        # Shape: [*add_dim], [*add_dim, num_times] -> [*add_dim, num_times]
        self.xi_1 = torch.einsum(einsum_eq, dy_2_bc / det, y_1_value) \
                    - torch.einsum(einsum_eq, dy_1_bc / det, y_2_value)
        self.xi_2 = torch.einsum(einsum_eq, y_1_bc / det, y_2_value) \
                    - torch.einsum(einsum_eq, y_2_bc / det, y_1_value)
        self.xi_3 = torch.einsum(einsum_eq, dy_1_bc / det, y_2_value) \
                    - torch.einsum(einsum_eq, dy_2_bc / det, y_1_value)
        self.xi_4 = torch.einsum(einsum_eq, y_2_bc / det, y_1_value) \
                    - torch.einsum(einsum_eq, y_1_bc / det, y_2_value)
        self.dxi_1 = torch.einsum(einsum_eq, dy_2_bc / det, dy_1_value) \
                     - torch.einsum(einsum_eq, dy_1_bc / det, dy_2_value)
        self.dxi_2 = torch.einsum(einsum_eq, y_1_bc / det, dy_2_value) \
                     - torch.einsum(einsum_eq, y_2_bc / det, dy_1_value)
        self.dxi_3 = torch.einsum(einsum_eq, dy_1_bc / det, dy_2_value) \
                     - torch.einsum(einsum_eq, dy_2_bc / det, dy_1_value)
        self.dxi_4 = torch.einsum(einsum_eq, y_2_bc / det, dy_1_value) \
                     - torch.einsum(einsum_eq, y_1_bc / det, dy_2_value)

        # Generate blocked basis boundary condition
        # Shape: [*add_dim, num_basis_g] ->
        # [*add_dim, num_dof, num_dof * num_basis_g]
        pos_basis_bc_multi_dofs = \
            torch.zeros((*self.add_dim, self.num_dof, self.num_params))
        vel_basis_bc_multi_dofs = \
            torch.zeros((*self.add_dim, self.num_dof, self.num_params))

        # Generated blocked basis
        # Shape: [*add_dim, num_times, num_basis_g] ->
        # [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        pos_basis_multi_dofs = \
            torch.zeros(*self.add_dim, num_times * self.num_dof,
                        self.num_params)
        vel_basis_multi_dofs = \
            torch.zeros(*self.add_dim, num_times * self.num_dof,
                        self.num_params)

        for i in range(self.num_dof):
            row_indices = slice(i * num_times,
                                (i + 1) * num_times)
            col_indices = slice(i * self.num_basis_g,
                                (i + 1) * self.num_basis_g)
            pos_basis_bc_multi_dofs[..., i, col_indices] = pos_basis_bc
            vel_basis_bc_multi_dofs[..., i, col_indices] = vel_basis_bc
            pos_basis_multi_dofs[..., row_indices, col_indices] = pos_basis
            vel_basis_multi_dofs[..., row_indices, col_indices] = vel_basis

        self.pos_basis_bc_multi_dofs = pos_basis_bc_multi_dofs
        self.vel_basis_bc_multi_dofs = vel_basis_bc_multi_dofs
        self.pos_basis_multi_dofs = pos_basis_multi_dofs
        self.vel_basis_multi_dofs = vel_basis_multi_dofs

        # Compute position and velocity determined part (part 1 and 2)
        # Position and velocity part 1 and part 2
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_dof]
        #            -> [*add_dim, num_dof, num_times]
        pos_det = torch.einsum('...j,...i->...ij', self.xi_1, self.bc_pos) \
                  + torch.einsum('...j,...i->...ij', self.xi_2, self.bc_vel)
        vel_det = torch.einsum('...j,...i->...ij', self.dxi_1, self.bc_pos) \
                  + torch.einsum('...j,...i->...ij', self.dxi_2, self.bc_vel)

        # Reshape: [*add_dim, num_dof, num_times]
        #       -> [*add_dim, num_dof * num_times]
        self.pos_det = torch.reshape(pos_det, [*self.add_dim, -1])
        self.vel_det = torch.reshape(vel_det, [*self.add_dim, -1])

        # Compute position and velocity variant part (part 3)
        # Position and velocity part 3_1 and 3_2
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_dof, num_basis_g * num_dof]
        #            -> [*add_dim, num_dof, num_times, num_basis_g * num_dof]
        pos_vary_ = torch.einsum('...j,...ik->...ijk', self.xi_3,
                                 self.pos_basis_bc_multi_dofs) + \
                    torch.einsum('...j,...ik->...ijk', self.xi_4,
                                 self.vel_basis_bc_multi_dofs)
        vel_vary_ = torch.einsum('...j,...ik->...ijk', self.dxi_3,
                                 self.pos_basis_bc_multi_dofs) + \
                    torch.einsum('...j,...ik->...ijk', self.dxi_4,
                                 self.vel_basis_bc_multi_dofs)

        # Reshape: [*add_dim, num_dof, num_times, num_basis_g * num_dof]
        #       -> [*add_dim, num_dof * num_times, num_basis_g * num_dof]
        pos_vary_ = \
            torch.reshape(pos_vary_, [*self.add_dim, -1, self.num_params])
        vel_vary_ = \
            torch.reshape(vel_vary_, [*self.add_dim, -1, self.num_params])

        self.pos_vary_ = pos_vary_ + self.pos_basis_multi_dofs
        self.vel_vary_ = vel_vary_ + self.vel_basis_multi_dofs


class MPFactory:
    @staticmethod
    def init_mp(config):
        """
        Create a MP instance given configuration

        Args:
            config: config dict

        Returns:
            MP instance
        """
        num_dof = config["num_dof"]
        mp_type = config["mp"]["type"]
        mp_config = config["mp"]["args"]
        duration = mp_config["tau"]

        # Get phase generator
        if mp_type == "promp":
            phase_gn = LinearPhaseGenerator(1 / duration)
        elif mp_type == "dmp" or mp_type == "idmp":
            phase_gn = ExpDecayPhaseGenerator(duration,
                                              mp_config["alpha_phase"])
        else:
            raise NotImplementedError

        # Get basis generator and mp class
        basis_gn_class = MPFactory.get_basis_gn_class(mp_type)
        mp_class = MPFactory.get_mp_class(mp_type=mp_type)

        # Initialize basis generator and mp
        basis_gn = basis_gn_class(
            phase_generator=phase_gn,
            num_basis=mp_config["num_basis"],
            duration=duration,
            basis_bandwidth_factor=mp_config["basis_bandwidth_factor"],
            num_basis_outside=mp_config["num_basis_outside"])
        mp: MPInterface = mp_class(basis_gn=basis_gn, num_dof=num_dof,
                                   **mp_config)
        return mp

    @staticmethod
    def get_mp_class(mp_type):
        mp_d = {"promp": ProMP,
                "dmp": DMP,
                "idmp": IDMP}
        return mp_d[mp_type]

    @staticmethod
    def get_basis_gn_class(mp_type):
        basis_gn_d = {"promp": NormalizedRBFBasisGenerator,
                      "dmp": DMPBasisGenerator,
                      "idmp": DMPBasisGenerator}
        return basis_gn_d[mp_type]
