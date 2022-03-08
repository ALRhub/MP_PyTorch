"""
@brief:     Movement Primitives interfaces in PyTorch
"""
from abc import ABC
from abc import abstractmethod
from typing import Union

import torch
from torch.distributions import MultivariateNormal

from mp_pytorch.basis_generator import BasisGenerator
import mp_pytorch.util as util


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

    @property
    def tau(self):
        return self.basis_gn.tau

    def clear_computation_result(self):
        """
        Clear runtime computation result

        Returns:
            None
        """

        self.pos = None
        self.vel = None

    def set_add_dim(self, add_dim: Union[list, torch.Size]):
        """
        Set additional batch dimension
        Args:
            add_dim: additional batch dimension

        Returns: None

        """
        self.add_dim = add_dim
        self.clear_computation_result()

    def set_mp_times(self, times: torch.Tensor):
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

    def set_params(self, params: torch.Tensor):
        """
        Set MP params
        Args:
            params: parameters

        Returns: None

        """

        # Shape of params
        # [*add_dim, total_num_params]

        num_basis_params = self.basis_gn.total_num_params
        if num_basis_params > 0:
            self.basis_gn.set_params(params[..., :num_basis_params])

        self.params = params[..., num_basis_params:]
        self.clear_computation_result()

    def get_params(self) -> torch.Tensor:
        """
        Return all learnable parameters
        Returns:
            parameters
        """
        # Shape of params
        # [*add_dim, total_num_params]
        params = self.basis_gn.get_params()
        params = torch.cat([params, self.params], dim=-1)
        return params

    def set_boundary_conditions(self, bc_time: torch.Tensor,
                                bc_pos: torch.Tensor,
                                bc_vel: torch.Tensor):
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
            params: parameters
            bc_time: boundary condition time
            bc_pos: boundary condition position
            bc_vel: boundary condition velocity
            kwargs: other keyword arguments

        Returns: None

        """
        if params is not None:
            self.set_params(params)
        if times is not None:
            self.set_mp_times(times)
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
            result["pos"] = self.get_traj_pos()
        else:
            result["pos"] = None

        # Velocity
        if get_vel:
            result["vel"] = self.get_traj_vel()
        else:
            result["vel"] = None

        # Return
        return result

    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        pass

    @property
    def total_num_params(self) -> int:
        """
        Returns: number of parameters of current class plus parameters of all
        attributes
        """
        return self.num_params + self.basis_gn.total_num_params

    @abstractmethod
    def get_traj_pos(self, times=None, params=None,
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
    def get_traj_vel(self, times=None, params=None,
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
    def learn_mp_params_from_trajs(self, times: torch.Tensor,
                                   trajs: torch.Tensor,
                                   reg=1e-9):
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

    def set_mp_params_variances(self, params_L: Union[torch.Tensor, None]):
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
            params: parameters
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
            flat_shape: if flatten the dimensions of Dof and time
            reg: regularization term

        Returns:
            results in dictionary
        """
        # Initialize result dictionary
        result = dict()

        # pos
        if get_pos:
            result["pos"] = self.get_traj_pos(flat_shape=flat_shape)
        else:
            result["pos"] = None

        # vel
        if get_vel:
            result["vel"] = self.get_traj_vel(flat_shape=flat_shape)
        else:
            result["vel"] = None

        # pos_cov
        if get_pos_cov:
            result["pos_cov"] = self.get_traj_pos_cov(reg=reg)
        else:
            result["pos_cov"] = None

        # pos_std
        if get_pos_std:
            result["pos_std"] = self.get_traj_pos_std(flat_shape=flat_shape,
                                                      reg=reg)
        else:
            result["pos_std"] = None

        # vel_cov
        if get_vel_cov:
            result["vel_cov"] = self.get_traj_vel_cov(reg=reg)
        else:
            result["vel_cov"] = None

        # vel_std
        if get_vel_std:
            result["vel_std"] = self.get_traj_vel_std(flat_shape=flat_shape,
                                                      reg=reg)
        else:
            result["vel_std"] = None

        return result

    @abstractmethod
    def get_traj_pos(self, times=None, params=None,
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
            flat_shape: if flatten the dimensions of Dof and time

        Returns:
            pos
        """
        pass

    @abstractmethod
    def get_traj_pos_cov(self, times=None, params_L=None,
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
    def get_traj_pos_std(self, times=None, params_L=None,
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
            flat_shape: if flatten the dimensions of Dof and time
            reg: regularization term

        Returns:
            pos std
        """
        pass

    @abstractmethod
    def get_traj_vel(self, times=None, params=None,
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
            flat_shape: if flatten the dimensions of Dof and time

        Returns:
            vel
        """
        pass

    @abstractmethod
    def get_traj_vel_cov(self, times=None, params_L=None,
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
    def get_traj_vel_std(self, times=None, params_L=None,
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
            flat_shape: if flatten the dimensions of Dof and time
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
            flat_shape: if flatten the dimensions of Dof and time

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

        params_super = self.basis_gn.get_params()
        if params_super.nelement() != 0:
            params_super = util.add_expand_dim(params_super,
                                               [-2], [num_smp])
            params_smp = torch.cat([params_super, params_smp], dim=-1)

        # Add additional sample axis to boundary condition
        bc_time = util.add_expand_dim(bc_time, [num_add_dim], [num_smp])
        bc_pos = util.add_expand_dim(bc_pos, [num_add_dim], [num_smp])
        bc_vel = util.add_expand_dim(bc_vel, [num_add_dim], [num_smp])

        # Update inputs
        self.update_mp_inputs(times, params_smp, None, bc_time, bc_pos, bc_vel)

        # Get sample trajectories
        pos_smp = self.get_traj_pos(flat_shape=flat_shape)

        return pos_smp
