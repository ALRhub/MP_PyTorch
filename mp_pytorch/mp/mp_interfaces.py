"""
@brief:     Movement Primitives interfaces in PyTorch
"""
import copy
from abc import ABC
from abc import abstractmethod
from typing import Iterable
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch.distributions import MultivariateNormal

import mp_pytorch.util as util
from mp_pytorch.basis_gn import BasisGenerator


class MPInterface(ABC):
    @abstractmethod
    def __init__(self,
                 basis_gn: BasisGenerator,
                 num_dof: int,
                 weights_scale: Union[float, Iterable] = 1.,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 **kwargs):
        """
        Constructor interface
        Args:
            basis_gn: basis generator
            num_dof: number of dof
            weights_scale: scaling for the parameters weights
            dtype: torch.dtype = torch.float32,
            device: torch.device = 'cpu',
            **kwargs: keyword arguments
        """
        self.dtype = dtype
        self.device = device

        # Additional batch dimension
        self.add_dim = list()

        # The basis generators
        self.basis_gn = basis_gn

        # Number of DoFs
        self.num_dof = num_dof

        # Scaling of weights
        self.weights_scale = \
            torch.as_tensor(weights_scale, dtype=self.dtype, device=self.device)
        assert self.weights_scale.ndim <= 1, \
            "weights_scale should be float or 1-dim vector"

        # Value caches
        # Compute values at these time points
        self.times = None

        # Learnable parameters
        self.params = None

        # Initial conditions
        self.init_time = None
        self.init_pos = None
        self.init_vel = None

        # Runtime computation results, shall be reset every time when
        # inputs are reset
        self.pos = None
        self.vel = None

        # Flag of if the MP instance is finalized
        self.is_finalized = False

        # Local parameters bound
        self.local_params_bound = kwargs.get("params_bound", None)
        if not self.local_params_bound:
            self.local_params_bound = torch.zeros([2, self._num_local_params],
                                                  dtype=self.dtype,
                                                  device=self.device)
            self.local_params_bound[0, :] = -torch.inf
            self.local_params_bound[1, :] = torch.inf
        else:
            self.local_params_bound = torch.as_tensor(self.local_params_bound,
                                                      dtype=self.dtype,
                                                      device=self.device)
        assert list(self.local_params_bound.shape) == [2,
                                                       self._num_local_params]

    @property
    def learn_tau(self):
        return self.phase_gn.learn_tau

    @property
    def learn_delay(self):
        return self.phase_gn.learn_delay

    @property
    def tau(self):
        return self.phase_gn.tau

    @property
    def num_basis(self):
        return self.basis_gn.num_basis

    @property
    def phase_gn(self):
        return self.basis_gn.phase_generator

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

    def set_times(self, times: Union[torch.Tensor, np.ndarray]):
        """
        Set MP time points
        Args:
            times: desired time points

        Returns:
            None
        """

        # Shape of times
        # [*add_dim, num_times]

        self.times = torch.as_tensor(times, dtype=self.dtype,
                                     device=self.device)
        self.clear_computation_result()

    def set_duration(self, duration: Optional[float], dt: float,
                     include_init_time: bool = False):
        """
        Set MP time points of a duration. The times start from init_time or 0

        Args:
            duration: desired duration of trajectory
            dt: control frequency
            include_init_time: if the duration includes the bc time step.
        Returns:
            None
        """

        # Shape of times
        # [*add_dim, num_times]

        if duration is None:
            duration = round(self.tau.max().item() / dt) * dt

        # dt = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        times = torch.linspace(0, duration, round(duration / dt) + 1,
                               dtype=self.dtype, device=self.device)
        times = util.add_expand_dim(times, list(range(len(self.add_dim))),
                                    self.add_dim)

        if self.init_time is not None:
            times = times + self.init_time[..., None]
        if include_init_time:
            self.set_times(times)
        else:
            self.set_times(times[..., 1:])

    def set_params(self,
                   params: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Set MP params
        Args:
            params: parameters

        Returns: unused parameters

        """

        # Shape of params
        # [*add_dim, num_params]

        params = torch.as_tensor(params, dtype=self.dtype, device=self.device)

        # Check number of params
        assert params.shape[-1] == self.num_params

        # Set additional batch size
        self.set_add_dim(list(params.shape[:-1]))

        remaining_params = self.basis_gn.set_params(params)
        self.params = remaining_params[..., :self._num_local_params]
        self.clear_computation_result()
        return remaining_params[..., self._num_local_params:]

    def set_initial_conditions(self, init_time: Union[torch.Tensor, np.ndarray],
                                init_pos: Union[torch.Tensor, np.ndarray],
                                init_vel: Union[torch.Tensor, np.ndarray]):
        """
        Set initial conditions in a batched manner

        Args:
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity

        Returns:
            None
        """

        # Shape of init_time:
        # [*add_dim]
        #
        # Shape of init_pos:
        # [*add_dim, num_dof]
        #
        # Shape of init_vel:
        # [*add_dim, num_dof]

        self.init_time = torch.as_tensor(init_time, dtype=self.dtype,
                                       device=self.device)
        self.init_pos = torch.as_tensor(init_pos, dtype=self.dtype,
                                      device=self.device)
        init_vel = torch.as_tensor(init_vel, dtype=self.dtype, device=self.device)

        # If velocity is non-zero, then cannot wait
        if torch.count_nonzero(init_vel) != 0:
            assert torch.all(self.init_time - self.phase_gn.delay >= 0), \
                f"Cannot set non-zero initial velocity {init_vel} if initial condition time" \
                f"value(s) {self.init_time} is (are) smaller than delay value(s) {self.phase_gn.delay}"
        self.init_vel = init_vel
        self.clear_computation_result()

    def update_inputs(self, times=None, params=None,
                      init_time=None, init_pos=None, init_vel=None, **kwargs):
        """
        Update MP
        Args:
            times: desired time points
            params: parameters
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            kwargs: other keyword arguments

        Returns: None

        """
        if params is not None:
            self.set_params(params)
        if times is not None:
            self.set_times(times)
        if all([data is not None for data in {init_time, init_pos, init_vel}]):
            self.set_initial_conditions(init_time, init_pos, init_vel)

    def get_params(self) -> torch.Tensor:
        """
        Return all learnable parameters
        Returns:
            parameters
        """
        # Shape of params
        # [*add_dim, num_params]
        params = self.basis_gn.get_params()
        params = torch.cat([params, self.params], dim=-1)
        return params

    def get_params_bounds(self) -> torch.Tensor:
        """
        Return all learnable parameters' bounds
        Returns:
            parameters bounds
        """
        # Shape of params_bounds
        # [2, num_params]

        params_bounds = self.basis_gn.get_params_bounds()
        params_bounds = torch.cat([params_bounds, self.local_params_bound],
                                  dim=1)
        return params_bounds

    def get_trajs(self, get_pos: bool = True, get_vel: bool = True) -> dict:
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
        result["pos"] = self.get_traj_pos() if get_pos else None

        # Velocity
        result["vel"] = self.get_traj_vel() if get_vel else None

        # Return
        return result

    @property
    def _num_local_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        return self.num_basis * self.num_dof

    @property
    def num_params(self) -> int:
        """
        Returns: number of parameters of current class plus parameters of all
        attributes
        """
        return self._num_local_params + self.basis_gn.num_params

    @abstractmethod
    def get_traj_pos(self, times=None, params=None,
                     init_time=None, init_pos=None, init_vel=None):
        """
        Get trajectory position
        Args:
            times: time points
            params: learnable parameters
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity

        Returns:
            pos
        """
        pass

    @abstractmethod
    def get_traj_vel(self, times=None, params=None,
                     init_time=None, init_pos=None, init_vel=None):
        """
        Get trajectory velocity

        Args:
            times: time points
            params: learnable parameters
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity

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

    def finalize(self):
        """
        Mark the MP as finalized so that the parameters cannot be
        updated any more
        Returns: None

        """
        self.is_finalized = True

    def reset(self):
        """
        Unmark the finalization
        Returns: None

        """
        self.basis_gn.reset()
        self.is_finalized = False

    @abstractmethod
    def _show_scaled_basis(self, *args, **kwargs):
        pass

    def show_scaled_basis(self, plot=False):
        """
        External call of show basis, it will make a hard copy of the current mp,
        and feed artificial time sequence.

        The current mp will not get influenced.

        Args:
            plot: if to plot the basis

        Returns:

        """
        # Make a hard copy to show basis and do not change other settings of the
        # original mp instance
        try:
            copied_mp = copy.deepcopy(self)
        except RuntimeError:
            print("Please do not use this function during NN training. "
                  "The deepcopy cannot work when there is a computation graph.")
            return
        return copied_mp._show_scaled_basis(plot)


class ProbabilisticMPInterface(MPInterface):
    def __init__(self,
                 basis_gn: BasisGenerator,
                 num_dof: int,
                 weights_scale: float = 1.,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 **kwargs):
        """
        Constructor interface
        Args:
            basis_gn: basis generator
            num_dof: number of dof
            weights_scale: scaling for the parameters weights
            dtype: torch data type
            device: torch device to run on
            **kwargs: keyword arguments
        """

        super().__init__(basis_gn, num_dof, weights_scale, dtype, device,
                         **kwargs)

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

    def set_mp_params_variances(self,
                                params_L: Union[
                                    torch.Tensor, None, np.ndarray]):
        """
        Set variance of MP params
        Args:
            params_L: cholesky of covariance matrix of the MP parameters

        Returns: None

        """
        # Shape of params_L
        # [*add_dim, num_mp_params, num_mp_params]

        self.params_L = torch.as_tensor(
            params_L) if params_L is not None else params_L
        self.clear_computation_result()

    def update_inputs(self, times=None, params=None, params_L=None,
                      init_time=None, init_pos=None, init_vel=None, **kwargs):
        """
        Set MP
        Args:
            times: desired time points
            params: parameters
            params_L: cholesky of covariance matrix of the MP parameters
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            kwargs: other keyword arguments

        Returns: None

        """
        super().update_inputs(times, params, init_time, init_pos, init_vel)
        if params_L is not None:
            self.set_mp_params_variances(params_L)

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

    def get_trajs(self, get_pos=True, get_pos_cov=True, get_pos_std=True,
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
        result["pos"] = self.get_traj_pos(
            flat_shape=flat_shape) if get_pos else None

        # vel
        result["vel"] = self.get_traj_vel(
            flat_shape=flat_shape) if get_vel else None

        # pos_cov
        result["pos_cov"] = self.get_traj_pos_cov(
            reg=reg) if get_pos_cov else None

        # pos_std
        result["pos_std"] = self.get_traj_pos_std(flat_shape=flat_shape,
                                                  reg=reg) if get_pos_std else None

        # vel_cov
        result["vel_cov"] = self.get_traj_vel_cov(
            reg=reg) if get_vel_cov else None

        # vel_std
        result["vel_std"] = self.get_traj_vel_std(flat_shape=flat_shape,
                                                  reg=reg) if get_vel_std else None

        return result

    @abstractmethod
    def get_traj_pos(self, times=None, params=None,
                     init_time=None, init_pos=None, init_vel=None,
                     flat_shape=False):
        """
        Get trajectory position
        Args:
            times: time points
            params: learnable parameters
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            flat_shape: if flatten the dimensions of Dof and time

        Returns:
            pos
        """
        pass

    @abstractmethod
    def get_traj_pos_cov(self, times=None, params_L=None,
                         init_time=None, init_pos=None, init_vel=None,
                         reg: float = 1e-4):
        """
        Get trajectory covariance
        Returns: cov

        Args:
            times: time points
            params_L: learnable parameters' variance
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            reg: regularization term

        Returns:
            pos cov
        """
        pass

    @abstractmethod
    def get_traj_pos_std(self, times=None, params_L=None,
                         init_time=None, init_pos=None, init_vel=None,
                         flat_shape=False, reg: float = 1e-4):
        """
        Get trajectory standard deviation
        Args:
            times: time points
            params_L: learnable parameters' variance
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            flat_shape: if flatten the dimensions of Dof and time
            reg: regularization term

        Returns:
            pos std
        """
        pass

    @abstractmethod
    def get_traj_vel(self, times=None, params=None,
                     init_time=None, init_pos=None, init_vel=None,
                     flat_shape=False):
        """
        Get trajectory velocity
        Returns: vel

        Args:
            times: time points
            params: learnable parameters
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            flat_shape: if flatten the dimensions of Dof and time

        Returns:
            vel
        """
        pass

    @abstractmethod
    def get_traj_vel_cov(self, times=None, params_L=None,
                         init_time=None, init_pos=None, init_vel=None,
                         reg: float = 1e-4):
        """
        Get trajectory covariance
        Args:
            times: time points
            params_L: learnable parameters' variance
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            reg: regularization term

        Returns:
            vel cov
        """
        pass

    @abstractmethod
    def get_traj_vel_std(self, times=None, params_L=None,
                         init_time=None, init_pos=None, init_vel=None,
                         flat_shape=False, reg: float = 1e-4):
        """
        Get trajectory standard deviation
        Args:
            times: time points
            params_L: learnable parameters' variance
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            flat_shape: if flatten the dimensions of Dof and time
            reg: regularization term

        Returns:
            vel std
        """
        pass

    def sample_trajectories(self, times=None, params=None, params_L=None,
                            init_time=None, init_pos=None, init_vel=None,
                            num_smp=1, flat_shape=False):
        """
        Sample trajectories from MP

        Args:
            times: time points
            params: learnable parameters
            params_L: learnable parameters' variance
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            num_smp: num of trajectories to be sampled
            flat_shape: if flatten the dimensions of Dof and time

        Returns:
            sampled trajectories
        """

        # Shape of pos_smp
        # [*add_dim, num_smp, num_times, num_dof]
        # or [*add_dim, num_smp, num_dof * num_times]

        if all([data is None for data in {times, params, params_L, init_time,
                                          init_pos, init_vel}]):
            times = self.times
            params = self.params
            params_L = self.params_L
            init_time = self.init_time
            init_pos = self.init_pos
            init_vel = self.init_vel

        num_add_dim = params.ndim - 1

        # Add additional sample axis to time
        # Shape [*add_dim, num_smp, num_times]
        times_smp = util.add_expand_dim(times, [num_add_dim], [num_smp])

        # Sample parameters, shape [num_smp, *add_dim, num_mp_params]
        params_smp = MultivariateNormal(loc=params,
                                        scale_tril=params_L,
                                        validate_args=False).rsample([num_smp])

        # Switch axes to [*add_dim, num_smp, num_mp_params]
        params_smp = torch.einsum('i...j->...ij', params_smp)

        params_super = self.basis_gn.get_params()
        if params_super.nelement() != 0:
            params_super_smp = util.add_expand_dim(params_super, [-2],
                                                   [num_smp])
            params_smp = torch.cat([params_super_smp, params_smp], dim=-1)

        # Add additional sample axis to initial condition
        if init_time is not None:
            init_time_smp = util.add_expand_dim(init_time, [num_add_dim], [num_smp])
            init_pos_smp = util.add_expand_dim(init_pos, [num_add_dim], [num_smp])
            init_vel_smp = util.add_expand_dim(init_vel, [num_add_dim], [num_smp])
        else:
            init_time_smp = None
            init_pos_smp = None
            init_vel_smp = None

        # Update inputs
        self.reset()
        self.update_inputs(times_smp, params_smp, None,
                           init_time_smp, init_pos_smp, init_vel_smp)

        # Get sample trajectories
        pos_smp = self.get_traj_pos(flat_shape=flat_shape)
        vel_smp = self.get_traj_vel(flat_shape=flat_shape)

        # Recover old inputs
        if params_super.nelement() != 0:
            params = torch.cat([params_super, params], dim=-1)
        self.reset()
        self.update_inputs(times, params, None, init_time, init_pos, init_vel)

        return pos_smp, vel_smp
