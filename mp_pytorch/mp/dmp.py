"""
@brief:     Dynamic Movement Primitives in PyTorch
"""
from typing import Iterable
from typing import Union
from typing import Tuple

import numpy as np
import torch
from mp_pytorch.util import to_nps
from mp_pytorch.basis_gn import BasisGenerator
from .mp_interfaces import MPInterface


class DMP(MPInterface):
    """DMP in PyTorch"""

    def __init__(self,
                 basis_gn: BasisGenerator,
                 num_dof: int,
                 weights_scale: Union[float, Iterable] = 1.,
                 goal_scale: float = 1.,
                 alpha: float = 25,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 **kwargs):
        """
        Constructor of DMP
        Args:
            basis_gn: basis function value generator
            num_dof: number of Degrees of Freedoms
            weights_scale: scaling for the parameters weights
            goal_scale: scaling for the goal
            dtype: torch data type
            device: torch device to run on
            kwargs: keyword arguments
        """

        super().__init__(basis_gn, num_dof, weights_scale, dtype, device,
                         **kwargs)

        # Number of parameters
        self.num_basis_g = self.num_basis + 1

        # Control parameters
        self.alpha = alpha
        self.beta = self.alpha / 4

        # Goal scale
        self.goal_scale = goal_scale
        self.weights_goal_scale = self.get_weights_goal_scale()

    @property
    def _num_local_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        return super()._num_local_params + self.num_dof

    def get_weights_goal_scale(self) -> torch.Tensor:
        """
        Returns: the weights and goal scaling vector
        """
        w_g_scale = torch.zeros(self.num_basis_g)
        w_g_scale[:-1] = self.weights_scale
        w_g_scale[-1] = self.goal_scale
        return w_g_scale

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

        init_time = torch.as_tensor(init_time, dtype=self.dtype, device=self.device)
        init_pos = torch.as_tensor(init_pos, dtype=self.dtype, device=self.device)
        init_vel = torch.as_tensor(init_vel, dtype=self.dtype, device=self.device)

        assert list(init_time.shape) == [*self.add_dim], \
            f"shape of initial condition time {list(init_time.shape)} " \
            f"does not match batch dimension {[*self.add_dim]}"
        assert list(init_pos.shape) == list(init_vel.shape) \
               and list(init_vel.shape) == [*self.add_dim, self.num_dof], \
            f"shape of initial condition position {list(init_pos.shape)} " \
            f"and initial condition velocity do not match {list(init_vel.shape)}"
        super().set_initial_conditions(init_time, init_pos, init_vel)

    def get_traj_pos(self, times=None, params=None,
                     init_time=None, init_pos=None, init_vel=None):
        """
        Compute trajectories at desired time points

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params: learnable parameters
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity

        Returns:
            pos
        """

        # Shape of pos
        # [*add_dim, num_times, num_dof]

        # Update inputs
        self.update_inputs(times, params, init_time, init_pos, init_vel)

        # Reuse result if existing
        if self.pos is not None:
            return self.pos

        # Check initial condition, the desired times should start from
        # initial condition time steps or plus dt
        if not torch.allclose(self.init_time, self.times[..., 0]):
            assert torch.allclose(self.times[..., 1] + self.init_time, 2 * self.times[..., 0]), \
                f"The start time value {self.times[..., 1]} should be either init_time {self.init_time} or init_time + dt."
            times_include_init = torch.cat([self.init_time[..., None], self.times], dim=-1)

            # Recursively call itself
            self.get_traj_pos(times_include_init)

            # Remove the init_time from the result
            self.pos = self.pos[..., 1:, :]
            self.vel = self.vel[..., 1:, :]
            self.times = self.times[..., 1:]
            return self.pos

        # Scale basis functions
        weights_goal_scale = self.weights_goal_scale.repeat(self.num_dof)

        # Split weights and goal
        # Shape of w:
        # [*add_dim, num_dof, num_basis]
        # Shape of g:
        # [*add_dim, num_dof, 1]
        w, g = self._split_weights_goal(self.params * weights_goal_scale)

        # Get basis, shape [*add_dim, num_times, num_basis]
        basis = self.basis_gn.basis(self.times)

        # Get canonical phase x, shape [*add_dim, num_times]
        canonical_x = self.phase_gn.phase(self.times)

        # Get forcing function
        # Einsum shape: [*add_dim, num_times]
        #               [*add_dim, num_times, num_basis]
        #               [*add_dim, num_dof, num_basis]
        #            -> [*add_dim, num_times, num_dof]
        f = torch.einsum('...i,...ik,...jk->...ij', canonical_x, basis, w)

        # Initialize trajectory position, velocity
        pos = torch.zeros([*self.add_dim, self.times.shape[-1], self.num_dof],
                          dtype=self.dtype, device=self.device)
        vel = torch.zeros([*self.add_dim, self.times.shape[-1], self.num_dof],
                          dtype=self.dtype, device=self.device)

        pos[..., 0, :] = self.init_pos
        vel[..., 0, :] = self.init_vel * self.phase_gn.tau[..., None]

        # Get scaled time increment steps
        scaled_times = self.phase_gn.left_bound_linear_phase(self.times)
        scaled_dt = torch.diff(scaled_times, dim=-1)

        # Apply Euler Integral
        for i in range(scaled_dt.shape[-1]):
            acc = (self.alpha * (self.beta * (g - pos[..., i, :])
                                 - vel[..., i, :]) + f[..., i, :])
            vel[..., i + 1, :] = \
                vel[..., i, :] + torch.einsum('...,...i->...i',
                                              scaled_dt[..., i], acc)
            pos[..., i + 1, :] = \
                pos[..., i, :] + torch.einsum('...,...i->...i',
                                              scaled_dt[..., i],
                                              vel[..., i + 1, :])

        # Unscale velocity to original time space
        vel /= self.phase_gn.tau[..., None, None]

        # Store pos and vel
        self.pos = pos
        self.vel = vel

        return pos

    def get_traj_vel(self, times=None, params=None,
                     init_time=None, init_pos=None, init_vel=None):
        """
        Get trajectory velocity

        Refer setting functions for desired shape of inputs

        Args:
            times: time points, can be None
            params: learnable parameters, can be None
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity

        Returns:
            vel
        """

        # Shape of vel
        # [*add_dim, num_times, num_dof]

        # Update inputs
        self.update_inputs(times, params, init_time, init_pos, init_vel)

        # Reuse result if existing
        if self.vel is not None:
            return self.vel

        # Recompute otherwise
        # Velocity is computed together with position in DMP
        self.get_traj_pos()
        return self.vel

    def learn_mp_params_from_trajs(self, times: torch.Tensor,
                                   trajs: torch.Tensor,
                                   reg: float = 1e-9):
        raise NotImplementedError

    def _split_weights_goal(self, wg):
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

    def _show_scaled_basis(self, plot=False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        tau = self.phase_gn.tau
        delay = self.phase_gn.delay
        assert tau.ndim == 0 and delay.ndim == 0
        times = torch.linspace(delay - tau, delay + 2 * tau, steps=1000,
                               device=self.device, dtype=self.dtype)
        self.set_add_dim([])
        self.set_times(times)

        weights_scale = self.weights_scale
        canonical_x = self.phase_gn.phase(self.times)
        # Get basis
        # Shape: [*add_dim, num_times, num_basis]
        basis_values = self.basis_gn.basis(times) * canonical_x[..., None] * weights_scale

        # Enforce all variables to numpy
        times, basis_values, delay, tau = \
            to_nps(times, basis_values, delay, tau)

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
