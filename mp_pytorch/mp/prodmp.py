from typing import Iterable
from typing import Tuple
from typing import Union
import logging
import numpy as np
import torch
import logging

import mp_pytorch.util
from mp_pytorch.basis_gn import ProDMPBasisGenerator
from .promp import ProMP


class ProDMP(ProMP):
    """Integral form of DMPs"""

    def __init__(self,
                 basis_gn: ProDMPBasisGenerator,
                 num_dof: int,
                 weights_scale: Union[float, Iterable] = 1.,
                 goal_scale: float = 1.,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 **kwargs):
        """
        Constructor of ProDMP
        Args:
            basis_gn: basis function value generator
            num_dof: number of Degrees of Freedoms
            weights_scale: scaling for the parameters weights
            goal_scale: scaling for the goal
            dtype: torch data type
            device: torch device to run on
            kwargs: keyword arguments

        Keyword Arguments:
            auto_scale_basis: apply scale to all basis to make max magnitude = 1
        """
        if not isinstance(basis_gn, ProDMPBasisGenerator):
            raise ValueError(
                f'ProDMP requires a ProDMP basis generator, {type(basis_gn)} is not supported.')

        # Disable learning of weights or goal
        self.disable_weights = kwargs.get("disable_weights", False)
        self.disable_goal = kwargs.get("disable_goal", False)
        assert not (self.disable_weights and self.disable_goal), \
            "Cannot disable both weights and goal learning."

        # Super init
        super().__init__(basis_gn, num_dof, weights_scale, dtype, device,
                         **kwargs)

        # Number of parameters
        self.num_basis_g = self.num_basis + 1

        # Goal scale
        self.auto_scale_basis = kwargs.get("auto_scale_basis", False)
        self.goal_scale = goal_scale
        self.weights_goal_scale = self.get_weights_goal_scale(
            self.auto_scale_basis)

        # Runtime intermediate variables shared by different getting functions
        self.y1 = None
        self.y2 = None
        self.dy1 = None
        self.dy2 = None
        self.y1_init = None
        self.y2_init = None
        self.dy1_init = None
        self.dy2_init = None

        self.pos_init = None
        self.vel_init = None
        self.pos_H_single = None
        self.vel_H_single = None

        self.pos_H_multi = None
        self.vel_H_multi = None

        # Padding params when weights or goal is disabled
        padding_basis = self.basis_gn.num_basis if self.disable_weights else 0
        padding_goal = 1 if self.disable_goal else 0
        if self.disable_weights or self.disable_goal:
            self.padding = torch.nn.ConstantPad2d(
                (padding_basis, padding_goal, 0, 0), 0)
            logging.warning(
                "Padding ProDMP is being used. Only the traj position"
                " and velocity can be computed correctly. The other "
                "entities are not guaranteed.")
        else:
            self.padding = lambda x: x

    @property
    def _num_local_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        if self.disable_weights:
            # Use goal only
            return self.num_dof
        if self.disable_goal:
            # Use basis only
            return super()._num_local_params
        else:
            # Use both basis and goal
            return super()._num_local_params + self.num_dof

    def get_weights_goal_scale(self, auto_scale_basis=False) -> torch.Tensor:
        """
        Compute scaling factors of weights and goal
        Args:
            auto_scale_basis: if compute the scaling factors automatically

        Returns: the weights and goal scaling vector
        """
        if auto_scale_basis:
            w_g_scale = self.basis_gn.get_basis_scale_factors()
            w_g_scale[:-1] = w_g_scale[:-1] * self.weights_scale
            w_g_scale[-1] = w_g_scale[-1] * self.goal_scale
        else:
            w_g_scale = torch.zeros(self.num_basis_g)
            w_g_scale[:-1] = self.weights_scale
            w_g_scale[-1] = self.goal_scale
        return w_g_scale

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

        # Get general solution values at desired time points
        # Shape [*add_dim, num_times]
        self.y1, self.y2, self.dy1, self.dy2 = \
            self.basis_gn.general_solution_values(times)

        super().set_times(times)

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

        init_time = torch.as_tensor(init_time, dtype=self.dtype,
                                    device=self.device)
        init_pos = torch.as_tensor(init_pos, dtype=self.dtype,
                                   device=self.device)
        init_vel = torch.as_tensor(init_vel, dtype=self.dtype,
                                   device=self.device)

        assert list(init_time.shape) == [*self.add_dim]
        assert list(init_pos.shape) == list(init_vel.shape) and list(
            init_vel.shape) == [*self.add_dim, self.num_dof]

        init_time = torch.as_tensor(init_time, dtype=self.dtype,
                                    device=self.device)
        y1_init, y2_init, dy1_init, dy2_init = self.basis_gn.general_solution_values(
            init_time[..., None])

        self.y1_init = y1_init.squeeze(-1)
        self.y2_init = y2_init.squeeze(-1)
        self.dy1_init = dy1_init.squeeze(-1)
        self.dy2_init = dy2_init.squeeze(-1)

        super().set_initial_conditions(init_time, init_pos, init_vel)

    def get_traj_pos(self, times=None, params=None,
                     init_time=None, init_pos=None, init_vel=None,
                     flat_shape=False):
        """
        Compute trajectory pos

        Refer setting functions for desired shape of inputs

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

        # Shape of pos
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, params, None, init_time, init_pos, init_vel)

        # Reuse result if existing
        if self.pos is not None:
            pos = self.pos

        else:

            # Recompute otherwise
            self.compute_intermediate_terms_single()

            # Reshape (and pad) params to [*add_dim, num_dof, num_basis_g]
            params = self.params.reshape([*self.add_dim, self.num_dof, -1])
            params = self.padding(params)

            # Scale basis functions
            pos_H_single = self.pos_H_single * self.weights_goal_scale

            # Position and velocity variant (part 3)
            # Einsum shape: [*add_dim, num_times, num_basis_g],
            #               [*add_dim, num_dof, num_basis_g]
            #            -> [*add_dim, num_dof, num_times]
            # Reshape to -> [*add_dim, num_dof * num_times]
            pos_linear = \
                torch.einsum('...jk,...ik->...ij', pos_H_single, params)
            pos_linear = torch.reshape(pos_linear, [*self.add_dim, -1])
            pos = self.pos_init + pos_linear
            self.pos = pos

        if not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            pos = pos.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            pos = torch.einsum('...ji->...ij', pos)

        return pos

    def get_traj_pos_cov(self, times=None, params_L=None, init_time=None,
                         init_pos=None, init_vel=None, reg: float = 1e-4):
        """
        Compute and return position covariance

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params_L: learnable parameters' variance
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            reg: regularization term

        Returns:
            pos_cov
        """

        # Shape of pos_cov
        # [*add_dim, num_dof * num_times, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, None, params_L, init_time, init_pos, init_vel)

        # Reuse result if existing
        if self.pos_cov is not None:
            return self.pos_cov

        # Recompute otherwise
        if self.params_L is None:
            return None

        # Get multi dof basis
        self.compute_intermediate_terms_multi_dof()

        # Scale basis functions
        weights_goal_scale = self.weights_goal_scale.repeat(self.num_dof)
        pos_H_multi = self.pos_H_multi * weights_goal_scale

        # Uncertainty of position
        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g],
        #               [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
        pos_cov = torch.einsum('...ik,...kl,...jl->...ij',
                               pos_H_multi, self.params_cov, pos_H_multi)

        # Determine regularization term to make traj_cov positive definite
        traj_cov_reg = reg
        reg_term_pos = torch.max(torch.einsum('...ii->...i',
                                              pos_cov)).item() * traj_cov_reg

        # Add regularization term for numerical stability
        self.pos_cov = pos_cov + torch.eye(pos_cov.shape[-1]) * reg_term_pos

        return self.pos_cov

    def get_traj_pos_std(self, times=None, params_L=None, init_time=None,
                         init_pos=None,
                         init_vel=None, flat_shape=False, reg: float = 1e-4):
        """
        Compute trajectory standard deviation

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params_L: learnable parameters' variance
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            flat_shape: if flatten the dimensions of Dof and time
            reg: regularization term

        Returns:
            pos_std
        """

        # Shape of pos_std
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, None, params_L, init_time, init_pos, init_vel)

        # Reuse result if existing
        if self.pos_std is not None:
            pos_std = self.pos_std

        else:
            # Recompute otherwise
            pos_cov = self.get_traj_pos_cov(reg=reg)
            if pos_cov is None:
                pos_std = None
            else:
                # Shape [*add_dim, num_dof * num_times]
                pos_std = torch.sqrt(torch.einsum('...ii->...i', pos_cov))

            self.pos_std = pos_std

        if pos_std is not None and not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            pos_std = pos_std.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            pos_std = torch.einsum('...ji->...ij', pos_std)

        return pos_std

    def get_traj_vel(self, times=None, params=None,
                     init_time=None, init_pos=None, init_vel=None,
                     flat_shape=False):
        """
        Compute trajectory velocity

        Refer setting functions for desired shape of inputs

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

        # Shape of vel
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, params, None, init_time, init_pos, init_vel)

        # Reuse result if existing
        if self.vel is not None:
            vel = self.vel
        else:
            # Recompute otherwise
            self.compute_intermediate_terms_single()

            # Reshape (and pad) params to [*add_dim, num_dof, num_basis_g]
            params = self.params.reshape([*self.add_dim, self.num_dof, -1])
            params = self.padding(params)

            # Scale basis functions
            vel_H_single = self.vel_H_single * self.weights_goal_scale

            # Position and velocity variant (part 3)
            # Einsum shape: [*add_dim, num_times, num_basis_g],
            #               [*add_dim, num_dof, num_basis_g]
            #            -> [*add_dim, num_dof, num_times]
            # Reshape to -> [*add_dim, num_dof * num_times]
            vel_linear = \
                torch.einsum('...jk,...ik->...ij', vel_H_single, params)
            vel_linear = torch.reshape(vel_linear, [*self.add_dim, -1])
            vel = self.vel_init + vel_linear

            # Unscale velocity to original time scale space
            vel = vel / self.phase_gn.tau[..., None]
            self.vel = vel

        if not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            vel = vel.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            vel = torch.einsum('...ji->...ij', vel)

        return vel

    def get_traj_vel_cov(self, times=None, params_L=None, init_time=None,
                         init_pos=None,
                         init_vel=None, reg: float = 1e-4):
        """
        Get trajectory velocity covariance

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params_L: learnable parameters' variance
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            reg: regularization term

        Returns:
            vel_cov
        """

        # Shape of vel_cov
        # [*add_dim, num_dof * num_times, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, None, params_L, init_time, init_pos, init_vel)

        # Reuse result if existing
        if self.vel_cov is not None:
            return self.vel_cov

        # Recompute otherwise
        if self.params_L is None:
            return None

        # Get multi dof basis
        self.compute_intermediate_terms_multi_dof()

        # Scale basis functions
        weights_goal_scale = self.weights_goal_scale.repeat(self.num_dof)
        vel_H_multi = self.vel_H_multi * weights_goal_scale

        # Uncertainty of velocity
        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g],
        #               [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
        vel_cov = torch.einsum('...ik,...kl,...jl->...ij',
                               vel_H_multi, self.params_cov,
                               vel_H_multi)

        # Determine regularization term to make traj_cov positive definite
        traj_cov_reg = reg
        reg_term_vel = torch.max(torch.einsum('...ii->...i',
                                              vel_cov)).item() * traj_cov_reg

        # Add regularization term for numerical stability
        vel_cov = vel_cov + torch.eye(vel_cov.shape[-1]) * reg_term_vel

        # Unscale velocity to original time scale space
        self.vel_cov = vel_cov / self.phase_gn.tau[..., None, None] ** 2

        return self.vel_cov

    def get_traj_vel_std(self, times=None, params_L=None, init_time=None,
                         init_pos=None,
                         init_vel=None, flat_shape=False, reg: float = 1e-4):
        """
        Compute trajectory velocity standard deviation

        Refer setting functions for desired shape of inputs

        Args:
            times: time points
            params_L: learnable parameters' variance
            init_time: initial condition time
            init_pos: initial condition position
            init_vel: initial condition velocity
            flat_shape: if flatten the dimensions of Dof and time
            reg: regularization term

        Returns:
            vel_std

        """

        # Shape of vel_std
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_inputs(times, None, params_L, init_time, init_pos, init_vel)

        # Reuse result if existing
        if self.vel_std is not None:
            vel_std = self.vel_std
        else:
            # Recompute otherwise
            vel_cov = self.get_traj_vel_cov(reg=reg)
            if vel_cov is None:
                vel_std = None
            else:
                # Shape [*add_dim, num_dof * num_times]
                vel_std = torch.sqrt(torch.einsum('...ii->...i', vel_cov))
            self.vel_std = vel_std

        if vel_std is not None and not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            vel_std = vel_std.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            vel_std = torch.einsum('...ji->...ij', vel_std)

        return vel_std

    def learn_mp_params_from_trajs(self, times: torch.Tensor,
                                   trajs: torch.Tensor,
                                   reg: float = 1e-9, **kwargs) -> dict:
        """
        Learn DMP weights and goal given trajectory position
        Use the initial position and velocity as initial condition

        Args:
            times: trajectory time points
            trajs: trajectory position in batch
            reg: regularization term
            kwargs: keyword arguments

        Returns:
            param_dict: dictionary of parameters containing
                - params (w + g)
                - init_time
                - init_pos
                - init_vel
        """
        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of trajs:
        # [*add_dim, num_times, num_dof]
        #
        # Shape of learned params
        # [*add_dim, num_dof * num_basis_g]

        # Assert trajs shape
        assert trajs.shape[:-1] == times.shape
        assert trajs.shape[-1] == self.num_dof

        trajs = torch.as_tensor(trajs, dtype=self.dtype, device=self.device)

        # Get initial conditions
        dt = self.basis_gn.scaled_dt * self.phase_gn.tau

        if all([key in kwargs.keys()
                for key in ["init_time", "init_pos", "init_vel"]]):
            logging.warning("ProDMP uses the given initial conditions")
            init_time = kwargs["init_time"]
            init_pos = kwargs["init_pos"]
            init_vel = kwargs["init_vel"]
        else:
            init_time = times[..., 0]
            init_pos = trajs[..., 0, :]
            init_vel = torch.diff(trajs, dim=-2)[..., 0, :] / dt

        # Setup stuff
        self.set_add_dim(list(trajs.shape[:-2]))
        self.set_times(times)
        self.set_initial_conditions(init_time, init_pos, init_vel)

        self.compute_intermediate_terms_single()
        self.compute_intermediate_terms_multi_dof()

        weights_goal_scale = self.weights_goal_scale.repeat(self.num_dof)
        pos_H_multi = self.pos_H_multi * weights_goal_scale

        # Solve this: Aw = B -> w = A^{-1} B
        # Einsum_shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        A = torch.einsum('...ki,...kj->...ij', pos_H_multi, pos_H_multi)
        # todo, check here
        A += torch.eye(self.num_dof * self.num_basis_g) * reg

        # Swap axis and reshape: [*add_dim, num_times, num_dof]
        #                     -> [*add_dim, num_dof, num_times]
        trajs = torch.einsum("...ij->...ji", trajs)

        # Reshape [*add_dim, num_dof, num_times]
        #      -> [*add_dim, num_dof * num_times]
        trajs = trajs.reshape([*self.add_dim, -1])

        # Position minus initial condition terms,
        pos_wg = trajs - self.pos_init

        # Einsum_shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times]
        #            -> [*add_dim, num_dof * num_basis_g]
        B = torch.einsum('...ki,...k->...i', pos_H_multi, pos_wg)

        # Shape of weights: [*add_dim, num_dof * num_basis_g]
        params = torch.linalg.solve(A, B)

        # Check if parameters basis or phase generator exist
        if self.basis_gn.num_params > 0:
            params_super = self.basis_gn.get_params()
            params = torch.cat([params_super, params], dim=-1)

        self.set_params(params)
        self.set_mp_params_variances(None)

        return {"params": params,
                "init_time": init_time,
                "init_pos": init_pos,
                "init_vel": init_vel}

    def compute_intermediate_terms_single(self):
        # Determinant of initial condition,
        # Shape: [*add_dim]
        det = self.y1_init * self.dy2_init - self.y2_init * self.dy1_init
        # Compute coefficients to form up traj position and velocity
        # Shape: [*add_dim], [*add_dim, num_times] -> [*add_dim, num_times]
        xi_1 = torch.einsum("...,...i->...i", self.dy2_init / det, self.y1) \
               - torch.einsum("...,...i->...i", self.dy1_init / det, self.y2)
        xi_2 = torch.einsum("...,...i->...i", self.y1_init / det, self.y2) \
               - torch.einsum("...,...i->...i", self.y2_init / det, self.y1)
        xi_3 = torch.einsum("...,...i->...i", self.dy1_init / det, self.y2) \
               - torch.einsum("...,...i->...i", self.dy2_init / det, self.y1)
        xi_4 = torch.einsum("...,...i->...i", self.y2_init / det, self.y1) \
               - torch.einsum("...,...i->...i", self.y1_init / det, self.y2)
        dxi_1 = torch.einsum("...,...i->...i", self.dy2_init / det, self.dy1) \
                - torch.einsum("...,...i->...i", self.dy1_init / det, self.dy2)
        dxi_2 = torch.einsum("...,...i->...i", self.y1_init / det, self.dy2) \
                - torch.einsum("...,...i->...i", self.y2_init / det, self.dy1)
        dxi_3 = torch.einsum("...,...i->...i", self.dy1_init / det, self.dy2) \
                - torch.einsum("...,...i->...i", self.dy2_init / det, self.dy1)
        dxi_4 = torch.einsum("...,...i->...i", self.y2_init / det, self.dy1) \
                - torch.einsum("...,...i->...i", self.y1_init / det, self.dy2)

        # Generate basis initial condition values
        # [*add_dim, num_basis_g]
        pos_basis_init = self.basis_gn.basis(self.init_time[..., None]).squeeze(
            -2)
        vel_basis_init = self.basis_gn.vel_basis(
            self.init_time[..., None]).squeeze(
            -2)

        # Scale init_vel
        init_vel = self.init_vel * self.phase_gn.tau[..., None]

        # Compute position and velocity initial condition part
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_dof]
        #            -> [*add_dim, num_dof, num_times]
        pos_det = torch.einsum('...j,...i->...ij', xi_1, self.init_pos) \
                  + torch.einsum('...j,...i->...ij', xi_2, init_vel)
        vel_det = torch.einsum('...j,...i->...ij', dxi_1, self.init_pos) \
                  + torch.einsum('...j,...i->...ij', dxi_2, init_vel)

        # Reshape: [*add_dim, num_dof, num_times]
        #       -> [*add_dim, num_dof * num_times]
        self.pos_init = torch.reshape(pos_det, [*self.add_dim, -1])
        self.vel_init = torch.reshape(vel_det, [*self.add_dim, -1])

        # Compute position and velocity linear basis part
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_basis_g]
        #            -> [*add_dim, num_times, num_basis_g]
        self.pos_H_single = \
            torch.einsum('...i,...j->...ij', xi_3, pos_basis_init) \
            + torch.einsum('...i,...j->...ij', xi_4, vel_basis_init) \
            + self.basis_gn.basis(self.times)
        self.vel_H_single = \
            torch.einsum('...i,...j->...ij', dxi_3, pos_basis_init) \
            + torch.einsum('...i,...j->...ij', dxi_4, vel_basis_init) \
            + self.basis_gn.vel_basis(self.times)

    def compute_intermediate_terms_multi_dof(self):
        # Determinant of initial condition,
        # Shape: [*add_dim]
        det = self.y1_init * self.dy2_init - self.y2_init * self.dy1_init

        # Compute coefficients to form up traj position and velocity
        # Shape: [*add_dim], [*add_dim, num_times] -> [*add_dim, num_times]
        xi_3 = torch.einsum("...,...i->...i", self.dy1_init / det, self.y2) \
               - torch.einsum("...,...i->...i", self.dy2_init / det, self.y1)
        xi_4 = torch.einsum("...,...i->...i", self.y2_init / det, self.y1) \
               - torch.einsum("...,...i->...i", self.y1_init / det, self.y2)
        dxi_3 = torch.einsum("...,...i->...i", self.dy1_init / det, self.dy2) \
                - torch.einsum("...,...i->...i", self.dy2_init / det, self.dy1)
        dxi_4 = torch.einsum("...,...i->...i", self.y2_init / det, self.dy1) \
                - torch.einsum("...,...i->...i", self.y1_init / det, self.dy2)

        # Generate blocked basis initial condition values
        # [*add_dim, num_dof, num_dof * num_basis_g]
        pos_basis_init_multi_dofs = self.basis_gn.basis_multi_dofs(
            self.init_time[..., None], self.num_dof)
        vel_basis_init_multi_dofs = self.basis_gn.vel_basis_multi_dofs(
            self.init_time[..., None], self.num_dof)

        # Compute position and velocity variant part (part 3)
        # Position and velocity part 3_1 and 3_2
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_dof, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof, num_times, num_dof * num_basis_g]
        pos_H_ = torch.einsum('...j,...ik->...ijk', xi_3,
                              pos_basis_init_multi_dofs) + \
                 torch.einsum('...j,...ik->...ijk', xi_4,
                              vel_basis_init_multi_dofs)
        vel_H_ = torch.einsum('...j,...ik->...ijk', dxi_3,
                              pos_basis_init_multi_dofs) + \
                 torch.einsum('...j,...ik->...ijk', dxi_4,
                              vel_basis_init_multi_dofs)
        # Reshape: [*add_dim, num_dof, num_times, num_dof * num_basis_g]
        #       -> [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        # todo, check here
        pos_H_ = torch.reshape(pos_H_, [*self.add_dim, -1,
                                        self.num_dof * self.num_basis_g])
        vel_H_ = torch.reshape(vel_H_, [*self.add_dim, -1,
                                        self.num_dof * self.num_basis_g])

        self.pos_H_multi = \
            pos_H_ + self.basis_gn.basis_multi_dofs(self.times, self.num_dof)
        self.vel_H_multi = \
            vel_H_ + self.basis_gn.vel_basis_multi_dofs(self.times,
                                                        self.num_dof)

    def _show_scaled_basis(self, plot=False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        tau = self.phase_gn.tau
        delay = self.phase_gn.delay
        assert tau.ndim == 0 and delay.ndim == 0
        times = torch.linspace(delay - tau, delay + 2 * tau, steps=1000,
                               device=self.device, dtype=self.dtype)
        self.set_add_dim([])
        self.set_times(times)
        self.set_initial_conditions(
            init_time=torch.zeros([], device=self.device,
                                  dtype=self.dtype) + delay,
            init_pos=torch.zeros([self.num_dof], device=self.device,
                                 dtype=self.dtype),
            init_vel=torch.zeros([self.num_dof], device=self.device,
                                 dtype=self.dtype),
        )

        self.compute_intermediate_terms_single()

        weights_goal_scale = self.weights_goal_scale

        # Get basis
        # Shape: [*add_dim, num_times, num_basis]
        basis_values = self.pos_H_single * weights_goal_scale

        # Enforce all variables to numpy
        times, basis_values, delay, tau = \
            mp_pytorch.util.to_nps(times, basis_values, delay, tau)

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
