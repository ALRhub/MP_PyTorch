"""
@brief:     Probabilistic Movement Primitives in PyTorch
"""
import logging
from typing import Iterable
from typing import Union
from typing import Tuple

import numpy as np
import torch
from mp_pytorch.util import to_nps
from mp_pytorch.basis_gn import BasisGenerator
from .mp_interfaces import ProbabilisticMPInterface


class ProMP(ProbabilisticMPInterface):
    """ProMP in PyTorch"""

    def __init__(self,
                 basis_gn: BasisGenerator,
                 num_dof: int,
                 weights_scale: Union[float, Iterable] = 1.,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 **kwargs):
        """
        Constructor of ProMP
        Args:
            basis_gn: basis function value generator
            num_dof: number of Degrees of Freedoms
            weights_scale: scaling for the parameters weights
            dtype: torch data type
            device: torch device to run on
            **kwargs: keyword arguments
        """

        super().__init__(basis_gn, num_dof, weights_scale, dtype, device,
                         **kwargs)

        # Some legacy code for a smooth start from 0
        self.has_zero_padding = hasattr(self.basis_gn, 'num_basis_zero_start')
        if self.has_zero_padding:
            # if no zero start/ zero goal, use weights as it is
            self.padding = torch.nn.ConstantPad2d(
                (self.basis_gn.num_basis_zero_start,
                 self.basis_gn.num_basis_zero_goal, 0, 0), 0)
            logging.warning(
                "Zero Padding ProMP is being used. Only the traj position"
                " and velocity can be computed correctly. The other "
                "entities are not guaranteed.")
        else:
            self.padding = lambda x: x

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

        times = torch.as_tensor(times, dtype=self.dtype, device=self.device)
        super().set_times(times)

    def set_mp_params_variances(self, params_L: Union[torch.Tensor, None]):
        """
        Set variance of MP params
        Args:
            params_L: cholesky of covariance matrix of the MP parameters

        Returns: None

        """
        # Shape of params_L:
        # [*add_dim, num_dof * num_basis, num_dof * num_basis]

        if params_L is not None:
            assert list(params_L.shape) == [*self.add_dim,
                                            self._num_local_params,
                                            self._num_local_params]
        super().set_mp_params_variances(params_L)

    def get_traj_pos(self, times=None, params=None,
                     init_time=None, init_pos=None, init_vel=None,
                     flat_shape=False):
        """
        Get trajectory position

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
            assert self.params is not None

            # Reshape params
            # [*add_dim, num_dof * num_basis] -> [*add_dim, num_dof, num_basis]
            params = self.params.reshape(*self.add_dim, self.num_dof, -1)

            # Padding if necessary, this is a legacy case
            # [*add_dim, num_dof, num_basis]
            # -> [*add_dim, num_dof, num_basis + num_padding]
            params = self.padding(params)
            if self.weights_scale.ndim != 0:
                weights_scale = self.padding(self.weights_scale[None])[0]
            else:
                weights_scale = self.padding(torch.ones([1, self.num_basis]) *
                                             self.weights_scale)[0]

            # Get basis
            # Shape: [*add_dim, num_times, num_basis]
            basis_single_dof = \
                self.basis_gn.basis(self.times) * weights_scale

            # Einsum shape: [*add_dim, num_times, num_basis],
            #               [*add_dim, num_dof, num_basis]
            #            -> [*add_dim, num_times, num_dof]
            pos = torch.einsum('...ik,...jk->...ij', basis_single_dof, params)

            # Padding if necessary, this is a legacy case
            pos += self.init_pos[..., None, :] if self.has_zero_padding else 0

            self.pos = pos

        if flat_shape:
            # Switch axes to [*add_dim, num_dof, num_times]
            pos = torch.einsum('...ji->...ij', pos)

            # Reshape to [*add_dim, num_dof * num_times]
            pos = pos.reshape(*self.add_dim, -1)

        return pos

    def get_traj_pos_cov(self, times=None, params_L=None,
                         init_time=None, init_pos=None, init_vel=None,
                         reg: float = 1e-4):
        """
        Compute position covariance

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

        # Otherwise recompute result
        if self.params_L is None:
            return None

        # Get weights scale
        if self.weights_scale.ndim == 0:
            weights_scale = self.weights_scale
        else:
            weights_scale = self.weights_scale.repeat(self.num_dof)

        # Get basis of all Dofs
        # Shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dofs = self.basis_gn.basis_multi_dofs(
            self.times, self.num_dof) * weights_scale

        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        #               [*add_dim, num_dof * num_basis, num_dof * num_basis]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis]
        #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
        pos_cov = torch.einsum('...ik,...kl,...jl->...ij',
                               basis_multi_dofs,
                               self.params_cov,
                               basis_multi_dofs)

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
        Compute position standard deviation

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
            # Otherwise recompute
            if self.pos_cov is not None:
                pos_cov = self.pos_cov
            else:
                pos_cov = self.get_traj_pos_cov()

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
        Get trajectory velocity

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
            pos = self.get_traj_pos()

            vel = torch.zeros_like(pos, dtype=self.dtype, device=self.device)
            vel[..., :-1, :] = torch.diff(pos, dim=-2) \
                               / torch.diff(self.times)[..., None]
            vel[..., -1, :] = vel[..., -2, :]

            self.vel = vel

        if flat_shape:
            # Switch axes to [*add_dim, num_dof, num_times]
            vel = torch.einsum('...ji->...ij', vel)

            # Reshape to [*add_dim, num_dof * num_times]
            vel = vel.reshape(*self.add_dim, -1)

        return vel

    def get_traj_vel_cov(self, times=None, params_L=None, init_time=None,
                         init_pos=None,
                         init_vel=None, reg: float = 1e-4):
        """
        Get velocity covariance

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
        self.vel_cov = None
        return self.vel_cov

    def get_traj_vel_std(self, times=None, params_L=None, init_time=None,
                         init_pos=None,
                         init_vel=None, flat_shape=False, reg: float = 1e-4):
        """
        Get trajectory standard deviation

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
        self.vel_std = None
        return self.vel_std

    def learn_mp_params_from_trajs(self, times: torch.Tensor,
                                   trajs: torch.Tensor,
                                   reg: float = 1e-9, **kwargs) -> dict:
        """
        Learn ProMP weights from demonstration

        Args:
            times: trajectory time points
            trajs: trajectory from which weights should be learned
            reg: regularization term
            kwargs: keyword arguments

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

        self.set_add_dim(list(trajs.shape[:-2]))
        self.set_times(times)

        # Get weights scale
        if self.weights_scale.ndim == 0:
            weights_scale = self.weights_scale
        else:
            weights_scale = self.weights_scale.repeat(self.num_dof)

        # Get multiple dof basis function values
        # Tensor [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dofs = self.basis_gn.basis_multi_dofs(
            times, self.num_dof) * weights_scale

        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis],
        #               [*add_dim, num_dof * num_times, num_dof * num_basis],
        #            -> [*add_dim, num_dof * num_basis, num_dof * num_basis]
        A = torch.einsum('...ki,...kj->...ij', basis_multi_dofs,
                         basis_multi_dofs)
        A += torch.eye(self._num_local_params) * reg

        # Reorder axis [*add_dim, num_times, num_dof]
        #           -> [*add_dim, num_dof, num_times]
        trajs = torch.as_tensor(trajs, dtype=self.dtype, device=self.device)
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

        # Check if parameters basis or phase generator exist
        if self.basis_gn.num_params > 0:
            params_super = self.basis_gn.get_params()
            params = torch.cat([params_super, params], dim=-1)

        self.set_params(params)
        self.set_mp_params_variances(None)

        return {"params": params}

    def _show_scaled_basis(self, plot=False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        tau = self.phase_gn.tau
        delay = self.phase_gn.delay
        assert tau.ndim == 0 and delay.ndim == 0
        times = torch.linspace(delay - tau, delay + 2 * tau, steps=1000,
                               device=self.device, dtype=self.dtype)

        if self.weights_scale.ndim != 0:
            weights_scale = self.padding(self.weights_scale[None])[0]
        else:
            weights_scale = self.padding(torch.ones([1, self.num_basis]) *
                                         self.weights_scale)[0]

        # Get basis
        # Shape: [*add_dim, num_times, num_basis]
        basis_values = \
            self.basis_gn.basis(times) * weights_scale

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
