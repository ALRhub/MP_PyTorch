"""
@brief:     Probabilistic Movement Primitives in PyTorch
"""
from typing import Union
import torch

from mp_pytorch import BasisGenerator
from .mp_interfaces import ProbabilisticMPInterface


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

        # Runtime variables
        self.basis_multi_dofs = None

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

        super().set_mp_times(times)

        # Shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        self.basis_multi_dofs = \
            self.basis_gn.basis_multi_dofs(times, self.num_dof)

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
            flat_shape: if flatten the dimensions of Dof and time

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

    def get_traj_pos_cov(self, times=None, params_L=None, bc_time=None,
                         bc_pos=None,
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

    def get_traj_pos_std(self, times=None, params_L=None, bc_time=None,
                         bc_pos=None,
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
            flat_shape: if flatten the dimensions of Dof and time
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
            pos_cov = self.get_traj_pos_cov()

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

    def get_traj_vel(self, times=None, params=None,
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
            flat_shape: if flatten the dimensions of Dof and time

        Returns:
            vel
        """
        # todo interpolation?
        self.vel = None
        return self.vel

    def get_traj_vel_cov(self, times=None, params_L=None, bc_time=None,
                         bc_pos=None,
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

    def get_traj_vel_std(self, times=None, params_L=None, bc_time=None,
                         bc_pos=None,
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
            flat_shape: if flatten the dimensions of Dof and time
            reg: regularization term

        Returns:
            vel_std
        """
        self.vel_std = None
        return self.vel_std

    def learn_mp_params_from_trajs(self, times: torch.Tensor,
                                   trajs: torch.Tensor,
                                   reg: float = 1e-9) -> dict:
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

        self.set_add_dim(list(trajs.shape[:-2]))
        self.set_mp_times(times)

        # Get multiple dof basis function values
        # Tensor [*add_dim, num_dof * num_times, num_dof * num_basis]
        basis_multi_dofs = self.basis_multi_dofs

        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis],
        #               [*add_dim, num_dof * num_times, num_dof * num_basis],
        #            -> [*add_dim, num_dof * num_basis, num_dof * num_basis]
        A = torch.einsum('...ki,...kj->...ij', basis_multi_dofs,
                         basis_multi_dofs)
        A += torch.eye(self._num_local_params) * reg

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

        # Check if parameters basis or phase generator exist
        if self.basis_gn.num_params > 0:
            params_super = self.basis_gn.get_params()
            params = torch.cat([params_super, params], dim=-1)

        self.set_params(params)
        self.set_mp_params_variances(None)

        return {"params": params}
