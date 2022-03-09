"""
@brief:     Dynamic Movement Primitives in PyTorch
"""

import torch

from mp_pytorch import BasisGenerator
from mp_pytorch import LinearPhaseGenerator
from mp_pytorch.mp_interfaces import MPInterface
from mp_pytorch.phase_generator import ExpDecayPhaseGenerator


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

        # Control parameters
        self.alpha = kwargs["alpha"]
        self.beta = self.alpha / 4

    @property
    def _num_local_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        return super()._num_local_params + self.num_dof

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

        assert list(bc_time.shape) == [*self.add_dim]
        assert list(bc_pos.shape) == list(bc_vel.shape) \
               and list(bc_vel.shape) == [*self.add_dim, self.num_dof]
        super().set_boundary_conditions(bc_time, bc_pos, bc_vel)

    def get_traj_pos(self, times=None, params=None,
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

        # Get canonical phase x, shape [*add_dim, num_times]
        canonical_x = self.phase_gn.phase(self.times)

        # Get forcing function
        # Einsum shape: [*add_dim, num_times]
        #               [*add_dim, num_times, num_basis]
        #               [*add_dim, num_dof, num_basis]
        #            -> [*add_dim, num_times, num_dof]
        f = torch.einsum('...i,...ik,...jk->...ij', canonical_x, basis, w)

        # Initialize trajectory position, velocity
        pos = torch.zeros([*self.add_dim, self.times.shape[-1], self.num_dof])
        vel = torch.zeros([*self.add_dim, self.times.shape[-1], self.num_dof])

        # Check boundary condition, the desired times should start from
        # boundary condition time steps
        assert torch.all(torch.abs(self.bc_time - self.times[..., 0]) < 1e-8), \
            "The first time step's value should be same to bc_time."
        pos[..., 0, :] = self.bc_pos
        vel[..., 0, :] = self.bc_vel

        # Get scaled time increment steps
        scaled_times = LinearPhaseGenerator.phase(self.phase_gn, self.times)
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

        # Store pos and vel
        self.pos = pos
        self.vel = vel

        return pos

    def get_traj_vel(self, times=None, params=None,
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
        self.get_traj_pos()
        return self.vel

    def learn_mp_params_from_trajs(self, times: torch.Tensor,
                                   trajs: torch.Tensor,
                                   reg: float = 1e-9):
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
