import torch

from mp_pytorch import IDMPBasisGenerator
from .promp import ProMP


class IDMP(ProMP):
    """Integral form of DMPs"""

    def __init__(self,
                 basis_gn: IDMPBasisGenerator,
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

        # Number of parameters
        self.num_basis_g = self.num_basis + 1

        # Runtime intermediate variables shared by different getting functions
        self.y1 = None
        self.y2 = None
        self.dy1 = None
        self.dy2 = None
        self.vel_basis_multi_dofs = None

        self.pos_det = None
        self.vel_det = None
        self.pos_vary_ = None
        self.vel_vary_ = None

    @property
    def _num_local_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        return super()._num_local_params + self.num_dof

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

        # Get general solution values at desired time points
        # Shape [*add_dim, num_times]
        self.y1, self.y2, self.dy1, self.dy2 = \
            self.basis_gn.general_solution_values(times)

        # Generated blocked basis of multi dofs, shape:
        # [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        self.vel_basis_multi_dofs = \
            self.basis_gn.vel_basis_multi_dofs(times, self.num_dof)

        super().set_mp_times(times)

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

        # Update shared intermediate variables
        self.compute_bc_intermediate_variables()

    def get_traj_pos(self, times=None, params=None,
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

        # Recompute otherwise
        # Position and velocity variant (part 3)
        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g],
        #               [*add_dim, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof * num_times]
        pos_vary = torch.einsum('...ij,...j->...i', self.pos_vary_, self.params)

        self.pos = self.pos_det + pos_vary

        if not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            self.pos = self.pos.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            self.pos = torch.einsum('...ji->...ij', self.pos)

        return self.pos

    def get_traj_pos_cov(self, times=None, params_L=None, bc_time=None,
                         bc_pos=None,
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
        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g],
        #               [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis_g]
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

    def get_traj_pos_std(self, times=None, params_L=None, bc_time=None,
                         bc_pos=None,
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

        # Recompute otherwise
        pos_cov = self.get_traj_pos_cov(reg=reg)
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
        Compute trajectory velocity

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

        # Shape of vel
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        if times is not None:
            self.set_mp_times(times)
        if params is not None:
            self.set_params(params)

        # Reuse result if existing
        if self.vel is not None:
            return self.vel

        # Recompute otherwise
        # Position and velocity variant (part 3)
        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g],
        #               [*add_dim, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof * num_times]
        vel_vary = torch.einsum('...ij,...j->...i', self.vel_vary_, self.params)

        vel = self.vel_det + vel_vary

        # Unscale velocity to original time scale space
        self.vel = vel / self.phase_gn.tau[..., None]

        if not flat_shape:
            # Reshape to [*add_dim, num_dof, num_times]
            self.vel = self.vel.reshape(*self.add_dim, self.num_dof, -1)

            # Switch axes to [*add_dim, num_times, num_dof]
            self.vel = torch.einsum('...ji->...ij', self.vel)

        return self.vel

    def get_traj_vel_cov(self, times=None, params_L=None, bc_time=None,
                         bc_pos=None,
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
        # Einsum shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g],
        #               [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof * num_times, num_dof * num_times]
        vel_cov = torch.einsum('...ik,...kl,...jl->...ij',
                               self.vel_vary_, self.params_cov, self.vel_vary_)

        # Determine regularization term to make traj_cov positive definite
        traj_cov_reg = reg
        reg_term_vel = torch.max(torch.einsum('...ii->...i',
                                              vel_cov)).item() * traj_cov_reg

        # Add regularization term for numerical stability
        vel_cov = vel_cov + torch.eye(vel_cov.shape[-1]) * reg_term_vel

        # Unscale velocity to original time scale space
        self.vel_cov = vel_cov / self.phase_gn.tau[..., None, None] ** 2

        return self.vel_cov

    def get_traj_vel_std(self, times=None, params_L=None, bc_time=None,
                         bc_pos=None,
                         bc_vel=None, flat_shape=False, reg: float = 1e-4):
        """
        Compute trajectory velocity standard deviation

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

        # Shape of vel_std
        # [*add_dim, num_times, num_dof] or [*add_dim, num_dof * num_times]

        # Update inputs
        self.update_mp_inputs(times, None, params_L, bc_time, bc_pos, bc_vel)

        # Reuse result if existing
        if self.vel_std is not None:
            return self.vel_std

        # Recompute otherwise
        vel_cov = self.get_traj_vel_cov(reg=reg)
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

    def learn_mp_params_from_trajs(self, times: torch.Tensor,
                                   trajs: torch.Tensor,
                                   reg: float = 1e-9) -> dict:
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
        #
        # Shape of learned params
        # [*add_dim, num_dof * num_basis_g]

        # Assert trajs shape
        assert trajs.shape[:-1] == times.shape
        assert trajs.shape[-1] == self.num_dof

        trajs = torch.Tensor(trajs)

        # Get boundary conditions
        dt = self.basis_gn.scaled_dt * self.phase_gn.tau
        bc_time = times[..., 0]
        bc_pos = trajs[..., 0, :]
        bc_vel = torch.diff(trajs, dim=-2)[..., 0, :] / dt

        # Setup stuff
        self.set_add_dim(list(trajs.shape[:-2]))
        self.set_mp_times(times)
        self.set_boundary_conditions(bc_time, bc_pos, bc_vel)

        # Solve this: Aw = B -> w = A^{-1} B
        # Einsum_shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof * num_basis_g, num_dof * num_basis_g]
        A = torch.einsum('...ki,...kj->...ij', self.pos_vary_, self.pos_vary_)
        A += torch.eye(self._num_local_params) * reg

        # Swap axis and reshape: [*add_dim, num_times, num_dof]
        #                     -> [*add_dim, num_dof, num_times]
        trajs = torch.einsum("...ij->...ji", trajs)

        # Reshape [*add_dim, num_dof, num_times]
        #      -> [*add_dim, num_dof * num_times]
        trajs = trajs.reshape([*self.add_dim, -1])

        # Position minus boundary condition terms,
        pos_wg = trajs - self.pos_det

        # Einsum_shape: [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        #               [*add_dim, num_dof * num_times]
        #            -> [*add_dim, num_dof * num_basis_g]
        B = torch.einsum('...ki,...k->...i', self.pos_vary_, pos_wg)

        # Shape of weights: [*add_dim, num_dof * num_basis_g]
        params = torch.linalg.solve(A, B)

        # Check if parameters basis or phase generator exist
        if self.basis_gn.num_params > 0:
            params_super = self.basis_gn.get_params()
            params = torch.cat([params_super, params], dim=-1)

        self.set_params(params)
        self.set_mp_params_variances(None)

        return {"params": params,
                "bc_time": bc_time,
                "bc_pos": bc_pos,
                "bc_vel": bc_vel}

    def compute_bc_intermediate_variables(self):
        """
        Evaluate boundary condition intermediate shared variables values

        Returns:
            None
        """

        # Extract boundary condition values
        # Shape [*add_dim, 1]
        y1_bc, y2_bc, dy1_bc, dy2_bc = \
            self.basis_gn.general_solution_values(self.bc_time[..., None])

        # Shape [*add_dim, 1] -> # Shape [*add_dim]
        y1_bc = y1_bc.squeeze(-1)
        y2_bc = y2_bc.squeeze(-1)
        dy1_bc = dy1_bc.squeeze(-1)
        dy2_bc = dy2_bc.squeeze(-1)

        # Determinant of boundary condition,
        # Shape: [*add_dim]
        det = y1_bc * dy2_bc - y2_bc * dy1_bc

        # Compute coefficients to form up traj position and velocity
        # Shape: [*add_dim], [*add_dim, num_times] -> [*add_dim, num_times]
        xi_1 = torch.einsum("...,...i->...i", dy2_bc / det, self.y1) \
               - torch.einsum("...,...i->...i", dy1_bc / det, self.y2)
        xi_2 = torch.einsum("...,...i->...i", y1_bc / det, self.y2) \
               - torch.einsum("...,...i->...i", y2_bc / det, self.y1)
        xi_3 = torch.einsum("...,...i->...i", dy1_bc / det, self.y2) \
               - torch.einsum("...,...i->...i", dy2_bc / det, self.y1)
        xi_4 = torch.einsum("...,...i->...i", y2_bc / det, self.y1) \
               - torch.einsum("...,...i->...i", y1_bc / det, self.y2)
        dxi_1 = torch.einsum("...,...i->...i", dy2_bc / det, self.dy1) \
                - torch.einsum("...,...i->...i", dy1_bc / det, self.dy2)
        dxi_2 = torch.einsum("...,...i->...i", y1_bc / det, self.dy2) \
                - torch.einsum("...,...i->...i", y2_bc / det, self.dy1)
        dxi_3 = torch.einsum("...,...i->...i", dy1_bc / det, self.dy2) \
                - torch.einsum("...,...i->...i", dy2_bc / det, self.dy1)
        dxi_4 = torch.einsum("...,...i->...i", y2_bc / det, self.dy1) \
                - torch.einsum("...,...i->...i", y1_bc / det, self.dy2)

        # Generate blocked basis boundary condition values
        # [*add_dim, num_dof, num_dof * num_basis_g]
        pos_basis_bc_multi_dofs = self.basis_gn.basis_multi_dofs(
            self.bc_time[..., None], self.num_dof)
        vel_basis_bc_multi_dofs = self.basis_gn.vel_basis_multi_dofs(
            self.bc_time[..., None], self.num_dof)

        # Scale bc_vel
        bc_vel = self.bc_vel * self.phase_gn.tau[..., None]

        # Compute position and velocity determined part (part 1 and 2)
        # Position and velocity part 1 and part 2
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_dof]
        #            -> [*add_dim, num_dof, num_times]
        pos_det = torch.einsum('...j,...i->...ij', xi_1, self.bc_pos) \
                  + torch.einsum('...j,...i->...ij', xi_2, bc_vel)
        vel_det = torch.einsum('...j,...i->...ij', dxi_1, self.bc_pos) \
                  + torch.einsum('...j,...i->...ij', dxi_2, bc_vel)

        # Reshape: [*add_dim, num_dof, num_times]
        #       -> [*add_dim, num_dof * num_times]
        self.pos_det = torch.reshape(pos_det, [*self.add_dim, -1])
        self.vel_det = torch.reshape(vel_det, [*self.add_dim, -1])

        # Compute position and velocity variant part (part 3)
        # Position and velocity part 3_1 and 3_2
        # Einsum shape: [*add_dim, num_times],
        #               [*add_dim, num_dof, num_dof * num_basis_g]
        #            -> [*add_dim, num_dof, num_times, num_dof * num_basis_g]
        pos_vary_ = torch.einsum('...j,...ik->...ijk', xi_3,
                                 pos_basis_bc_multi_dofs) + \
                    torch.einsum('...j,...ik->...ijk', xi_4,
                                 vel_basis_bc_multi_dofs)
        vel_vary_ = torch.einsum('...j,...ik->...ijk', dxi_3,
                                 pos_basis_bc_multi_dofs) + \
                    torch.einsum('...j,...ik->...ijk', dxi_4,
                                 vel_basis_bc_multi_dofs)

        # Reshape: [*add_dim, num_dof, num_times, num_dof * num_basis_g]
        #       -> [*add_dim, num_dof * num_times, num_dof * num_basis_g]
        pos_vary_ = \
            torch.reshape(pos_vary_,
                          [*self.add_dim, -1, self._num_local_params])
        vel_vary_ = \
            torch.reshape(vel_vary_,
                          [*self.add_dim, -1, self._num_local_params])

        self.pos_vary_ = pos_vary_ + self.basis_multi_dofs
        self.vel_vary_ = vel_vary_ + self.vel_basis_multi_dofs
