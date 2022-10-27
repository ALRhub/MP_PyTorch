from typing import Union

import numpy as np
import torch

from .phase_generator import PhaseGenerator


class ExpDecayPhaseGenerator(PhaseGenerator):
    def __init__(self,
                 tau: float = 1.0,
                 delay: float = 0.0,
                 alpha_phase: float = 3.0,
                 learn_tau: bool = False,
                 learn_delay: bool = False,
                 learn_alpha_phase: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 *args, **kwargs):
        """
        Constructor for exponential decay phase generator
        Args:
            tau: trajectory length scaling factor
            delay: time to wait before execute
            alpha_phase: decaying factor: tau * dx/dt = -alpha_phase * x
            learn_tau: if tau is learnable parameter
            learn_delay: if delay is learnable parameter
            learn_alpha_phase: if alpha_phase is a learnable parameter
            dtype: torch data type
            device: torch device to run on
            *args: other arguments list
            **kwargs: other keyword arguments
        """
        super(ExpDecayPhaseGenerator, self).__init__(tau=tau, delay=delay,
                                                     learn_tau=learn_tau,
                                                     learn_delay=learn_delay,
                                                     dtype=dtype, device=device,
                                                     *args, **kwargs)

        self.alpha_phase = torch.tensor(alpha_phase, dtype=self.dtype,
                                        device=self.device)
        self.learn_alpha_phase = learn_alpha_phase

        if learn_alpha_phase:
            self.alpha_phase_bound = kwargs.get("alpha_phase_bound",
                                                [1e-5, torch.inf])
            assert len(self.alpha_phase_bound) == 2

    @property
    def _num_local_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        return super()._num_local_params + int(self.learn_alpha_phase)

    def set_params(self,
                   params: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Set parameters of current object and attributes
        Args:
            params: parameters to be set

        Returns:
            Unused parameters
        """
        params = torch.as_tensor(params, dtype=self.dtype, device=self.device)

        is_finalized = self.is_finalized

        remaining_params = super().set_params(params)

        iterator = 0
        if self.learn_alpha_phase:
            if is_finalized:
                pass
            else:
                self.alpha_phase = remaining_params[..., iterator]
            iterator += 1
        self.finalize()
        return remaining_params[..., iterator:]

    def get_params(self) -> torch.Tensor:
        """
        Return all learnable parameters
        Returns:
            parameters
        """
        # Shape of params
        # [*add_dim, num_params]
        params = super().get_params()
        if self.learn_alpha_phase:
            params = torch.cat([params, self.alpha_phase[..., None]], dim=-1)
        return params

    def get_params_bounds(self) -> torch.Tensor:
        """
        Return all learnable parameters' bounds
        Returns:
            parameters bounds
        """
        # Shape of params_bounds
        # [2, num_params]

        params_bounds = super().get_params_bounds()
        if self.learn_alpha_phase:
            alpha_phase_bound = \
                torch.as_tensor(self.alpha_phase_bound, dtype=self.dtype,
                                device=self.device)[..., None]
            params_bounds = torch.cat([params_bounds, alpha_phase_bound], dim=1)
        return params_bounds

    def left_bound_linear_phase(self, times):
        """
        Compute left bounded linear phase in [0, +inf]
        Returns:
            linear phase in Tensor
        """
        # Shape of time
        # [*add_dim, num_times]

        left_bound_Linear_phase = torch.clip(
            (times - self.delay[..., None]) / self.tau[..., None], min=0)
        return left_bound_Linear_phase

    def phase(self, times: torch.Tensor):
        """
        Compute phase
        Args:
            times: times Tensor

        Returns:
            phase in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]

        phase = torch.exp(
            -self.alpha_phase[..., None] * self.left_bound_linear_phase(times))
        return phase

    def phase_to_time(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Inverse operation, compute times given phase
        Args:
            phases: phases in Tensor

        Returns:
            times in Tensor
        """
        l_phases = torch.log(phases) / (-self.alpha_phase[..., None])
        times = l_phases * self.tau[..., None] + self.delay[..., None]

        return times

    def linear_phase_to_time(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Inverse operation, linearly compute times given phase
        Args:
            phases: phases in Tensor

        Returns:
            times in Tensor
        """
        times = phases * self.tau[..., None] + self.delay[..., None]
        return times

    def unbound_linear_phase(self, times):
        """
        Compute unbounded linear phase [-inf, +inf]
        Args:
            times: times in Tensor

        Returns:
            phase in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]

        linear_phase = (times - self.delay[..., None]) / self.tau[..., None]
        return linear_phase

    def unbound_phase(self, times: torch.Tensor) -> torch.Tensor:
        """
        Compute unbounded phase
        Args:
            times: times in Tensor

        Returns:
            phase in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]
        phase = torch.exp(
            -self.alpha_phase[..., None] * self.unbound_linear_phase(times))
        return phase
