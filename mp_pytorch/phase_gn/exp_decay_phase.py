import torch
from .linear_phase import LinearPhaseGenerator


class ExpDecayPhaseGenerator(LinearPhaseGenerator):
    def __init__(self,
                 tau: float = 1.0,
                 delay: float = 0.0,
                 alpha_phase: float = 3.0,
                 learn_tau: bool = False,
                 learn_delay: bool = False,
                 learn_alpha_phase: bool = False):
        """
        Constructor for exponential decay phase generator
        Args:
            tau: trajectory length scaling factor
            delay: time to wait before execute
            alpha_phase: decaying factor: tau * dx/dt = -alpha_phase * x
            learn_tau: if tau is learnable parameter
            learn_delay: if delay is learnable parameter
            learn_alpha_phase: if alpha_phase is a learnable parameter
        """
        self.alpha_phase = torch.tensor(alpha_phase).float()
        self.learn_alpha_phase = learn_alpha_phase

        super(ExpDecayPhaseGenerator, self).__init__(tau=tau, delay=delay,
                                                     learn_tau=learn_tau,
                                                     learn_delay=learn_delay)

    @property
    def _num_local_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        n_param = super()._num_local_params
        if self.learn_alpha_phase:
            n_param += 1

        return n_param

    def set_params(self, params: torch.Tensor) -> torch.Tensor:
        """
        Set parameters of current object and attributes
        Args:
            params: parameters to be set

        Returns:
            Unused parameters
        """
        remaining_params = super().set_params(params)

        iterator = 0
        if self.learn_alpha_phase:
            self.alpha_phase = remaining_params[..., iterator]
            iterator += 1
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

        phase = torch.exp(-self.alpha_phase[..., None] * super().phase(times))
        return phase

    def phase_to_time(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Inverse operation, compute times given phase
        Args:
            phases: phases in Tensor

        Returns:
            times in Tensor
        """
        times = super().phase_to_time(torch.log(phases) /
                                      (-self.alpha_phase[..., None]))

        return times