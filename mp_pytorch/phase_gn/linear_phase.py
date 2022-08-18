import torch
from .phase_generator import PhaseGenerator


class LinearPhaseGenerator(PhaseGenerator):
    def __init__(self, tau: float = 1.0, delay: float = 0.0,
                 learn_tau: bool = False,
                 learn_delay: bool = False):
        """
        Constructor for linear phase generator
        Args:
            tau: trajectory length scaling factor
            delay: time to wait before execute
            learn_tau: if tau is learnable parameter
            learn_delay: if delay is learnable parameter
        """
        super(LinearPhaseGenerator, self).__init__(tau=tau, delay=delay,
                                                   learn_tau=learn_tau,
                                                   learn_delay=learn_delay)

    def phase(self, times: torch.Tensor) -> torch.Tensor:
        """
        Compute phase
        Args:
            times: times in Tensor

        Returns:
            phase in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]

        phase = torch.clip(self.unbound_phase(times), 0, 1)
        return phase

    def phase_to_time(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Inverse operation, compute times given phase
        Args:
            phases: phases in Tensor

        Returns:
            times in Tensor
        """
        times = phases * self.tau[..., None] + self.delay[..., None]
        return times

    def unbound_phase(self, times: torch.Tensor) -> torch.Tensor:
        """
        Compute phase
        Args:
            times: times in Tensor

        Returns:
            phase in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]

        phase = (times - self.delay[..., None]) / self.tau[..., None]
        return phase
