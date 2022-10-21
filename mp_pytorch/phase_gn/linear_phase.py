import torch

from .phase_generator import PhaseGenerator


class LinearPhaseGenerator(PhaseGenerator):
    """Linear phase generator"""

    def phase(self, times: torch.Tensor) -> torch.Tensor:
        """
        Compute bounded phase in [0, 1]
        Args:
            times: times in Tensor

        Returns:
            phase in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]

        phase = torch.clip(
            (times - self.delay[..., None]) / self.tau[..., None], 0, 1)
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
        Compute unbounded phase
        Args:
            times: times in Tensor

        Returns:
            phase in Tensor

        """
        # Shape of time
        # [*add_dim, num_times]

        phase = (times - self.delay[..., None]) / self.tau[..., None]
        return phase
