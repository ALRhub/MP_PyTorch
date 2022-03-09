"""
@brief:     Phase generators in PyTorch
"""

from abc import ABC
from abc import abstractmethod

import torch
import torch.nn.functional as F


# Classes of Phase Generator
class PhaseGenerator(ABC):
    @abstractmethod
    def __init__(self, tau: float = 1.0, wait: float = 0.0,
                 learn_tau: bool = False, learn_wait: bool = False,
                 *args, **kwargs):
        """
            Basis class constructor
        Args:
            tau: trajectory length scaling factor
            wait: time to wait before execute
            learn_tau: if tau is learnable parameter
            learn_wait: if wait is learnable parameter
            *args: other arguments list
            **kwargs: other keyword arguments
        """
        self.tau = torch.tensor(tau).float()
        self.wait = torch.tensor(wait).float()
        self.learn_tau = learn_tau
        self.learn_wait = learn_wait

    @abstractmethod
    def phase(self, times: torch.Tensor) -> torch.Tensor:
        """
        Basis class phase interface
        Args:
            times: times in Tensor

        Returns: phases in Tensor

        """
        pass

    @property
    def num_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        n_param = 0
        if self.learn_tau:
            n_param += 1
        if self.learn_wait:
            n_param += 1
        return n_param

    @property
    def total_num_params(self) -> int:
        """
        Returns: number of parameters of current class plus parameters of all
        attributes
        """
        return self.num_params

    def set_params(self, params: torch.Tensor) -> torch.Tensor:
        """
        Set parameters of current object and attributes
        Args:
            params: parameters to be set

        Returns:
            Unused parameters
        """
        iterator = 0
        if self.learn_tau:
            tau = params[..., iterator]
            assert tau.min() > 0
            self.tau = tau
            iterator += 1
        if self.learn_wait:
            wait = params[..., iterator]
            assert wait.min() > 0
            self.wait = wait
            iterator += 1
        remaining_params = params[..., iterator:]
        return remaining_params

    def get_params(self) -> torch.Tensor:
        """
        Return all learnable parameters
        Returns:
            parameters
        """
        # Shape of params
        # [*add_dim, total_num_params]

        params = torch.Tensor([])
        if self.learn_tau:
            params = torch.cat([params, self.tau[..., None]], dim=-1)
        if self.learn_wait:
            params = torch.cat([params, self.wait[..., None]], dim=-1)
        return params


class LinearPhaseGenerator(PhaseGenerator):
    def __init__(self, tau: float = 1.0, wait: float = 0.0,
                 learn_tau: bool = False,
                 learn_wait: bool = False):
        """
        Constructor for linear phase generator
        Args:
            tau: trajectory length scaling factor
            wait: time to wait before execute
            learn_tau: if tau is learnable parameter
            learn_wait: if wait is learnable parameter
        """
        super(LinearPhaseGenerator, self).__init__(tau=tau, wait=wait,
                                                   learn_tau=learn_tau,
                                                   learn_wait=learn_wait)

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

        phase = (times - self.wait[..., None]) / self.tau[..., None]
        return phase


class ExpDecayPhaseGenerator(LinearPhaseGenerator):
    def __init__(self,
                 tau: float = 1.0,
                 wait: float = 0.0,
                 alpha_phase: float = 3.0,
                 learn_tau: bool = False,
                 learn_wait: bool = False,
                 learn_alpha_phase: bool = False):
        """
        Constructor for exponential decay phase generator
        Args:
            tau: trajectory length scaling factor
            wait: time to wait before execute
            alpha_phase: decaying factor: tau * dx/dt = -alpha_phase * x
            learn_tau: if tau is learnable parameter
            learn_wait: if wait is learnable parameter
            learn_alpha_phase: if alpha_phase is a learnable parameter
        """
        self.alpha_phase = torch.tensor(alpha_phase).float()
        self.learn_alpha_phase = learn_alpha_phase

        super(ExpDecayPhaseGenerator, self).__init__(tau=tau, wait=wait,
                                                     learn_tau=learn_tau,
                                                     learn_wait=learn_wait)

    @property
    def num_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        n_param = super().num_params
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
        # [*add_dim, total_num_params]
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

        phase = torch.exp(-self.alpha_phase * super().phase(times))
        return phase
