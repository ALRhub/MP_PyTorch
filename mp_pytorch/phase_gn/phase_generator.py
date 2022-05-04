"""
@brief:     Phase generators in PyTorch
"""

from abc import ABC
from abc import abstractmethod
import torch


# Classes of Phase Generator


class PhaseGenerator(ABC):
    @abstractmethod
    def __init__(self, tau: float = 1.0, delay: float = 0.0,
                 learn_tau: bool = False, learn_delay: bool = False,
                 *args, **kwargs):
        """
            Basis class constructor
        Args:
            tau: trajectory length scaling factor
            delay: time to wait before execute
            learn_tau: if tau is learnable parameter
            learn_delay: if delay is learnable parameter
            *args: other arguments list
            **kwargs: other keyword arguments
        """
        self.tau = torch.tensor(tau)
        self.delay = torch.tensor(delay)
        self.learn_tau = learn_tau
        self.learn_delay = learn_delay

    @abstractmethod
    def phase(self, times: torch.Tensor) -> torch.Tensor:
        """
        Basis class phase interface
        Args:
            times: times in Tensor

        Returns: phases in Tensor

        """
        pass

    @abstractmethod
    def phase_to_time(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Inverse operation, compute times given phase
        Args:
            phases: phases in Tensor

        Returns:
            times in Tensor
        """
        pass

    @property
    def _num_local_params(self) -> int:
        """
        Returns: number of parameters of current class
        """
        n_param = 0
        if self.learn_tau:
            n_param += 1
        if self.learn_delay:
            n_param += 1
        return n_param

    @property
    def num_params(self) -> int:
        """
        Returns: number of parameters of current class plus parameters of all
        attributes
        """
        return self._num_local_params

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
        if self.learn_delay:
            delay = params[..., iterator]
            assert delay.min() >= 0
            self.delay = delay
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
        # [*add_dim, num_params]

        params = torch.Tensor([])
        if self.learn_tau:
            params = torch.cat([params, self.tau[..., None]], dim=-1)
        if self.learn_delay:
            params = torch.cat([params, self.delay[..., None]], dim=-1)
        return params


