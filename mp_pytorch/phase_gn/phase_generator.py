"""
@brief:     Phase generators in PyTorch
"""
from abc import ABC
from abc import abstractmethod
from typing import Union

import numpy as np
import torch


# Classes of Phase Generator


class PhaseGenerator(ABC):

    def __init__(self,
                 tau: float = 1.0,
                 delay: float = 0.0,
                 learn_tau: bool = False,
                 learn_delay: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 *args, **kwargs):
        """
            Basis class constructor
        Args:
            tau: trajectory length scaling factor
            delay: time to wait before execute
            learn_tau: if tau is learnable parameter
            learn_delay: if delay is learnable parameter
            dtype: torch data type
            device: torch device to run on
            *args: other arguments list
            **kwargs: other keyword arguments
        """
        self.dtype = dtype
        self.device = device

        self.tau = torch.as_tensor(tau, dtype=self.dtype, device=self.device)
        self.delay = torch.as_tensor(delay, dtype=self.dtype,
                                     device=self.device)
        self.learn_tau = learn_tau
        self.learn_delay = learn_delay

        if learn_tau:
            self.tau_bound = kwargs.get("tau_bound", [1e-5, torch.inf])
            assert len(self.tau_bound) == 2
        if learn_delay:
            self.delay_bound = kwargs.get("delay_bound", [0, torch.inf])
            assert len(self.delay_bound) == 2

        self.is_finalized = False

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
    def unbound_phase(self, times: torch.Tensor) -> torch.Tensor:
        """
        Basis class unbound phase interface
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
        return int(self.learn_tau) + int(self.learn_delay)

    @property
    def num_params(self) -> int:
        """
        Returns: number of parameters of current class plus parameters of all
        attributes
        """
        return self._num_local_params

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

        iterator = 0
        is_finalized = self.is_finalized

        if self.learn_tau:
            tau = params[..., iterator]
            assert tau.min() > 0
            if is_finalized:
                pass
            else:
                self.tau = tau
            iterator += 1
        if self.learn_delay:
            delay = params[..., iterator]
            assert delay.min() >= 0
            if is_finalized:
                pass
            else:
                self.delay = delay
            iterator += 1
        remaining_params = params[..., iterator:]

        self.finalize()
        return remaining_params

    def get_params(self) -> torch.Tensor:
        """
        Return all learnable parameters
        Returns:
            parameters
        """
        # Shape of params
        # [*add_dim, num_params]

        params = torch.as_tensor([], dtype=self.dtype, device=self.device)
        if self.learn_tau:
            params = torch.cat([params, self.tau[..., None]], dim=-1)
        if self.learn_delay:
            params = torch.cat([params, self.delay[..., None]], dim=-1)
        return params

    def get_params_bounds(self) -> torch.Tensor:
        """
        Return all learnable parameters' bounds
        Returns:
            parameters bounds
        """
        # Shape of params_bounds
        # [2, num_params]

        params_bounds = torch.zeros([2, 0], dtype=self.dtype,
                                    device=self.device)
        if self.learn_tau:
            tau_bound = torch.as_tensor(self.tau_bound, dtype=self.dtype,
                                        device=self.device)[..., None]
            params_bounds = torch.cat([params_bounds, tau_bound], dim=1)
        if self.learn_delay:
            delay_bound = torch.as_tensor(self.delay_bound, dtype=self.dtype,
                                          device=self.device)[..., None]
            params_bounds = torch.cat([params_bounds, delay_bound], dim=1)
        return params_bounds

    def finalize(self):
        """
        Mark the phase generator as finalized so that the parameters cannot be
        updated any more
        Returns: None

        """
        self.is_finalized = True

    def reset(self):
        """
        Unmark the finalization
        Returns: None

        """
        self.is_finalized = False
