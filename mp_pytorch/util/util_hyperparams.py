"""
    Utilities of hyper-parameters and randomness
"""

import random
import numpy as np
import torch
from addict import Dict


class HyperParametersPool:
    def __init__(self):
        raise RuntimeError("Do not instantiate this class.")

    @staticmethod
    def set_hyperparameters(hp_dict: Dict):
        """
        Set runtime hyper-parameters
        Args:
            hp_dict: dictionary of hyper-parameters

        Returns:
            None
        """
        if hasattr(HyperParametersPool, "_hp_dict"):
            raise RuntimeError("Hyper-parameters already exist")
        else:
            # Initialize hyper-parameters dictionary
            HyperParametersPool._hp_dict = hp_dict

            # Setup random seeds globally
            seed = hp_dict.get("seed", 1234)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    @staticmethod
    def hp_dict():
        """
        Get runtime hyper-parameters
        Returns:
            hp_dict: dictionary of hyper-parameters
        """
        if not hasattr(HyperParametersPool, "_hp_dict"):
            return None
        else:
            hp_dict = HyperParametersPool._hp_dict
            return hp_dict


def np_rng():
    """
    Returns: random numpy number generator
    """
    return HyperParametersPool.np_rng()


def torch_rng():
    """
    Returns: random torch number generator
    """
    return HyperParametersPool.torch_rng()


def decide_hyperparameter(obj: any,
                          run_time_value: any,
                          parameter_key: str,
                          parameter_default: any) -> any:
    """
    A helper function to determine function's hyper-parameter
    Args:
        obj: the object asking for hyper-parameter
        run_time_value: runtime value, will be used if it is not None
        parameter_key: the key to search in the hyper-parameters pool
        parameter_default: use this value if neither runtime nor config value

    Returns:
        the parameter following the preference
        - if runtime value is given, use it
        - else if find it in the config pool, use that one
        - else use the default value
    """
    if run_time_value is not None:
        return run_time_value
    elif hasattr(obj, parameter_key):
        return getattr(obj, parameter_key)
    else:
        hp_dict = HyperParametersPool.hp_dict()
        if hp_dict is not None \
                and parameter_key in hp_dict.keys():
            actual_value = hp_dict.get(parameter_key)
            setattr(obj, parameter_key, actual_value)
            return actual_value
        else:
            return parameter_default
