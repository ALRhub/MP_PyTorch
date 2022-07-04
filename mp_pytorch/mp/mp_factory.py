from typing import Literal

from mp_pytorch.basis_gn import ProDMPBasisGenerator
from mp_pytorch.basis_gn import NormalizedRBFBasisGenerator
from mp_pytorch.basis_gn import ZeroPaddingNormalizedRBFBasisGenerator
from mp_pytorch.phase_gn import ExpDecayPhaseGenerator
from mp_pytorch.phase_gn import LinearPhaseGenerator
from .dmp import DMP
from .prodmp import ProDMP
from .promp import ProMP


class MPFactory:
    @staticmethod
    def init_mp(mp_type: Literal["promp", "dmp", "prodmp"],
                mp_args: dict,
                num_dof: int = 1,
                tau: float = 3,
                delay: float = 0,
                learn_tau: bool = False,
                learn_delay: bool = False):
        """
        This is a helper class to initialize MPs,
        You can also directly initialize the MPs without using this class

        Create an MP instance given configuration

        Args:
            mp_type: type of movement primitives
            mp_args: arguments to a specific mp, refer each MP class
            num_dof: the number of degree of freedoms
            tau: default length of the trajectory
            delay: default delay before executing the trajectory
            learn_tau: if the length is a learnable parameter
            learn_delay: if the delay is a learnable parameter

        Returns:
            MP instance
        """

        # Get phase generator
        if mp_type == "promp":
            phase_gn = LinearPhaseGenerator(tau=tau, delay=delay,
                                            learn_tau=learn_tau,
                                            learn_delay=learn_delay)
            basis_gn = NormalizedRBFBasisGenerator(
                phase_generator=phase_gn,
                num_basis=mp_args["num_basis"],
                basis_bandwidth_factor=mp_args["basis_bandwidth_factor"],
                num_basis_outside=mp_args["num_basis_outside"])
            mp = ProMP(basis_gn=basis_gn, num_dof=num_dof, **mp_args)

        elif mp_type == 'zero_padding_promp':
            phase_gn = LinearPhaseGenerator(tau=tau,
                                            learn_tau=learn_tau,
                                            learn_delay=learn_delay)
            basis_gn = ZeroPaddingNormalizedRBFBasisGenerator(
                phase_generator=phase_gn,
                num_basis=mp_args["num_basis"],
                num_basis_zero_start=mp_args['num_basis_zero_start'],
                num_basis_zero_goal=mp_args['num_basis_zero_goal'],
                basis_bandwidth_factor=mp_args["basis_bandwidth_factor"]
            )
            mp = ProMP(basis_gn=basis_gn, num_dof=num_dof, **mp_args)

        elif mp_type == "dmp":
            phase_gn = ExpDecayPhaseGenerator(tau=tau, delay=delay,
                                              learn_tau=learn_tau,
                                              learn_delay=learn_delay,
                                              alpha_phase=mp_args[
                                                  "alpha_phase"])
            basis_gn = NormalizedRBFBasisGenerator(
                phase_generator=phase_gn,
                num_basis=mp_args["num_basis"],
                basis_bandwidth_factor=mp_args["basis_bandwidth_factor"],
                num_basis_outside=mp_args["num_basis_outside"])
            mp = DMP(basis_gn=basis_gn, num_dof=num_dof, **mp_args)
        elif mp_type == "prodmp":
            phase_gn = ExpDecayPhaseGenerator(tau=tau, delay=delay,
                                              learn_tau=learn_tau,
                                              learn_delay=learn_delay,
                                              alpha_phase=mp_args[
                                                  "alpha_phase"])
            basis_gn = ProDMPBasisGenerator(
                phase_generator=phase_gn,
                num_basis=mp_args["num_basis"],
                basis_bandwidth_factor=mp_args["basis_bandwidth_factor"],
                num_basis_outside=mp_args["num_basis_outside"],
                dt=mp_args["dt"],
                alpha=mp_args["alpha"])
            mp = ProDMP(basis_gn=basis_gn, num_dof=num_dof, **mp_args)
        else:
            raise NotImplementedError

        return mp
