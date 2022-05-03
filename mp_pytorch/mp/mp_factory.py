from mp_pytorch.basis_gn import IDMPBasisGenerator
from mp_pytorch.basis_gn import NormalizedRBFBasisGenerator
from mp_pytorch.phase_gn import ExpDecayPhaseGenerator
from mp_pytorch.phase_gn import LinearPhaseGenerator
from .dmp import DMP
from .idmp import IDMP
from .mp_interfaces import MPInterface
from .promp import ProMP


class MPFactory:
    @staticmethod
    def init_mp(config):
        """
        Create an MP instance given configuration

        Args:
            config: config dict

        Returns:
            MP instance
        """
        num_dof = config["num_dof"]
        tau = config["tau"]
        delay = config["delay"]
        mp_type = config["mp_type"]
        mp_config = config["mp_args"]
        learn_tau = config.get("learn_tau", False)
        learn_delay = config.get("learn_delay", False)

        # Get phase generator
        if mp_type == "promp":
            phase_gn = LinearPhaseGenerator(tau=tau,delay=delay,
                                            learn_tau=learn_tau,
                                            learn_delay=learn_delay)
            basis_gn = NormalizedRBFBasisGenerator(
                phase_generator=phase_gn,
                num_basis=mp_config["num_basis"],
                basis_bandwidth_factor=mp_config["basis_bandwidth_factor"],
                num_basis_outside=mp_config["num_basis_outside"])
            mp = ProMP(basis_gn=basis_gn, num_dof=num_dof, **mp_config)

        elif mp_type == "dmp":
            phase_gn = ExpDecayPhaseGenerator(tau=tau,delay=delay,
                                              learn_tau=learn_tau,
                                              learn_delay=learn_delay,
                                              alpha_phase=mp_config[
                                                  "alpha_phase"])
            basis_gn = NormalizedRBFBasisGenerator(
                phase_generator=phase_gn,
                num_basis=mp_config["num_basis"],
                basis_bandwidth_factor=mp_config["basis_bandwidth_factor"],
                num_basis_outside=mp_config["num_basis_outside"])
            mp = DMP(basis_gn=basis_gn, num_dof=num_dof, **mp_config)
        elif mp_type == "idmp":
            phase_gn = ExpDecayPhaseGenerator(tau=tau,delay=delay,
                                              learn_tau=learn_tau,
                                              learn_delay=learn_delay,
                                              alpha_phase=mp_config[
                                                  "alpha_phase"])
            basis_gn = IDMPBasisGenerator(
                phase_generator=phase_gn,
                num_basis=mp_config["num_basis"],
                basis_bandwidth_factor=mp_config["basis_bandwidth_factor"],
                num_basis_outside=mp_config["num_basis_outside"],
                dt=mp_config["dt"],
                alpha=mp_config["alpha"])
            mp = IDMP(basis_gn=basis_gn, num_dof=num_dof, **mp_config)
        else:
            raise NotImplementedError

        return mp
