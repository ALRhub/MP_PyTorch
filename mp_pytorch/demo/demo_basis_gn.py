from matplotlib import pyplot as plt

from mp_pytorch.basis_gn import NormalizedRBFBasisGenerator
from mp_pytorch.basis_gn import ProDMPBasisGenerator
from mp_pytorch.phase_gn import ExpDecayPhaseGenerator
from mp_pytorch.phase_gn import LinearPhaseGenerator


def demo_norm_rbf_basis():
    phase_gn = LinearPhaseGenerator(tau=3, delay=1,
                                    learn_tau=False, learn_delay=False)
    basis_gn = NormalizedRBFBasisGenerator(phase_generator=phase_gn,
                                           num_basis=10,
                                           basis_bandwidth_factor=3,
                                           num_basis_outside=0)
    basis_gn.show_basis(plot=True)


def demo_prodmp_basis():
    phase_gn = ExpDecayPhaseGenerator(tau=3, delay=1, alpha_phase=3,
                                      learn_tau=False, learn_delay=False,
                                      learn_alpha_phase=False)
    basis_gn = ProDMPBasisGenerator(phase_generator=phase_gn,
                                    num_basis=10,
                                    basis_bandwidth_factor=3,
                                    num_basis_outside=0)
    basis_gn.show_basis(plot=True)


if __name__ == "__main__":
    demo_norm_rbf_basis()
    demo_prodmp_basis()
