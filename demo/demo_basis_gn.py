from matplotlib import pyplot as plt

from basis_gn import NormalizedRBFBasisGenerator
from basis_gn import ProDMPBasisGenerator
from phase_gn import ExpDecayPhaseGenerator
from phase_gn import LinearPhaseGenerator


def demo_norm_rbf_basis():
    phase_gn = LinearPhaseGenerator(tau=3, delay=1,
                                    learn_tau=False, learn_delay=False)
    basis_gn = NormalizedRBFBasisGenerator(phase_generator=phase_gn,
                                           num_basis=14,
                                           basis_bandwidth_factor=3,
                                           num_basis_outside=2)
    times, basis_values = basis_gn.show_basis()

    plt.figure()
    for i in range(basis_values.shape[-1]):
        plt.plot(times, basis_values[:, i], label=f"basis_{i}")
        plt.grid()

    plt.legend()
    plt.show()


def demo_prodmp_basis():
    phase_gn = ExpDecayPhaseGenerator(tau=3, delay=1, alpha_phase=3,
                                      learn_tau=False, learn_delay=False,
                                      learn_alpha_phase=False)
    basis_gn = ProDMPBasisGenerator(phase_generator=phase_gn,
                                    num_basis=14,
                                    basis_bandwidth_factor=3,
                                    num_basis_outside=2)
    times, basis_values = basis_gn.show_basis()

    plt.figure()
    for i in range(basis_values.shape[-1]-1):
        plt.plot(times, basis_values[:, i], label=f"basis_{i}")
        plt.grid()

    plt.legend()
    plt.show()


if __name__ == "__main__":
    demo_norm_rbf_basis()
    demo_prodmp_basis()
