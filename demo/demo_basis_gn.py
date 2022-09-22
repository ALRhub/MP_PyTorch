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
    phase_gn = ExpDecayPhaseGenerator(tau=1, delay=0, alpha_phase=3,
                                      learn_tau=False, learn_delay=False,
                                      learn_alpha_phase=False)
    basis_gn = ProDMPBasisGenerator(phase_generator=phase_gn,
                                    num_basis=10,
                                    basis_bandwidth_factor=3,
                                    num_basis_outside=0)
    times, basis_values, vel_basis_values = basis_gn.show_basis(plot=False)
    fig_1 = plt.figure(figsize=[5, 3], dpi=200)
    for i in range(basis_values.shape[-1] - 1):
        plt.plot(times, basis_values[:, i], label=f"w_basis_{i}")
    plt.grid(alpha=0.5)
    axes = plt.gca()
    axes.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
    # plt.legend()
    # plt.show()
    fig_2 = plt.figure(figsize=[5, 3], dpi=200)
    plt.plot(times, basis_values[:, -1], label=f"goal_basis")
    plt.grid(alpha=0.5)
    # plt.legend()
    # plt.show()

    fig_3 = plt.figure(figsize=[5, 3], dpi=200)
    for i in range(basis_values.shape[-1] - 1):
        plt.plot(times, vel_basis_values[:, i], label=f"w_basis_{i}")
    plt.grid(alpha=0.5)
    axes = plt.gca()
    axes.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
    # plt.legend()
    # plt.show()
    fig_4 = plt.figure(figsize=[5, 3], dpi=200)
    plt.plot(times, vel_basis_values[:, -1], label=f"goal_basis")
    plt.grid(alpha=0.5)

    fig_1.savefig("/tmp/p_basis_w.pdf", dpi=200, bbox_inches="tight")
    fig_2.savefig("/tmp/p_basis_g.pdf", dpi=200, bbox_inches="tight")
    fig_3.savefig("/tmp/v_basis_w.pdf", dpi=200, bbox_inches="tight")
    fig_4.savefig("/tmp/v_basis_g.pdf", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    # demo_norm_rbf_basis()
    demo_prodmp_basis()
