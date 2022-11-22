"""
@breif: Demo of the ProDMPs with autoscaling.
"""

from matplotlib import pyplot as plt

from mp_pytorch.demo import get_mp_utils
from mp_pytorch.mp import MPFactory

def test_prodmp_scaling(auto_scale=True, manual_w_scale=1., manual_g_scale=1.):
    config, time, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("prodmp", True, True)
    config['mp_args']['autoscale'] = auto_scale
    config['mp_args']['weights_scale'] = manual_w_scale
    config['mp_args']['goal_scale'] = manual_g_scale
    mp = MPFactory.init_mp(**config)
    time, basis_values = mp.basis_gn.show_basis(plot=False)
    basis_values = basis_values * mp.get_weights_goal_scale(auto_scale)

    plot_scaled_basis(time, basis_values,
                      "auto scaling: {}, w_scale: {}, g_scale: {}".format(auto_scale, manual_w_scale, manual_g_scale))


def plot_scaled_basis(time, basis_values, title):
    fig, axes = plt.subplots(1, 2, sharex=True, squeeze=False)
    for i in range(basis_values.shape[-1] - 1):
        axes[0, 0].plot(time, basis_values[:, i], label="w_basis_{}".format(i))
    axes[0, 0].grid()
    axes[0, 0].legend()

    axes[0, 1].plot(time, basis_values[:, -1], label=f"goal_basis")
    axes[0, 1].grid()
    axes[0, 1].legend()

    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    test_prodmp_scaling(auto_scale=False, manual_w_scale=1., manual_g_scale=1.)
    test_prodmp_scaling(auto_scale=True, manual_w_scale=1., manual_g_scale=1.)
    test_prodmp_scaling(auto_scale=True, manual_w_scale=0.3, manual_g_scale=0.3)