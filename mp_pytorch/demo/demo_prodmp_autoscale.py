"""
@breif: Demo of the ProDMPs with autoscaling.
"""

from matplotlib import pyplot as plt

from mp_pytorch.demo import get_mp_utils
from mp_pytorch.mp import MPFactory


def test_prodmp_scaling(auto_scale=True, manual_w_scale=1., manual_g_scale=1.):
    config, time, params, params_L, init_time, init_pos, init_vel, demos = \
        get_mp_utils("prodmp", True, True)
    config['mp_args']['auto_scale_basis'] = auto_scale
    config['mp_args']['weights_scale'] = manual_w_scale
    config['mp_args']['goal_scale'] = manual_g_scale
    mp = MPFactory.init_mp(**config)
    mp.show_scaled_basis(True)


if __name__ == "__main__":
    test_prodmp_scaling(auto_scale=False, manual_w_scale=1., manual_g_scale=1.)
    test_prodmp_scaling(auto_scale=True, manual_w_scale=1., manual_g_scale=1.)
    test_prodmp_scaling(auto_scale=True, manual_w_scale=0.3, manual_g_scale=0.3)
