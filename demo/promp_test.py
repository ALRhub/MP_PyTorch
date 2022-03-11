"""
@brief: testing MPs
"""

from matplotlib import pyplot as plt

import mp_pytorch.util as util
from mp_pytorch import MPFactory
from mp_pytorch import ProMP
from demo.data_for_demo import get_mp_utils


def test_promp():
    util.print_wrap_title("test_promp")

    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("promp", True, True)

    mp = MPFactory.init_mp(config)
    assert isinstance(mp, ProMP)
    mp.update_mp_inputs(times=times, params=params, params_L=params_L,
                        bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    traj_dict = mp.get_mp_trajs(get_pos=True, get_pos_cov=True,
                                get_pos_std=True, get_vel=True,
                                get_vel_cov=True, get_vel_std=True)
    # Pos
    util.print_line_title("pos")
    print("traj_dict[pos].shape", traj_dict["pos"].shape)
    util.debug_plot(times[0], [traj_dict["pos"][0, :, 0]], title="promp_mean")

    # Pos_cov
    util.print_line_title("pos_cov")
    pass

    # Pos_std
    util.print_line_title("pos_std")
    plt.figure()
    print("traj_dict[pos_std].shape", traj_dict["pos_std"].shape)
    util.fill_between(times[0], traj_dict["pos"][0, :, 0],
                      traj_dict["pos_std"][0, :, 0], draw_mean=True)
    plt.title("promp std")
    plt.show()

    # Vel
    util.print_line_title("vel")
    assert traj_dict["vel"] is None

    # Vel_cov
    util.print_line_title("vel_cov")
    assert traj_dict["vel_cov"] is None

    # Vel_std
    util.print_line_title("vel_std")
    assert traj_dict["vel_std"] is None

    # Sample trajectories
    util.print_line_title("sample trajectories")
    num_smp = 5
    samples = mp.sample_trajectories(num_smp=num_smp)
    print("samples.shape", samples.shape)
    util.debug_plot(times[0], [samples[0, i, :, 0] for i in range(num_smp)],
                    title="promp_samples")

    # Learn weights
    util.print_line_title("learn weights")
    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("promp", False, False)

    mp = MPFactory.init_mp(config)
    mp.update_mp_inputs(times=times, params=params, params_L=params_L,
                        bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    params_dict = mp.learn_mp_params_from_trajs(times, demos)
    # Reconstruct demos using learned weights
    rec_demo = mp.get_traj_pos(times, **params_dict)
    util.debug_plot(times[0], [demos[0, :, 0], rec_demo[0, :, 0]],
                    labels=["demos", "rec_demos"],
                    title="ProMP demos vs. rec_demos")


def main():
    test_promp()


if __name__ == "__main__":
    main()
