"""
@brief: testing MPs
"""

from matplotlib import pyplot as plt

import mp_pytorch.util as util
from mp_pytorch.demo import get_mp_utils
from mp_pytorch.mp import MPFactory
from mp_pytorch.mp import ProDMP


def test_prodmp():
    util.print_wrap_title("test_prodmp")
    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("prodmp", True, True)
    mp = MPFactory.init_mp(**config)
    mp.update_inputs(times=times, params=params, params_L=params_L,
                     bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    assert isinstance(mp, ProDMP)
    traj_dict = mp.get_trajs(get_pos=True, get_pos_cov=True,
                             get_pos_std=True, get_vel=True,
                             get_vel_cov=True, get_vel_std=True)
    # Pos
    util.print_line_title("pos")
    print(traj_dict["pos"].shape)
    util.debug_plot(times[0], [traj_dict["pos"][0, :, 0]], title="prodmp_pos")

    # Pos_cov
    util.print_line_title("pos_cov")
    pass

    # Pos_std
    util.print_line_title("pos_std")
    plt.figure()
    util.fill_between(times[0], traj_dict["pos"][0, :, 0],
                      traj_dict["pos_std"][0, :, 0], draw_mean=True)
    plt.title("prodmp pos std")
    plt.show()

    # Vel
    util.print_line_title("vel")
    util.debug_plot(times[0], [traj_dict["vel"][0, :, 0]], title="prodmp_vel")

    # Vel_cov
    util.print_line_title("vel_cov")
    pass

    # Vel_std
    util.print_line_title("vel_std")
    plt.figure()
    print("traj_dict[vel_std].shape", traj_dict["vel_std"].shape)
    util.fill_between(times[0], traj_dict["vel"][0, :, 0],
                      traj_dict["vel_std"][0, :, 0], draw_mean=True)
    plt.title("prodmp vel std")
    plt.show()

    # Sample trajectories
    util.print_line_title("sample trajectories")
    num_smp = 50
    samples, samples_vel = mp.sample_trajectories(num_smp=num_smp)
    print("samples.shape", samples.shape)
    util.debug_plot(times[0], [samples[0, i, :, 0] for i in range(num_smp)],
                    title="prodmp_samples")

    # Parameters demo
    util.print_line_title("params_bounds")
    low, high = mp.get_params_bounds()
    print("Lower bound", low, sep="\n")
    print("Upper bound", high, sep="\n")
    print(mp.get_params_bounds().shape)

    # Learn weights
    util.print_line_title("learn weights")
    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("prodmp", False, False)

    mp = MPFactory.init_mp(**config)
    mp.update_inputs(times=times, params=params, params_L=params_L,
                     bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    params_dict = mp.learn_mp_params_from_trajs(times, demos)

    # Reconstruct demos using learned weights
    rec_demo = mp.get_traj_pos(times, **params_dict)
    util.debug_plot(times[0], [demos[0, :, 0], rec_demo[0, :, 0]],
                    labels=["demos", "rec_demos"],
                    title="ProDMP demos vs. rec_demos")


def main():
    test_prodmp()


if __name__ == "__main__":
    main()
