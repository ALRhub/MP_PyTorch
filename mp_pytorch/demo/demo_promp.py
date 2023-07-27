"""
@brief: testing MPs
"""

from matplotlib import pyplot as plt

import mp_pytorch.util as util
from mp_pytorch.demo import get_mp_utils
from mp_pytorch.mp import MPFactory
from mp_pytorch.mp import ProMP


def test_promp():
    util.print_wrap_title("test_promp")

    config, times, params, params_L, init_time, init_pos, init_vel, demos = \
        get_mp_utils("promp", True, True)

    mp = MPFactory.init_mp(**config)
    assert isinstance(mp, ProMP)
    mp.update_inputs(times=times, params=params, params_L=params_L,
                     init_time=init_time, init_pos=init_pos, init_vel=init_vel)
    traj_dict = mp.get_trajs(get_pos=True, get_pos_cov=True,
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
    print("traj_dict[vel].shape", traj_dict["vel"].shape)
    util.debug_plot(times[0], [traj_dict["vel"][0, :, 0]],
                    title="promp_vel_mean")

    # Vel_cov
    util.print_line_title("vel_cov")
    assert traj_dict["vel_cov"] is None

    # Vel_std
    util.print_line_title("vel_std")
    assert traj_dict["vel_std"] is None

    # Sample trajectories
    util.print_line_title("sample trajectories")
    num_smp = 50
    samples, samples_vel = mp.sample_trajectories(num_smp=num_smp)
    print("samples.shape", samples.shape)
    util.debug_plot(times[0], [samples[0, i, :, 0] for i in range(num_smp)],
                    title="promp_samples")

    # Parameters demo
    util.print_line_title("params_bounds")
    low, high = mp.get_params_bounds()
    print("Lower bound", low, sep="\n")
    print("Upper bound", high, sep="\n")
    print(mp.get_params_bounds().shape)

    # Learn weights
    util.print_line_title("learn weights")
    config, times, params, params_L, init_time, init_pos, init_vel, demos = \
        get_mp_utils("promp", False, False)

    mp = MPFactory.init_mp(**config)
    mp.update_inputs(times=times, params=params, params_L=params_L,
                     init_time=init_time, init_pos=init_pos, init_vel=init_vel)
    params_dict = mp.learn_mp_params_from_trajs(times, demos)
    # Reconstruct demos using learned weights
    rec_demo = mp.get_traj_pos(times, **params_dict)
    util.debug_plot(times[0], [demos[0, :, 0], rec_demo[0, :, 0]],
                    labels=["demos", "rec_demos"],
                    title="ProMP demos vs. rec_demos")

    # Show scaled basis
    mp.show_scaled_basis(plot=True)


def test_zero_padding_promp():
    util.print_wrap_title("test_zero_padding_promp")

    config, times, params, params_L, init_time, init_pos, init_vel, demos = \
        get_mp_utils("zero_padding_promp", False, False)

    mp = MPFactory.init_mp(**config)
    assert isinstance(mp, ProMP)
    mp.update_inputs(times=times, params=params, params_L=params_L,
                     init_time=init_time, init_pos=init_pos, init_vel=init_vel)

    # Pos
    util.print_line_title("zero padding pos")
    pos = mp.get_traj_pos()
    print("traj_dict[pos].shape", pos.shape)
    util.debug_plot(times[0], [pos[0, :, 0]], title="zero_promp_mean")

    # Vel
    util.print_line_title("zero padding vel")
    vel = mp.get_traj_vel()
    print("traj_dict[vel].shape", vel.shape)
    util.debug_plot(times[0], [vel[0, :, 0]], title="zero_promp_vel_mean")

    # Show scaled basis
    mp.show_scaled_basis(plot=True)


def main():
    test_promp()
    test_zero_padding_promp()


if __name__ == "__main__":
    main()
