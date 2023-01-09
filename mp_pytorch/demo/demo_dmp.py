"""
@brief: testing MPs
"""

import mp_pytorch.util as util
from mp_pytorch.demo import get_mp_utils
from mp_pytorch.mp import MPFactory


def test_dmp():
    util.print_wrap_title("test_dmp")
    config, times, params, params_L, init_time, init_pos, init_vel, demos = \
        get_mp_utils("dmp", True, True)

    mp = MPFactory.init_mp(**config)

    # params_L here is redundant, but it will not fail the update func

    # Uncomment this line below if you want to exclude init_time from prediction
    # times = times[..., 1:]

    mp.update_inputs(times=times, params=params, params_L=params_L,
                     init_time=init_time, init_pos=init_pos, init_vel=init_vel)

    traj_dict = mp.get_trajs(get_pos=True, get_vel=True)

    # Pos
    util.print_line_title("pos")
    print(traj_dict["pos"].shape)
    util.debug_plot(times[0], [traj_dict["pos"][0, :, 0]], title="dmp_pos")

    # Vel
    util.print_line_title("vel")
    util.debug_plot(times[0], [traj_dict["vel"][0, :, 0]], title="dmp_vel")

    # Parameters demo
    util.print_line_title("params_bounds")
    low, high = mp.get_params_bounds()
    print("Lower bound", low, sep="\n")
    print("Upper bound", high, sep="\n")
    print(mp.get_params_bounds().shape)

    # Show scaled basis
    config, times, params, params_L, init_time, init_pos, init_vel, demos = \
        get_mp_utils("dmp", False, False)

    mp = MPFactory.init_mp(**config)
    mp.show_scaled_basis(plot=True)


def main():
    test_dmp()


if __name__ == "__main__":
    main()
