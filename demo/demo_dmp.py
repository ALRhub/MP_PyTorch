"""
@brief: testing MPs
"""

import mp_pytorch.util as util
from demo_mp_config import get_mp_utils
from mp_pytorch import MPFactory


def test_dmp():
    util.print_wrap_title("test_dmp")
    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("dmp", True, True)

    mp = MPFactory.init_mp(**config)

    # params_L here is redundant, but it will not fail the update func
    mp.update_mp_inputs(times=times, params=params, params_L=params_L,
                        bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)

    traj_dict = mp.get_mp_trajs(get_pos=True, get_vel=True)

    # Pos
    util.print_line_title("pos")
    print(traj_dict["pos"].shape)
    util.debug_plot(times[0], [traj_dict["pos"][0, :, 0]], title="dmp_pos")

    # Vel
    util.print_line_title("vel")
    util.debug_plot(times[0], [traj_dict["vel"][0, :, 0]], title="dmp_vel")

    # Parameters demo
    util.print_line_title("params_bounds")
    print(mp.get_params_bounds())
    print(mp.get_params_bounds().shape)


def main():
    test_dmp()


if __name__ == "__main__":
    main()
