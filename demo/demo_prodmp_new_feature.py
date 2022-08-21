"""
@brief: testing MPs
"""

from matplotlib import pyplot as plt

import mp_pytorch.util as util
from demo_mp_config import get_mp_utils
from mp_pytorch.mp import MPFactory
from mp_pytorch.mp import ProDMP


def test_prodmp():
    util.print_wrap_title("test_prodmp")
    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("prodmp", False, False)
    mp = MPFactory.init_mp(**config)
    mp.update_inputs(times=times, params=params, params_L=params_L,
                     bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    assert isinstance(mp, ProDMP)
    traj_dict1 = mp.get_trajs(get_pos=True, get_pos_cov=True,
                              get_pos_std=True, get_vel=True,
                              get_vel_cov=True, get_vel_std=True)

    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("prodmp", False, False)

    mp2 = MPFactory.init_mp(**config)
    mp2.update_inputs(times=times, params=params * -1, params_L=params_L,
                      bc_time=bc_time, bc_pos=bc_pos * -1, bc_vel=bc_vel * -1)
    traj_dict2 = mp2.get_trajs(get_pos=True, get_pos_cov=True,
                               get_pos_std=True, get_vel=True,
                               get_vel_cov=True, get_vel_std=True)

    # Pos
    util.print_line_title("pos")
    util.debug_plot(times[0], [traj_dict["pos"][0, :, 0]
                               for traj_dict in [traj_dict1, traj_dict2]],
                    title="prodmp_pos")

    # Pos_cov
    util.print_line_title("pos_cov")
    pass

    # Pos_std
    util.print_line_title("pos_std")
    plt.figure()
    util.fill_between(times[0], traj_dict1["pos"][0, :, 0],
                      traj_dict1["pos_std"][0, :, 0], draw_mean=True)
    util.fill_between(times[0], traj_dict2["pos"][0, :, 0],
                      traj_dict2["pos_std"][0, :, 0], draw_mean=True)

    plt.title("prodmp pos std")
    plt.show()

    # Vel
    util.print_line_title("vel")
    util.debug_plot(times[0], [traj_dict["vel"][0, :, 0]
                               for traj_dict in [traj_dict1, traj_dict2]],
                    title="prodmp_vel")

    # Vel_cov
    util.print_line_title("vel_cov")
    pass

    # Vel_std
    util.print_line_title("vel_std")
    plt.figure()

    util.fill_between(times[0], traj_dict1["vel"][0, :, 0],
                      traj_dict1["vel_std"][0, :, 0], draw_mean=True)
    util.fill_between(times[0], traj_dict2["vel"][0, :, 0],
                      traj_dict2["vel_std"][0, :, 0], draw_mean=True)
    plt.title("prodmp vel std")
    plt.show()

    # Sample trajectories
    # util.print_line_title("sample trajectories")
    # num_smp = 50
    # samples, samples_vel = mp.sample_trajectories(num_smp=num_smp)
    # print("samples.shape", samples.shape)
    # util.debug_plot(times[0], [samples[0, i, :, 0] for i in range(num_smp)],
    #                 title="prodmp_samples")


def main():
    test_prodmp()


if __name__ == "__main__":
    main()
