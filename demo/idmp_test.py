"""
@brief: testing MPs
"""

from matplotlib import pyplot as plt

import mp_pytorch.util as util
from mp_pytorch import IDMP
from mp_pytorch import MPFactory
from data_for_demo import get_mp_utils


def test_dmp_vs_idmp():
    idmp_config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("idmp", True, True)
    idmp = MPFactory.init_mp(idmp_config)
    idmp_pos = idmp.get_traj_pos(times, params, bc_time, bc_pos, bc_vel)
    idmp_vel = idmp.get_traj_vel(times, params, bc_time, bc_pos, bc_vel)

    dmp_config = get_mp_utils("dmp", True, True)[0]
    dmp = MPFactory.init_mp(dmp_config)
    dmp_pos = dmp.get_traj_pos(times, params, bc_time, bc_pos, bc_vel)
    dmp_vel = dmp.get_traj_vel(times, params, bc_time, bc_pos, bc_vel)

    util.debug_plot(times[0], [dmp_pos[0, :, 0], idmp_pos[0, :, 0]],
                    ["dmp_pos", "idmp_pos"], "pos comparison")
    util.debug_plot(times[0], [dmp_vel[0, :, 0], idmp_vel[0, :, 0]],
                    ["dmp_vel", "idmp_vel"], "pos comparison")


def test_idmp():
    util.print_wrap_title("test_idmp")
    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("idmp", True, True)
    mp = MPFactory.init_mp(config)
    mp.update_mp_inputs(times=times, params=params, params_L=params_L,
                        bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    assert isinstance(mp, IDMP)
    traj_dict = mp.get_mp_trajs(get_pos=True, get_pos_cov=True,
                                get_pos_std=True, get_vel=True,
                                get_vel_cov=True, get_vel_std=True)
    # Pos
    util.print_line_title("pos")
    print(traj_dict["pos"].shape)
    util.debug_plot(times[0], [traj_dict["pos"][0, :, 0]], title="idmp_pos")

    # Pos_cov
    util.print_line_title("pos_cov")
    pass

    # Pos_std
    util.print_line_title("pos_std")
    plt.figure()
    util.fill_between(times[0], traj_dict["pos"][0, :, 0],
                      traj_dict["pos_std"][0, :, 0], draw_mean=True)
    plt.title("idmp pos std")
    plt.show()

    # Vel
    util.print_line_title("vel")
    util.debug_plot(times[0], [traj_dict["vel"][0, :, 0]], title="idmp_vel")

    # Vel_cov
    util.print_line_title("vel_cov")
    pass

    # Vel_std
    util.print_line_title("vel_std")
    plt.figure()
    print("traj_dict[vel_std].shape", traj_dict["vel_std"].shape)
    util.fill_between(times[0], traj_dict["vel"][0, :, 0],
                      traj_dict["vel_std"][0, :, 0], draw_mean=True)
    plt.title("idmp vel std")
    plt.show()

    # Sample trajectories
    util.print_line_title("sample trajectories")
    num_smp = 5
    samples = mp.sample_trajectories(num_smp=num_smp)
    print("samples.shape", samples.shape)
    util.debug_plot(times[0], [samples[0, i, :, 0] for i in range(num_smp)],
                    title="idmp_samples")

    # Learn weights
    util.print_line_title("learn weights")
    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("idmp", False, False)

    mp = MPFactory.init_mp(config)
    mp.update_mp_inputs(times=times, params=params, params_L=params_L,
                        bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    params_dict = mp.learn_mp_params_from_trajs(times, demos)

    # Reconstruct demos using learned weights
    rec_demo = mp.get_traj_pos(times, **params_dict)
    util.debug_plot(times[0], [demos[0, :, 0], rec_demo[0, :, 0]],
                    labels=["demos", "rec_demos"],
                    title="IDMP demos vs. rec_demos")


def main():
    test_idmp()
    test_dmp_vs_idmp()


if __name__ == "__main__":
    main()
