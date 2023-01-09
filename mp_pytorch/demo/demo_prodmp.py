"""
@brief: testing MPs
"""

import torch
from matplotlib import pyplot as plt

import mp_pytorch.util as util
from mp_pytorch.demo import get_mp_utils
from mp_pytorch.mp import MPFactory
from mp_pytorch.mp import ProDMP


def test_prodmp():
    util.print_wrap_title("test_prodmp")
    config, times, params, params_L, init_time, init_pos, init_vel, demos = \
        get_mp_utils("prodmp", True, True)
    mp = MPFactory.init_mp(**config)
    mp.update_inputs(times=times, params=params, params_L=params_L,
                     init_time=init_time, init_pos=init_pos, init_vel=init_vel)
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
    config, times, params, params_L, init_time, init_pos, init_vel, demos = \
        get_mp_utils("prodmp", False, False)

    mp = MPFactory.init_mp(**config)
    params_dict = mp.learn_mp_params_from_trajs(times, demos)

    # Reconstruct demos using learned weights
    rec_demo = mp.get_traj_pos(times, **params_dict)
    util.debug_plot(times[0], [demos[0, :, 0], rec_demo[0, :, 0]],
                    labels=["demos", "rec_demos"],
                    title="ProDMP demos vs. rec_demos")

    des_init_pos = torch.zeros_like(demos[:, 0]) - 0.25
    des_init_vel = torch.zeros_like(demos[:, 0])

    params_dict = \
        mp.learn_mp_params_from_trajs(times, demos, init_time=times[:, 0],
                                      init_pos=des_init_pos, init_vel=des_init_vel)

    # Reconstruct demos using learned weights
    rec_demo = mp.get_traj_pos(times, **params_dict)
    util.debug_plot(times[0], [demos[0, :, 0], rec_demo[0, :, 0]],
                    labels=["demos", "rec_demos"],
                    title="ProDMP demos vs. rec_demos")

    # Show scaled basis
    mp.show_scaled_basis(plot=True)


def test_prodmp_disable_weights():
    util.print_wrap_title("test_prodmp_disable_weights")
    learn_tau = True
    learn_delay = True

    config, times, params, _, init_time, init_pos, init_vel, demos = \
        get_mp_utils("prodmp", learn_tau, learn_delay)

    # Disable weights
    config["mp_args"]["disable_weights"] = True
    num_dof = config["num_dof"]
    add_dim = params.shape[:-1]
    goal = 2
    params = torch.ones([*add_dim, num_dof]) * goal
    if learn_delay:
        params = torch.cat([torch.ones([*add_dim, 1]) * 1, params], dim=-1)
    if learn_tau:
        params = torch.cat([torch.ones([*add_dim, 1]) * 3, params], dim=-1)

    mp = MPFactory.init_mp(**config)
    mp.update_inputs(times=times, params=params, params_L=None,
                     init_time=init_time, init_pos=init_pos, init_vel=init_vel)
    traj_dict = mp.get_trajs(get_pos=True, get_pos_cov=False,
                             get_pos_std=False, get_vel=True,
                             get_vel_cov=False, get_vel_std=False)

    # Pos
    util.print_line_title("pos")
    print(traj_dict["pos"].shape)
    util.debug_plot(times[0], [traj_dict["pos"][0, :, 0]],
                    title="prodmp_pos, disable weights")

    # Vel
    util.print_line_title("vel")
    util.debug_plot(times[0], [traj_dict["vel"][0, :, 0]],
                    title="prodmp_vel, disable weights")


def test_prodmp_disable_goal():
    util.print_wrap_title("test_prodmp_disable_goals")
    learn_tau = True
    learn_delay = True

    config, times, params, _, init_time, init_pos, init_vel, demos = \
        get_mp_utils("prodmp", learn_tau, learn_delay)

    # Disable weights
    config["mp_args"]["disable_goal"] = True
    num_dof = config["num_dof"]
    add_dim = params.shape[:-1]
    goal = 2
    params = \
        torch.ones([*add_dim, num_dof * config['mp_args']['num_basis']]) * 500

    if learn_delay:
        params = torch.cat([torch.ones([*add_dim, 1]) * 1, params], dim=-1)
    if learn_tau:
        params = torch.cat([torch.ones([*add_dim, 1]) * 3, params], dim=-1)

    mp = MPFactory.init_mp(**config)
    mp.update_inputs(times=times, params=params, params_L=None,
                     init_time=init_time, init_pos=init_pos, init_vel=init_vel)
    traj_dict = mp.get_trajs(get_pos=True, get_pos_cov=False,
                             get_pos_std=False, get_vel=True,
                             get_vel_cov=False, get_vel_std=False)

    # Pos
    util.print_line_title("pos")
    print(traj_dict["pos"].shape)
    util.debug_plot(times[0], [traj_dict["pos"][0, :, 0]],
                    title="prodmp_pos, disable goal")

    # Vel
    util.print_line_title("vel")
    util.debug_plot(times[0], [traj_dict["vel"][0, :, 0]],
                    title="prodmp_vel, disable goal")


def main():
    test_prodmp()
    test_prodmp_disable_weights()
    test_prodmp_disable_goal()


if __name__ == "__main__":
    main()
