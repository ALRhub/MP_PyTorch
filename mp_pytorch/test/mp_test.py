"""
@brief: testing MPs
"""

import torch
from addict import Dict
from matplotlib import pyplot as plt

import mp_pytorch.util as util
from mp_pytorch import MPFactory
from mp_pytorch import ProMP


def get_mp_utils(mp_type: str):
    torch.manual_seed(0)
    config = Dict()
    config.num_dof = 2
    config.mp.args.num_basis = 10
    config.mp.args.basis_bandwidth_factor = 2
    config.mp.args.num_basis_outside = 1
    config.mp.args.alpha = 25
    config.mp.args.tau = 3
    config.mp.args.alpha_phase = 2
    config.mp.args.dt = 0.001
    config.mp.type = mp_type

    num_traj = 3
    num_dof = config.num_dof
    num_basis = config.mp.args.num_basis
    num_param = num_dof * num_basis

    scale_factor = 1
    if "dmp" in mp_type:
        num_param += num_dof
        scale_factor = 100

    times = util.add_expand_dim(torch.linspace(1, 3,
                                               int(2 / config.mp.args.dt + 1)),
                                [0], [num_traj])
    params = torch.randn([num_traj, num_param]) * scale_factor
    lct = torch.distributions.transforms.LowerCholeskyTransform(cache_size=0)
    params_L = lct(torch.randn([num_traj, num_param, num_param])) \
               * 0.01 * scale_factor

    bc_time = times[:, 0]
    bc_pos = torch.randn([num_traj, num_dof])
    bc_vel = torch.randn([num_traj, num_dof])

    demos = torch.zeros([*times.shape, num_dof])
    for i in range(num_dof):
        demos[..., i] = torch.sin(times + i)

    return config.to_dict(), times, params, params_L, bc_time, bc_pos, \
           bc_vel, demos


def test_promp():
    util.print_wrap_title("test_promp")
    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("promp")
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
    params_dict = mp.learn_mp_params_from_trajs(times, demos)
    # Reconstruct demos using learned weights
    rec_demo = mp.get_pos(times, **params_dict)
    util.debug_plot(times[0], [demos[0, :, 0], rec_demo[0, :, 0]],
                    labels=["demos", "rec_demos"],
                    title="ProMP demos vs. rec_demos")


def test_dmp():
    util.print_wrap_title("test_dmp")
    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("dmp")
    mp = MPFactory.init_mp(config)

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


def test_idmp():
    util.print_wrap_title("test_idmp")
    config, times, params, params_L, bc_time, bc_pos, bc_vel, demos = \
        get_mp_utils("idmp")
    mp = MPFactory.init_mp(config)
    mp.update_mp_inputs(times=times, params=params, params_L=params_L,
                        bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
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
    params_dict = mp.learn_mp_params_from_trajs(times, demos)

    # Reconstruct demos using learned weights
    rec_demo = mp.get_pos(times, **params_dict)
    util.debug_plot(times[0], [demos[0, :, 0], rec_demo[0, :, 0]],
                    labels=["demos", "rec_demos"],
                    title="IDMP demos vs. rec_demos")


def main():
    test_promp()
    test_dmp()
    test_idmp()


if __name__ == "__main__":
    main()
