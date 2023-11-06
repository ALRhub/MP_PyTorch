import torch
from addict import Dict

import mp_pytorch.util as util
from mp_pytorch.mp import MPFactory
import tikzplotlib


def get_mp_utils(mp_type: str, learn_tau=False, learn_delay=False,
                 relative_goal=False):
    torch.manual_seed(0)
    config = Dict()

    config.num_dof = 2
    config.tau = 3
    config.learn_tau = learn_tau
    config.learn_delay = learn_delay

    config.mp_args.num_basis = 5
    config.mp_args.basis_bandwidth_factor = 2
    config.mp_args.num_basis_outside = 0
    config.mp_args.alpha = 25
    config.mp_args.alpha_phase = 0.5
    config.mp_args.dt = 0.01
    config.mp_args.weights_scale = torch.ones([config.mp_args.num_basis])
    # config.mp_args.weights_scale = 10
    config.mp_args.goal_scale = 1
    config.mp_args.relative_goal = relative_goal
    config.mp_type = mp_type

    if mp_type == "zero_padding_promp":
        config.mp_args.num_basis_zero_start = int(
            0.4 * config.mp_args.num_basis)
        config.mp_args.num_basis_zero_goal = 0

    # Generate parameters
    num_param = config.num_dof * config.mp_args.num_basis
    params_scale_factor = 100
    params_L_scale_factor = 10

    if "dmp" in config.mp_type:
        num_param += config.num_dof
        params_scale_factor = 1000
        params_L_scale_factor = 0.3

    # assume we have 3 trajectories in a batch
    num_traj = 3
    num_t = int(3 / config.mp_args.dt) * 2 + 1

    # Get parameters
    torch.manual_seed(0)

    # initial position
    init_pos = 5 * torch.ones([num_traj, config.num_dof])

    params = torch.randn([num_traj, num_param]) * params_scale_factor
    # params = torch.ones([num_traj, num_param]) * params_scale_factor

    if "dmp" in config.mp_type:
        params[:,
        config.mp_args.num_basis::config.mp_args.num_basis + 1] *= 0.001
        if relative_goal:
            params[:, config.mp_args.num_basis::config.mp_args.num_basis + 1] -= \
                init_pos

    if config.learn_delay:
        torch.manual_seed(0)
        delay = torch.rand([num_traj, 1])
        params = torch.cat([delay, params], dim=-1)
    else:
        delay = 0

    if config.learn_tau:
        torch.manual_seed(0)
        tau = torch.rand([num_traj, 1]) + 4
        params = torch.cat([tau, params], dim=-1)
        times = util.tensor_linspace(0, tau + delay, num_t).squeeze(-1)
    else:
        times = util.tensor_linspace(0, torch.ones([num_traj, 1]) * config.tau
                                     + delay, num_t).squeeze(-1)

    lct = torch.distributions.transforms.LowerCholeskyTransform(cache_size=0)
    torch.manual_seed(0)
    params_L = lct(torch.randn([num_traj, num_param, num_param])) \
               * params_L_scale_factor

    init_time = times[:, 0]

    if config.learn_delay:
        init_vel = torch.zeros_like(init_pos)
    else:
        init_vel = -5 * torch.ones([num_traj, config.num_dof])

    demos = torch.zeros([*times.shape, config.num_dof])
    for i in range(config.num_dof):
        demos[..., i] = torch.sin(2 * times + i) + 5

    return config.to_dict(), times, params, params_L, init_time, init_pos, \
           init_vel, demos


def test_prodmp():
    # Learn weights
    util.print_line_title("learn weights")
    config, times, params, params_L, init_time, init_pos, init_vel, demos = \
        get_mp_utils("prodmp", False, False, False)

    config['mp_args']['auto_scale_basis'] = True
    config['mp_args']['weights_scale'] = 2
    config['mp_args']['goal_scale'] = 2
    mp = MPFactory.init_mp(**config)
    mp.learn_mp_params_from_trajs(times, demos)

    lct = torch.distributions.transforms.LowerCholeskyTransform(cache_size=0)
    cov = torch.randn([3, 12, 12]) * 0.1
    cov[..., -1, :] = cov[..., -1, :]
    cov[..., -1] = cov[..., -1]
    params_L = lct(cov)
    # params_L[..., -1, :] = params_L[..., -1, :] * 0.001
    mp.params_L = params_L
    # for seed in range(10):
    #     print(seed)
    #     torch.manual_seed(seed)
    #     num_smp = 10
    #     smp_traj = mp.sample_trajectories(num_smp=num_smp)[0]
    #     util.debug_plot(times[0], [smp_traj[0, i, :, 0] for i in range(num_smp)])

    torch.manual_seed(2)
    num_smp = 10
    smp_traj = mp.sample_trajectories(num_smp=num_smp)[0] * 0.1 - 0.5
    # util.debug_plot(times[0], [smp_traj[0, i, :, 0] for i in [1, 5, 6, 7, 9]])

    # BBRL
    import matplotlib.pyplot as plt
    plt.figure()
    # for i in [1, 5, 6, 7, 9]:
    for i in [9]:
        plt.plot(times[0][::5], smp_traj[0, i, :, 0][::5], linewidth=2)
    plt.grid(alpha=0.5)
    plt.xlabel("Time [s]", fontsize=20)
    plt.ylabel("Torque [N·m]", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.show()
    tikzplotlib.save("test_prodmp.tex")
    plt.close()

    # TCP
    plt.figure()
    for i in [1, 5, 6, 7]:
        plt.plot(times[0][::5], smp_traj[0, i, :, 0][::5], linewidth=2,
                 color='gray', alpha=0.5)
    plt.plot(times[0][::5][:20], smp_traj[0, 9, :, 0][::5][:20], linewidth=2)
    plt.plot(times[0][::5][20:40], smp_traj[0, 9, :, 0][::5][20:40],
             linewidth=2)
    plt.plot(times[0][::5][40:60], smp_traj[0, 9, :, 0][::5][40:60],
             linewidth=2)
    plt.plot(times[0][::5][60:80], smp_traj[0, 9, :, 0][::5][60:80],
             linewidth=2)
    plt.plot(times[0][::5][80:100], smp_traj[0, 9, :, 0][::5][80:100],
             linewidth=2)
    plt.plot(times[0][::5][100:], smp_traj[0, 9, :, 0][::5][100:], linewidth=2)

    plt.scatter(times[0][::5][::20], smp_traj[0, 9, :, 0][::5][::20], s=100,
                color='black', zorder=1000)
    plt.grid(alpha=0.5)
    plt.xlabel("Time [s]", fontsize=20)
    plt.ylabel("Torque [N·m]", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.show()
    tikzplotlib.save("tcp.tex")
    plt.close()

    # PPO
    ppo_traj = smp_traj[0, 9, :, 0]
    ppo_noise = torch.randn(ppo_traj.shape) * 0.15
    fig = plt.figure()
    plt.plot(times[0][::5], (ppo_traj)[::5], label="Mean")
    plt.plot(times[0][::5], (ppo_traj + ppo_noise)[::5], label="Mean + Noise")
    plt.grid(alpha=0.5)
    plt.xlabel("Time [s]", fontsize=20)
    plt.ylabel("Torque [N·m]", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15, loc='upper right', ncol=1)
    # plt.show()
    tikzplotlib.save("test_ppo.tex", figure=fig)
    plt.close()


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


if __name__ == "__main__":
    test_prodmp()
