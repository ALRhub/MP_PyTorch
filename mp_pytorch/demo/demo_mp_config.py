import torch
from addict import Dict

import mp_pytorch.util as util


def get_mp_utils(mp_type: str, learn_tau=False, learn_delay=False):
    torch.manual_seed(0)
    config = Dict()

    config.num_dof = 2
    config.tau = 3
    config.learn_tau = learn_tau
    config.learn_delay = learn_delay

    config.mp_args.num_basis = 10
    config.mp_args.basis_bandwidth_factor = 2
    config.mp_args.num_basis_outside = 0
    config.mp_args.alpha = 25
    config.mp_args.alpha_phase = 2
    config.mp_args.dt = 0.01
    config.mp_args.weights_scale = torch.ones([config.mp_args.num_basis])
    # config.mp_args.weights_scale = 10
    config.mp_args.goal_scale = 1
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

    params = torch.randn([num_traj, num_param]) * params_scale_factor

    if "dmp" in config.mp_type:
        params[:, config.mp_args.num_basis::config.mp_args.num_basis] *= 0.001

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
    init_pos = 5 * torch.ones([num_traj, config.num_dof])
    if config.learn_delay:
        init_vel = torch.zeros_like(init_pos)
    else:
        init_vel = -5 * torch.ones([num_traj, config.num_dof])

    demos = torch.zeros([*times.shape, config.num_dof])
    for i in range(config.num_dof):
        demos[..., i] = torch.sin(2 * times + i)

    return config.to_dict(), times, params, params_L, init_time, init_pos, \
           init_vel, demos
