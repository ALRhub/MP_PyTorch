import torch
from addict import Dict

from mp_pytorch import util
from mp_pytorch.mp import MPFactory


def get_mp_config(relative_goal=False, disable_goal=False):
    """
    Get the config of DMPs for testing

    Args:
        relative_goal: if True, the goal is relative to the initial position
        disable_goal:

    Returns:
        config in dictionary
    """

    torch.manual_seed(0)

    config = Dict()
    config.mp_type = "prodmp"
    config.num_dof = 2
    config.tau = 3
    config.learn_tau = True
    config.learn_delay = True

    config.mp_args.num_basis = 4
    config.mp_args.basis_bandwidth_factor = 2
    config.mp_args.num_basis_outside = 0
    config.mp_args.alpha = 25
    config.mp_args.alpha_phase = 2
    config.mp_args.dt = 0.001
    config.mp_args.relative_goal = relative_goal
    config.mp_args.disable_goal = disable_goal
    config.mp_args.weights_scale = torch.ones([4]) * 1
    config.mp_args.goal_scale = 1

    # assume we have 3 trajectories in a batch
    num_traj = 3

    # Get trajectory scaling
    tau, delay = 4, 1
    scale_delay = torch.Tensor([tau, delay])
    scale_delay = util.add_expand_dim(scale_delay, [0], [num_traj])

    # Get times
    num_t = int(config.tau / config.mp_args.dt) * 2 + 1
    times = util.tensor_linspace(0, (tau + delay), num_t).squeeze(-1)
    times = util.add_expand_dim(times, [0], [num_traj])

    # Get IC
    init_time = times[:, 0]
    init_pos_scalar = 1
    init_pos = init_pos_scalar * torch.ones([num_traj, config.num_dof])
    init_vel = torch.zeros_like(init_pos)

    # Get params
    goal = init_pos_scalar
    if relative_goal:
        goal -= init_pos_scalar
    if not disable_goal:
        params_list = [100, 200, 300, -100, goal]
    else:
        params_list = [100, 200, 300, -100]
    params = torch.Tensor(params_list * config.num_dof)
    params = util.add_expand_dim(params, [0], [num_traj])
    params = torch.cat([scale_delay, params], dim=-1)

    return config, params, times, init_time, init_pos, init_vel


def get_prodmp_results(relative_goal, disable_goal=False):
    config, params, times, init_time, init_pos, init_vel = get_mp_config(
        relative_goal, disable_goal)
    mp = MPFactory.init_mp(**config)
    mp.update_inputs(times, params, None, init_time, init_pos, init_vel)
    result_dict = mp.get_trajs()
    return result_dict


if __name__ == "__main__":
    no_relative_goal_results = get_prodmp_results(False)
    relative_goal_results = get_prodmp_results(True)
    disable_goal_results = get_prodmp_results(True, True)

    for key in no_relative_goal_results.keys():
        print(key)
        if no_relative_goal_results[key] is None:
            print("None")
        elif torch.allclose(no_relative_goal_results[key],
                            relative_goal_results[key])\
                and torch.allclose(no_relative_goal_results[key],
                                   disable_goal_results[key]):
            print("PASS")
        else:
            print("FAIL")
