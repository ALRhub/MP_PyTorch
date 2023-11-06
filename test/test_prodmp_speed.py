import torch
from addict import Dict

from mp_pytorch import util
from mp_pytorch.mp import MPFactory


def get_mp_config():
    """
    Get the config of DMPs for testing

    Args:
        mp_type: "dmp" or "prodmp"

    Returns:
        config in dictionary
    """

    device = torch.device("cuda")

    torch.manual_seed(0)

    config = Dict()
    config.num_dof = 2
    config.tau = 3
    config.learn_tau = True
    config.learn_delay = True

    config.mp_args.num_basis = 5
    config.mp_args.basis_bandwidth_factor = 2
    config.mp_args.num_basis_outside = 0
    config.mp_args.alpha = 25
    config.mp_args.alpha_phase = 2
    config.mp_args.dt = 0.02
    config.mp_args.weights_scale = 1
    config.mp_args.goal_scale = 1

    # assume we have 3 trajectories in a batch
    num_traj = 3

    # Get trajectory scaling
    tau, delay = 4, 1
    scale_delay = torch.Tensor([tau, delay]).to(device)
    scale_delay = util.add_expand_dim(scale_delay, [0], [num_traj])

    # Get params
    params = torch.Tensor([100, 200, 300, -100, -200, -2] * config.num_dof).to(device)
    params.requires_grad = True
    params = util.add_expand_dim(params, [0], [num_traj])
    params = torch.cat([scale_delay, params], dim=-1).to(device)

    # Get times
    num_t = int(config.tau / config.mp_args.dt) * 2 + 1
    times = util.tensor_linspace(0, (tau + delay), num_t).squeeze(-1)
    times = util.add_expand_dim(times, [0], [num_traj])
    times = times.to(device)
    # Get IC
    init_time = times[:, 0]
    init_pos = 5 * torch.ones([num_traj, config.num_dof]).to(device)
    init_vel = torch.zeros_like(init_pos).to(device)

    return config, params, times, init_time, init_pos, init_vel


def speed_test():
    device = 'cuda'

    # Get config
    config, params, times, init_time, init_pos, init_vel = get_mp_config()

    # Initialize the DMP and ProDMP
    config.mp_type = "dmp"
    dmp = MPFactory.init_mp(**config.to_dict(), device=device)
    config.mp_type = "prodmp"
    prodmp = MPFactory.init_mp(**config.to_dict(), device=device)

    def traj_gen_func_dmp(params):
        params += 0.01
        dmp.update_inputs(times=times, params=params,
                          init_time=init_time, init_pos=init_pos + 0.01,
                          init_vel=init_vel)

        dmp_pos = dmp.get_traj_pos()
        dmp_vel = dmp.get_traj_vel()

    def traj_gen_func_prodmp(params):
        params += 0.01
        prodmp.update_inputs(times=times, params=params, params_L=None,
                             init_time=init_time, init_pos=init_pos + 0.01,
                             init_vel=init_vel)

        prodmp_pos = prodmp.get_traj_pos()
        prodmp_vel = prodmp.get_traj_vel()

    # Get trajectory
    print("dmp: ")
    util.how_fast(100, traj_gen_func_dmp, params)
    print("prodmp: ")
    util.how_fast(100, traj_gen_func_prodmp, params)


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    speed_test()
