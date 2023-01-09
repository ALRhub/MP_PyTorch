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

    torch.manual_seed(0)

    config = Dict()
    config.num_dof = 2
    config.tau = 3
    config.learn_tau = True
    config.learn_delay = True

    config.mp_args.num_basis = 9
    config.mp_args.basis_bandwidth_factor = 2
    config.mp_args.num_basis_outside = 0
    config.mp_args.alpha = 25
    config.mp_args.alpha_phase = 2
    config.mp_args.dt = 0.001
    config.mp_args.weights_scale = torch.ones([9]) * 1
    config.mp_args.goal_scale = 1

    # assume we have 3 trajectories in a batch
    num_traj = 3

    # Get trajectory scaling
    tau, delay = 4, 1
    scale_delay = torch.Tensor([tau, delay])
    scale_delay = util.add_expand_dim(scale_delay, [0], [num_traj])

    # Get params
    params = torch.Tensor([100, 200, 300, -100, -200, -300,
                           100, 200, 300, -2] * config.num_dof)
    params = util.add_expand_dim(params, [0], [num_traj])
    params = torch.cat([scale_delay, params], dim=-1)

    # Get times
    num_t = int(config.tau / config.mp_args.dt) * 2 + 1
    times = util.tensor_linspace(0, (tau + delay), num_t).squeeze(-1)
    times = util.add_expand_dim(times, [0], [num_traj])

    # Get IC
    init_time = times[:, 0]
    init_pos = 5 * torch.ones([num_traj, config.num_dof])
    init_vel = torch.zeros_like(init_pos)

    return config, params, times, init_time, init_pos, init_vel


def test_dmp_vs_prodmp_identical(plot=False):
    # Get config
    config, params, times, init_time, init_pos, init_vel = get_mp_config()

    # Initialize the DMP and ProDMP
    config.mp_type = "dmp"
    dmp = MPFactory.init_mp(**config.to_dict())
    config.mp_type = "prodmp"
    prodmp = MPFactory.init_mp(**config.to_dict())

    # Get trajectory
    dmp.update_inputs(times=times, params=params,
                      init_time=init_time, init_pos=init_pos, init_vel=init_vel)

    prodmp.update_inputs(times=times, params=params, params_L=None,
                         init_time=init_time, init_pos=init_pos, init_vel=init_vel)

    dmp_pos = dmp.get_traj_pos()
    dmp_vel = dmp.get_traj_vel()
    prodmp_pos = prodmp.get_traj_pos()
    prodmp_vel = prodmp.get_traj_vel()

    if plot:
        util.debug_plot(x=None, y=[dmp_pos[0, :, 0], prodmp_pos[0, :, 0]],
                        labels=["dmp", "prodmp"], title="DMP vs. ProDMP")

        util.debug_plot(x=None, y=[dmp_vel[0, :, 0], prodmp_vel[0, :, 0]],
                        labels=["dmp", "prodmp"], title="DMP vs. ProDMP")

    # Compute error
    error = dmp_pos - prodmp_pos
    print(f"Desired_max_error: {0.000406}, "
          f"Actual_error: {error.max()}")
    assert error.max() < 4.1e-3


if __name__ == "__main__":
    test_dmp_vs_prodmp_identical(True)
