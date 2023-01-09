import torch
from addict import Dict

from mp_pytorch.mp import MPFactory
from mp_pytorch import util


def get_mp_config():
    """
    Get the config of MPs for testing

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

    # Get params_L
    diag = torch.Tensor([10, 20, 30, 10, 20, 30,
                         10, 20, 30, 4] * config.num_dof)
    off_diag = torch.linspace(-9.5, 9.4, 190)
    params_L = util.build_lower_matrix(diag, off_diag)
    params_L = util.add_expand_dim(params_L, [0], [num_traj])

    # Get times
    num_t = int(config.tau / config.mp_args.dt) * 2 + 1
    times = util.tensor_linspace(0, (tau + delay), num_t).squeeze(-1)
    times = util.add_expand_dim(times, [0], [num_traj])

    # Get IC
    init_time = times[:, 0]
    init_pos = 5 * torch.ones([num_traj, config.num_dof])
    init_vel = torch.zeros_like(init_pos)

    return config, params, params_L, times, init_time, init_pos, init_vel


def dmp_quantitative_test(plot=False):
    config, params, params_L, times, init_time, init_pos, init_vel = get_mp_config()
    config.mp_type = "dmp"
    dmp = MPFactory.init_mp(**config.to_dict())
    dmp.update_inputs(times=times, params=params,
                      init_time=init_time, init_pos=init_pos, init_vel=init_vel)
    pos = dmp.get_traj_pos()
    vel = dmp.get_traj_vel()

    if plot:
        util.debug_plot(x=None, y=[pos[0, :, 0]], title="DMP pos")
        util.debug_plot(x=None, y=[vel[0, :, 0]], title="DMP vel")

    # Quantitative testing
    assert torch.abs(pos[0, 100, 0] - 5) < 1e-9
    assert torch.abs(pos[0, 1000, 0] - 5) < 1e-9
    assert torch.abs(pos[0, 2000, 0] - 1.2169) < 3.71e-5
    assert torch.abs(pos[0, 3000, 0] + 0.9573) < 3.6e-5
    assert torch.abs(pos[0, 4000, 0] + 2.0863) < 3.78e-5
    assert torch.abs(pos[0, 5000, 0] + 2.2132) < 2.56e-5
    assert torch.abs(pos[0, 6000, 0] + 1.8799) < 2.146e-6
    return True


def promp_quantitative_test(plot=False):
    config, params, params_L, times, init_time, init_pos, init_vel = get_mp_config()
    config.mp_type = "promp"

    # Fix the number of basis
    config.mp_args.num_basis += 1

    promp = MPFactory.init_mp(**config.to_dict())

    promp.update_inputs(times=times, params=params, params_L=params_L,
                        init_time=init_time, init_pos=init_pos, init_vel=init_vel)
    pos = promp.get_traj_pos()
    vel = promp.get_traj_vel()
    pos_flat = promp.get_traj_pos(flat_shape=True)
    pos_cov = promp.get_traj_pos_cov()
    mvn = torch.distributions.MultivariateNormal(loc=pos_flat,
                                                 covariance_matrix=pos_cov,
                                                 validate_args=False)

    if plot:
        util.debug_plot(x=None, y=[pos[0, :, 0]], title="ProMP pos")
        util.debug_plot(x=None, y=[vel[0, :, 0]], title="ProMP vel")

    # Quantitative testing
    assert torch.abs(pos[0, 100, 0] - 129.1609) < 4.6e-5
    assert torch.abs(pos[0, 1000, 0] - 129.1609) < 4.6e-5
    assert torch.abs(pos[0, 2000, 0] - 219.7397) < 4.6e-5
    assert torch.abs(pos[0, 3000, 0] + 111.4337) < 3.1e-5
    assert torch.abs(pos[0, 4000, 0] + 145.4950) < 3.1e-5
    assert torch.abs(pos[0, 5000, 0] - 203.8375) < 3.1e-5
    assert torch.abs(pos[0, 6000, 0] - 80.8178) < 3.82

    assert torch.abs(mvn.log_prob(pos_flat)[0] - 801.7334) < 1e-1
    return True


def prodmp_quantitative_test(plot=True):
    config, params, params_L, times, init_time, init_pos, init_vel = get_mp_config()
    config.mp_type = "prodmp"
    prodmp = MPFactory.init_mp(**config.to_dict())
    prodmp.update_inputs(times=times, params=params, params_L=params_L,
                         init_time=init_time, init_pos=init_pos, init_vel=init_vel)
    pos = prodmp.get_traj_pos()
    vel = prodmp.get_traj_vel()
    pos_flat = prodmp.get_traj_pos(flat_shape=True)
    pos_cov = prodmp.get_traj_pos_cov()
    mvn = torch.distributions.MultivariateNormal(loc=pos_flat,
                                                 covariance_matrix=pos_cov,
                                                 validate_args=False)

    if plot:
        util.debug_plot(x=None, y=[pos[0, :, 0]], title="ProDMP pos")
        util.debug_plot(x=None, y=[vel[0, :, 0]], title="ProDMP vel")

    # Quantitative testing
    assert torch.abs(pos[0, 100, 0] - 5) < 1e-9
    assert torch.abs(pos[0, 1000, 0] - 5) < 1e-9
    assert torch.abs(pos[0, 2000, 0] - 1.2203) < 4.37e-5
    assert torch.abs(pos[0, 3000, 0] + 0.9576) < 3.9e-5
    assert torch.abs(pos[0, 4000, 0] + 2.0867) < 3.56e-5
    assert torch.abs(pos[0, 5000, 0] + 2.2136) < 3.49e-5
    assert torch.abs(pos[0, 6000, 0] + 1.8799) < 4.73e-5

    assert torch.abs(mvn.log_prob(pos_flat)[0] - 774.2725) < 6.11e-5
    return True


if __name__ == "__main__":
    dmp_quantitative_test(plot=True)
    promp_quantitative_test(plot=True)
    prodmp_quantitative_test(plot=True)
