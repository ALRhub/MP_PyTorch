import torch

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

    # Get BC
    bc_time = times[:, 0]
    bc_pos = 5 * torch.ones([num_traj, config.num_dof])
    bc_vel = torch.zeros_like(bc_pos)

    return config, params, times, bc_time, bc_pos, bc_vel


def test_numerical_dmp(device_name: str):
    config, params, times, bc_time, bc_pos, bc_vel = get_mp_config()
    # Initialize the DMP
    config.mp_type = "dmp"
    mp = MPFactory.init_mp(**config.to_dict())

    # Get trajectory
    mp.update_inputs(times=times, params=params,
                      bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    pos = mp.get_traj_pos()
    vel = mp.get_traj_vel()

    util.debug_plot(x=None, y=[pos[0, :, 0]],
                    labels=[config.mp_type], title=config.mp_type)

    util.debug_plot(x=None, y=[vel[0, :, 0]],
                    labels=[config.mp_type], title=config.mp_type)
    torch.save(pos, f'{device_name}_{config.mp_type}_pos.pt')
    torch.save(vel, f'{device_name}_{config.mp_type}_vel.pt')


def test_numerical_prodmp(device_name: str):
    config, params, times, bc_time, bc_pos, bc_vel = get_mp_config()
    # Initialize the DMP
    config.mp_type = "prodmp"
    mp = MPFactory.init_mp(**config.to_dict())

    y_1_value = mp.basis_gn.y_1_value
    y_2_value = mp.basis_gn.y_2_value
    pc_pos_basis = mp.basis_gn.pc_pos_basis
    pc_vel_basis = mp.basis_gn.pc_vel_basis

    torch.save(pc_pos_basis, f'{device_name}_{config.mp_type}_pc_pos_basis.pt')
    torch.save(pc_vel_basis, f'{device_name}_{config.mp_type}_pc_vel_basis.pt')
    torch.save(y_1_value, f'{device_name}_{config.mp_type}_y_1_value.pt')
    torch.save(y_2_value, f'{device_name}_{config.mp_type}_y_2_value.pt')

    # Get trajectory
    mp.update_inputs(times=times, params=params,
                      bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    pos = mp.get_traj_pos()
    vel = mp.get_traj_vel()

    util.debug_plot(x=None, y=[pos[0, :, 0]],
                    labels=[config.mp_type], title=config.mp_type)

    util.debug_plot(x=None, y=[vel[0, :, 0]],
                    labels=[config.mp_type], title=config.mp_type)
    torch.save(pos, f'{device_name}_{config.mp_type}_pos.pt')
    torch.save(vel, f'{device_name}_{config.mp_type}_vel.pt')


if __name__ == '__main__':
    device = "Intel_6700K"
    test_numerical_dmp(device)
    test_numerical_prodmp(device)
