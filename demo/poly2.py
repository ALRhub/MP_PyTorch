import numpy as np
import mp_pytorch.util as util
import torch
from addict import Dict

from mp_pytorch.mp import MPFactory


def get_polynomial_value(coefficients, x):
    return np.polyval(coefficients, x)


def get_times():
    return np.linspace(0, 3, 301)


def get_group_traj(via_points_x,
                   via_points_y_,
                   y_last_min,
                   y_last_max):
    poly_coefficients_list = []
    trajs_list = []
    times = get_times()
    for y_last in range(y_last_min, y_last_max, 1):
        via_points_y_[-1] = y_last
        poly_coefficients = np.polyfit(via_points_x, via_points_y_, 3)
        poly_coefficients_list.append(poly_coefficients)
        traj = np.polyval(poly_coefficients, times)
        trajs_list.append(traj)
    trajs_group = np.asarray(trajs_list)

    # Plot
    num_traj = len(trajs_list)
    util.debug_plot(util.add_expand_dim(times, [0], [num_traj]).T,
                    trajs_group.T)
    # print(poly_coefficients_list)
    return trajs_group


def get_mp_config():
    torch.manual_seed(0)
    config = Dict()
    config.num_dof = 1
    config.tau = 3
    config.mp_args.num_basis = 10
    config.mp_args.basis_bandwidth_factor = 2
    config.mp_args.num_basis_outside = 0
    config.mp_args.alpha = 25
    config.mp_args.alpha_phase = 2
    config.mp_args.dt = 0.01
    config.mp_type = "prodmp"
    return config.to_dict()


def fit_mp(traj_group):
    # Pre-processing
    config = get_mp_config()
    mp = MPFactory.init_mp(**config)

    num_traj1 = traj_group.shape[0]
    times1 = util.add_expand_dim(torch.Tensor(get_times()), [0], [num_traj1])
    mp.learn_mp_params_from_trajs(times1, torch.Tensor(traj_group)[..., None])
    util.debug_plot(times1.T, mp.get_traj_pos().squeeze().T)

    return mp.get_params()


def compute_cov(params):
    cov = torch.cov(params.T)
    reg = 10  # ProDMP
    # reg = 1e-4  # ProMP
    mean_params = params.mean(dim=0)
    L = torch.linalg.cholesky(cov + torch.eye(cov.shape[0]) * reg)
    config = get_mp_config()
    mp = MPFactory.init_mp(**config)
    bc_time = torch.tensor(0)
    bc_pos = torch.tensor([0])
    bc_vel = torch.tensor([0])
    times = torch.Tensor(get_times())
    mp.update_inputs(times=times, params=mean_params, params_L=L,
                     bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    trajs, _ = mp.sample_trajectories(num_smp=10)
    util.debug_plot(x=None, y=[trajs.squeeze().T])


if __name__ == "__main__":
    traj_group1 = get_group_traj(np.array([-0.01, 0, 1, 2, 3]),
                                 np.array([0, 0, 2, 2, 0]), -8, 13)
    traj_group2 = get_group_traj(np.array([-0.01, 0, 1, 2, 3]),
                                 np.array([0, 0, -2, -2, 0]), -12, 9)
    params1 = fit_mp(traj_group1)
    params2 = fit_mp(traj_group2)

    compute_cov(params1)
