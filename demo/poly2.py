import numpy as np
from matplotlib import pyplot as plt

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
                   y_last_max,
                   plot=False):
    poly_coefficients_list = []
    trajs_list = []
    times = get_times()
    for y_last in range(y_last_min, y_last_max, 1):
        via_points_y_[-1] = y_last
        poly_coefficients = np.polyfit(via_points_x, via_points_y_,
                                       via_points_x.shape[0] - 2)
        poly_coefficients_list.append(poly_coefficients)
        traj = np.polyval(poly_coefficients, times)
        trajs_list.append(traj)
    trajs_group = np.asarray(trajs_list)

    # Plot
    if plot:
        num_traj = len(trajs_list)
        util.debug_plot(util.add_expand_dim(times, [0], [num_traj]).T,
                        trajs_group.T)
        print(poly_coefficients_list)
    return trajs_group


def get_mp_config():
    torch.manual_seed(0)
    config = Dict()
    config.num_dof = 1
    config.tau = 3
    config.mp_args.num_basis = 20
    config.mp_args.basis_bandwidth_factor = 2
    config.mp_args.num_basis_outside = 0
    config.mp_args.alpha = 25
    config.mp_args.alpha_phase = 2
    config.mp_args.dt = 0.01
    config.mp_type = "prodmp"
    return config.to_dict()


def fit_mp(traj_group, plot=False):
    # Pre-processing
    config = get_mp_config()
    mp = MPFactory.init_mp(**config)

    num_traj1 = traj_group.shape[0]
    times1 = util.add_expand_dim(torch.Tensor(get_times()), [0], [num_traj1])
    mp.learn_mp_params_from_trajs(times1, torch.Tensor(traj_group)[..., None])
    if plot:
        util.debug_plot(times1.T, mp.get_traj_pos().squeeze().T)

    return mp.get_params()


def compute_cov(params, plot=False):
    cov = torch.cov(params.T)
    # reg = 10  # ProDMP
    reg = 5e-6  # ProMP
    mean_params = params.mean(dim=0)
    L = torch.linalg.cholesky(cov + torch.eye(cov.shape[0]) * reg)
    if plot:
        config = get_mp_config()
        mp = MPFactory.init_mp(**config)
        bc_time = torch.tensor(0)
        bc_pos = torch.tensor([0])
        bc_vel = torch.tensor([0])
        times = torch.Tensor(get_times())
        mp.update_inputs(times=times, params=mean_params, params_L=L,
                         bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
        trajs, _ = mp.sample_trajectories(num_smp=100)
        util.debug_plot(x=None, y=[trajs.squeeze().T])
        traj_mean = mp.get_traj_pos()
        traj_std = mp.get_traj_pos_std()
        plt.figure()
        util.fill_between(x=times, y_mean=traj_mean.squeeze(),
                          y_std=traj_std.squeeze(), axis=None, draw_mean=True)
        plt.show()
    return mean_params, L


def blending(mean1, L1, mean2, L2):
    pass


def combination(mean1, L1, mean2, L2):
    config = get_mp_config()
    mp = MPFactory.init_mp(**config)
    bc_time = torch.tensor(0)
    bc_pos = torch.tensor([0])
    bc_vel = torch.tensor([1])
    times = torch.Tensor(get_times())
    ############################################################################
    # Group 1
    mp.update_inputs(times=times, params=mean1, params_L=L1,
                     bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    traj_mean1 = mp.get_traj_pos().squeeze()
    traj_cov1 = mp.get_traj_pos_cov()
    std1 = torch.sqrt(torch.einsum('...ii->...i', traj_cov1))

    plt.figure()
    util.fill_between(x=times, y_mean=traj_mean1, y_std=std1, axis=None,
                      draw_mean=True, color='b')

    ############################################################################
    # Group 2
    mp.update_inputs(times=times, params=mean2, params_L=L2,
                     bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    traj_mean2 = mp.get_traj_pos().squeeze()
    traj_cov2 = mp.get_traj_pos_cov()

    std2 = torch.sqrt(torch.einsum('...ii->...i', traj_cov2))
    util.fill_between(x=times, y_mean=traj_mean2, y_std=std2, axis=None,
                      draw_mean=True, color='r')

    # plt.show()

    ############################################################################
    inv_cov1 = torch.inverse(traj_cov1)
    inv_cov2 = torch.inverse(traj_cov2)
    cov_combine = (inv_cov1 + inv_cov2).inverse()

    mean_combine = cov_combine @ (inv_cov1 @ traj_mean1 + inv_cov2 @ traj_mean2)
    combine_std = torch.sqrt(torch.einsum('...ii->...i', cov_combine))

    # plt.figure()
    util.fill_between(x=times, y_mean=mean_combine,
                      y_std=combine_std, axis=None, draw_mean=True, color='g')
    plt.ylim([-5, 5])
    plt.xlim([-0, 3])
    plt.show()


if __name__ == "__main__":
    plot = False

    traj_group1 = get_group_traj(np.array([-0.01, 0, 0.9, 2.1, 3]),
                                 np.array([0, 0, 0, 0, 0]), -6, 7,
                                 plot=plot)
    traj_group2 = get_group_traj(np.array([-0.001, 0, 1.35, 3, 4]),
                                 np.array([0, 0, -1, 1, 0]), -12, 9,
                                 plot=plot)
    params1 = fit_mp(traj_group1, plot=plot)
    params2 = fit_mp(traj_group2, plot=plot)

    mean1, L1 = compute_cov(params1, plot)
    mean2, L2 = compute_cov(params2, plot)

    combination(mean1, L1, mean2, L2)
