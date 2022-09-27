import torch
from addict import Dict
from matplotlib import pyplot as plt

from mp_pytorch.mp import MPFactory

from mp_pytorch import util
from mp_pytorch.mp import ProMP


def get_mp_config():
    """
    Get the config of DMPs for testing

    Args:
        mp_type: "dmp" or "prodmp"

    Returns:
        config in dictionary
    """

    torch.manual_seed(100)

    config = Dict()
    config.num_dof = 2
    config.tau = 3
    config.learn_tau = False
    config.learn_delay = False

    config.mp_args.num_basis = 9
    config.mp_args.basis_bandwidth_factor = 2
    config.mp_args.num_basis_outside = 0
    config.mp_args.alpha = 25
    config.mp_args.alpha_phase = 2
    config.mp_args.dt = 0.002

    # assume we have 3 trajectories in a batch
    num_traj = 3

    # Get trajectory scaling
    tau, delay = 3, 0
    scale_delay = torch.Tensor([tau, delay])
    scale_delay = util.add_expand_dim(scale_delay, [0], [num_traj])

    # Get params
    params = torch.Tensor([100, 200, 3000, -1000, -1000, -1500,
                           1000, 2000, 3000, -1] * config.num_dof)
    params = util.add_expand_dim(params, [0], [num_traj])
    # params = torch.cat([scale_delay, params], dim=-1)

    # Get times
    num_t = int(config.tau / config.mp_args.dt) * 2 + 1
    times = util.tensor_linspace(0, (tau + delay), num_t).squeeze(-1)
    times = util.add_expand_dim(times, [0], [num_traj])

    # Get BC
    bc_time = times[:, 0]
    bc_pos = 5 * torch.ones([num_traj, config.num_dof])
    bc_vel = torch.zeros_like(bc_pos)

    return config, params, times, bc_time, bc_pos, bc_vel


def test_dmp_vs_prodmp_identical(plot=False):
    # Get config
    config, params, times, bc_time, bc_pos, bc_vel = get_mp_config()

    # Initialize the DMP and ProDMP
    config.mp_type = "dmp"
    dmp = MPFactory.init_mp(**config.to_dict())
    config.mp_type = "prodmp"
    prodmp = MPFactory.init_mp(**config.to_dict())

    config.mp_type = "promp"
    # config.mp_args.num_basis = 5
    promp = MPFactory.init_mp(**config.to_dict())

    # Get trajectory
    dmp.update_inputs(times=times, params=params,
                      bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)

    prodmp.update_inputs(times=times, params=params, params_L=None,
                         bc_time=bc_time, bc_pos=bc_pos, bc_vel=bc_vel)
    util.run_time_test(lock=True)
    dmp_pos = dmp.get_traj_pos()
    util.run_time_test(lock=False)
    dmp_vel = dmp.get_traj_vel()

    util.run_time_test(lock=True)
    prodmp_pos = prodmp.get_traj_pos()
    util.run_time_test(lock=False)

    prodmp_vel = prodmp.get_traj_vel()
    promp.learn_mp_params_from_trajs(times=times, trajs=dmp_pos)
    promp_pos = promp.get_traj_pos()

    if plot:
        # util.debug_plot(x=None, y=[dmp_pos[0, :, 0], prodmp_pos[0, :, 0]],
        #                 labels=["dmp", "prodmp"], title="DMP vs. ProDMP")
        #
        fig1 = plt.figure(figsize=(7, 5))
        plt.plot(times[0].numpy(), dmp_pos[0, :, 0].numpy(), label="DMPs",
                 linewidth=3, color='k')
        plt.plot(times[0].numpy(), prodmp_pos[0, :, 0].numpy(),
                 label="ProDMPs", linestyle="--", linewidth=3, dashes=(5, 5),
                 color='gold')
        plt.plot(times[0].numpy(), promp_pos[0, :, 0].numpy(),
                 label="ProMPs", linestyle="--", linewidth=3, dashes=(10, 5),
                 color='red')
        plt.legend(handlelength=5, borderpad=1.2, labelspacing=1.2)
        plt.ylim([-5.3, 5.3])
        plt.grid(True)
        plt.show()
        # fig1.savefig("/tmp/pos1.pdf", dpi=200, bbox_inches="tight")
        # util.debug_plot(x=None, y=[dmp_vel[0, :, 0], prodmp_vel[0, :, 0]],
        #                 labels=["dmp", "prodmp"], title="DMP vs. ProDMP")

        # fig2 = plt.figure(figsize=(7, 5))
        # plt.plot(times[0].numpy(), dmp_vel[0, :, 0].numpy(), label="DMPs",
        #          linewidth=3)
        # plt.plot(times[0].numpy(), prodmp_vel[0, :, 0].numpy(),
        #          label="ProDMPs", linestyle="--", linewidth=3)
        # plt.legend()
        # plt.show()
        # fig2.savefig("/tmp/vel1.pdf", dpi=200, bbox_inches="tight")

    # return

    ############################################################################
    # Get mean of the traj
    demo_traj = dmp_pos[0, :]

    # Get params of the traj
    promp_params = promp.learn_mp_params_from_trajs(times[0], demo_traj)["params"]

    # Get L of the traj
    params_std = torch.randn(promp_params.shape)
    promp_params_L = util.build_lower_matrix(params_std, None)

    # Get conditioning time and position
    cond_time = times[0, 1000:1001]
    cond_pos = demo_traj[1000]
    cond_pos_L = util.build_lower_matrix(torch.randn([2]), None)*0.01

    # Apply conditioning
    params_new, params_L_new = \
        promp_conditioning(promp, promp_params, promp_params_L, cond_time, cond_pos, cond_pos_L)

    new_times = times[0, 1000:]

    # Plot
    promp.update_inputs(times=new_times, params=params_new, params_L=params_L_new)
    promp_samples, _ = promp.sample_trajectories(num_smp=100)
    fig9 = plt.figure(figsize=(7, 5))
    # old traj
    plt.plot(times[0, :1000].numpy(), dmp_pos[0, :1000, 0].numpy(), linewidth=3,
             color='k', label='Old Trajectory')

    # new traj_mean + std
    # plt.plot(new_times[0].numpy(), prodmp_pos[0, :, 0].numpy(), linewidth=3)
    # util.fill_between(x=new_times[0].numpy(),
    #                   y_mean=prodmp_pos[0, :, 0].numpy(),
    #                   y_std=prodmp_std[0, :, 0].numpy())

    # samples
    plt.plot(new_times.numpy(), promp_samples[:, :, 0].numpy().T, linewidth=3, zorder=10)
    # plt.scatter(x=cond_time.numpy(), y=cond_pos[0].numpy(), zorder=0)
    plt.show()


    return
    ############################################################################
    new_bc_pos = dmp_pos[:, 1000]
    new_bc_vel = dmp_vel[:, 1000]
    new_bc_time = times[:, 1000]
    new_times = times[:, 1000:]
    params_std = torch.abs(params)
    params_std = torch.randn(params.shape)
    params_L = util.build_lower_matrix(params_std, None)

    prodmp.update_inputs(times=new_times, params_L=params_L,
                         bc_time=new_bc_time, bc_pos=new_bc_pos,
                         bc_vel=new_bc_vel)

    prodmp_pos = prodmp.get_traj_pos()
    prodmp_std = prodmp.get_traj_pos_std()
    prodmp_samples, _ = prodmp.sample_trajectories(num_smp=50)

    fig3 = plt.figure(figsize=(7, 5))
    # old traj
    plt.plot(times[0, :1000].numpy(), dmp_pos[0, :1000, 0].numpy(), linewidth=3,
             color='k', label='Old Trajectory')

    # new traj_mean + std
    # plt.plot(new_times[0].numpy(), prodmp_pos[0, :, 0].numpy(), linewidth=3)
    # util.fill_between(x=new_times[0].numpy(),
    #                   y_mean=prodmp_pos[0, :, 0].numpy(),
    #                   y_std=prodmp_std[0, :, 0].numpy())

    # samples
    plt.plot(new_times[0].numpy(), prodmp_samples[0, :, :, 0].numpy().T,
             linewidth=3)
    plt.ylim([-5.3, 5.3])
    plt.axvline(x=times[0, 1000].numpy(), alpha=0.7, linewidth=3, color='b',
                label='Replanning time')
    plt.grid(True)
    # plt.legend()
    # plt.show()
    plt.legend(handlelength=5, borderpad=1.2, labelspacing=1.2)
    fig3.savefig("/tmp/sample.pdf", dpi=200, bbox_inches="tight")


def promp_conditioning(promp: ProMP, params, params_L,
                       time, des_pos, des_pos_L):
    # Shape of params:
    # [*add_dim, num_dof * num_basis]
    #
    # Shape of params_L:
    # [*add_dim, num_dof * num_basis, num_dof * num_basis]
    #
    # Shape of time:
    # [*add_dim, num_times]
    #
    # Shape of des_pos:
    # [*add_dim, num_dof * num_times]
    #
    # Shape of des_pos_L:
    # [*add_dim, num_dof * num_times, num_dof * num_times]

    # Einsum shape:
    # [*add_dim, num_dof * num_basis, num_dof * num_basis],
    # [*add_dim, num_dof * num_basis, num_dof * num_basis],
    # -> [*add_dim, num_dof * num_basis, num_dof * num_basis]
    params_cov = torch.einsum('...ij,...kj->...ik', params_L, params_L)

    # Einsum shape:
    # [*add_dim, num_dof * num_times, num_dof * num_times],
    # [*add_dim, num_dof * num_times, num_dof * num_times],
    # -> [*add_dim, num_dof * num_times, num_dof * num_times]
    des_pos_cov = torch.einsum('...ij,...kj->...ik', des_pos_L, des_pos_L)

    # Shape:
    # [*add_dim, num_dof * num_times, num_dof * num_basis]
    m_basis = promp.basis_gn.basis_multi_dofs(time, num_dof=promp.num_dof)

    # Einsum shape:
    # [*add_dim, num_dof * num_times, num_dof * num_basis],
    # [*add_dim, num_dof * num_basis, num_dof * num_basis],
    # [*add_dim, num_dof * num_times, num_dof * num_basis],
    # -> [*add_dim, num_dof * num_times, num_dof * num_times]
    temp1 = des_pos_cov + torch.einsum('...ik,...kl,...jl->...ij',
                                       m_basis, params_cov, m_basis)

    # Shape remains:
    # [*add_dim, num_dof * num_times, num_dof * num_times]
    temp1 = torch.linalg.inv(temp1)

    # Einsum shape
    # [*add_dim, num_dof * num_times, num_dof * num_basis],
    # [*add_dim, num_dof * num_basis]
    # -> [*add_dim, num_dof * num_times]
    temp2 = des_pos - torch.einsum('...ij,...j->...i', m_basis, params)

    # Einsum shape:
    # [*add_dim, num_dof * num_basis, num_dof * num_basis],
    # [*add_dim, num_dof * num_times, num_dof * num_basis],
    # [*add_dim, num_dof * num_times, num_dof * num_times],
    # [*add_dim, num_dof * num_times],
    # -> [*add_dim, num_dof * num_basis]
    params_new = params + torch.einsum('...ji,...kj,...kl,...l->...i',
                                       params_cov, m_basis, temp1, temp2)

    # Einsum shape:
    # [*add_dim, num_dof * num_basis, num_dof * num_basis],
    # [*add_dim, num_dof * num_times, num_dof * num_basis],
    # [*add_dim, num_dof * num_times, num_dof * num_times],
    # [*add_dim, num_dof * num_times, num_dof * num_basis],
    # [*add_dim, num_dof * num_basis, num_dof * num_basis],
    # -> [*add_dim, num_dof * num_basis, num_dof * num_basis]
    params_new_cov = params_cov - torch.einsum('...ij,...kj,...kl,...lm,...mn'
                                               '->...in', params_cov, m_basis,
                                               temp1, m_basis, params_cov)
    params_new_L = torch.linalg.cholesky(params_new_cov)

    return params_new, params_new_L


if __name__ == "__main__":
    test_dmp_vs_prodmp_identical(True)
