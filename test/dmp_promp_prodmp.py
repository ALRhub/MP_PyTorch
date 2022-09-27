import torch
from addict import Dict
from matplotlib import pyplot as plt

from mp_pytorch import util
from mp_pytorch.mp import MPFactory
from mp_pytorch.mp import ProMP
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})

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

    config.mp_args.num_basis = 5
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

    # Get params_L
    params_std = torch.randn(params.shape)
    params_L = util.build_lower_matrix(params_std, None)

    # Get times
    num_t = int(config.tau / config.mp_args.dt) + 1
    times = util.tensor_linspace(0, (tau + delay), num_t).squeeze(-1)
    times = util.add_expand_dim(times, [0], [num_traj])

    # Get BC
    bc_time = times[:, 0]
    bc_pos = 5 * torch.ones([num_traj, config.num_dof])
    bc_vel = torch.zeros_like(bc_pos)

    # Demo
    demos = torch.zeros([*times.shape, config.num_dof])
    for i in range(config.num_dof):
        demos[..., i] = torch.sin(2 * times + i)

    return config, params, params_L, times, bc_time, bc_pos, bc_vel, demos


def fit_demo():
    config = get_mp_config()[0]
    times = get_mp_config()[3]
    demos = get_mp_config()[-1]

    # Initialize the DMP, ProDMP and ProMP
    config.mp_type = "dmp"
    dmp = MPFactory.init_mp(**config.to_dict())

    config.mp_type = "prodmp"
    prodmp = MPFactory.init_mp(**config.to_dict())

    config.mp_type = "promp"
    promp = MPFactory.init_mp(**config.to_dict())

    # Fit trajectory
    prodmp.learn_mp_params_from_trajs(times, demos)
    promp.learn_mp_params_from_trajs(times, demos)
    dmp.update_inputs(times=times, params=prodmp.params, bc_time=prodmp.bc_time,
                      bc_pos=prodmp.bc_pos, bc_vel=prodmp.bc_vel)

    # Get reconstructed trajectory
    dmp_pos = dmp.get_traj_pos()
    prodmp_pos = prodmp.get_traj_pos()
    promp_pos = promp.get_traj_pos()

    # Plot
    fig_rec = plt.figure(figsize=(5, 3.5), dpi=200)
    plt.plot(times[0].numpy(), demos[0, :, 0].numpy(), label="Demo",
             linewidth=5, color="k", zorder=0)

    plt.plot(times[0].numpy(), dmp_pos[0, :, 0].numpy(), label="DMPs",
             linewidth=3, color="r")
    plt.plot(times[0].numpy(), prodmp_pos[0, :, 0].numpy(), label="ProDMPs",
             linewidth=3, color="gold", dashes=(5, 5))
    plt.plot(times[0].numpy(), promp_pos[0, :, 0].numpy(), label="ProMPs",
             linewidth=3, color="deepskyblue", dashes=(5, 5))
    plt.legend(handlelength=3, labelspacing=1)
    # plt.ylim([-5.3, 5.3])
    plt.grid(True)

    # Change the content in below
    axins_y = plt.gca().inset_axes([0.1, 0.05, 0.30303, 0.35])

    plt.scatter(times[0, 50], demos[0, 50, 0].numpy() - 0.05, s=1500,
                facecolor='none',
                linewidth=2, edgecolor='r', zorder=100)
    axins_y.plot(times[0, :100].numpy(), demos[0, :100, 0].numpy(),
                 label="Demos", linewidth=5, color="k")
    axins_y.plot(times[0, :100].numpy(), dmp_pos[0, :100, 0].numpy(),
                 label="DMPs", linewidth=3, color="r")
    axins_y.plot(times[0, :100].numpy(), prodmp_pos[0, :100, 0].numpy(),
                 label="ProDMPs", linewidth=3, color="gold", dashes=(3, 3))
    axins_y.plot(times[0, :100].numpy(), promp_pos[0, :100, 0].numpy(),
                 label="ProMPs", linewidth=3, color="deepskyblue", dashes=(3, 3))

    # axins_y.set_xlim(1450, 1550)
    # axins_y.set_ylim(0.245, 0.29)
    axins_y.set_xticklabels([])
    axins_y.set_yticklabels([])
    axins_y.xaxis.set_visible(False)
    axins_y.yaxis.set_visible(False)

    plt.show()
    fig_rec.savefig(f"/tmp/fig_rec.pdf", dpi=200, bbox_inches="tight")


def prodmp_bc():
    config, _, params_L, times, _, _, _, demos = get_mp_config()
    config.mp_type = "prodmp"
    prodmp = MPFactory.init_mp(**config.to_dict())
    prodmp.learn_mp_params_from_trajs(times[:, 500:], demos[:, 500:])
    params = prodmp.params
    params_std = torch.randn(params.shape) * 7
    params_L = util.build_lower_matrix(params_std, None)
    prodmp.update_inputs(params=params, params_L=params_L)
    prodmp_samples, _ = prodmp.sample_trajectories(num_smp=50)

    fig_cond = plt.figure(figsize=(7, 5))
    plt.plot(times[0, :501].numpy(), demos[0, :501, 0].numpy(), linewidth=3,
             color='k', label='Old Trajectory')

    plt.plot(times[0, 500:].numpy(), prodmp_samples[0, :, :, 0].numpy().T,
             linewidth=3, zorder=10)
    plt.scatter(x=prodmp.bc_time[0].numpy(), y=prodmp.bc_pos[0, 0].numpy(),
                zorder=0)
    plt.show()
    old_times = times[0, :501].numpy()
    old_traj = demos[0, :501, 0].numpy()
    plot_times = times[0, 500:].numpy()
    plot_trajs = prodmp_samples[0, :, :, 0].numpy().T

    return fig_cond, old_times, old_traj, plot_times, plot_trajs


def promp_bc_conditioning():
    config, _, _, times, _, _, _, demos = get_mp_config()

    config.mp_type = "promp"
    promp = MPFactory.init_mp(**config.to_dict())
    params = promp.learn_mp_params_from_trajs(times, demos)["params"]

    params_std = torch.randn(params.shape)
    params_L = util.build_lower_matrix(params_std, None)
    promp.update_inputs(params_L=params_L)

    # Via-point
    cond_time = times[:, 500:501]
    cond_pos = demos[:, 500]
    cond_pos_L = util.build_lower_matrix(torch.randn([times.shape[0], 2]),
                                         None) * 0.07

    # Apply conditioning
    params_new, params_L_new = \
        promp_conditioning(promp, params, params_L, cond_time, cond_pos,
                           cond_pos_L)

    new_times = times[:, 500:]
    promp.update_inputs(times=new_times, params=params_new,
                        params_L=params_L_new)
    promp_samples, _ = promp.sample_trajectories(num_smp=100)

    fig_cond = plt.figure(figsize=(7, 5))
    plt.plot(times[0, :501].numpy(), demos[0, :501, 0].numpy(), linewidth=3,
             color='k', label='Previous Trajectory')

    plt.plot(new_times[0].numpy(), promp_samples[0, :, :, 0].numpy().T,
             linewidth=3, zorder=10)
    plt.scatter(x=cond_time[0, 0].numpy(), y=cond_pos[0, 0].numpy(), zorder=0)

    old_times = times[0, :501].numpy()
    old_traj = demos[0, :501, 0].numpy()
    plot_times = times[0, 500:].numpy()
    plot_trajs = promp_samples[0, :, :, 0].numpy().T
    return fig_cond, old_times, old_traj, plot_times, plot_trajs


if __name__ == "__main__":
    fit_demo()
    promp_fig_cond, promp_old_times, promp_old_traj, promp_plot_times, \
    promp_plot_trajs = promp_bc_conditioning()
    prodmp_fig_cond, prodmp_old_times, prodmp_old_traj, prodmp_plot_times, \
    prodmp_plot_trajs = prodmp_bc()
    fig, axes = plt.subplots(2, 1, figsize=(5, 3.5), sharex=True, squeeze=True,
                             gridspec_kw={'height_ratios': [1, 1]}, )

    # ProDMP
    axes[0].plot(prodmp_old_times, prodmp_old_traj + 0.1, "k", linewidth=3)
    axes[0].plot(prodmp_plot_times, prodmp_plot_trajs + 0.1, linewidth=2,
                 zorder=100)
    axes[0].set_xlim([0.85, 1.15])
    axes[0].set_ylim([0.6, 1.2])
    axes[0].grid(True, alpha=0.5)
    axes[0].axvline(x=1.0, linewidth=3, color="r")

    # ProMP
    axes[1].plot(promp_old_times, promp_old_traj + 0.1, "k", linewidth=3,
                 label="Previous Traj.")
    axes[1].plot(promp_plot_times, promp_plot_trajs + 0.1, linewidth=2,
                 zorder=100)
    axes[1].set_xlim([0.85, 1.15])
    axes[1].set_ylim([0.6, 1.2])
    axes[1].grid(True, alpha=0.5)
    axes[1].axvline(x=1.0, linewidth=3, color="r", label="Replan Time")
    axes[1].legend(loc="lower left", handlelength=0.8, labelspacing=0.9)

    plt.show()
    fig.savefig(f"/tmp/promp_vs_prodmp.pdf", dpi=200, bbox_inches="tight")
