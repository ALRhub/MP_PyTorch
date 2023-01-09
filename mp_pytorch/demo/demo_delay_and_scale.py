import matplotlib.pyplot as plt
import torch
from addict import Dict

from mp_pytorch.demo import get_mp_utils
from mp_pytorch.mp import MPFactory
from mp_pytorch.mp import ProMP
from mp_pytorch import util


def get_mp_scale_and_delay_util(mp_type: str, tau: float, delay: float):
    config, _, params, params_L, _, init_pos, _, _ = get_mp_utils(mp_type, False,
                                                                False)
    config = Dict(config)
    config.tau = tau
    config.delay = delay
    num_traj = params.shape[0]
    num_t = int((tau + delay) / config.mp_args.dt) * 2 + 1
    times = util.tensor_linspace(0, torch.ones([num_traj, 1])
                                 * tau + delay, num_t).squeeze(-1)
    init_time = times[:, 0] + delay
    init_vel = torch.zeros_like(init_pos)

    return config.to_dict(), times, params, params_L, init_time, init_pos, init_vel


def test_static_delay_and_scale():
    tau_list = [1.0, 2.0, 3.0]
    delay_list = [0.0, 1.0, 2.0]
    mp_list = ["promp", "dmp", "prodmp"]
    time_max = tau_list[-1] + delay_list[-1]

    for mp_type in mp_list:

        fig, axes = plt.subplots(len(tau_list), len(delay_list), sharex='all',
                                 sharey='all', squeeze=False)
        fig.suptitle(f"Static scale and delay of {mp_type}")
        for i, tau in enumerate(tau_list):
            for j, delay in enumerate(delay_list):
                config, _, params, params_L, init_time, init_pos, init_vel = \
                    get_mp_scale_and_delay_util(mp_type, tau, delay)
                config = Dict(config)
                num_traj = params.shape[0]
                num_t = int(time_max / config.mp_args.dt) * 2 + 1

                times = util.tensor_linspace(0, torch.ones(
                    [num_traj, 1]) * time_max, num_t).squeeze(-1)

                mp = MPFactory.init_mp(**config)

                init_time = times[:, 0]
                mp.update_inputs(times=times, params=params,
                                 params_L=params_L,
                                 init_time=init_time, init_pos=init_pos,
                                 init_vel=init_vel)
                traj_pos = mp.get_traj_pos()[0, :, 0]
                traj_pos = util.to_np(traj_pos)

                times = util.to_np(times[0])
                axes[i, j].plot(times, traj_pos)

                if isinstance(mp, ProMP):
                    traj_std = mp.get_traj_pos_std()[0, :, 0]
                    traj_std = util.to_np(traj_std)
                    util.fill_between(times, traj_pos, traj_std, axes[i, j])

                axes[i, j].axvline(x=delay, linestyle='--', color='r',
                                   alpha=0.3)
                axes[i, j].axvline(x=tau + delay, linestyle='--', color='r',
                                   alpha=0.3)
                axes[i, j].grid(alpha=0.2)
                axes[i, j].title.set_text(f"Scale: {tau}s, Delay: {delay}s")

        plt.show()


def test_learnable_delay_and_scale():
    tau_list = [1.0, 2.0, 3.0]
    delay_list = [0.0, 1.0, 2.0]
    mp_list = ["promp", "dmp", "prodmp"]
    for mp_type in mp_list:
        config = get_mp_utils(mp_type, learn_tau=True, learn_delay=True)[0]
        config = Dict(config)
        # Generate parameters
        num_param = config.num_dof * config.mp_args.num_basis
        params_scale_factor = 100
        params_L_scale_factor = 10

        if "dmp" in config.mp_type:
            num_param += config.num_dof
            params_scale_factor = 1000
            params_L_scale_factor = 0.1

        # assume we have 3 trajectories in a batch
        num_traj = len(tau_list) * len(delay_list)
        time_max = tau_list[-1] + delay_list[-1]
        num_t = int(time_max / config.mp_args.dt) * 2 + 1
        times = util.tensor_linspace(0, torch.ones([num_traj, 1]) * time_max,
                                     num_t).squeeze(-1)

        torch.manual_seed(0)
        params = torch.randn([1, num_param]).expand([num_traj, num_param]) \
                 * params_scale_factor
        if "dmp" in config.mp_type:
            params[:, config.mp_args.num_basis::config.mp_args.num_basis] \
                *= 0.001

        lct = torch.distributions.transforms.LowerCholeskyTransform(
            cache_size=0)
        params_L = lct(torch.randn([1, num_param, num_param]).expand(
            [num_traj, num_param, num_param])) * params_L_scale_factor

        tau_delay = torch.zeros([num_traj, 2])
        for i, tau in enumerate(tau_list):
            for j, delay in enumerate(delay_list):
                tau_delay[i * len(tau_list) + j] = torch.Tensor([tau, delay])
        params = torch.cat([tau_delay, params], dim=-1)

        init_time = times[:, 0]
        init_pos = 5 * torch.ones([num_traj, config.num_dof])
        init_vel = torch.zeros_like(init_pos)

        mp = MPFactory.init_mp(**config)
        mp.update_inputs(times=times, params=params,
                         params_L=params_L,
                         init_time=init_time, init_pos=init_pos,
                         init_vel=init_vel)

        traj_pos = mp.get_traj_pos()[..., 0]
        traj_pos = util.to_np(traj_pos)

        times = util.to_np(times)

        fig, axes = plt.subplots(len(tau_list), len(delay_list), sharex='all',
                                 sharey='all', squeeze=False)
        fig.suptitle(f"Learnable scale and delay of {mp_type}")
        for i, tau in enumerate(tau_list):
            for j, delay in enumerate(delay_list):
                axes[i, j].plot(times[i * len(tau_list) + j],
                                traj_pos[i * len(tau_list) + j])

                if isinstance(mp, ProMP):
                    traj_std = mp.get_traj_pos_std()[..., 0]
                    traj_std = util.to_np(traj_std)
                    util.fill_between(times[i * len(tau_list) + j],
                                      traj_pos[i * len(tau_list) + j],
                                      traj_std[i * len(tau_list) + j],
                                      axes[i, j])
                axes[i, j].axvline(x=delay, linestyle='--', color='r',
                                   alpha=0.3)
                axes[i, j].axvline(x=tau + delay, linestyle='--', color='r',
                                   alpha=0.3)
                axes[i, j].title.set_text(f"Scale: {tau}s, Delay: {delay}s")
                axes[i, j].grid(alpha=0.2)

        plt.show()


if __name__ == '__main__':
    test_static_delay_and_scale()
    test_learnable_delay_and_scale()
