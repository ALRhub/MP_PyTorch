import matplotlib.pyplot as plt
import torch
from addict import Dict
from mp_pytorch import MPFactory

from demo import get_mp_utils
from mp_pytorch import util


def test_static_delay_and_scale():
    delay_list = torch.Tensor([0.0, 1.0, 2.0])
    tau_list = torch.Tensor([1.0, 2.0, 3.0])
    mp_list = ["promp", "dmp", "idmp"]
    for mp_type in mp_list:

        fig, axes = plt.subplots(len(delay_list), len(tau_list), sharex=True,
                                 sharey=True, squeeze=False)

        for i, delay in enumerate(delay_list):
            for j, tau in enumerate(tau_list):
                config, _, params, _, _, bc_pos, _, _ = \
                    get_mp_utils(mp_type, False, False)
                config = Dict(config)
                config.tau = tau
                config.delay = delay

                num_traj = params.shape[0]
                bc_vel = torch.zeros_like(bc_pos)
                num_t = int((tau + delay) / config.mp_args.dt) * 2 + 1
                times = util.tensor_linspace(0, torch.ones([num_traj, 1])
                                             * tau + delay, num_t).squeeze(-1)
                mp = MPFactory.init_mp(config)
                bc_time = times[:, 0] + delay
                mp.update_mp_inputs(times=times, params=params,
                                    params_L=None,
                                    bc_time=bc_time, bc_pos=bc_pos,
                                    bc_vel=bc_vel)
                traj_pos = mp.get_traj_pos()[0, :, 0]
                traj_pos = util.to_np(traj_pos)
                times = util.to_np(times[0])
                axes[i, j].plot(times, traj_pos)
        plt.show()

if __name__ == '__main__':
    test_static_delay_and_scale()