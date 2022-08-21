import torch

import mp_pytorch.util as util
import numpy as np
import matplotlib.pyplot as plt

from demo.demo_mp_config import get_mp_utils
from mp_pytorch.mp import MPFactory


def polynomial(data, poly_coeffs):
    a, b, c, d = poly_coeffs
    return a * pow(data, 3) + b * pow(data, 2) + c * data + d


# Group1
point1 = np.array([0, 0])
point2 = np.array([1, 2])
point3 = np.array([2, 2])

point4_y = np.arange(-10, 11, 1)
point4_x = 3

via_point_group1 = np.zeros([21, 4, 2])
via_point_group1[:, 0] = point1
via_point_group1[:, 1] = point2
via_point_group1[:, 2] = point3
via_point_group1[:, 3, 0] = point4_x
via_point_group1[:, 3, 1] = point4_y

print(via_point_group1)
poly_group1 = []
for i, via_points in enumerate(via_point_group1):
    poly_group1.append(
        np.polyfit(x=via_points[:, 0], y=via_points[:, 1], deg=3))
poly_group1 = np.asarray(poly_group1)
print(poly_group1)
poly_object_list = []

time_pts = np.linspace(0, 3, 301)
plt.figure()
traj_group1 = np.zeros([21, 301])
for i in range(traj_group1.shape[0]):
    traj_group1[i] = polynomial(time_pts, poly_group1[i]) + np.random.rand(1) *1e-1
    plt.plot(time_pts, traj_group1[i])

plt.ylim([-5, 5])
plt.xlim([-0, 3])
plt.title("demos")
plt.show()
times1 = util.add_expand_dim(torch.Tensor(time_pts), [0], [21])
traj_group1 = torch.Tensor(traj_group1)[..., None]

config = get_mp_utils("prodmp", False, False)[0]
config["num_dof"] = 1
mp = MPFactory.init_mp(**config)
mp.learn_mp_params_from_trajs(times1, traj_group1)
params = mp.get_params()
mean_params = torch.mean(params, dim=0, keepdim=False)
cov_params = torch.cov(params.T)
mp.set_mp_params_variances(torch.cholesky(cov_params))
traj_dict = mp.get_trajs()



# Pos
util.print_line_title("pos")
print(traj_dict["pos"].shape)
util.debug_plot(times1.T, [traj_dict["pos"][..., 0].T], title="prodmp_rec_pos")

# Pos_std
util.print_line_title("pos_std")
plt.figure()
util.fill_between(times[0], traj_dict["pos"][0, :, 0],
                  traj_dict["pos_std"][0, :, 0], draw_mean=True)
plt.title("prodmp pos std")
plt.show()
