from mp_pytorch.basis_gn import torch
from mp_pytorch.mp import MPFactory

DEFAULT_VALUES = dict(
    num_dof=2,
    tau=1.0,
    learn_tau=True,
    device="cuda",
    mp_type="promp",
    mp_args=dict(
        num_basis=3,
        dt=0.01,
        relative_goal=True,
        alpha=20.0,
        alpha_phase=1.0,
        basis_bandwidth_factor=1.0,
        num_basis_outside=0,
    ),
)

class SimplePredictor(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, learn_tau: bool):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.ReLU(),
            # torch.nn.Linear(32, 32),
            # torch.nn.ReLU(),
            # torch.nn.Linear(32, output_size),
            # torch.nn.ReLU()
        )

        self.learn_tau = learn_tau
        self.min_tau = 0.3
        self.max_tau = 3.0

    def forward(self, x):
        output = self.net(x)
        return output + 1


def print_computational_graph(grad_fn, level: int = 0):
    for next_grad_fn, _ in grad_fn.next_functions:
        next_grad_fn_str = "." * level + repr(next_grad_fn)
        if hasattr(next_grad_fn, "_raw_saved_self"):
            try:
                hasattr(next_grad_fn, "_saved_self")
            except RuntimeError:
                next_grad_fn_str += ",self=MISSING"
            else:
                if next_grad_fn._saved_self is None:
                    next_grad_fn_str += f",self=None"
                else:
                    next_grad_fn_str += f",self=[{next_grad_fn._saved_self.shape}]"

        if hasattr(next_grad_fn, "_raw_saved_other"):
            try:
                hasattr(next_grad_fn, "_saved_other")
            except RuntimeError:
                next_grad_fn_str += ",other=MISSING"
            else:
                if next_grad_fn._saved_other is None:
                    next_grad_fn_str += f",other=None"
                else:
                    next_grad_fn_str += f",other=[{next_grad_fn._saved_other.shape}]"

        print(next_grad_fn_str)
        if next_grad_fn is not None:
            print_computational_graph(next_grad_fn, level + 1)


if __name__ == "__main__":
    mp = MPFactory.init_mp(**DEFAULT_VALUES)

    batch_size = 1
    output_size = DEFAULT_VALUES["num_dof"] * DEFAULT_VALUES["mp_args"][
        "num_basis"] + 1 * DEFAULT_VALUES["learn_tau"]

    # predictor = torch.abs((torch.randn((1, output_size)).to(DEFAULT_VALUES["device"])))
    # predictor = torch.nn.Parameter(data=predictor)
    # optimizer = torch.optim.Adam([predictor,], lr=1e-3)

    input_size = 3
    predictor = SimplePredictor(input_size, output_size,
                                DEFAULT_VALUES["learn_tau"]).to(DEFAULT_VALUES["device"])
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)




    prediction_times = torch.arange(0, 1, DEFAULT_VALUES["mp_args"]["dt"]).to(
        DEFAULT_VALUES["device"]).unsqueeze(0).repeat(batch_size, 1)

    for iteration in range(3):

        referenc_trajectory = torch.randn((batch_size,
                                           prediction_times.shape[-1],
                                           DEFAULT_VALUES["num_dof"])).to(
            DEFAULT_VALUES["device"])

        initial_position = torch.randn_like(referenc_trajectory[:, 0, :])
        initial_velocity = torch.randn_like(referenc_trajectory[:, 0, :])
        initial_time = torch.zeros_like(referenc_trajectory[:, 0, 0])
        prediction_times = torch.arange(0, 1,
                                        DEFAULT_VALUES["mp_args"]["dt"]).to(
            DEFAULT_VALUES["device"]).unsqueeze(0).repeat(batch_size, 1)

        # Predict the parameters of the MP
        if isinstance(predictor, torch.nn.Module):
            params = predictor(torch.randn((batch_size, input_size)).to(DEFAULT_VALUES["device"]))
        else:
            params = predictor
        # Predict trajectory
        mp.update_inputs(
            times=prediction_times,
            init_pos=initial_position,
            init_vel=initial_velocity,
            init_time=initial_time,
            params=torch.randn_like(params),
            params_L=None,
        )
        mp.phase_gn.set_params(params[..., 0])
        trajectory = mp.get_traj_pos()

        # Compute the loss
        loss = trajectory.mean()

        # print_computational_graph(loss.grad_fn)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        # print(loss.item())
