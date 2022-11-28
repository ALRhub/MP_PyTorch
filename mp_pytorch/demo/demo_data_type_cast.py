import numpy as np
import torch

import mp_pytorch.util as util


def test_to_ts():
    util.print_wrap_title("test_to_ts")

    a = torch.Tensor([1, 2, 3])
    b = torch.Tensor([1, 2, 3]).double()
    c = 3.14
    d = np.array([1, 2, 3])  # This is a float 64 array
    e = np.array([1, 2, 3], dtype=float)

    util.print_line_title("Original data")
    for data in [a, b, c, d, e]:
        print(f"data: {data}")

    for data_type in [torch.float32, torch.float64]:
        for device in ["cpu", "cuda"]:
            util.print_line_title(f"data_type: {data_type}, device: {device}")
            for data in [a, b, c, d, e]:
                tensor_data = util.to_ts(data, data_type, device)
                print(tensor_data)
                print(tensor_data.device)
                print(tensor_data.type(), "\n")


def test_to_tss():
    util.print_wrap_title("test_to_tss")
    a = torch.Tensor([1, 2, 3])
    b = torch.Tensor([1, 2, 3]).double()
    c = 3.14
    d = np.array([1, 2, 3])  # This is a float 64 array
    e = np.array([1, 2, 3], dtype=float)

    util.print_line_title("Original data")
    for data in [a, b, c, d, e]:
        print(f"data: {data}")

    util.print_line_title("Casted data")
    a, b, c, d, e = util.to_tss(a, b, c, d, e, dtype=torch.float64,
                                device="cuda")
    for data in [a, b, c, d, e]:
        util.print_line()
        print(data)
        print(data.device)
        print(data.type(), "\n")


if __name__ == '__main__':
    test_to_ts()
    test_to_tss()
