"""
    Utilities of geometry computation
"""
from typing import Union
import numpy as np
import torch
import mp_pytorch.util as util

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def euler2quat(euler: Union[np.ndarray, torch.Tensor]) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    Convert Euler Angles to Quaternions.  See rotation.py for notes
    Args:
        euler: Euler angle

    Returns:
        Quaternion, WXYZ
    """
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    if util.is_np(euler):
        quat = np.zeros(euler.shape[:-1] + (4,), dtype=np.float64)
    elif util.is_ts(euler):
        quat = torch.zeros(euler.shape[:-1] + (4,))
    else:
        raise NotImplementedError
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat: Union[np.ndarray, torch.Tensor]) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    Convert Rotation Matrix to Euler Angles.

    Args:
        mat: rotation matrix

    Returns:
        euler angle
    """
    use_torch = False
    if util.is_ts(mat):
        mat = util.to_np(mat)
        use_torch = True

    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(
        mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.zeros(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy),
        -np.arctan2(-mat[..., 0, 2], cy)
    )
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    if use_torch:
        euler = torch.Tensor(euler)
    return euler


def quat2mat(quat: Union[np.ndarray, torch.Tensor]) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    Convert Quaternion to Euler Angles.

    Args:
        quat: quaternion, WXYZ

    Returns:
        rotation matrix
    """
    use_torch = False
    if util.is_ts(quat):
        quat = util.to_np(quat)
        use_torch = True

    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    result = np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat,
                      np.eye(3))

    if use_torch:
        return torch.Tensor(result)
    else:
        return result


def quat2euler(quat: Union[np.ndarray, torch.Tensor]) \
        -> Union[np.ndarray, torch.Tensor]:
    """
    Convert Quaternion to Euler Angles.
    Args:
        quat: quaternion, WXYZ

    Returns:
        euler angles
    """
    return mat2euler(quat2mat(quat))
