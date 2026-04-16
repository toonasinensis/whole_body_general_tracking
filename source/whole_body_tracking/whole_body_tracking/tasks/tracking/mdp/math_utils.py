import torch

from isaaclab.utils.math import matrix_from_quat


@torch.jit.script
def quat_to_6d(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to 6D rotation representation.
    Returns a tensor of shape (..., 6)
    """
    # 使用你原来的函数得到旋转矩阵
    R = matrix_from_quat(quaternions)  # (..., 3, 3)

    # 提取前两列作为6D表示
    # col0 = R[..., :, 0]   # shape (..., 3)
    # col1 = R[..., :, 1]   # shape (..., 3)
    # 拼接成 (..., 6)
    return torch.cat((R[..., :, 0], R[..., :, 1]), dim=-1)
