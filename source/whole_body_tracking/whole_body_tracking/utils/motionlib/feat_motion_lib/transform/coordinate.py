from __future__ import annotations

import torch

try:
    from whole_body_tracking.utils.motionlib.smpl_math_utils.smpl_math_utils import (
        angle_axis_to_rotation_matrix,
        rotation_matrix_to_angle_axis,
    )
except ModuleNotFoundError:  # pragma: no cover - test/runtime fallback without importing whole_body_tracking package
    from smpl_math_utils import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

_Y_UP_TO_Z_UP = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)


def yup_to_zup_points(x: torch.Tensor) -> torch.Tensor:
    return x @ _Y_UP_TO_Z_UP.T.to(x.device)


def yup_to_zup_root_pose(pose_aa: torch.Tensor) -> torch.Tensor:
    pose_aa = pose_aa.clone()
    rot = angle_axis_to_rotation_matrix(pose_aa[:, :3])
    rot = _Y_UP_TO_Z_UP.to(pose_aa.device) @ rot
    pose_aa[:, :3] = rotation_matrix_to_angle_axis(rot)
    return pose_aa
