from __future__ import annotations

import numpy as np
import torch

try:
    from whole_body_tracking.utils.motionlib.smpl_math_utils.smpl_math_utils import angle_axis_to_rotation_matrix
except ModuleNotFoundError:  # pragma: no cover - test/runtime fallback without importing whole_body_tracking package
    from smpl_math_utils import angle_axis_to_rotation_matrix

from ..transform.coordinate import yup_to_zup_root_pose
from ..types import MotionData
from .index import build_motion_index, flatten_indices

_SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21]


# TODO check smpl motion store

class SmplMotionStore:
    def __init__(self, clips: list[MotionData], up_axis: str = "yup", device: str = "cpu") -> None:
        if up_axis not in ("yup", "zup"):
            raise ValueError(f"up_axis must be 'yup' or 'zup', got {up_axis!r}")
        if not clips:
            raise ValueError("clips list is empty")

        self.up_axis = up_axis
        self.device = device
        self.poses_flat = torch.cat([clip.pose_aa for clip in clips], dim=0).to(device)
        self.joints_flat = torch.cat([clip.smpl_joints for clip in clips], dim=0).to(device)
        self.transl_flat = torch.cat([clip.transl for clip in clips], dim=0).to(device)
        self.frame_counts = np.array([clip.num_frames for clip in clips], dtype=np.int64)
        self.motion_fps = np.array([clip.fps for clip in clips], dtype=np.float32)
        self.motion_source_fps = np.array([clip.source_fps for clip in clips], dtype=np.float32)
        self.index = build_motion_index(self.frame_counts.tolist(), device=device)

    def to_device(self, device: str | torch.device) -> None:
        self.device = str(device)
        self.poses_flat = self.poses_flat.to(device)
        self.joints_flat = self.joints_flat.to(device)
        self.transl_flat = self.transl_flat.to(device)
        self.index = build_motion_index(self.frame_counts.tolist(), device=str(device))

    def _idx(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return flatten_indices(self.index.start_idx, motion_ids, motion_steps)

    # TODO 确认是否需要预先计算这些信息，以减小 RL 训练时的计算负担
    def get_pose(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        pose = self.poses_flat[self._idx(motion_ids, motion_steps)]
        if self.up_axis == "yup":
            return yup_to_zup_root_pose(pose)
        return pose

    def get_joints(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self.joints_flat[self._idx(motion_ids, motion_steps)]

    def get_transl(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        transl = self.transl_flat[self._idx(motion_ids, motion_steps)]
        if self.up_axis == "yup":
            from ..transform.coordinate import yup_to_zup_points

            return yup_to_zup_points(transl)
        return transl

    def get_global_positions(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self.get_joints(motion_ids, motion_steps) + self.get_transl(motion_ids, motion_steps).unsqueeze(1)

    def get_global_rotations(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        pose = self.get_pose(motion_ids, motion_steps)
        batch_size = pose.shape[0]
        local_rot = angle_axis_to_rotation_matrix(pose.reshape(batch_size * 24, 3)).reshape(batch_size, 24, 3, 3)
        global_rot = torch.empty_like(local_rot)
        for joint_id, parent_id in enumerate(_SMPL_PARENTS):
            if parent_id < 0:
                global_rot[:, joint_id] = local_rot[:, joint_id]
            else:
                global_rot[:, joint_id] = global_rot[:, parent_id] @ local_rot[:, joint_id]
        return global_rot

    def get_num_motions(self) -> int:
        return len(self.frame_counts)

    def get_num_frames(self, motion_id: int) -> int:
        if motion_id < 0 or motion_id >= len(self.frame_counts):
            raise IndexError(f"motion_id {motion_id} out of range [0, {len(self.frame_counts)})")
        return int(self.frame_counts[motion_id])

    def get_motion_fps(self, motion_id: int) -> float:
        return float(self.motion_fps[motion_id])

    def get_motion_duration(self, motion_id: int) -> float:
        fps = self.get_motion_fps(motion_id)
        num_frames = self.get_num_frames(motion_id)
        return (num_frames - 1) / fps if num_frames > 1 else 0.0

    def sample_random(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        motion_ids = torch.randint(0, self.get_num_motions(), (batch_size,))
        frame_counts = torch.from_numpy(self.frame_counts[motion_ids.numpy()])
        motion_steps = (torch.rand(batch_size) * frame_counts).long()
        return motion_ids, motion_steps
