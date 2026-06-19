from __future__ import annotations

from collections.abc import Sequence

import torch

try:
    from whole_body_tracking.utils.motionlib.smpl_math_utils.smpl_math_utils import (
        angle_axis_to_quaternion,
        quaternion_to_rotation_matrix,
    )
except ModuleNotFoundError:  # pragma: no cover - direct package usage fallback
    from smpl_math_utils import angle_axis_to_quaternion, quaternion_to_rotation_matrix

from ..config import UnifiedLoadConfig
from ..errors import MotionNotLoadedError
from ..store import motion_ids_from_timestamps
from ..types import LoadReport, UnifiedMotionState
from .config_adapter import cfg_to_unified_load_config
from .orchestration import execute_unified_load
from .smpl_motion_lib import SmplMotionLib


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    result = q.clone()
    result[..., 1:] = -result[..., 1:]
    return result / q.square().sum(dim=-1, keepdim=True).clamp(min=1.0e-12)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., 1:]
    q_w = q[..., 0:1]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2.0 * (q_w * uv + uuv)


class UnifiedMotionLib:
    """
        统一的运动数据接口，提供对机器人运动数据和对应 SMPL 数据的访问。
        设计目标是简化下游任务（如 RL 训练）对运动数据的使用，隐藏数据加载、预处理和索引管理的复杂性。    
    """

    def __init__(
        self,
        body_indexes: Sequence[int],
        motion_anchor_body_index: int,
        joint_names: Sequence[str] | None = None,
        motion_body_names: Sequence[str] | None = None,
        all_body_names: Sequence[str] | None = None,
        device: str = "cpu",
    ) -> None:
        self.body_indexes = list(body_indexes)
        self.motion_anchor_body_index = motion_anchor_body_index
        self.joint_names = list(joint_names) if joint_names is not None else None
        self.motion_body_names = list(motion_body_names) if motion_body_names is not None else None
        self.all_body_names = list(all_body_names) if all_body_names is not None else None
        self.device = device

        self.joint_pos: torch.Tensor | None = None
        self.joint_vel: torch.Tensor | None = None
        self._body_pos_w_sel: torch.Tensor | None = None
        self._body_quat_w_sel: torch.Tensor | None = None
        self._body_lin_vel_w_sel: torch.Tensor | None = None
        self._body_ang_vel_w_sel: torch.Tensor | None = None

        self.fps: float | None = None
        self.time_step_total: int | None = None
        self.file_names: list[str] | None = None
        self.motion_num: int | None = None
        self.frame_list: torch.Tensor | None = None
        self.time_step_start_idx: torch.Tensor | None = None
        self.time_step_end_idx: torch.Tensor | None = None

        self._smpl_lib: SmplMotionLib | None = None
        self._sample_counter = 0

    def reset(self) -> None:
        self.joint_pos = None
        self.joint_vel = None
        self._body_pos_w_sel = None
        self._body_quat_w_sel = None
        self._body_lin_vel_w_sel = None
        self._body_ang_vel_w_sel = None
        self.fps = None
        self.time_step_total = None
        self.file_names = None
        self.motion_num = None
        self.frame_list = None
        self.time_step_start_idx = None
        self.time_step_end_idx = None
        self._smpl_lib = None

    def load_from_cfg(self, cfg) -> LoadReport:
        load_config = cfg_to_unified_load_config(
            cfg,
            sample_counter=self._sample_counter,
            device=self.device,
        )
        return self.load(load_config)

    def load(self, load_config: UnifiedLoadConfig) -> LoadReport:
        self.reset()
        outcome = execute_unified_load(
            body_indexes=self.body_indexes,
            joint_names=self.joint_names,
            motion_body_names=self.motion_body_names,
            all_body_names=self.all_body_names,
            load_config=load_config,
        )
        self._sample_counter = outcome.next_sample_counter
        self._apply_motion_state(outcome.state)
        self._smpl_lib = outcome.smpl_lib
        return outcome.report

    def _apply_motion_state(self, state: UnifiedMotionState) -> None:
        self.joint_pos = state.joint_pos
        self.joint_vel = state.joint_vel
        self._body_pos_w_sel = state.body_pos_w
        self._body_quat_w_sel = state.body_quat_w
        self._body_lin_vel_w_sel = state.body_lin_vel_w
        self._body_ang_vel_w_sel = state.body_ang_vel_w
        self.frame_list = state.frame_list
        self.motion_num = state.motion_num
        self.fps = state.fps
        self.time_step_total = state.time_step_total
        self.time_step_start_idx = state.time_step_start_idx
        self.time_step_end_idx = state.time_step_end_idx
        self.file_names = state.file_names

    def _require_robot_tensor(self, tensor: torch.Tensor | None, name: str) -> torch.Tensor:
        if tensor is None:
            raise MotionNotLoadedError(f"Robot motion data not loaded. Cannot access '{name}'.")
        return tensor

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._require_robot_tensor(self._body_pos_w_sel, "body_pos_w")

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._require_robot_tensor(self._body_quat_w_sel, "body_quat_w")

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._require_robot_tensor(self._body_lin_vel_w_sel, "body_lin_vel_w")

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._require_robot_tensor(self._body_ang_vel_w_sel, "body_ang_vel_w")

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_pos_z(self) -> torch.Tensor:
        return self.anchor_pos_w[:, 2:3]

    @property
    def smpl_joints(self) -> torch.Tensor | None:
        return self._smpl_lib.joints_flat if self._smpl_lib is not None else None

    @property
    def smpl_transl(self) -> torch.Tensor | None:
        return self._smpl_lib.transl_flat if self._smpl_lib is not None else None

    @property
    def smpl_poses(self) -> torch.Tensor | None:
        return self._smpl_lib.poses_flat if self._smpl_lib is not None else None

    def _require_smpl(self) -> SmplMotionLib:
        if self._smpl_lib is None:
            raise MotionNotLoadedError("SMPL data not loaded.")
        return self._smpl_lib

    # 索引指定帧的 smpl 数据 
    # motion_ids: motion id
    # motion_steps: motion step id (frame id within the motion)
    def get_smpl_joints(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self._require_smpl().get_smpl_joints(motion_ids, motion_steps)

    def get_smpl_transl(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self._require_smpl().get_smpl_transl(motion_ids, motion_steps)

    def get_smpl_pose(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self._require_smpl().get_smpl_pose(motion_ids, motion_steps)

    def get_smpl_global_position(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self._require_smpl().get_smpl_global_position(motion_ids, motion_steps)

    def get_smpl_root_quat_w(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        raw_aa = self.get_smpl_pose(motion_ids, motion_steps)[..., :3]
        q = angle_axis_to_quaternion(raw_aa.reshape(-1, 3))
        q_base_conj = torch.tensor([[0.5, -0.5, -0.5, -0.5]], device=q.device, dtype=q.dtype).expand(q.shape[0], -1)
        return _quat_mul(q, q_base_conj).view(raw_aa.shape[:-1] + (4,))

    def get_smpl_root_quat_w_dif_l(
        self,
        motion_ids: torch.Tensor,
        motion_steps: torch.Tensor,
        robot_anchor_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        smpl_root = self.get_smpl_root_quat_w(motion_ids, motion_steps)
        root_rot_dif = _quat_mul(_quat_inv(robot_anchor_quat_w), smpl_root)
        mat = quaternion_to_rotation_matrix(root_rot_dif)
        return mat[..., :2].reshape(root_rot_dif.shape[:-1] + (6,))

    def get_smpl_joints_local(
        self,
        motion_ids: torch.Tensor,
        motion_steps: torch.Tensor,
    ) -> torch.Tensor:
        ref_joints = self.get_smpl_joints(motion_ids, motion_steps)
        root_quat = self.get_smpl_root_quat_w(motion_ids, motion_steps)
        root_quat = root_quat.unsqueeze(-2).expand(ref_joints.shape[:-1] + (4,))
        return _quat_apply(_quat_inv(root_quat.reshape(-1, 4)), ref_joints.reshape(-1, 3)).view_as(ref_joints)

    def motion_ids_from_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        if self.time_step_end_idx is None or self.time_step_total is None or self.motion_num is None:
            raise MotionNotLoadedError("UnifiedMotionLib not loaded. Call load() or load_from_cfg() first.")
        return motion_ids_from_timestamps(
            timestamps=timestamps,
            end_idx=self.time_step_end_idx,
            total_frames=self.time_step_total,
            motion_num=self.motion_num,
        )
