from __future__ import annotations

import datetime
import numpy as np
import os
from pathlib import Path

# from whole_body_parkour.data import DATA_ASSET_DIR
import random
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from tqdm import tqdm
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import (  # RED_ARROW_X_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_rotate_inverse as quat_apply_inverse, # sim5.1 to sim4.5
    quat_error_magnitude,
)

from whole_body_tracking.utils.motionlib.feat_motion_lib.facade.unified_motion_lib import UnifiedMotionLib

from ..math_utils import quat_to_6d

from .motion_cache import MotionReferenceCache
from .motion_reset import MotionCommandResetter
from .motion_selection import create_motion_selection_policy
from .motion_timeline import MotionCommandTimeline
from .motion_viser import MotionCommandDebugVisualizer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader(UnifiedMotionLib):
    def __init__(
        self,
        cfg: MotionCommandCfg,
        body_indexes: Sequence[int] = (0,),
        motion_anchor_body_index: int = 0,
        joint_names: Sequence[str] | None = None,
        motion_body_names: Sequence[str] | None = None,
        all_body_names: Sequence[str] | None = None,
        device: str = "cpu",
    ):
        super().__init__(
            body_indexes=body_indexes,
            motion_anchor_body_index=motion_anchor_body_index,
            joint_names=joint_names,
            motion_body_names=motion_body_names,
            all_body_names=all_body_names,
            device=device,
        )
        self.cfg = cfg
        self.json_path = None
        self.death_path = None
        self.first_init = False

    def resample_motionloader(self, device):
        self.device = device
        return self.load_from_cfg(self.cfg)


"""
Motion Command 承担
1 - 各个核心模块内部变量的暴露功能
2 - 定义核心模块之间的工作流程
3 - 各模块对其他模块的内部变量只有读的权力 没有写的权力
4 - 各模块通过对 MotionCommand 声明的 property 来访问其他模块
5 - MotionCommand 原生的变量作为共享变量 可供其他模块读写
"""

# TODO renaming the variables for clearity
class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        self.debug_visualizer: MotionCommandDebugVisualizer | None = None
        super().__init__(cfg, env)

        #region robot body indexing
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        # TODO use more robust anchor index assigning logic
        assert self.robot_anchor_body_index == 0
        assert self.motion_anchor_body_index == 0
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        #endregion robot body indexing

        #region Future-frame indexing helpers.
        future_step_num = getattr(self.cfg, "future_step_num", [0])
        if future_step_num is None or len(future_step_num) == 0:
            future_step_num = [0]
        self._future_step_offsets = torch.tensor(future_step_num, device=self.device, dtype=torch.long)
        #endregion Future-frame indexing helpers.

        self.motion = MotionLoader(
            self.cfg,
            body_indexes=self.body_indexes.tolist(),
            motion_anchor_body_index=self.motion_anchor_body_index,
            joint_names=list(self.robot.data.joint_names),
            motion_body_names=list(self.cfg.body_names),
            all_body_names=list(self.robot.body_names),
            device=self.device,
        )

        #region functional modules from sub_modules
        self.timeline = MotionCommandTimeline(self.num_envs, self._future_step_offsets, self.device)
        self.reference_cache = MotionReferenceCache(self.num_envs, len(cfg.body_names), self.device)
        
        self.resetter = MotionCommandResetter(self.num_envs, self.cfg, self.robot, self.device)
        # TODO env classes mask 应该转移为 env 的原生变量
        self.envs_classes_mask = self.resetter.envs_classes_mask
        setattr(self.env, "envs_classes_mask", self.resetter.envs_classes_mask)
        
        self.debug_visualizer = MotionCommandDebugVisualizer(self.cfg, self, self.device)

        # TODO selection policy 这个类设计的不好，需要想办法改掉
        self.selection_policy = create_motion_selection_policy(self.cfg, num_envs=self.num_envs, device=self.device)
        self.adaptive_sampler = getattr(self.selection_policy, "sampler", None)
        #endregion functional modules from sub_modules
        
        #region for motion bins analyzing
        self.use_new_motion_pre_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Evaluation bookkeeping belongs to the command, not the timeline cursor.
        self.eval_cycle_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.fixed_eval_motion_ids: torch.Tensor | None = None
        self.history_success_rate_dict = {}
        #endregion for motion bins analyzing

        # Load motions from dataset and prepare functional components
        self.resample_motion_files(self.env)

        #region metrics
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.adaptive_sample:
            self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_top1_prob_max"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_top1_prob_bin"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_top1_prob_mean"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_top1_prob_min"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["prob_max_over_uniform"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["prob_uniform"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["failures_min"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["failures_mean"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["failures_max"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["failures_max_over_uniform"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["num_concentrate_bins"] = torch.zeros(self.num_envs, device=self.device)
        #endregion metrics

        self.command_step_count = 0
        self.fixed_eval_motion_ids = getattr(self.selection_policy, "fixed_eval_motion_ids", None)
        if self.cfg.debug_vis: self.debug_visualizer.set_enabled(True)
        
        self._refresh_reference_cache() #align reference motion to current robots' xy_yaw coordination

        """
        TODO
        将不必要的属性 alias 删除
        将功能性的计算代码写入其他类
        command 只作为接口和流程管理器使用
        """

    #region normal property
    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [
                # self.joint_pos_future.view(self.num_envs, -1),
                # self.joint_vel_future.view(self.num_envs, -1),
                self.joint_pos,
                self.joint_vel,
                self.anchor_lin_vel_b,
                self.anchor_ang_vel_b,
                self.anchor_project_gravity,
                self.anchor_pos_z,
            ],
            dim=1,
        )

    # TODO(refactor): The reference-motion queries below still mix "raw reference
    # frame lookup" and "consumer-facing feature access" inside MotionCommand.
    # Split them into:
    # 1. ReferenceMotionAccessor: current/future frame queries from MotionLoader
    #    driven by timeline state.
    # 2. ReferenceFeatureView (or MotionCommandFeatures): observation/reward/debug
    #    facing derived features built on top of the accessor.
    @property
    def motion_ids(self) -> torch.Tensor:
        return self.timeline.motion_ids

    @property
    def local_time_steps(self) -> torch.Tensor:
        return self.timeline.local_time_steps
    
    @property
    def body_pos_relative_w(self) -> torch.Tensor:
        return self.reference_cache.body_pos_relative_w

    @property
    def body_quat_relative_w(self) -> torch.Tensor:
        return self.reference_cache.body_quat_relative_w

    @property
    def anchor_pos_z(self):
        return self.anchor_pos_w[:, 2:3]

    @property
    def anchor_project_gravity(self) -> torch.Tensor:
        project_gravity = quat_apply_inverse(self.anchor_quat_w, self.robot.data.GRAVITY_VEC_W)
        return project_gravity

    @property
    def anchor_6d_rotation(self) -> torch.Tensor:
        six_d_rotation = quat_to_6d(self.anchor_quat_w)
        return six_d_rotation

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.global_time_steps]  # (num_envs, num_joints, 3)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.global_time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.global_time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.global_time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.global_time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.global_time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        result = (
            torch.index_select(self.motion.anchor_pos_w, dim=0, index=self.global_time_steps)
            + self._env.scene.env_origins
        )
        return result

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.anchor_quat_w[self.global_time_steps]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.anchor_lin_vel_w[self.global_time_steps]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.anchor_ang_vel_w[self.global_time_steps]

    @property
    def global_time_steps(self) -> torch.Tensor:
        """Global timestamps for indexing the concatenated motion buffers."""
        # TODO(refactor): Move timeline-driven reference indexing helpers such as
        # global_time_steps/motion_num_steps/global_future_steps into ReferenceMotionAccessor.
        return self.timeline.global_time_steps(self.motion)

    @property
    def num_future_frames(self) -> int:
        return self.timeline.num_future_frames

    @property
    def global_start_steps(self) -> torch.Tensor:
        """Per-env global start index of the current motion."""
        return self.timeline.global_start_steps(self.motion)

    @property
    def motion_num_steps(self) -> torch.Tensor:
        """Per-env number of frames for the current motion."""
        return self.timeline.motion_num_steps(self.motion)

    @property
    def future_time_steps_init(self) -> torch.Tensor:
        """Future step offsets (local frame deltas) as a 1D tensor."""
        return self._future_step_offsets

    @property
    def expanded_future_motion_ids(self) -> torch.Tensor:
        """Motion ids for all future reference frames, flattened."""
        return self.timeline.expanded_future_motion_ids()

    @property
    def global_future_steps(self) -> torch.Tensor:
        """Compute absolute (global) time-step indices for all future reference frames.

        Clamps to the last valid frame of each motion to avoid out-of-bounds access.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames,)``.
        """
        return self.timeline.global_future_steps(self.motion)

    @property
    def anchor_pos_w_future(self) -> torch.Tensor:
        """Future reference anchor position in world frame for each env."""
        return self.motion.anchor_pos_w[self.global_future_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def joint_pos_future(self) -> torch.Tensor:
        """Future reference joint positions for each env."""
        return self.motion.joint_pos[self.global_future_steps]

    @property
    def joint_vel_future(self) -> torch.Tensor:
        """Future reference joint velocities for each env."""
        return self.motion.joint_vel[self.global_future_steps]

    @property
    def anchor_quat_w_future(self) -> torch.Tensor:
        """Future reference anchor orientation in world frame for each env."""
        return self.motion.anchor_quat_w[self.global_future_steps]

    @property
    def joint_vel_multi_future(self) -> torch.Tensor:
        """Return reference joint velocities for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * ...)``.
        """
        return self.motion.joint_vel[self.global_future_steps].view(self.num_envs, -1)

    @property
    def has_smpl_data(self) -> bool:
        return (
            getattr(self.motion, "smpl_joints", None) is not None
            and getattr(self.motion, "smpl_transl", None) is not None
            and getattr(self.motion, "smpl_poses", None) is not None
        )

    @property
    def smpl_joints(self) -> torch.Tensor:
        return self.motion.smpl_joints[self.global_time_steps]

    @property
    def smpl_transl(self) -> torch.Tensor:
        return self.motion.smpl_transl[self.global_time_steps]

    @property
    def smpl_poses(self) -> torch.Tensor:
        return self.motion.smpl_poses[self.global_time_steps]

    @property
    def smpl_poses_future(self) -> torch.Tensor:
        return self.motion.smpl_poses[self.global_future_steps]

    @property
    def smpl_joints_future(self) -> torch.Tensor:
        return self.motion.smpl_joints[self.global_future_steps]

    @property
    def smpl_transl_future(self) -> torch.Tensor:
        return self.motion.smpl_transl[self.global_future_steps]

    @property
    def smpl_global_position(self) -> torch.Tensor | None:
        if not self.has_smpl_data:
            return None
        smpl_global = self.motion.get_smpl_global_position(self.motion_ids, self.local_time_steps)
        return smpl_global + self._env.scene.env_origins[:, None, :]

    @property
    def smpl_global_position_future(self) -> torch.Tensor | None:
        if not self.has_smpl_data:
            return None
        local_max = (self.motion_num_steps - 1).clamp(min=0)
        future_local = torch.clip(
            self.local_time_steps[:, None] + self.future_time_steps_init[None, :],
            max=local_max[:, None],
        )
        smpl_global = self.motion.get_smpl_global_position(self.expanded_future_motion_ids, future_local.reshape(-1))
        smpl_global = smpl_global.view(self.num_envs, self.num_future_frames, 24, 3)
        return smpl_global + self._env.scene.env_origins[:, None, None, :]

    def get_smpl_joints(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self.motion.get_smpl_joints(motion_ids, motion_steps)

    def get_smpl_transl(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self.motion.get_smpl_transl(motion_ids, motion_steps)

    def get_smpl_pose(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self.motion.get_smpl_pose(motion_ids, motion_steps)

    def get_smpl_global_position(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self.motion.get_smpl_global_position(motion_ids, motion_steps)

    @property
    def smpl_root_quat_w(self) -> torch.Tensor:
        return self.motion.get_smpl_root_quat_w(self.motion_ids, self.local_time_steps)

    @property
    def smpl_root_quat_w_multi_future(self) -> torch.Tensor:
        local_max = (self.motion_num_steps - 1).clamp(min=0)
        future_local = torch.clip(
            self.local_time_steps[:, None] + self.future_time_steps_init[None, :],
            max=local_max[:, None],
        ).reshape(-1)
        return self.motion.get_smpl_root_quat_w(self.expanded_future_motion_ids, future_local).view(
            self.num_envs, self.num_future_frames, 4
        )

    @property
    def smpl_root_quat_w_dif_l_multi_future(self) -> torch.Tensor:
        local_max = (self.motion_num_steps - 1).clamp(min=0)
        future_local = torch.clip(
            self.local_time_steps[:, None] + self.future_time_steps_init[None, :],
            max=local_max[:, None],
        ).reshape(-1)
        robot_anchor = self.robot_anchor_quat_w[:, None, :].expand(-1, self.num_future_frames, -1)
        return self.motion.get_smpl_root_quat_w_dif_l(
            self.expanded_future_motion_ids,
            future_local,
            robot_anchor.reshape(-1, 4),
        ).view(self.num_envs, -1)

    @property
    def smpl_joints_local_multi_future(self) -> torch.Tensor:
        local_max = (self.motion_num_steps - 1).clamp(min=0)
        future_local = torch.clip(
            self.local_time_steps[:, None] + self.future_time_steps_init[None, :],
            max=local_max[:, None],
        ).reshape(-1)
        return self.motion.get_smpl_joints_local(self.expanded_future_motion_ids, future_local).view(
            self.num_envs, self.num_future_frames, 24, 3
        )

    @property
    def anchor_lin_vel_b(self) -> torch.Tensor:
        anchor_lin_vel_b = quat_apply_inverse(self.anchor_quat_w, self.anchor_lin_vel_w)
        return anchor_lin_vel_b

    @property
    def anchor_ang_vel_b(self) -> torch.Tensor:
        anchor_ang_vel_b = quat_apply_inverse(self.anchor_quat_w, self.anchor_ang_vel_w)
        return anchor_ang_vel_b

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    # endregion normal property

    #region to be deprecated
    def motion_ids_from_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Get motion ids for each timestamp in the current concatenated motion buffer."""
        return self.motion.motion_ids_from_timestamps(timestamps)

    def _set_time_from_global_timestamps(self, env_ids: Sequence[int], timestamps: torch.Tensor) -> None:
        """Set (motion_ids, local time_steps, local end) from global timestamps."""
        if len(env_ids) == 0:
            return
        selection = self.timeline.build_selection_from_global_timestamps(self.motion, timestamps)
        self.timeline.apply_selection(env_ids, selection)

    def _setup_fixed_eval_motion_assignment(self) -> None:
        """Compatibility wrapper for deterministic eval assignment."""
        self.selection_policy.bind_motion_source(
            self.motion,
            timeline=self.timeline,
            decimation=self.env.cfg.decimation,
            sim_dt=self.env.cfg.sim.dt,
        )
        self.fixed_eval_motion_ids = getattr(self.selection_policy, "fixed_eval_motion_ids", None)
    #endregion to be deprecated

    #region core functions
    def _refresh_reference_cache(self) -> None:
        self.reference_cache.refresh(
            anchor_pos_w=self.anchor_pos_w,
            anchor_quat_w=self.anchor_quat_w,
            body_pos_w=self.body_pos_w,
            body_quat_w=self.body_quat_w,
            robot_anchor_pos_w=self.robot_anchor_pos_w,
            robot_anchor_quat_w=self.robot_anchor_quat_w,
        )

    def resample_motion_files(self, env):
        self.motion.resample_motionloader(device=self.device)
        self.use_new_motion_pre_env[:] = False
        self.resample_time = 0
        self.timeline.clear_timeline()
        self.selection_policy.bind_motion_source(
            self.motion,
            timeline=self.timeline,
            decimation=env.cfg.decimation,
            sim_dt=env.cfg.sim.dt,
        )
        self.fixed_eval_motion_ids = getattr(self.selection_policy, "fixed_eval_motion_ids", None)
        if self.selection_policy.counts_eval_cycles:
            self.eval_cycle_count.zero_()

    def _resample_command(self, env_ids: Sequence[int], *, allow_failure_accounting: bool = True):
        """_resample_command will be called multiple times in each step
        
        step(action)
        │
        ├─ [1] process_action
        ├─ [2] physics × decimation
        ├─ [3] termination_manager.compute()
        ├─ [4] reward_manager.compute()          ← 用的是上一步 command
        │
        ├─ [5] 若有 env 终止:
        │       command_manager.reset(env_ids)
        │         └─ _resample()   ★ 路径 1（仅终止 env）
        │
        ├─ [6] command_manager.compute(dt)
        │       ├─ time_left -= dt    # 判断 command 何时 resample
        │       ├─ if time_left <= 0:
        │       │     _resample()  ★ 路径 2（到期 env）
        │       └─ _update_command()
        │             └─ (OrientationCommand) _resample()  ★ 路径 3
        │
        ├─ [7] interval events
        └─ [8] observation_manager.compute()     ← 用的是本步最终 command
        """
        if len(env_ids) == 0:
            return
        
        selection = self.selection_policy.select(
            env_ids,
            motion_source=self.motion,
            timeline=self.timeline,
            terminated=self._env.termination_manager.terminated,
            metrics=self.metrics,
            allow_failure_accounting=allow_failure_accounting,
        )
        self.timeline.apply_selection(env_ids, selection)

        self.resetter.apply(
            env_ids,
            body_pos_w=self.body_pos_w,
            body_quat_w=self.body_quat_w,
            body_lin_vel_w=self.body_lin_vel_w,
            body_ang_vel_w=self.body_ang_vel_w,
            joint_pos=self.joint_pos,
            joint_vel=self.joint_vel,
            env_origins=self._env.scene.env_origins,
        )
        self._refresh_reference_cache()

    def _update_command(self):
        """
            Called every control step, update commands for envs that are out of time
            在 reset() 之后先处理终止超时的 env

            1. process_action(action)
            2. physics 循环 (decimation 次 sim.step)
            3. termination 计算
            4. reward 计算          ← 用的是上一步的 command
            5. 终止 env 的 reset    ← 可能触发 command_manager.reset → _resample
            6. command_manager.compute  ← _update_command() 在这里
            7. interval events
            8. observation 计算     ← 用的是本步 _update_command 后的 command
            9. return

            env.step(action)
            └─ command_manager.compute(dt=step_dt)      # 仅 RL env 的 step 中有
                └─ term.compute(dt)
                        └─ _update_command()            ← 唯一标准入口
        """
        self.command_step_count += 1
        self.timeline.step()
        # NOTE why not padding with the last frame if exceeding the frame count ?
        env_ids = self.timeline.expired_env_ids(self.cfg.max_future_step)  # 防止溢出

        # Count completed cycles only on real end-of-clip events during stepping.
        if self.selection_policy.counts_eval_cycles:
            if len(env_ids) > 0:
                self.eval_cycle_count[env_ids] += 1
        
        # 重新生成指令，在本项目中resample_interval 设置的很大，故这段代码可以认为未执行过
        # TODO 重新写这段逻辑以获得更好的可读性
        if self.cfg.resample_interval != -1:  # change the reference motion every resample_interval control steps
            self.resample_time += 1
            if self.resample_time >= self.cfg.resample_interval:
                # --debug
                print("resample command called for resample_interval")
                # --debug
                self.resample_motion_files(self.env)
                env_ids = torch.arange(self.num_envs, device=self.device)
                # motion source 已重载，不能把旧 episode failure 计入新 motion bins。
                self._resample_command(env_ids, allow_failure_accounting=False)
                # Keep aligned reference caches in sync for both stepped and reset envs.
                self._refresh_reference_cache()
                self.selection_policy.step_post_update()
                return
        
        # 这里只更新 future commands 溢出的 env_ids
        # -- debug
        # print("_update_command() called")
        # print("env_ids in _update_command(): ", env_ids)
        # print(self._env.termination_manager.terminated[env_ids])
        # -- debug
        self._resample_command(env_ids, allow_failure_accounting=False)

        # Keep aligned reference caches in sync for both stepped and reset envs.
        self._refresh_reference_cache()

        self.selection_policy.step_post_update()
    #endregion core functions

    #region debug
    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.mean(torch.abs(self.joint_pos - self.robot_joint_pos), dim=-1)
        self.metrics["error_joint_vel"] = torch.mean(torch.abs(self.joint_vel - self.robot_joint_vel), dim=-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if self.debug_visualizer is None:
            return
        self.debug_visualizer.set_enabled(debug_vis)

    def _debug_vis_callback(self, event):
        if self.debug_visualizer is None:
            return
        self.debug_visualizer.render()
    #endregion debug


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command.
    TODO class 应该细分
    """

    class_type: type = MotionCommand

    # basic configs
    eval_mode: bool = False
    fixed_eval_motion_ids: bool = False
    adaptive_sample: bool = True
    asset_name: str = MISSING
    max_motion_num: int = 999999
    resample_interval: int = 300000000000   # 训练时暂不需要 resample
    
    motion_file: str = MISSING # 改为 motion_folder
    dataset_txt: str = None
    
    # smpl configs
    smpl_file_path: str | None = None
    target_fps: float = 50.0
    up_axis: str = "yup"
    max_frame_diff: int = 2

    # reference motion obs configs
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    # future_step_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    future_step_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    max_future_step = max(future_step_num)

    # env classification
    envs_classes_ratio: dict = {"lying": 0.3, "range": 0.7}
    # 初始化具体参数
    pose_range: dict[str, tuple[float, float]] = {}
    pose_range_lying_height_range: tuple[float, float] = (0.25, 0.45)

    # randomization configs
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # adaptive sampling configs
    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    motion_ratio = [0.2, 0.6, 0.2] # 不同难度的动作采样加权平均
    adaptive_uniform_ratio: float = 0.5
    adaptive_alpha: float = 0.001
    motion_sampling_start_frame: int = 5
    adaptive_sample_rewind_min_bins: int = 0
    adaptive_sample_rewind_bins: int = 1

    failure_cap: bool = True
    failure_cap_beta: float = 200.0
    failure_most_hard_cap_beta: float = 10000.0

    # adaptive bins log configs
    # 每隔多少步导出一次 adaptive bins 概率和 bin->motion 反查映射，-1 表示不保存
    log_save_path: str = "train_logs"
    fail_count_save_interval: int = 500 * 24
    save_adaptive_bins: bool = True
    adaptive_bins_file_prefix: str = "adaptive_bins"

    # visualizer configs
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    current_anchor_lin_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/current/anchor_lin_vel"
    )
    goal_anchor_lin_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal/anchor_lin_vel"
    )
    current_anchor_lin_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    goal_anchor_lin_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # Debug printing for anchor velocity in _debug_vis_callback.
    debug_anchor_speed: bool = True
    debug_anchor_speed_scale: float = 1.0

    # 为了分布式训练
    distributed: bool = False
    local_rank: int = -1
    total_rank: int = -1


__all__ = ["MotionLoader", "MotionCommand", "MotionCommandCfg"]
