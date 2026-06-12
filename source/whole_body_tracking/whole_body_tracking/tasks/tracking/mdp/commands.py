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

from .math_utils import quat_to_6d
from .motion_command_runtime import (
    MotionCommandResetter,
    MotionCommandDebugVisualizer,
    MotionCommandTimeline,
    MotionReferenceCache,
    create_motion_selection_policy,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(
        self,
        cfg: MotionCommandCfg,
        body_indexes: Sequence[int] = [0],
        motion_anchor_body_index: int = 0,
        device: str = "cpu",
    ):
        self.device = device
        self.cfg = cfg
        self.body_indexes = body_indexes
        self.motion_anchor_body_index = motion_anchor_body_index
        self.motion_num = None
        self.json_path = None
        self.death_path = None
        self.first_init = False

        self.joint_pos = None
        self.joint_vel = None
        self._body_pos_w_sel = None
        self._body_quat_w_sel = None
        self._body_lin_vel_w_sel = None
        self._body_ang_vel_w_sel = None

        self.time_step_total = None  # self.joint_pos.shape[0]
        self.file_names = None
        self.time_step_end_idx = None
        self.frame_list = None  # 确保在同一 device
        self.time_step_start_idx = None
        self.motion_num = None
        if self.cfg.eval_mode:
            self.sample_counter = 0

    def _find_npz_files(self, dir_path: Path, motion_num: int, dataset_txt=None):
        """随机选择一个子文件夹（若存在），返回其中所有 .npz 文件路径"""
        if dataset_txt is not None:
            with open(dataset_txt) as f:
                relative_paths = [line.strip() for line in f if line.strip()]
                # import ipdb;ipdb.set_trace()
            npz_files = [dir_path + "/" + rel_path for rel_path in relative_paths]
            random.seed(42)  # 方便对比试验
            random.shuffle(npz_files)
        else:
            dir_path = Path(dir_path)
            npz_files = list(dir_path.rglob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(f"No .npz files found in {dir_path}")
            """从 npz 文件中均匀随机采样 motion_num 个"""
            npz_files.sort()

        if len(npz_files) > motion_num and motion_num != -1:  # 动作文件需要采样（内存不够）或者人为指定把所有数据拿出来
            if self.cfg.eval_mode:
                start_idx = self.sample_counter * motion_num  # % len(npz_files)
                end_idx = min(start_idx + motion_num, len(npz_files))
                if start_idx >= len(npz_files) or (end_idx - start_idx) < motion_num:
                    raise ValueError("Not enough npz files for eval_mode sampling. all data if eval finished.")
                sampled_files = npz_files[start_idx:end_idx]
                print("eval_mode process: ", (float(start_idx) / len(npz_files)))
                print("当前本地时间:", datetime.datetime.now().strftime("%H%M"))
                self.sample_counter += 1
            else:
                # 采样后保持排序顺序：先按motion号排序，再按z_scale排序
                sampled_files = sorted(random.sample(npz_files, motion_num))
        elif self.cfg.distributed:
            total_motion_num = len(npz_files)
            subset_motion_num = total_motion_num // self.cfg.total_rank
            start_idx = subset_motion_num * self.cfg.local_rank
            end_idx = subset_motion_num * (self.cfg.local_rank + 1)
            print(f"第{self.cfg.local_rank}个GPU的数据集为数据{start_idx}->{end_idx}, 共{subset_motion_num}个")
            sampled_files = npz_files[start_idx:end_idx]
        else:
            print(f"加载全部共{len(npz_files)}个NPZ动作数据")
            sampled_files = npz_files
        return sampled_files

    def load_and_cat_npz_with_filenames(self, dir_path, motion_num=25, device="cpu", dataset_txt=None):
        """
        加载 npz 文件，并按 batch 拼接，避免一次性占用 GPU 显存。
        batch_size: 每次拼接的帧数
        """

        batch_size = 1024
        npz_file_paths = self._find_npz_files(dir_path, motion_num, dataset_txt)
        rel_npz_file_names = []
        tensor_keys = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
        tensor_lists = {k: [] for k in tensor_keys}
        fps_list = []
        frames_per_file = []

        # 先在 CPU 上读取数据，保持 list
        for f_path in tqdm(npz_file_paths, desc="Processing files"):
            try:
                data = np.load(f_path, allow_pickle=True)
                for k in tensor_keys:
                    tensor = torch.from_numpy(data[k]).float()  # CPU tensor
                    tensor_lists[k].append(tensor)
                fps_list.append(data["fps"])
                frames_per_file.append(data["joint_pos"].shape[0])
                rel_npz_file_names.append(os.path.relpath(f_path, dir_path))
            except Exception as e:
                print(f"   路径: {f_path}")
                print(f"   错误: {e}")
                # 可选：跳过这个文件继续
                continue

        if not frames_per_file:
            raise RuntimeError(f"No valid .npz motion files loaded from {dir_path}")

        fps_values = [float(fps) for fps in fps_list]
        assert len(set(fps_values)) == 1, "All fps in npz files must be the same."
        fps = fps_values[0]

        if device != "cpu":
            print(f"[Before batch cat] GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

        # 按 batch 拼接，逐步拷贝到 GPU（如果 device != "cpu"）

        data_dict = {k: [] for k in tensor_keys}
        for k in tensor_keys:
            batch_tensors = []
            for tensor in tensor_lists[k]:
                num_frames = tensor.shape[0]
                for start in range(0, num_frames, batch_size):
                    end = min(start + batch_size, num_frames)
                    batch = tensor[start:end]
                    if device != "cpu":
                        batch = batch.to(device)
                    batch_tensors.append(batch)
            # 最后将所有 batch 拼成一个 tensor
            data_dict[k] = torch.cat(batch_tensors, dim=0)

        if device != "cpu":
            print(f"[After batch cat] GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

        return data_dict, rel_npz_file_names, fps, frames_per_file

    def resample_motionloader(self, device):
        data_dict, file_names, fps, frame_list = self.load_and_cat_npz_with_filenames(
            self.cfg.motion_file, self.cfg.max_motion_num, device, self.cfg.dataset_txt
        )  # frame list是每个motion file长度的list
        self.fps = fps

        self.joint_pos = data_dict["joint_pos"]
        self.joint_vel = data_dict["joint_vel"]

        self._body_pos_w_sel = data_dict["body_pos_w"][:, self.body_indexes]
        self._body_quat_w_sel = data_dict["body_quat_w"][:, self.body_indexes]
        self._body_lin_vel_w_sel = data_dict["body_lin_vel_w"][:, self.body_indexes]
        self._body_ang_vel_w_sel = data_dict["body_ang_vel_w"][:, self.body_indexes]
        # print("Loaded motions with total frames:", sum(frame_list), frame_list)
        if device != "cpu":
            print(f"[data_dict] GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        print(" \n \n \n")

        self.time_step_total = sum(frame_list)  # self.joint_pos.shape[0]
        self.file_names = file_names
        print("file nums:", "     ", len(self.file_names))
        print("frame nums:", "     ", (self._body_pos_w_sel.shape[0]))

        self.frame_list = torch.tensor(frame_list, device=device)
        self.motion_num = len(frame_list)

        self.time_step_end_idx = torch.cumsum(self.frame_list, dim=0)
        self.time_step_start_idx = torch.cat(
            [torch.tensor([0], device=self.frame_list.device), self.time_step_end_idx[:-1]]
        )

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w_sel  # 直接返回缓存，O(1)

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w_sel

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w_sel

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w_sel

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        # motion_anchor_body_index 是 int，整数索引返回 view 而非 copy
        return self._body_pos_w_sel[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._body_quat_w_sel[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w_sel[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w_sel[:, self.motion_anchor_body_index]

    @property
    def anchor_pos_z(self) -> torch.Tensor:
        return self.anchor_pos_w[:, 2:3]

    def motion_ids_from_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Map global frame timestamps to motion ids.
        The global timestamps index the concatenated motion tensor. This function returns
        the corresponding motion id for each timestamp based on ``time_step_end_idx``.
        Args
            timestamps: (num_envs, ) global frame timestamps for each environment
        Returns
            (num_envs, ) corresponding motion ids for each environment
        """
        if self.time_step_end_idx is None:
            raise RuntimeError("MotionLoader is not initialized. Call resample_motionloader first.")
        if timestamps.dtype != torch.long:
            timestamps = timestamps.long()
        timestamps = torch.clamp(timestamps, min=0, max=int(self.time_step_total) - 1)
        # General case: end_idx is exclusive; use right=True so boundary timestamps map to the next motion.
        motion_ids = torch.bucketize(timestamps, self.time_step_end_idx, right=True)
        return torch.clamp(motion_ids, min=0, max=self.motion_num - 1)


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

        #region functional modules from motion_command_runtime
        self.motion = MotionLoader(self.cfg, self.body_indexes, self.motion_anchor_body_index, device=self.device)
        self.timeline = MotionCommandTimeline(self.num_envs, self._future_step_offsets, self.device)
        self.reference_cache = MotionReferenceCache(self.num_envs, len(cfg.body_names), self.device)
        self.resetter = MotionCommandResetter(self.cfg, self.robot, self.device)
        self.debug_visualizer = MotionCommandDebugVisualizer(self.cfg, self, self.device)

        self.motion_ids = self.timeline.motion_ids
        self.local_time_steps = self.timeline.local_time_steps
        self.eval_cycle_count = self.timeline.eval_cycle_count
        self.frame_end_per_env = self.timeline.frame_end_per_env
        self.body_pos_relative_w = self.reference_cache.body_pos_relative_w
        self.body_quat_relative_w = self.reference_cache.body_quat_relative_w

        self.selection_policy = create_motion_selection_policy(self.cfg, num_envs=self.num_envs, device=self.device)
        self.adaptive_sampler = getattr(self.selection_policy, "sampler", None)
        #endregion functional modules from motion_command_runtime
        
        #region for motion bins analyzing
        self.bin_count = 0
        self.kernel = torch.zeros(0, dtype=torch.float, device=self.device)
        self.success_motion = torch.zeros(0, dtype=torch.float32, device=self.device)
        self.bin_failed_count = torch.zeros(0, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(0, dtype=torch.float, device=self.device)
        self.use_new_motion_pre_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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

    #region normal property
    @property
    def command(self) -> torch.Tensor:  
        # TODO Consider again if this is the best observation
        # TODO(refactor): Move observation-facing feature composition into a dedicated
        # ReferenceFeatureView/MotionCommandFeatures facade so MotionCommand only
        # owns IsaacLab lifecycle orchestration.
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
        # global_time_steps/motion_num_steps/future_time_steps into ReferenceMotionAccessor.
        return self.timeline.global_time_steps(self.motion)

    @property
    def num_future_frames(self) -> int:
        return self.timeline.num_future_frames

    @property
    def motion_start_time_steps(self) -> torch.Tensor:
        """Per-env global start index of the current motion."""
        return self.timeline.motion_start_time_steps(self.motion)

    @property
    def motion_num_steps(self) -> torch.Tensor:
        """Per-env number of frames for the current motion."""
        return self.timeline.motion_num_steps(self.motion)

    @property
    def future_time_steps_init(self) -> torch.Tensor:
        """Future step offsets (local frame deltas) as a 1D tensor."""
        return self._future_step_offsets

    @property
    def future_motion_ids(self) -> torch.Tensor:
        """Motion ids for all future reference frames, flattened."""
        return self.timeline.future_motion_ids()

    @property
    def future_time_steps(self) -> torch.Tensor:
        """Compute absolute (global) time-step indices for all future reference frames.

        Clamps to the last valid frame of each motion to avoid out-of-bounds access.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames,)``.
        """
        return self.timeline.future_time_steps(self.motion)

    @property
    def anchor_pos_w_future(self) -> torch.Tensor:
        """Future reference anchor position in world frame for each env."""
        return self.motion.anchor_pos_w[self.future_time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def joint_pos_future(self) -> torch.Tensor:
        """Future reference joint positions for each env."""
        return self.motion.joint_pos[self.future_time_steps]

    @property
    def joint_vel_future(self) -> torch.Tensor:
        """Future reference joint velocities for each env."""
        return self.motion.joint_vel[self.future_time_steps]

    @property
    def anchor_quat_w_future(self) -> torch.Tensor:
        """Future reference anchor orientation in world frame for each env."""
        return self.motion.anchor_quat_w[self.future_time_steps]

    @property
    def joint_vel_multi_future(self) -> torch.Tensor:
        """Return reference joint velocities for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * ...)``.
        """
        return self.motion.joint_vel[self.future_time_steps].view(self.num_envs, -1)

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
        selection = self.timeline.selection_from_global_timestamps(self.motion, timestamps)
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
        self.timeline.invalidate()
        self.selection_policy.bind_motion_source(
            self.motion,
            timeline=self.timeline,
            decimation=env.cfg.decimation,
            sim_dt=env.cfg.sim.dt,
        )
        self.fixed_eval_motion_ids = getattr(self.selection_policy, "fixed_eval_motion_ids", None)
        if self.adaptive_sampler is not None:
            # Keep legacy attribute names accessible while the sampler owns the state.
            self.bin_count = self.adaptive_sampler.bin_count
            self.bin_failed_count = self.adaptive_sampler.bin_failed_count
            self._current_bin_failed = self.adaptive_sampler._current_bin_failed
            self.kernel = self.adaptive_sampler.kernel
            self.success_motion = self.adaptive_sampler.success_motion

    def _resample_command(self, env_ids: Sequence[int], *, allow_failure_accounting: bool = True):
        """_resample_command will be called multiple times in each step
        1. Called from _update_command
        2. Called directly in the IsaacLab simulator
        2.1. after check_termination()
        2.2. after reset_all() maybe?
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
        )
        self._refresh_reference_cache()

    def _update_command(self):
        """
        Called every control step, update commands for envs that are out of time
        """
        self.command_step_count += 1
        self.timeline.step()
        # NOTE why not padding with the last frame if exceeding the frame count ?
        env_ids = self.timeline.expired_env_ids(self.cfg.max_future_step)  # 防止溢出

        # Count completed cycles only on real end-of-clip events during stepping.
        if self.selection_policy.counts_eval_cycles:
            if len(env_ids) > 0:
                self.eval_cycle_count[env_ids] += 1
        if self.cfg.resample_interval != -1:  # change the reference motion every resample_interval control steps
            self.resample_time += 1
            if self.resample_time >= self.cfg.resample_interval:
                self.resample_motion_files(self.env)
                env_ids = torch.arange(self.num_envs, device=self.device)
                self._resample_command(env_ids, allow_failure_accounting=False)
                # Keep aligned reference caches in sync for both stepped and reset envs.
                self._refresh_reference_cache()
                self.selection_policy.step_post_update(
                    motion_source=self.motion,
                    command_step_count=self.command_step_count,
                )
                return
        self._resample_command(env_ids)

        # Keep aligned reference caches in sync for both stepped and reset envs.
        self._refresh_reference_cache()

        self.selection_policy.step_post_update(
            motion_source=self.motion,
            command_step_count=self.command_step_count,
        )
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
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    # basic configs
    eval_mode: bool = False
    fixed_eval_motion_ids: bool = False
    adaptive_sample: bool = True
    asset_name: str = MISSING
    max_motion_num: int = 999999
    resample_interval: int = 300000000000
    motion_file: str = MISSING
    dataset_txt: str = None  # "~/dataset/g1-mimic-npz/dataset.txt"

    # reference motion obs configs
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    future_step_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    max_future_step = max(future_step_num)
    
    # randomization configs
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    
    # adaptive sampling configs
    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    motion_ratio = [0.2, 0.79, 0.01]  # 预留参数，暂时不使用 # 越来越难
    adaptive_uniform_ratio: float = 0.5
    adaptive_alpha: float = 0.001

    failure_cap: bool = True
    failure_cap_beta: float = 200.0
    failure_most_hard_cap_beta: float = 5000.0

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
