from __future__ import annotations

import datetime
import json
import math
import numpy as np
import os

# from whole_body_parkour.data import DATA_ASSET_DIR
import random
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from tqdm import tqdm
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import (  # RED_ARROW_X_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: F401
from isaaclab.utils.math import quat_from_euler_xyz  # noqa: F401
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .math_utils import quat_to_6d

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
                start_idx = min(start_idx, len(npz_files) - 1)  # 防止越界
                end_idx = min(start_idx + motion_num, len(npz_files) - 1)  # 防止越界
                if start_idx > len(npz_files) or start_idx == len(npz_files) - 1:
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
        rel_npz_file_names = [os.path.relpath(f, dir_path) for f in npz_file_paths]
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
            except Exception as e:
                print(f"   路径: {f_path}")
                print(f"   错误: {e}")
                # 可选：跳过这个文件继续
                continue

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


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # Per-env local frame index within the currently selected motion.
        self.local_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Per-env motion id in the current concatenated motion buffer.
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.motion = MotionLoader(self.cfg, self.body_indexes, self.motion_anchor_body_index, device=self.device)

        self.use_new_motion_pre_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.resample_motion_files(self.env)
        # self.motion.update_last_motion_data()  # when init , init last motion data

        self.frame_end_per_env = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        self.history_success_rate_dict = {}
        self.command_step_count = 0
        self._last_bins_export_step = -1

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
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

        # Future-frame indexing helpers.
        future_step_num = getattr(self.cfg, "future_step_num", [0])
        if future_step_num is None or len(future_step_num) == 0:
            future_step_num = [0]
        self._future_step_offsets = torch.tensor(future_step_num, device=self.device, dtype=torch.long)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
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

    # region normal property
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

    def motion_ids_from_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Get motion ids for each timestamp in the current concatenated motion buffer."""
        return self.motion.motion_ids_from_timestamps(timestamps)

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
        start = self.motion.time_step_start_idx[self.motion_ids]
        return start + self.local_time_steps

    @property
    def num_future_frames(self) -> int:
        return int(self._future_step_offsets.numel())

    @property
    def motion_start_time_steps(self) -> torch.Tensor:
        """Per-env global start index of the current motion."""
        return self.motion.time_step_start_idx[self.motion_ids]

    @property
    def motion_num_steps(self) -> torch.Tensor:
        """Per-env number of frames for the current motion."""
        start = self.motion.time_step_start_idx[self.motion_ids]
        end = self.motion.time_step_end_idx[self.motion_ids]
        return end - start

    @property
    def future_time_steps_init(self) -> torch.Tensor:
        """Future step offsets (local frame deltas) as a 1D tensor."""
        return self._future_step_offsets

    @property
    def future_motion_ids(self) -> torch.Tensor:
        """Motion ids for all future reference frames, flattened."""
        return self.motion_ids[:, None].expand(-1, self.num_future_frames).reshape(-1)

    @property
    def future_time_steps(self) -> torch.Tensor:
        """Compute absolute (global) time-step indices for all future reference frames.

        Clamps to the last valid frame of each motion to avoid out-of-bounds access.

        Returns:
            Flattened tensor of shape ``(num_envs * num_future_frames,)``.
        """
        start = self.motion_start_time_steps
        local_max = (self.motion_num_steps - 1).clamp(min=0)
        future_local = torch.clip(
            self.local_time_steps[:, None] + self.future_time_steps_init[None, :],
            max=local_max[:, None],
        )
        return (start[:, None] + future_local).long()

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

    def _set_time_from_global_timestamps(self, env_ids: Sequence[int], timestamps: torch.Tensor) -> None:
        """Set (motion_ids, local time_steps, local end) from global timestamps."""
        if len(env_ids) == 0:
            return

        if timestamps.dtype != torch.long:
            timestamps = timestamps.long()
        timestamps = torch.clamp(timestamps, min=0, max=int(self.motion.time_step_total) - 1)

        motion_ids = self.motion.motion_ids_from_timestamps(timestamps)
        start = self.motion.time_step_start_idx[motion_ids]
        end = self.motion.time_step_end_idx[motion_ids]
        local_t = timestamps - start

        self.motion_ids[env_ids] = motion_ids
        self.local_time_steps[env_ids] = local_t
        # store local end (exclusive) to drive resampling with local time_steps
        self.frame_end_per_env[env_ids] = end - start

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

    def resample_motion_files(self, env):
        self.motion.resample_motionloader(device=self.device)
        self.use_new_motion_pre_env[:] = False
        self.bin_count = (
            int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        )  # 1s motion frames for each bin
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        # NOTE self.kernel is not used in this script
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()
        self.resample_time = 0
        self.success_motion = torch.zeros(self.motion.motion_num, dtype=torch.float32, device=self.device)

    def _compute_sampling_probabilities(self) -> torch.Tensor:
        self.metrics["failures_max"][:] = self.bin_failed_count.max()
        self.metrics["failures_mean"][:] = self.bin_failed_count.mean()
        self.metrics["failures_min"][:] = self.bin_failed_count.min()
        self.metrics["failures_max_over_uniform"][:] = (
            self.bin_failed_count.max() * self.bin_count / (self.bin_failed_count.sum() + 1e-8)
        )
        clipped_bin_failed_count = torch.clamp(
            self.bin_failed_count, max=self.cfg.failure_most_hard_cap_beta * self.bin_failed_count.mean()
        )
        sampling_probabilities = torch.nn.functional.pad(
            clipped_bin_failed_count.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities = sampling_probabilities / (sampling_probabilities.sum() + 1e-12)

        sampling_probabilities_middle_hard = torch.clamp(
            sampling_probabilities, max=self.cfg.failure_cap_beta * sampling_probabilities.mean()
        )
        sampling_probabilities_middle_hard = sampling_probabilities_middle_hard / (
            sampling_probabilities_middle_hard.sum() + 1e-12
        )

        sampling_probabilities_most_hard = torch.clamp(
            sampling_probabilities, max=self.cfg.failure_most_hard_cap_beta * sampling_probabilities.mean()
        )
        sampling_probabilities_most_hard = sampling_probabilities_most_hard / (
            sampling_probabilities_most_hard.sum() + 1e-12
        )

        sampling_probabilities = self.cfg.motion_ratio[0] * (1 / float(self.bin_count)) + (
            self.cfg.motion_ratio[1] * sampling_probabilities_middle_hard
            + self.cfg.motion_ratio[2] * sampling_probabilities_most_hard
        )

        return sampling_probabilities

    def _get_bin_global_frame_range(self, bin_index: int) -> tuple[int, int]:
        total_frames = max(int(self.motion.time_step_total), 1)
        if total_frames == 1:
            return 0, 1

        scaled_total = total_frames - 1
        global_start = int(math.floor((bin_index * scaled_total) / float(self.bin_count)))
        global_end = int(math.ceil(((bin_index + 1) * scaled_total) / float(self.bin_count)))
        global_end = max(global_start + 1, min(global_end, total_frames))
        return global_start, global_end

    def _build_bin_motion_segments(self, global_start: int, global_end: int) -> list[dict]:
        start_idx = self.motion.time_step_start_idx.detach().cpu().tolist()
        end_idx = self.motion.time_step_end_idx.detach().cpu().tolist()
        file_names = list(getattr(self.motion, "file_names", []))
        if len(file_names) == 0:
            file_names = [f"motion_{i:05d}.npz" for i in range(int(self.motion.motion_num))]

        segments: list[dict] = []
        for motion_id, (motion_start, motion_end) in enumerate(zip(start_idx, end_idx)):
            overlap_start = max(global_start, int(motion_start))
            overlap_end = min(global_end, int(motion_end))
            if overlap_start >= overlap_end:
                continue
            segments.append(
                {
                    "motion_id": int(motion_id),
                    "motion_file": file_names[motion_id],
                    "global_frame_start": int(overlap_start),
                    "global_frame_end_exclusive": int(overlap_end),
                    "motion_local_frame_start": int(overlap_start - motion_start),
                    "motion_local_frame_end_exclusive": int(overlap_end - motion_start),
                }
            )
            if motion_end >= global_end:
                break
        return segments

    def _export_adaptive_bins(self) -> None:
        import time

        st1 = time.time()
        if not self.cfg.adaptive_sample or not self.cfg.save_adaptive_bins:
            return
        if self.bin_count <= 0 or self.motion.time_step_total is None:
            return

        sampling_probabilities = self._compute_sampling_probabilities().detach().cpu()
        bin_failed_count = self.bin_failed_count.detach().cpu()
        fps = float(getattr(self.motion, "fps", 0.0))
        rank = int(self.cfg.local_rank) if int(self.cfg.local_rank) >= 0 else 0

        # 只导出bin_failed_count最大的前100个bins
        topk = min(100, int(self.bin_count))
        # torch.topk返回值是(tensor, indices)
        _, top_indices = torch.topk(bin_failed_count, k=topk, largest=True, sorted=True)
        top_indices = top_indices.tolist()

        bins = []
        for bin_index in top_indices:
            global_start, global_end = self._get_bin_global_frame_range(bin_index)
            motion_segments = self._build_bin_motion_segments(global_start, global_end)
            first_segment = motion_segments[0] if motion_segments else None
            bins.append(
                {
                    "bin_index": int(bin_index),
                    "sampling_probability": float(sampling_probabilities[bin_index].item()),
                    "bin_failed_count": float(bin_failed_count[bin_index].item()),
                    "global_frame_start": int(global_start),
                    "global_frame_end_exclusive": int(global_end),
                    "global_time_s_start": float(global_start / fps) if fps > 0 else 0.0,
                    "global_time_s_end_exclusive": float(global_end / fps) if fps > 0 else 0.0,
                    "segment_count": int(len(motion_segments)),
                    "motion_id_at_bin_start": int(first_segment["motion_id"]) if first_segment else -1,
                    "motion_file": first_segment["motion_file"] if first_segment else None,
                    "motion_local_frame_start": int(first_segment["motion_local_frame_start"]) if first_segment else -1,
                    "motion_local_frame_end_exclusive": (
                        int(first_segment["motion_local_frame_end_exclusive"]) if first_segment else -1
                    ),
                    "motion_segments": motion_segments,
                }
            )

        # 按bin_failed_count降序排序bins
        bins_sorted_indices = sorted(range(len(bins)), key=lambda index: bins[index]["bin_failed_count"], reverse=True)
        payload = {
            "meta": {
                "step": int(self.command_step_count),
                "rank": rank,
                "distributed": bool(self.cfg.distributed),
                "motions_dir": str(self.cfg.motion_file),
                "log_save_path": str(self.cfg.log_save_path),
                "fps": fps,
                "motion_num": int(self.motion.motion_num),
                "total_frames": int(self.motion.time_step_total),
                "bin_count": int(self.bin_count),
                "adaptive_kernel_size": int(self.cfg.adaptive_kernel_size),
                "adaptive_lambda": float(self.cfg.adaptive_lambda),
                "adaptive_uniform_ratio": float(self.cfg.adaptive_uniform_ratio),
                "adaptive_alpha": float(self.cfg.adaptive_alpha),
            },
            "bins": bins,
            "bins_sorted": bins_sorted_indices,
        }

        save_dir = Path(self.cfg.log_save_path).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            save_dir / f"{self.cfg.adaptive_bins_file_prefix}_rank_{rank:02d}_step_{self.command_step_count:09d}.json"
        )
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._last_bins_export_step = self.command_step_count
        print(f"[MotionCommand] Wrote adaptive bins: {out_path}, {rank}, which take times: {time.time()-st1}")

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_joint_pos"] = torch.mean(torch.abs(self.joint_pos - self.robot_joint_pos), dim=-1)
        self.metrics["error_joint_vel"] = torch.mean(torch.abs(self.joint_vel - self.robot_joint_vel), dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """
        1. update bin failed history
        2. compute sampling probability and sample bins accordingly
        3. compute metrics
        """
        # NOTE there shall be a logic to justify
        # whether the env_ids are out of time, out of motion range, or failed (early termination)
        # if early termination, add to self._current_bin_failed
        # else no change
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            # Use the last valid frame (exclusive end indices shouldn't be used as timestamps).
            global_ts = torch.clamp(self.global_time_steps - 1, min=0, max=int(self.motion.time_step_total) - 1)
            current_bin_index = torch.clamp(
                (global_ts * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            # NOTE terminated envs are early terminated or out of motion range ?
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample

        sampling_probabilities = self._compute_sampling_probabilities()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        global_ts = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Map global timestamp -> (motion_id, local frame index, local end), avoiding mask+argmax.
        self._set_time_from_global_timestamps(env_ids, global_ts)
        if self.cfg.eval_mode:
            # 评估模式下，动作都从该 motion 的第一个帧进行（local=0）
            self.local_time_steps[env_ids] = 0
        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob_max"][:] = pmax
        self.metrics["prob_max_over_uniform"][:] = pmax / (1 / self.bin_count)
        self.metrics["prob_uniform"][:] = 1 / self.bin_count
        self.metrics["sampling_top1_prob_bin"][:] = imax.float() / self.bin_count
        self.metrics["sampling_top1_prob_mean"][:] = sampling_probabilities.mean()
        self.metrics["sampling_top1_prob_min"][:] = sampling_probabilities.min()
        self.metrics["num_concentrate_bins"][:] = (
            sampling_probabilities > self.cfg.failure_most_hard_cap_beta * 0.5 * (sampling_probabilities.mean())
        ).sum()  # 计算超过平均值10倍的数目

    def _resample_command(self, env_ids: Sequence[int]):
        """_resample_command will be called multiple times in each step
        1. Called from _update_command
        2. Called directly in the IsaacLab simulator
        2.1. after check_termination()
        2.2. after reset_all() maybe?
        """
        if len(env_ids) == 0:
            return
        if self.cfg.adaptive_sample:
            self._adaptive_sampling(env_ids)
        else:
            raise NotImplementedError
            # self.use_new_motion_pre_env[env_ids] = True

        # add noise to robot states when envs are reset
        # add noise to root state
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        # orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        # root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]
        # add noise to joint state
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_vel_limits = self.robot.data.joint_vel_limits[env_ids]
        max_ang_vel_root = 20.0
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        joint_vel[env_ids] = torch.clip(joint_vel[env_ids], -joint_vel_limits[:, :], joint_vel_limits[:, :])
        root_ang_vel[env_ids] = torch.clip(root_ang_vel[env_ids], -max_ang_vel_root, max_ang_vel_root)
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        # TODO: 切换动作文件时是否需要清空历史Observation
        # NOTE: what about the historical observation when resetting the environment?

    def _update_command(self):
        """
        Called every control step, update commands for envs that are out of time
        """
        self.command_step_count += 1
        self.local_time_steps += 1
        env_ids = torch.where(self.local_time_steps >= self.frame_end_per_env - self.cfg.max_future_step)[0]  # 防止溢出
        if self.cfg.resample_interval != -1:  # change the reference motion every resample_interval control steps
            self.resample_time += 1
            if self.resample_time >= self.cfg.resample_interval:
                self.resample_motion_files(self.env)
                env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)

        # compute the metrics
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )  # 把数据中xy yaw换成实际机器人的xy yaw

        # MAE update the bin failed history
        # NOTE we shall ensure that the _resample_command() function be called once before this function
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()
        if (
            self.cfg.save_adaptive_bins
            and self.cfg.fail_count_save_interval > 0
            and self.command_step_count % self.cfg.fail_count_save_interval == 0
            and self._last_bins_export_step != self.command_step_count
        ):
            self._export_adaptive_bins()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )
                self.future_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/future/anchor")
                )

                if self.cfg.debug_anchor_speed:
                    self.current_anchor_lin_vel_visualizer = VisualizationMarkers(
                        self.cfg.current_anchor_lin_vel_visualizer_cfg
                    )
                    self.goal_anchor_lin_vel_visualizer = VisualizationMarkers(
                        self.cfg.goal_anchor_lin_vel_visualizer_cfg
                    )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            self.future_anchor_visualizer.set_visibility(True)
            if self.cfg.debug_anchor_speed:
                self.current_anchor_lin_vel_visualizer.set_visibility(True)
                self.goal_anchor_lin_vel_visualizer.set_visibility(True)

            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                self.future_anchor_visualizer.set_visibility(False)
                if self.cfg.debug_anchor_speed:
                    self.current_anchor_lin_vel_visualizer.set_visibility(False)
                    self.goal_anchor_lin_vel_visualizer.set_visibility(False)

                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _resolve_velocity_to_arrow(
        self,
        velocity_w: torch.Tensor,
        default_scale: tuple[float, float, float],
        speed_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert 3D world velocity to arrow scale and world quaternion."""
        speed = torch.linalg.norm(velocity_w, dim=1)
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(velocity_w.shape[0], 1)
        arrow_scale[:, 0] *= speed * speed_scale

        # Rotate arrow local +X axis to velocity direction in world frame.
        eps = 1.0e-8
        direction = velocity_w / speed.unsqueeze(-1).clamp(min=eps)
        x_axis = torch.zeros_like(direction)
        x_axis[:, 0] = 1.0

        cross = torch.cross(x_axis, direction, dim=1)
        dot = torch.sum(x_axis * direction, dim=1).clamp(-1.0, 1.0)

        w = torch.sqrt(((1.0 + dot).clamp(min=0.0)) * 0.5)
        xyz = cross / (2.0 * w.unsqueeze(-1).clamp(min=eps))
        arrow_quat_w = torch.cat([w.unsqueeze(-1), xyz], dim=1)

        # Handle opposite direction (dot=-1): 180 deg around +Y axis.
        opposite = dot < (-1.0 + 1.0e-6)
        if torch.any(opposite):
            arrow_quat_w[opposite] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device)

        # For near-zero velocity, keep identity orientation.
        stationary = speed < 1.0e-6
        if torch.any(stationary):
            arrow_quat_w[stationary] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        arrow_quat_w = torch.nn.functional.normalize(arrow_quat_w, dim=1)
        return arrow_scale, arrow_quat_w

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        if hasattr(self, "future_anchor_visualizer"):
            self.future_anchor_visualizer.visualize(
                self.anchor_pos_w_future.view(-1, 3), self.anchor_quat_w_future.view(-1, 4)
            )

        if self.cfg.debug_anchor_speed:
            current_lin_vel_scale, current_lin_vel_quat = self._resolve_velocity_to_arrow(
                self.robot_anchor_lin_vel_w,
                self.current_anchor_lin_vel_visualizer.cfg.markers["arrow"].scale,
                self.cfg.debug_anchor_speed_scale,
            )
            goal_lin_vel_scale, goal_lin_vel_quat = self._resolve_velocity_to_arrow(
                self.anchor_lin_vel_w,
                self.goal_anchor_lin_vel_visualizer.cfg.markers["arrow"].scale,
                self.cfg.debug_anchor_speed_scale,
            )

            current_lin_vel_pos = self.robot_anchor_pos_w
            goal_lin_vel_pos = self.anchor_pos_w

            self.current_anchor_lin_vel_visualizer.visualize(
                current_lin_vel_pos,
                current_lin_vel_quat,
                current_lin_vel_scale,
            )
            self.goal_anchor_lin_vel_visualizer.visualize(
                goal_lin_vel_pos,
                goal_lin_vel_quat,
                goal_lin_vel_scale,
            )

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    dataset_txt: str = None  # "/home/xiechunyang/wt_ws/wt_wbc/dataset/g1-mimic-npz/dataset.txt"
    eval_mode: bool = False
    adaptive_sample: bool = True
    asset_name: str = MISSING
    max_motion_num: int = 999999
    resample_interval: int = 300000000000
    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    log_save_path: str = "train_logs"
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    future_step_num = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    max_future_step = max(future_step_num)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    motion_ratio = [0.2, 0.79, 0.01]  # 预留参数，暂时不使用 # 越来越难
    adaptive_uniform_ratio: float = 0.5
    adaptive_alpha: float = 0.001

    failure_cap: bool = True
    failure_cap_beta: float = 200.0
    failure_most_hard_cap_beta: float = 5000.0

    # 每隔多少步导出一次 adaptive bins 概率和 bin->motion 反查映射，-1 表示不保存
    fail_count_save_interval: int = 500 * 24
    save_adaptive_bins: bool = True
    adaptive_bins_file_prefix: str = "adaptive_bins"

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
