from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
    euler_xyz_from_quat,
    quat_apply_inverse
)
from tqdm import tqdm
from .math_utils import quat_to_6d
# from whole_body_parkour.data import DATA_ASSET_DIR
import random
import json
import datetime
import copy
from pathlib import Path
import copy
from pathlib import Path

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
 
class MotionLoader:
    def __init__(self, cfg: MotionCommandCfg, body_indexes: Sequence[int] = [0], device: str = "cpu"):
        self.device = device
        self.cfg = cfg
        self.body_indexes = body_indexes
        self.motion_num = None
        self.json_path = None
        self.death_path = None
        self.first_init = False

        self.joint_pos = None
        self.joint_vel = None
        self._body_pos_w = None
        self._body_quat_w = None
        self._body_lin_vel_w = None
        self._body_ang_vel_w = None

        # self.last_joint_pos = self.joint_pos
        # self.last_joint_vel = self.joint_vel
        # self._last_body_pos_w = self._body_pos_w
        # self._last_body_quat_w = self._body_quat_w
        # self._last_body_lin_vel_w = self._body_lin_vel_w
        # self._last_body_ang_vel_w = self._body_ang_vel_w

        self.time_step_total = None  # self.joint_pos.shape[0]
        self.file_names = None
        self.time_step_end_idx = None
        self.frame_list = None  # 确保在同一 device
        self.time_step_start_idx = None
        self.motion_num = None
        if self.cfg.eval_mode:
            self.sample_counter = 0
        
    def _find_npz_files(self, dir_path: Path, motion_num: int):
        """随机选择一个子文件夹（若存在），返回其中所有 .npz 文件路径"""
        dir_path = Path(dir_path)
        npz_files = list(dir_path.rglob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {dir_path}")
        """从 npz 文件中均匀随机采样 motion_num 个"""
        npz_files.sort()
         
        if len(npz_files) > motion_num and motion_num != -1:  # 动作文件需要采样（内存不够）或者人为指定把所有数据拿出来
            if self.cfg.eval_mode:
                start_idx = (self.sample_counter * motion_num) # % len(npz_files)
                start_idx = min(start_idx, len(npz_files)-1)  # 防止越界
                end_idx = min(start_idx + motion_num, len(npz_files)-1)  # 防止越界
                if start_idx > len(npz_files) or start_idx ==  len(npz_files)-1:
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
    
    def load_and_cat_npz_with_filenames(self, dir_path, motion_num=25, device="cpu"):
        """
        加载 npz 文件，并按 batch 拼接，避免一次性占用 GPU 显存。
        batch_size: 每次拼接的帧数
        """

        batch_size = 1024
        npz_file_paths = self._find_npz_files(dir_path, motion_num)
        rel_npz_file_names = [os.path.relpath(f, dir_path) for f in npz_file_paths]
        tensor_keys = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                    "body_lin_vel_w", "body_ang_vel_w"]
        tensor_lists = {k: [] for k in tensor_keys}
        fps_list = []
        frames_per_file = []

        # 先在 CPU 上读取数据，保持 list
        for f_path in tqdm(npz_file_paths, desc="Processing files"):            
            data = np.load(f_path, allow_pickle=True)
            for k in tensor_keys:
                tensor = torch.from_numpy(data[k]).float()  # CPU tensor
                tensor_lists[k].append(tensor)
            fps_list.append(data["fps"])
            frames_per_file.append(data["joint_pos"].shape[0])

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

    # def update_last_motion_data(self):
    #     if self.joint_pos is not None:
    #         print("Saving last motion data for continuity.")
    #         self.last_joint_pos = self.joint_pos.clone()
    #         self.last_joint_vel = self.joint_vel.clone()
    #         self._last_body_pos_w = self._body_pos_w.clone()
    #         self._last_body_quat_w = self._body_quat_w.clone()
    #         self._last_body_lin_vel_w = self._body_lin_vel_w.clone()
    #         self._last_body_ang_vel_w = self._body_ang_vel_w.clone()

    def resample_motionloader(self, device):

        data_dict, file_names, fps, frame_list = self.load_and_cat_npz_with_filenames(self.cfg.motion_file,\
                                                                                      self.cfg.max_motion_num, device) # frame list是每个motion file长度的list
        self.fps = fps

        # if self.cfg.resample_interval != -1:
            # save last data
            # self.update_last_motion_data()

        self.joint_pos = data_dict["joint_pos"] 
        self.joint_vel = data_dict["joint_vel"] 
        self._body_pos_w = data_dict["body_pos_w"] 
        self._body_quat_w = data_dict["body_quat_w"] 
        self._body_lin_vel_w = data_dict["body_lin_vel_w"]
        self._body_ang_vel_w = data_dict["body_ang_vel_w"]
        self._body_indexes = self.body_indexes
        # print("Loaded motions with total frames:", sum(frame_list), frame_list)
        if device != "cpu":
            print(f"[data_dict] GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

        print(" ")
        print(" ")
        print(" ")

        self.time_step_total = sum(frame_list)  # self.joint_pos.shape[0]
        self.file_names = file_names
        print("self.file_names: ", self.file_names)
        self.time_step_end_idx = []
        self.frame_list = torch.tensor(frame_list, device=device)  
        self.time_step_end_idx = torch.cumsum(self.frame_list, dim=0)
        self.time_step_start_idx = torch.cat(
            [torch.tensor([0], device=self.frame_list.device), self.time_step_end_idx[:-1]])
        self.motion_num = len(frame_list)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]
    
    @property
    def body_pos_z(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes, 2:3]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    def motion_ids_from_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Map global frame timestamps to motion ids.

        The global timestamps index the concatenated motion tensor. This function returns
        the corresponding motion id for each timestamp based on ``time_step_end_idx``.
        """
        if self.time_step_end_idx is None:
            raise RuntimeError("MotionLoader is not initialized. Call resample_motionloader first.")
        if timestamps.dtype != torch.long:
            timestamps = timestamps.long()
        timestamps = torch.clamp(timestamps, min=0, max=int(self.time_step_total) - 1)
        motion_ids = torch.bucketize(timestamps, self.time_step_end_idx, right=False)
        return torch.clamp(motion_ids, max=self.motion_num - 1)
    
 
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
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.motion = MotionLoader(self.cfg, self.body_indexes, device=self.device)
        
        self.use_new_motion_pre_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.resample_motion_files(self.env)

        # self.motion.update_last_motion_data()  # when init , init last motion data

        self.frame_end_per_env = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.history_success_rate_dict = {}

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
            self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)


    # 旧的command，没有姿态指令，anchor高度指令
    # @property
    # def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
    #     return torch.cat([self.joint_pos, self.joint_vel, self.anchor_lin_vel_w, self.anchor_ang_vel_w], dim=1)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, 
                          self.joint_vel, 
                          self.anchor_lin_vel_w, 
                          self.anchor_ang_vel_w,
                          self.anchor_project_gravity,
                        #   self.anchor_6d_rotation,
                          self.anchor_pos_z], dim=1)

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

    # # joint 系列
    # @future_motion_property("joint_pos", has_body_dim=False)
    # def future_joint_pos(self):
    #     pass

    # @future_motion_property("joint_vel", has_body_dim=False)
    # def future_joint_vel(self):
    #     pass

    # body_z系列
    # @future_motion_property("body_pos_z", has_body_dim=True)
    # def future_body_pos_z(self):
    #     pass

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        body_offsets_w = self.body_offsets_from_motion_ids(self.motion_ids_from_timestamps(self.time_steps))
        return self.motion.body_pos_w[self.time_steps] + body_offsets_w[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        body_offsets_w = self.body_offsets_from_motion_ids(self.motion_ids_from_timestamps(self.time_steps))
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + body_offsets_w

    def motion_ids_from_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Get motion ids for each timestamp in the current concatenated motion buffer."""
        return self.motion.motion_ids_from_timestamps(timestamps)

    def body_offsets_from_motion_ids(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Build world-frame body offsets from motion ids.

        Offsets are selected from terrain origins using modulo indexing:
        ``terrain_ids = motion_ids % num_terrain_origins``.
        """
        if motion_ids.dtype != torch.long:
            motion_ids = motion_ids.long()

        # Same logic as replay script:
        # terrain_origins = scene.terrain.terrain_origins.reshape(-1, 3)
        # terrain_ids = motion_ids % terrain_origins.shape[0]
        # offsets = terrain_origins[terrain_ids]
        terrain_origins = self._env.scene.terrain.terrain_origins.reshape(-1, 3)
        terrain_ids = motion_ids % terrain_origins.shape[0]
        offsets = terrain_origins[terrain_ids].to(self.device)
        return offsets

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    
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

    def resample_motion_files(self, env):
        self.motion.resample_motionloader(device=self.device)
        self.use_new_motion_pre_env[:] = False

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.resample_time = 0
        self.success_motion = torch.zeros(self.motion.motion_num, dtype=torch.float32, device=self.device)

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
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        beta = 200.0 # 限制最大失败率
        self.cfg.adaptive_uniform_ratio = 0.5 # 这个数越小平滑度越高 
        # 双卡lafan1可以使用参数为0.5，不会影响最终效果，但是训练会较慢
        # 八卡amass+lafan1 0.7
        clipped_bin_failed_count \
            = torch.clamp(self.bin_failed_count, 
                          max = beta * self.bin_failed_count.mean())
        sampling_probabilities = self.cfg.adaptive_uniform_ratio * clipped_bin_failed_count \
                                + (1 - self.cfg.adaptive_uniform_ratio) / float(self.bin_count)

        # sampling_probabilities = torch.nn.functional.pad(
        #     sampling_probabilities.unsqueeze(0).unsqueeze(0),
        #     (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
        #     mode="replicate",
        # )
        # sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        # self.time_steps[env_ids] = (
        #     sampled_bins 
        #     / self.bin_count 
        #     * (self.motion.time_step_total - 1)
        # ).long()

        mask = self.time_steps[env_ids].unsqueeze(1) <= self.motion.time_step_end_idx.unsqueeze(0)
        nearest_end_idx = mask.float().argmax(dim=1)  # [num_envs]
        # 对应的结束帧
        self.frame_end_per_env[env_ids] = self.motion.time_step_end_idx[nearest_end_idx]  # [num_envs]
        if self.cfg.eval_mode:
            self.time_steps[env_ids] = self.motion.time_step_start_idx[nearest_end_idx]   #评估模式下，动作都从第一个帧进行
        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.adaptive_sample:
            self._adaptive_sampling(env_ids)
        else:
            raise NotImplementedError
            # self.use_new_motion_pre_env[env_ids] = True
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

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_vel_limits = self.robot.data.joint_vel_limits[env_ids]
        max_ang_vel_root = 20.0
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        joint_vel[env_ids] = torch.clip(
            joint_vel[env_ids], -joint_vel_limits[:, :], joint_vel_limits[:, :]
        )
        root_ang_vel[env_ids] = torch.clip(root_ang_vel[env_ids], -max_ang_vel_root,max_ang_vel_root)
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        # TODO: 切换动作文件时是否需要清空历史Observation

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.frame_end_per_env)[0]
        if self.cfg.resample_interval != -1:
            self.resample_time += 1
            if self.resample_time >= self.cfg.resample_interval:
                self.resample_motion_files(self.env)
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat) # 把数据中xy yaw换成实际机器人的xy yaw

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
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
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    eval_mode: bool = False
    adaptive_sample: bool = True
    adaptive_sample_motion_file: bool = True
    asset_name: str = MISSING
    max_motion_num: int = 25
    resample_interval: int = 30000000000
    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    log_save_path: str = "train_logs"
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # 为了分布式训练
    distributed: bool = False
    local_rank: int = -1
    total_rank: int = -1

    