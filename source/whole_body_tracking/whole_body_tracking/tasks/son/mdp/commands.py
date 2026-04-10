from __future__ import annotations

import math
import os
from os.path import dirname, join

import numpy as np
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Union, Any, Dict, List, Optional

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
)

from whole_body_tracking.utils.motion_dataset import Motion_Bins_Dataset
from whole_body_tracking.utils.motion_dataloader import Motion_Bins_Dataloader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reorder_joint_data_for_isaac(joint_data: torch.Tensor) -> torch.Tensor:
    """
    将NPZ文件中的关节数据重排序为IsaacLab期望的顺序
    
    Args:
        joint_data: NPZ文件中的关节数据 (按NPZ顺序排列)
        npz_joint_order: NPZ文件中使用的关节顺序
        isaac_joint_names: IsaacLab机器人的关节顺序
    
    Returns:
        重排序后的关节数据 (按IsaacLab顺序排列)
    """
    # 固定的NPZ到IsaacLab映射索引（深度优先 -> 广度优先）
    NPZ_TO_ISAAC_INDICES =  [0, 6, 12, 1, 7, 13, 20, 2, 8, 14, 21, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26]
    return joint_data[:, NPZ_TO_ISAAC_INDICES]


def reorder_isaac_data_for_npz(isaac_data: torch.Tensor) -> torch.Tensor:
    """
    将IsaacLab的关节数据重排序为NPZ文件的顺序
    
    Args:
        isaac_data: IsaacLab中的关节数据 (按IsaacLab顺序排列)
        npz_joint_order: NPZ文件中使用的关节顺序
        isaac_joint_names: IsaacLab机器人的关节顺序
    
    Returns:
        重排序后的关节数据 (按NPZ顺序排列)
    """
    # 固定的IsaacLab到NPZ映射索引（广度优先 -> 深度优先）
    ISAAC_TO_NPZ_INDICES = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 23, 25, 6, 10, 14, 18, 22, 24, 26]
    # 三维数据: (num_envs, num_joints, 2) - 用于关节限制
    return isaac_data[:, ISAAC_TO_NPZ_INDICES, :]


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  
        # TODO Use the observation of SONIC
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

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
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

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

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Sample more often from time-regions where we usually fail.

        This uses the failure histogram in ``bin_failed_count`` (updated in ``_update_command``)
        to build a sampling distribution over time bins, then resamples time steps for the
        specified ``env_ids`` from that distribution.
        """
        # 1. Create sampling distribution (failure-biased + uniform prior)
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        # 2. Apply smoothing (non-causal convolution = looks left & right)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        # 3. Sample bin indices
        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        # 4. Convert sampled bin -> actual timestep (with sub-bin uniform jitter)
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        # 5. Compute metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample motion commands for the given environments.
        Env_ids are where the motion finishes or the episode (for any reason) is terminated.
        """
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        
        # fetch states from the motion buffer
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()
        # resample the joint and root states
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """Advance motion time, update failure statistics, and resample commands if needed."""
        # 1. Record failures for early terminations (exclude natural time-outs at motion end).
        episode_terminated = self._env.termination_manager.terminated
        last_valid_step = self.motion.time_step_total - 1
        early_terminated = episode_terminated & (self.time_steps < last_valid_step)
        if torch.any(early_terminated):
            failed_envs = torch.where(early_terminated)[0]
            # Map current local time_step -> bin index [0, bin_count-1]
            failed_bins = torch.clamp(
                (self.time_steps[failed_envs] * self.bin_count) // max(self.motion.time_step_total, 1),
                0,
                self.bin_count - 1,
            )
            # Accumulate failures for this step into the scratch buffer
            self._current_bin_failed += torch.bincount(failed_bins, minlength=self.bin_count).float()

        # 2. Advance local time
        self.time_steps += 1

        # 3. Resample when motion finishes or the episode (for any reason) is terminated.
        motion_finished = self.time_steps >= self.motion.time_step_total
        env_ids = torch.where(motion_finished | episode_terminated)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # 4. Update failure histogram with exponential moving average and clear scratch buffer.
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
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

    asset_name: str = MISSING
    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class MultiMotionCommand(CommandTerm):
    """Multi-motion tracking command with global bins sampling strategy.
    
    Key differences from old implementation:
    1. Global bins: All motions' bins are concatenated into a single global bin buffer
    2. Direct global sampling: Sample global bin index → uniform sample within bin → compute motion_id + time_step
    3. Vectorized operations: Similar to motion_buffer design, use offsets for efficient indexing
    
    Sampling workflow:
    1. Compute global bin probabilities (adaptive + uniform)
    2. Sample global bin indices using multinomial
    3. Uniform sample local timesteps within selected bins
    4. Use searchsorted to find motion_id from global timesteps
    5. Compute local time_steps from global timesteps
    """
    
    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Initialize robot and body indexes
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index  = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], 
            dtype=torch.long, device=self.device)

        print(f"[MultiMotionCommand] Robot  body names: {self.robot.body_names}")
        print(f"[MultiMotionCommand] Motion body names: {self.cfg.body_names}")
        print(f"[MultiMotionCommand] Body indexes: {self.body_indexes}")
        # Load motions from source dataset
        self._init_datasets()

        # Environment state: which motion and timestep each env is at
        # random init the motion ids
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # env_id -> motion_id
        # self.motion_ids = torch.randint(0, self.dataloader.num_motions, (self.num_envs,), device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Track if system has started (scalar, not per-env)
        self._has_started = False

        # Relative pose storage for tracking
        self.body_pos_relative_w  = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Motion-level bins setup: one bin per motion segment in the dataloader.
        # This keeps adaptive sampling defined over motion segments rather than time windows.
        self.bin_count = self.dataloader.num_motions
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)

        # Metrics
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
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top2_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top2_bin"] = torch.zeros(self.num_envs, device=self.device)
        print("[MultiMotionCommand] Initialization complete:")
        print(f"  - Loaded {self.dataloader.num_motions} motions")
        print(f"  - Total frames: {self.dataloader.time_step_total}")
        print(f"  - Total bins: {self.bin_count}")

    def _init_datasets(self):
        """ Initialize Motion_Bins_Dataset and Motion_Bins_Dataloader
        """
        print(f"[MultiMotionCommand] Loading dataset from: {self.cfg.dataset_dir}")
        self.dataset = Motion_Bins_Dataset(dataset_dir=self.cfg.dataset_dir)
        self.dataloader = Motion_Bins_Dataloader(
            dataset=self.dataset,
            body_indexes=self.body_indexes,
            device=self.device,
            world_size=self.cfg.distributed_world_size,
            rank=self.cfg.distributed_rank,
        )

    @property
    def command(self) -> torch.Tensor:
        """Return horizon future motion frames (joint pos + vel) similar to MotionCommandSonic.
        Shape: (num_envs, command_horizon * 2 * num_joints)
        The joint pos and vel in the future 10 frames with a fixed delta t are concatenated as the command for each environment.
        """
        # Compute local time steps of each environment
        horizon = self.cfg.command_horizon
        fps_env = self.dataloader.motion_fps[self.motion_ids]  # [num_envs]
        interval_steps = torch.clamp((fps_env / float(self.cfg.command_rate_hz)).round().long(), min=1)  # [num_envs]
        base_steps = self.time_steps  # [num_envs]
        # Offsets for each frame in the horizon, scaled per-env by interval_steps
        frame_ids = torch.arange(horizon, dtype=base_steps.dtype, device=self.device)  # [horizon] 10 frames
        offsets = interval_steps.unsqueeze(1) * frame_ids.unsqueeze(0)                 # [num_envs, horizon]
        future_global_time_steps = base_steps.unsqueeze(1) + offsets                   # [num_envs, horizon]
        # Clamp future timesteps to stay within the current motion for each env
        motion_end_steps = self.dataloader.motion_lengths[self.motion_ids] - 1 + self.dataloader.one_sec_frames # [num_envs]
        future_global_time_steps = torch.clamp(future_global_time_steps, max=motion_end_steps.unsqueeze(1))     # [num_envs, horizon]
        # Gather joint positions / velocities using matching [N, H] motion/time indices
        motion_ids_expanded = self.motion_ids.unsqueeze(1).expand_as(future_global_time_steps)              # [N, H]
        joint_pos = self.dataloader.motion_buffer.joint_pos[motion_ids_expanded, future_global_time_steps]  # [N, H, J]
        joint_vel = self.dataloader.motion_buffer.joint_vel[motion_ids_expanded, future_global_time_steps]  # [N, H, J]
        cmd = torch.cat([joint_pos, joint_vel], dim=-1)  # [N, H, 2*J]
        return cmd.view(self.num_envs, -1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """Target joint positions for all environments."""
        return self.dataloader.motion_buffer.joint_pos[self.motion_ids, self.time_steps] # num_envs, jnt_num

    @property
    def joint_vel(self) -> torch.Tensor:
        """Target joint velocities for all environments."""
        return self.dataloader.motion_buffer.joint_vel[self.motion_ids, self.time_steps] # num_envs, jnt_num

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Target body positions in world frame."""
        return self.dataloader.motion_buffer.body_pos_w[self.motion_ids, self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Target body quaternions in world frame."""
        return self.dataloader.motion_buffer.body_quat_w[self.motion_ids, self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Target body linear velocities in world frame."""
        return self.dataloader.motion_buffer.body_lin_vel_w[self.motion_ids, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Target body angular velocities in world frame."""
        return self.dataloader.motion_buffer.body_ang_vel_w[self.motion_ids, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """Target anchor body position in world frame."""
        return self.dataloader.motion_buffer.body_pos_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Target anchor body quaternion in world frame."""
        return self.dataloader.motion_buffer.body_quat_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """Target anchor body linear velocity in world frame."""
        return self.dataloader.motion_buffer.body_lin_vel_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """Target anchor body angular velocity in world frame."""
        return self.dataloader.motion_buffer.body_ang_vel_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """Current robot joint positions."""
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """Current robot joint velocities."""
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """Current robot body positions in world frame."""
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """Current robot body quaternions in world frame."""
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """Current robot body linear velocities in world frame."""
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """Current robot body angular velocities in world frame."""
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Current robot anchor body position in world frame."""
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Current robot anchor body quaternion in world frame."""
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        """Current robot anchor body linear velocity in world frame."""
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        """Current robot anchor body angular velocity in world frame."""
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    def _update_metrics(self):
        """ Update tracking error metrics. """        
        # anchor body tracking
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        # ee body tracking
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1)
        # joint tracking
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1) # [N, 29]
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    # NOTE current sampling mechanism is problematic, need to be modified
    def _adaptive_sampling(self, env_ids: Sequence[int] | torch.Tensor):
        """Adaptive sampling over motion-level bins.

        Workflow:
        1. Compute sampling probabilities (failure-weighted + uniform) from per-motion failure counts.
        2. Sample motion bin indices.
        3. Uniform sample timesteps within selected motions.
        4. Assign sampled motion_id and local time_step to the given envs.

        Bin failure counts are updated in ``_update_command`` for all terminated envs
        (including mid-motion failures), so harder motion segments get higher sampling probability.
        """
        #############################################
        # Convert env_ids to tensor if it is a list #
        #############################################
        # env_ids contains both early terminated and natural terminated envs
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        episode_failed = self._env.termination_manager.terminated[env_ids_tensor] # len(env_ids_tensor)
        if torch.any(episode_failed):
            # fail_bins is the motion ids list of the failed envs
            fail_bins = self.motion_ids[env_ids_tensor][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)
            # print(f"fail_bins shape: {fail_bins.shape}")
            # print(f"episode_failed envs number: {episode_failed.float().sum().item()}")
        ##########################################
        # Step 1: Compute sampling probabilities #
        ##########################################
        total_failed = self.bin_failed_count.sum()
        if total_failed <= 0:
            sampling_probabilities = torch.ones(self.bin_count, device=self.device) / self.bin_count
            failed_sampling_probabilities = sampling_probabilities.clone()
        else:
            clip_bin_failed_count = torch.minimum(
                self.bin_failed_count, total_failed / self.cfg.adaptive_cap
            )
            failed_sampling_probabilities = torch.nn.functional.normalize(
                clip_bin_failed_count, p=1, dim=0
            )
            sampling_probabilities = (
                self.cfg.adaptive_uniform_ratio * failed_sampling_probabilities
                + (1.0 - self.cfg.adaptive_uniform_ratio) / self.bin_count
            )
            sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        ##############################
        # Step 2: Sample global bins #
        ##############################
        sampled_motion_bins = torch.multinomial(sampling_probabilities, len(env_ids_tensor), replacement=True)  # [M]
        # Uniform sample within each bin
        random_offsets = sample_uniform(0.0, 1.0, (len(env_ids_tensor),), device=self.device)
        self.time_steps[env_ids_tensor] = (random_offsets * (self.dataloader.motion_lengths[sampled_motion_bins] - 1)).long()
        self.motion_ids[env_ids_tensor] = sampled_motion_bins    # id of a motion segment

        #########################################################
        # Step 3: Update sampling metrics (for debugging only) #
        #########################################################
        H = -(failed_sampling_probabilities * (failed_sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = failed_sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        # Top-1
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
        # Top-2
        failed_sampling_probabilities[imax] = 0.0
        p2max, i2max = failed_sampling_probabilities.max(dim=0)
        self.metrics["sampling_top2_prob"][:] = p2max
        self.metrics["sampling_top2_bin"][:] = i2max.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor):
        """Resample motion commands for given environments.

        1. will be called when envs earlyterminated or time out AUTOMATICALLY
        2. will be called when self.time_out < 0 [set as infinity in this class]
        3. will be called by _update_command() in this class, which will be called every time step AUTOMATICALLY

        so this function will be called twice in this class
        
        call self._adaptive_sampling(env_ids) to adjust sampling probabilities
        reset envs according to the sampled motion ids and time steps
        """
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        ###############################################################
        # Initialize robot state from sampled motion data with noise. #
        ###############################################################
        # Get motion data for resampled environments
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()
        # Add pose noise
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        # Add velocity noise
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]
        # Get joint positions and velocities
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        # Add joint position noise and clip to limits
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        isaac_soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        npz_soft_joint_pos_limits = reorder_isaac_data_for_npz(isaac_soft_joint_pos_limits)
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], 
            npz_soft_joint_pos_limits[:, :, 0], 
            npz_soft_joint_pos_limits[:, :, 1]
        )
        # Reorder joint data back to Isaac order and write to sim
        isaac_joint_pos = reorder_joint_data_for_isaac(joint_pos[env_ids])
        isaac_joint_vel = reorder_joint_data_for_isaac(joint_vel[env_ids])
        # Write to simulation (isaac_joint_* already indexed by env_ids)
        self.robot.write_joint_state_to_sim(isaac_joint_pos, isaac_joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """ Update command each timestep.
        To handle cases where motion bin is finished
        So envs of early termination or time out are not considered in this function
        """
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.dataloader.motion_lengths[self.motion_ids])[0]
        # print("Call resample command ...")
        self._resample_command(env_ids)

        ######################################
        # Update relative poses for tracking #
        ######################################
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # Update bin failure counts with exponential moving average
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()
        
        # Mark system as started after first update completes
        self._has_started = True

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
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
        """Debug visualization callback."""
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for multi-motion command with global bins sampling."""

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""
    # Dataset configuration
    dataset_dir: str = MISSING
    """Directory containing motion bins."""

    # Distributed training configuration
    distributed_world_size: int = 1
    """Total number of distributed processes."""
    distributed_rank: int = 0
    """Current process rank in distributed training."""
    distributed_data_split: bool = False
    """Whether to enable distributed data sharding across ranks."""
    # Body configuration
    anchor_body_name: str = MISSING
    """Name of the anchor body (usually root or pelvis)."""
    body_names: list[str] = MISSING
    """List of body names to track."""

    # Initialization noise ranges
    pose_range: dict[str, tuple[float, float]] = {}
    """Pose noise ranges for x, y, z, roll, pitch, yaw."""
    velocity_range: dict[str, tuple[float, float]] = {}
    """Velocity noise ranges for x, y, z, roll, pitch, yaw."""
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    """Joint position noise range."""

    # Adaptive sampling parameters
    adaptive_kernel_size: int = 1
    """Kernel size for convolution smoothing of bin probabilities."""
    adaptive_lambda: float = 0.8
    """Exponential decay factor for kernel weights."""
    adaptive_uniform_ratio: float = 0.9
    """Ratio of uniform sampling mixed with failure-based sampling."""
    adaptive_cap: int = 2
    """Cap for bin failure counts to prevent extreme probabilities."""
    adaptive_alpha: float = 0.001
    """EMA smoothing factor for bin failure counts."""

    # Command horizon configuration (must align with MotionCommandSonic)
    command_horizon: int = 10
    """Number of future frames to include in the command."""
    command_rate_hz: float = 10.0
    """Command rate in Hz, used to compute frame interval from per-motion FPS."""
    # Evaluation-specific parameters
    eval_target_attempts: int = 128
    """Number of evaluation attempts per motion (used by EvalMultiMotionCommand)."""

    # Visualization
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
