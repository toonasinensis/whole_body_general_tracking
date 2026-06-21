import math
import torch
from typing import Sequence

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform


#负责环境重置
# 提供
#   需要重置的环境 ID
#   参考动作的 body 位置和朝向
#   参考动作的 关节 位置和速度
#   相关参数
# 重置机器人的状态
class MotionCommandResetter:  #checked
    """Apply sampled reference motion states back to the simulator on reset/resample."""

    def __init__(self, cfg, robot: Articulation, device: str):
        # TODO cfg 边界不稳定
        self.cfg = cfg
        self.robot = robot
        self.device = device
        self._pose_init_method_env_ids: dict[str, torch.Tensor] = {}
        self._pose_init_method_num_envs = -1

    def _get_pose_init_method_env_ids(self, num_envs: int) -> dict[str, torch.Tensor]:
        if self._pose_init_method_num_envs == num_envs:
            return self._pose_init_method_env_ids

        ratios = getattr(self.cfg, "pose_init_method_ratios", {"range": 1.0})
        if not ratios:
            ratios = {"range": 1.0}

        total_ratio = sum(max(0.0, float(ratio)) for ratio in ratios.values())
        if total_ratio <= 0.0:
            ratios = {"range": 1.0}
            total_ratio = 1.0

        start = 0
        method_env_ids: dict[str, torch.Tensor] = {}
        items = list(ratios.items())
        for index, (method, ratio) in enumerate(items):
            ratio = max(0.0, float(ratio)) / total_ratio
            end = num_envs if index == len(items) - 1 else start + int(num_envs * ratio)
            method_env_ids[str(method)] = torch.arange(start, end, dtype=torch.long, device=self.device)
            start = end

        self._pose_init_method_env_ids = method_env_ids
        self._pose_init_method_num_envs = num_envs
        return method_env_ids

    def _apply_root_pose_randomization(
        self,
        root_pos: torch.Tensor,
        root_ori: torch.Tensor,
        env_ids: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        pose_ranges = torch.tensor(
            [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )
        method_env_ids = self._get_pose_init_method_env_ids(root_pos.shape[0])
        for mode, all_method_env_ids in method_env_ids.items():
            if all_method_env_ids.numel() == 0:
                continue
            method_mask = torch.isin(env_ids, all_method_env_ids)
            pose_env_ids = env_ids[method_mask]
            if pose_env_ids.numel() == 0:
                continue

            pose_noise = sample_uniform(
                pose_ranges[:, 0], pose_ranges[:, 1], (pose_env_ids.numel(), 6), device=self.device
            )
            self._apply_root_pose_randomization_mode(
                mode,
                root_pos,
                root_ori,
                pose_env_ids,
                pose_noise,
                env_origins,
            )

    def _apply_root_pose_randomization_mode(
        self,
        mode: str,
        root_pos: torch.Tensor,
        root_ori: torch.Tensor,
        pose_env_ids: torch.Tensor,
        pose_noise: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        if mode == "range":
            root_pos[pose_env_ids] += pose_noise[:, :3]
            roll = pose_noise[:, 3]
            pitch = pose_noise[:, 4]
        elif mode == "lying":
            root_pos[pose_env_ids, :2] += pose_noise[:, :2]
            height_range = torch.tensor(
                getattr(self.cfg, "pose_range_lying_height_range", (0.25, 0.45)),
                device=self.device,
            )
            root_pos[pose_env_ids, 2] = env_origins[pose_env_ids, 2] + sample_uniform(
                height_range[0], height_range[1], (pose_env_ids.numel(),), device=self.device
            )
            lie_ids = torch.randint(0, 4, (pose_env_ids.numel(),), device=self.device)
            roll = torch.zeros(pose_env_ids.numel(), device=self.device)
            pitch = torch.zeros(pose_env_ids.numel(), device=self.device)
            roll[lie_ids == 0] = math.pi / 2.0
            roll[lie_ids == 1] = -math.pi / 2.0
            pitch[lie_ids == 2] = math.pi / 2.0
            pitch[lie_ids == 3] = -math.pi / 2.0
        else:
            raise ValueError(f"Unsupported pose init method={mode!r}. Expected 'range' or 'lying'.")

        root_ori[pose_env_ids] = quat_mul(
            quat_from_euler_xyz(roll, pitch, pose_noise[:, 5]),
            root_ori[pose_env_ids],
        )

    def apply(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        *,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        # NOTE body_names[0] MUST be robot's anchor 
        # TODO use assigned anchor id to index anchor states
        root_pos = body_pos_w[:, 0].clone()
        root_ori = body_quat_w[:, 0].clone()
        root_lin_vel = body_lin_vel_w[:, 0].clone()
        root_ang_vel = body_ang_vel_w[:, 0].clone()

        # add noise
        self._apply_root_pose_randomization(root_pos, root_ori, env_ids, env_origins)

        velocity_ranges = torch.tensor(
            [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )
        velocity_noise = sample_uniform(
            velocity_ranges[:, 0], velocity_ranges[:, 1], (env_ids.numel(), 6), device=self.device
        )
        root_lin_vel[env_ids] += velocity_noise[:, :3]
        root_ang_vel[env_ids] += velocity_noise[:, 3:]

        joint_pos = joint_pos.clone()
        joint_vel = joint_vel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)

        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_vel_limits = self.robot.data.joint_vel_limits[env_ids]
        #TODO add this to config
        max_ang_vel_root = 20.0

        #clip to limits
        joint_pos[env_ids] = torch.clip(joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
        joint_vel[env_ids] = torch.clip(joint_vel[env_ids], -joint_vel_limits[:, :], joint_vel_limits[:, :])
        root_ang_vel[env_ids] = torch.clip(root_ang_vel[env_ids], -max_ang_vel_root, max_ang_vel_root)

        #write to sim
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
