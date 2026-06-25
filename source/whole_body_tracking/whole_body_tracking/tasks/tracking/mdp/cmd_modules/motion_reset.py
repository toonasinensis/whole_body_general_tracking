import math
import torch
from typing import Sequence

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform


class MotionCommandResetter:  #checked
    """Apply sampled reference motion states back to the simulator on reset/resample."""

    def __init__(self, num_envs:int, cfg, robot: Articulation, device: str):
        # TODO cfg 边界不稳定
        self.num_envs = num_envs
        self.cfg = cfg
        self.robot = robot
        self.device = device

        # tensordict
        self.envs_classes_mask = {}
        self._init_envs_mask(self.num_envs)

        """
        负责环境重置
        
        - 根据 cfg 设置 env_mask 
        - 根据不同的 env_mask 实现对应的初始化方案

        享有变量
        - self.envs_classes_mask 不同的训练环境的 mask
        - self.envs_classes_mask 被 motion_command terminaition rewards 可读但不可写

        TODO 
        - 将 self.cfg 实例化为专属的 Config 类
        - envs_classes_mask 
        """

    def _init_envs_mask(self, num_envs: int) -> None:
        """
        Build contiguous env masks from cfg.envs_classes_ratio.
        
        TODO 尝试使用 random env masks 初始化方案
        """
        self.envs_classes_mask.clear()
        ratio_sum = sum(self.cfg.envs_classes_ratio.values())
        start = 0
        class_items = list(self.cfg.envs_classes_ratio.items())
        for i, (key, ratio) in enumerate(class_items):
            ratio = ratio / ratio_sum
            count = num_envs - start if i == len(class_items) - 1 else int(ratio * num_envs)
            end = start + count
            mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            mask[start:end] = True
            self.envs_classes_mask[key] = mask
            start = end

    def _apply_lying_randomization(
        self,
        root_pos: torch.Tensor,
        root_ori: torch.Tensor,
        pose_env_ids: torch.Tensor,
        pose_noise: torch.Tensor,
        env_origins: torch.Tensor,
    ):
        # set root linear position
        root_pos[pose_env_ids, :2] += pose_noise[:, :2]
        # TODO set this as private confing
        height_range = torch.tensor(
            getattr(self.cfg, "pose_range_lying_height_range", (0.25, 0.45)),
            device=self.device,
        )
        root_pos[pose_env_ids, 2] = env_origins[pose_env_ids, 2] + sample_uniform(
            height_range[0], height_range[1], (pose_env_ids.numel(),), device=self.device
        )

        # set root angular position
        lie_ids = torch.randint(0, 4, (pose_env_ids.numel(),), device=self.device)
        roll = torch.zeros(pose_env_ids.numel(), device=self.device)
        pitch = torch.zeros(pose_env_ids.numel(), device=self.device)
        roll[lie_ids == 0] = math.pi / 2.0
        roll[lie_ids == 1] = -math.pi / 2.0
        pitch[lie_ids == 2] = math.pi / 2.0
        pitch[lie_ids == 3] = -math.pi / 2.0
        root_ori[pose_env_ids] = quat_mul(
            quat_from_euler_xyz(roll, pitch, pose_noise[:, 5]),
            root_ori[pose_env_ids],
        )

    def _apply_range_randomization(
        self,
        root_pos: torch.Tensor,
        root_ori: torch.Tensor,
        pose_env_ids: torch.Tensor,
        pose_noise: torch.Tensor,
        env_origins: torch.Tensor,
    ):
        root_pos[pose_env_ids] += pose_noise[:, :3]
        roll = pose_noise[:, 3]
        pitch = pose_noise[:, 4]
        root_ori[pose_env_ids] = quat_mul(
            quat_from_euler_xyz(roll, pitch, pose_noise[:, 5]),
            root_ori[pose_env_ids],
        )

    def add_root_pose_randomization(
        self,
        root_pos: torch.Tensor,
        root_ori: torch.Tensor,
        env_ids: torch.Tensor,
        env_origins: torch.Tensor,  
    ):
        """
        NOTE 设置 "roll", "pitch", "yaw" 为 0 以保证原来的 range 初始化方式不变
        TODO 未来改用纯 dict 方式 不显式指定 mode
        """
        # compute pose noise ranges
        pose_ranges = torch.tensor(
            [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )

        # set range mode env root pose
        pose_range_env_mask = self.envs_classes_mask["range"]
        pose_env_ids = env_ids[pose_range_env_mask[env_ids]]
        if pose_env_ids.numel() > 0:
            pose_noise = sample_uniform(
                pose_ranges[:, 0], pose_ranges[:, 1], (pose_env_ids.numel(), 6), device=self.device
            ) # num_env_ids, 6
            self._apply_range_randomization(
                root_pos,
                root_ori,
                pose_env_ids,
                pose_noise,
                env_origins,
            )

        # set lying mode env root pose
        pose_range_env_mask = self.envs_classes_mask["lying"]
        pose_env_ids = env_ids[pose_range_env_mask[env_ids]]
        if pose_env_ids.numel() > 0:
            pose_noise = sample_uniform(
                pose_ranges[:, 0], pose_ranges[:, 1], (pose_env_ids.numel(), 6), device=self.device
            ) # num_env_ids, 6
            self._apply_lying_randomization(
                root_pos,
                root_ori,
                pose_env_ids,
                pose_noise,
                env_origins,
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
        joint_pos = joint_pos.clone()
        joint_vel = joint_vel.clone()
        root_pos = body_pos_w[:, 0].clone()
        root_ori = body_quat_w[:, 0].clone()
        root_lin_vel = body_lin_vel_w[:, 0].clone()
        root_ang_vel = body_ang_vel_w[:, 0].clone()

        # add noise to root position
        self.add_root_pose_randomization(root_pos, root_ori, env_ids, env_origins)

        # add noise to root velocity
        velocity_ranges = torch.tensor(
            [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )
        velocity_noise = sample_uniform(
            velocity_ranges[:, 0], velocity_ranges[:, 1], (env_ids.numel(), 6), device=self.device
        )
        root_lin_vel[env_ids] += velocity_noise[:, :3]
        root_ang_vel[env_ids] += velocity_noise[:, 3:]

        # add noise to joint position
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
