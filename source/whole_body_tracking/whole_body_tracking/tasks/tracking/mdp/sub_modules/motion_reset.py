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

    def apply(
        self,
        env_ids: Sequence[int],
        *,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        if len(env_ids) == 0:
            return
        # NOTE body_names[0] MUST be robot's anchor 
        # TODO use assigned anchor id to index anchor states
        root_pos = body_pos_w[:, 0].clone()
        root_ori = body_quat_w[:, 0].clone()
        root_lin_vel = body_lin_vel_w[:, 0].clone()
        root_ang_vel = body_ang_vel_w[:, 0].clone()

        # add noise
        pose_ranges = torch.tensor(
            [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )
        pose_noise = sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += pose_noise[:, :3]
        root_ori[env_ids] = quat_mul(
            quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]),
            root_ori[env_ids],
        )

        velocity_ranges = torch.tensor(
            [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )
        velocity_noise = sample_uniform(
            velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=self.device
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

