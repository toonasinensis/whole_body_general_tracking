from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat


#负责将参考数据或者额外的数据和当前机器人状态对齐
# 根据机器人的 anchor 位置和朝向
# 重新对齐参考动作的 anchor 和 body 的位置和朝向
class MotionReferenceCache:  #checked
    """Cache reference poses aligned into the robot's anchor XY-yaw frame."""

    def __init__(self, num_envs: int, body_count: int, device: str): #wxyz
        self.body_pos_relative_w  = torch.zeros(num_envs, body_count, 3, device=device)
        self.body_quat_relative_w = torch.zeros(num_envs, body_count, 4, device=device)
        self.body_quat_relative_w[:, :, 0] = 1.0

    def refresh(
        self,
        *,
        anchor_pos_w: torch.Tensor,
        anchor_quat_w: torch.Tensor,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        robot_anchor_pos_w: torch.Tensor,
        robot_anchor_quat_w: torch.Tensor,
    ) -> None:
        """
            align reference motion to the robot's current XY position and yaw, 
            but keep the reference motion's absolute height profile.
            used for computing rewards.
        """
        body_count = body_pos_w.shape[1]
        anchor_pos_w_repeat  = anchor_pos_w[:, None, :].repeat(1, body_count, 1)  #envs_num, body_num, dim
        anchor_quat_w_repeat = anchor_quat_w[:, None, :].repeat(1, body_count, 1)
        robot_anchor_pos_w_repeat = robot_anchor_pos_w[:, None, :].repeat(1, body_count, 1)
        robot_anchor_quat_w_repeat = robot_anchor_quat_w[:, None, :].repeat(1, body_count, 1)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w[:] = quat_mul(delta_ori_w, body_quat_w)
        self.body_pos_relative_w[:]  = delta_pos_w + quat_apply(delta_ori_w, body_pos_w - anchor_pos_w_repeat)

    """负责环境重置逻辑"""
