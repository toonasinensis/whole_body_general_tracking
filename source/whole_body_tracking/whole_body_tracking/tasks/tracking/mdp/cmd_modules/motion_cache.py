from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat


class MotionReferenceCache:  #checked
    """Cache reference poses aligned into the robot's anchor XY-yaw frame."""

    def __init__(self, num_envs: int, body_count: int, device: str): #wxyz
        self.body_pos_relative_w  = torch.zeros(num_envs, body_count, 3, device=device)
        self.body_quat_relative_w = torch.zeros(num_envs, body_count, 4, device=device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        """
        负责将参考数据或和其他模态的数据和当前机器人状态对齐
        - 根据当前机器人的 anchor 位置和朝向
        - 重新对齐参考动作的 anchor 和 body 的位置和朝向        

        拥有对外变量
        self.body_pos_relative_w  : 参考动作帧根据当前环境中 robot root 状态 xy 平移 和 yaw 旋转对齐后的世界坐标系的位置表示
        self.body_quat_relative_w : 参考动作帧根据当前环境中 robot root 状态 xy 平移 和 yaw 旋转对齐后的世界坐标系的旋转表示

        它的值只应该由 MotionReferenceCache.refresh() 写入
        在每一步末尾更新 作为下一步的参考帧
        其他模块通过 MotionCommand property 只读
        """

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
        robot_anchor_pos_w_repeat  = robot_anchor_pos_w[:, None, :].repeat(1, body_count, 1)
        robot_anchor_quat_w_repeat = robot_anchor_quat_w[:, None, :].repeat(1, body_count, 1)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w[:] = quat_mul(delta_ori_w, body_quat_w)
        self.body_pos_relative_w[:]  = delta_pos_w + quat_apply(delta_ori_w, body_pos_w - anchor_pos_w_repeat)
