from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def update_push_with_entropy_and_eposidelength(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    entropy_threshold: float = 0.9,
    target_episode_length: float = 490.0,
) -> float:
    """
    如果智能体的采样熵（sampling_entropy）和平均存活时长达到了设定阈值，
    才将其 push 扰动打开。 如果低于该阈值，则将推力区间设置为 0。
    """
    motion_term = env.command_manager.get_term("motion")
    
    # 1. 提取当前平均指标
    mean_entropy = motion_term.metrics["sampling_entropy"].mean().item()
    
    # 获取本次【刚刚死亡/被截断】的机器人的存活步数均值
    # 注意：此时这些环境(env_ids)的 episode_length_buf 记录了它们完整的一次存活了多少步
    if len(env_ids) > 0:
        died_episode_length = env.episode_length_buf[env_ids].float().mean().item()
    else:
        died_episode_length = 0.0

    # 为了平滑，我们在环境(env)级别维护一个全局的滑动平均存活步数
    if not hasattr(env, "smooth_episode_length"):
        env.smooth_episode_length = 0.0
    
    if len(env_ids) > 0:
        # 使用滑动平均更新全局存活指标
        alpha = 0.05
        env.smooth_episode_length = (1 - alpha) * env.smooth_episode_length + alpha * died_episode_length

    # 2. 判断是否达标
    can_push = (mean_entropy > entropy_threshold) and (env.smooth_episode_length > target_episode_length)

    # 3. 动态调整 event 里的受力范围
    if "push_robot" in env.event_manager.active_terms:
        push_event = env.event_manager.get_term_cfg("push_robot")
        if can_push:
            from whole_body_tracking.tasks.tracking.tracking_env_cfg import VELOCITY_SMALL_RANGE
            push_event.params["velocity_range"] = VELOCITY_SMALL_RANGE
        else:
            push_event.params["velocity_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            }
        env.event_manager.set_term_cfg("push_robot", push_event)

    if "base_external_force_torque" in env.event_manager.active_terms:
        force_event = env.event_manager.get_term_cfg("base_external_force_torque")
        if can_push:
            force_event.params["force_range"] = {
                "x": (-600.0, 600.0), "y": (-600.0, 600.0), "z": (-350.0, 350.0)
            }
        else:
            force_event.params["force_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        env.event_manager.set_term_cfg("base_external_force_torque", force_event)

    return 1.0 if can_push else 0.0
