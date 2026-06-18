from __future__ import annotations
import torch
from typing import Protocol
from dataclasses import dataclass


#定义 MotionData 的基本属性
# 数据源应该至少有以下所定义的属性
# 以供本模块使用
# 本模块只调用这些暴露的属性，而不使用 MotionData 的其他属性
class MotionDataSource(Protocol):   #checked
    """ Motion ranges for motion indexing"""
    time_step_total: int
    time_step_start_idx: torch.Tensor
    time_step_end_idx: torch.Tensor

    def motion_ids_from_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Map global timestamps to motion ids."""


#定义选择动作的基本数据结构
# MotionData 根据此改变自身的 indexing 逻辑
@dataclass
class MotionSelection:  #checked
    """ Used to fetch motion frame for each env """
    motion_ids: torch.Tensor
    local_time_steps: torch.Tensor
    frame_end: torch.Tensor
