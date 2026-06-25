from __future__ import annotations
import torch
from typing import Protocol
from dataclasses import dataclass


class MotionDataSource(Protocol):   #checked
    """ Motion ranges for motion indexing"""
    time_step_total: int
    time_step_start_idx: torch.Tensor
    time_step_end_idx: torch.Tensor
    motion_num: int

    """
    定义 MotionData 的基本属性
    数据源应该至少有本接口所定义的属性以供其他模块使用
    本模块只调用这些暴露的属性，而不使用 MotionData 的其他属性    

    TODO
    重命名变量 对齐模块间的变量命名规则
    """

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

    """
    模块间传输 motion frame 的标准类

    TODO
    尝试预分配此部分空间
    """
