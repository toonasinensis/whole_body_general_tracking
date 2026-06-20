"""Storage and indexing helpers for loaded motion clips."""

from .paired_store import PairedMotionStore
from .robot_store import RobotMotionStore
from .smpl_store import SmplMotionStore
from .index import motion_ids_from_timestamps

__all__ = [
    "PairedMotionStore",
    "RobotMotionStore",
    "SmplMotionStore",
    "motion_ids_from_timestamps"
]

"""
    1. 对外依赖 types.py 中预定义的 数据类型
    2. 负责对 io 加载后的 motion clips 进行拼接和存储
"""
