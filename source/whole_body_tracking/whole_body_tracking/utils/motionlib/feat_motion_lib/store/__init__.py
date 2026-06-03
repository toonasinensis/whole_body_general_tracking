"""
A module for handling motion data storage and indexing.
"""

from .index import build_length_starts, build_motion_index, flatten_indices, motion_ids_from_timestamps
from .paired_store import PairedMotionStore
from .robot_store import RobotMotionStore
from .smpl_store import SmplMotionStore


"""
    checked by kiki on 2024-06-10
    这个模块定义了几个类和函数来处理 robot motion data 和 smpl motion data 的存储和索引
    RobotMotionStore 和 SmplMotionStore 分别封装了 robot motion data 和 smpl motion data 的连接和索引信息
    PairedMotionStore 用于同时存储和验证成对的 robot motion 和 smpl motion 数据
    
    TODO 需要确认 smpl motion data 的 get 函数是否正确，以及是否需要预计算并存储
"""
