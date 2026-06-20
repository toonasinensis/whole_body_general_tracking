"""File IO entry points for loaded motion clips."""

from .discovery import discover_npz_files, discover_pkl_files
from .robot_npz import load_robot_motion_file
from .smpl_pkl import load_smpl_motion_file

__all__ = [
    "discover_npz_files",
    "discover_pkl_files",
    "load_smpl_motion_file",
    "load_robot_motion_file",
]

"""
    "discover_npz_files": 读取 npz 文件的绝对路径  tuple[list[str], int]
    "discover_pkl_files": 读取 pkl 文件的绝对路径  tuple[list[str], int]
    "load_smpl_motion_file" : 从给定的文件路径中读取详细信息 RobotMotionData
    "load_robot_motion_file": 从给定的文件路径中读取详细信息 RobotMotionData
"""
