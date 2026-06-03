from .smpl_motion_lib import SmplMotionLib
from .unified_motion_lib import UnifiedMotionLib

"""
    本模块的核心对外接口是 UnifiedMotionLib 类，提供对机器人运动数据和对应 SMPL 数据的统一访问。
    设计目标是简化下游任务（如 RL 训练）对运动数据的使用，隐藏数据加载、预处理和索引管理的复杂性。
    主要功能包括：
    - 统一加载接口：通过 load() 和 load_from_cfg() 方法加载运动数据，支持从配置对象直接加载。
    - 设备管理：提供 to_device() 方法将数据迁移到指定设备。
    - 数据访问接口：提供属性和方法访问机器人运动数据和 SMPL 数据，如 joint_pos、body_pos_w、get_smpl_joints() 等。
    - 内部状态管理：维护当前加载的运动数据状态，支持 reset() 方法清除状态。
    - 错误处理：在访问数据前检查加载状态，未加载时抛出 MotionNotLoadedError 错误。
"""
