# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The D12-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the D12-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

D20_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/d20_v2/urdf/D20.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.14,
            "right_shoulder_roll_joint": -0.14,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_pitch_joint": 1.1,
            ".*_wrist_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.1,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_pitch_joint": 0.15,
            ".*_ankle_pitch_joint": -0.05,
            ".*_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
            effort_limit_sim={
                "waist_yaw_joint": 180,
                "waist_roll_joint": 180,
                "waist_pitch_joint": 180,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 29.8,
                "waist_roll_joint": 29.8,
                "waist_pitch_joint": 29.8,
            },
            stiffness={
                "waist_yaw_joint": 400.0,
                "waist_roll_joint": 100.0,
                "waist_pitch_joint": 100.0,
            },
            damping={
                "waist_yaw_joint": 4.0,
                "waist_roll_joint": 1.0,
                "waist_pitch_joint": 1.0,
            },
            armature={
                "waist_yaw_joint": 0.189,
                "waist_roll_joint": 0.378,
                "waist_pitch_joint": 0.378,
            },
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_hip_pitch_joint", ".*_knee_pitch_joint"],
            effort_limit_sim={
                ".*_hip_pitch_joint": 180,
                ".*_hip_roll_joint": 180,
                ".*_hip_yaw_joint": 180,
                ".*_knee_pitch_joint": 295,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 16.0,
                ".*_hip_roll_joint": 16.0,
                ".*_hip_yaw_joint": 16.0,
                ".*_knee_pitch_joint": 20.2,
            },
            stiffness={
                ".*_hip_pitch_joint": 300.0,
                ".*_hip_roll_joint": 300.0,
                ".*_hip_yaw_joint": 250.0,
                ".*_knee_pitch_joint": 300.0,
            },
            damping={
                ".*_hip_pitch_joint": 4.0,
                ".*_hip_roll_joint": 4.0,
                ".*_hip_yaw_joint": 3.0,
                ".*_knee_pitch_joint": 4.0,
            },
            armature={
                ".*_hip_pitch_joint": 0.01,
                ".*_hip_roll_joint": 0.126,
                ".*_hip_yaw_joint": 0.01,
                ".*_knee_pitch_joint": 0.15,
            },
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_pitch_joint",
                              ".*_wrist_roll_joint"],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 94,
                ".*_shoulder_roll_joint": 94,
                ".*_shoulder_yaw_joint": 94,
                ".*_elbow_pitch_joint": 94,
                ".*_wrist_roll_joint": 20,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 25.5,
                ".*_shoulder_roll_joint": 25.5,
                ".*_shoulder_yaw_joint": 25.5,
                ".*_elbow_pitch_joint": 25.5,
                ".*_wrist_roll_joint": 18,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 100,
                ".*_shoulder_roll_joint": 100,
                ".*_shoulder_yaw_joint": 100,
                ".*_elbow_pitch_joint": 100,
                ".*_wrist_roll_joint": 50,
            },
            damping={
                ".*_shoulder_pitch_joint": 2,
                ".*_shoulder_roll_joint": 2,
                ".*_shoulder_yaw_joint": 2,
                ".*_elbow_pitch_joint": 2,
                ".*_wrist_roll_joint": 1,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.032,
                ".*_shoulder_roll_joint": 0.032,
                ".*_shoulder_yaw_joint": 0.032,
                ".*_elbow_pitch_joint": 0.032,
                ".*_wrist_roll_joint": 0.005,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=95,
            velocity_limit_sim=25.5,
            stiffness={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            damping={
                ".*_ankle_pitch_joint": 1.0,
                ".*_ankle_roll_joint": 1.0,
            },
            armature={
                ".*_ankle_pitch_joint": 0.064,
                ".*_ankle_roll_joint": 0.064,
            },
        ),
    },
)

D20_V2_ACTION_SCALE = {}
for actuator in D20_V2_CFG.actuators.values():
    effort_limit_sim = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    joint_name_patterns = actuator.joint_names_expr
    if not isinstance(effort_limit_sim, dict):
        effort_limit_sim = {joint_name: effort_limit_sim for joint_name in joint_name_patterns}
    if not isinstance(stiffness, dict):
        stiffness = {joint_name: stiffness for joint_name in joint_name_patterns}
    for joint_name in joint_name_patterns:
        if joint_name in effort_limit_sim and joint_name in stiffness and stiffness[joint_name]:
            D20_V2_ACTION_SCALE[joint_name] = 0.25 * effort_limit_sim[joint_name] / stiffness[joint_name]
