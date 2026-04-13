import re

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.actuators.actuator_cfg import DelayedPDActuatorCfg_RobanS22
from whole_body_tracking.assets import ASSET_DIR

# ============================================================
# Motor armature parameters for RobanS2.2 (biped_s17)
# ============================================================
# NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
# DAMPING_RATIO = 2.0
#
# Motor types based on URDF effort limits:
#   - 150N motors  (leg_l/r 1,2,4):  Large leg actuators
#   - 80N  motor   (waist_yaw):      Waist actuator
#   - 70N  motors  (leg_l/r 3):      Leg yaw actuators
#   - 74N  motors  (leg_l/r 5,6):    Ankle actuators
#   - 37N  motors  (zarm_l/r 2,3,4): Arm actuators
#   - 14.1N motors (zarm_l/r 1):     Arm shoulder
#   - 12N  motor   (zhead_2):        Head pitch
#   - 1.5N motor   (zhead_1):        Head yaw
# ============================================================

PRESERVE_JOINT_ORDER_ASSET_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "waist_yaw_joint",
        "leg_l1_joint",
        "leg_l2_joint",
        "leg_l3_joint",
        "leg_l4_joint",
        "leg_l5_joint",
        "leg_l6_joint",
        "leg_r1_joint",
        "leg_r2_joint",
        "leg_r3_joint",
        "leg_r4_joint",
        "leg_r5_joint",
        "leg_r6_joint",
        "zarm_l1_joint",
        "zarm_l2_joint",
        "zarm_l3_joint",
        "zarm_l4_joint",
        "zarm_r1_joint",
        "zarm_r2_joint",
        "zarm_r3_joint",
        "zarm_r4_joint",
    ],
    preserve_order=True,
)


# RobanS22_CYLINDER_CFG is the configuration for the RobanS22 robot.
# It contains:
# - Basic URDF settings
# - Initial state: joint positions and velocities
# - Soft joint pos limit factor: factor to scale the joint positions
# - Actuators: configuration for the actuators
RobanS22_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/roban_s22/urdf/biped_s17.urdf",
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
        pos=(0.0, 0.0, 0.8),
        rot=(1, 0.0, 0.0, 0.0),
        joint_pos={
            "waist_yaw_joint": 0.0,
            "leg_l1_joint": 0.0,
            "leg_l2_joint": 0.0,
            "leg_l3_joint": 0.0,
            "leg_l4_joint": 0.0,
            "leg_l5_joint": 0.0,
            "leg_l6_joint": 0.0,
            "leg_r1_joint": 0.0,
            "leg_r2_joint": 0.0,
            "leg_r3_joint": 0.0,
            "leg_r4_joint": 0.0,
            "leg_r5_joint": 0.0,
            "leg_r6_joint": 0.0,
            "zarm_l1_joint": 0.0,
            "zarm_l2_joint": 0.0,
            "zarm_l3_joint": 0.0,
            "zarm_l4_joint": 0.0,
            "zarm_r1_joint": 0.0,
            "zarm_r2_joint": 0.0,
            "zarm_r3_joint": 0.0,
            "zarm_r4_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "motor": DelayedPDActuatorCfg_RobanS22(
            joint_names_expr=[
                "waist_yaw_joint",
                "leg_.*",
                "zarm_.*",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 80.0,
                "leg_[l,r]1_joint": 150.0,
                "leg_[l,r]2_joint": 150.0,
                "leg_[l,r]3_joint": 70.0,
                "leg_[l,r]4_joint": 150.0,
                "leg_[l,r]5_joint": 74.0,
                "leg_[l,r]6_joint": 74.0,
                "zarm_[l,r]1_joint": 14.1,
                "zarm_[l,r]2_joint": 37.0,
                "zarm_[l,r]3_joint": 37.0,
                "zarm_[l,r]4_joint": 37.0,
            },
            effort_limit_rated={
                "waist_yaw_joint": 35.0,
                "leg_[l,r]1_joint": 51.0,
                "leg_[l,r]2_joint": 51.0,
                "leg_[l,r]3_joint": 34.0,
                "leg_[l,r]4_joint": 51.0,
                "leg_[l,r]5_joint": 37.0,
                "leg_[l,r]6_joint": 37.0,
                "zarm_[l,r]1_joint": 6.1,
                "zarm_[l,r]2_joint": 11.0,
                "zarm_[l,r]3_joint": 11.0,
                "zarm_[l,r]4_joint": 11.0,
            },
            velocity_limit={
                "waist_yaw_joint": 12.0,
                "leg_[l,r]1_joint": 14.6,
                "leg_[l,r]2_joint": 14.6,
                "leg_[l,r]3_joint": 12.0,
                "leg_[l,r]4_joint": 14.6,
                "leg_[l,r]5_joint": 17.0,
                "leg_[l,r]6_joint": 17.0,
                "zarm_[l,r]1_joint": 10.5,
                "zarm_[l,r]2_joint": 15.0,
                "zarm_[l,r]3_joint": 15.0,
                "zarm_[l,r]4_joint": 15.0,
            },
            stiffness={
                "waist_yaw_joint": 40.1792,
                "leg_[l,r]1_joint": 90.1792,
                "leg_[l,r]2_joint": 150.0984,
                "leg_[l,r]3_joint": 40.1792,
                "leg_[l,r]4_joint": 150.0984,
                "leg_[l,r]5_joint": 34.2506,
                "leg_[l,r]6_joint": 34.2506,
                "zarm_[l,r]1_joint": 14.2506,
                "zarm_[l,r]2_joint": 14.2506,
                "zarm_[l,r]3_joint": 14.2506,
                "zarm_[l,r]4_joint": 14.2506,
            },
            damping={
                "waist_yaw_joint": 3.5579,
                "leg_[l,r]1_joint": 3.5579,
                "leg_[l,r]2_joint": 8.3088,
                "leg_[l,r]3_joint": 3.5579,
                "leg_[l,r]4_joint": 8.3088,
                "leg_[l,r]5_joint": 2.9072,
                "leg_[l,r]6_joint": 2.9072,
                "zarm_[l,r]1_joint": 1.9072,
                "zarm_[l,r]2_joint": 1.9072,
                "zarm_[l,r]3_joint": 1.9072,
                "zarm_[l,r]4_joint": 1.9072,
            },
            armature={
                "waist_yaw_joint": 0.03178707,
                "leg_[l,r]1_joint": 0.0493,
                "leg_[l,r]2_joint": 0.0493,
                "leg_[l,r]3_joint": 0.03178707,
                "leg_[l,r]4_joint": 0.0493,
                "leg_[l,r]5_joint": 0.036,
                "leg_[l,r]6_joint": 0.036,
                "zarm_[l,r]1_joint": 0.015,
                "zarm_[l,r]2_joint": 0.018,
                "zarm_[l,r]3_joint": 0.018,
                "zarm_[l,r]4_joint": 0.018,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "waist_yaw_joint": 0.2,
                "leg_[l,r]1_joint": 0.2,
                "leg_[l,r]2_joint": 0.2,
                "leg_[l,r]3_joint": 0.2,
                "leg_[l,r]4_joint": 0.2,
                "leg_[l,r]5_joint": 0.2,
                "leg_[l,r]6_joint": 0.2,
                "zarm_[l,r]1_joint": 0.2,
                "zarm_[l,r]2_joint": 0.2,
                "zarm_[l,r]3_joint": 0.2,
                "zarm_[l,r]4_joint": 0.2,
            },
            activation_vel=0.1,
            friction_dynamic=0,
        ),
    },
)


# Compute the action scale for joints of roban_s22 robot
RobanS22_ACTION_SCALE = {}
for a in RobanS22_CYLINDER_CFG.actuators.values():
    e_cfg = a.effort_limit_sim
    e_rated_cfg = a.effort_limit_rated
    s_cfg = a.stiffness
    name_patterns = a.joint_names_expr

    if not name_patterns:
        continue

    candidate_joint_names = []
    joint_names_list = PRESERVE_JOINT_ORDER_ASSET_CFG.joint_names or []
    for jn in joint_names_list:
        for pat in name_patterns:
            if re.fullmatch(pat, jn):
                candidate_joint_names.append(jn)
                break

    def _resolve_value(joint_name, cfg_value):
        if isinstance(cfg_value, dict):
            for pat, val in cfg_value.items():
                if re.fullmatch(pat, joint_name):
                    return val
            return None
        else:
            return cfg_value

    for joint_name in candidate_joint_names:
        e_val = _resolve_value(joint_name, e_cfg)
        s_val = _resolve_value(joint_name, s_cfg)
        e_rated_val = _resolve_value(joint_name, e_rated_cfg)
        if e_val is None or e_rated_val is None or s_val in (None, 0):
            continue
        RobanS22_ACTION_SCALE[joint_name] = 0.25 * float(e_rated_val * 0.6) / float(s_val)
