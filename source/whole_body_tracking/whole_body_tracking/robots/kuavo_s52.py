import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
import re

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.actuators.actuator_cfg import DelayedPDActuatorCfg_KuavoS52

KUAVO_S52_PRESERVE_JOINT_ORDER_ASSET_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        'leg_l1_joint',
        'leg_l2_joint',
        'leg_l3_joint',
        'leg_l4_joint',
        'leg_l5_joint',
        'leg_l6_joint',

        'leg_r1_joint',
        'leg_r2_joint',
        'leg_r3_joint',
        'leg_r4_joint',
        'leg_r5_joint',
        'leg_r6_joint',

        'waist_yaw_joint',

        'zarm_l1_joint',
        'zarm_l2_joint',
        'zarm_l3_joint',
        'zarm_l4_joint',
        'zarm_l5_joint',
        'zarm_l6_joint',
        'zarm_l7_joint',

        'zarm_r1_joint',
        'zarm_r2_joint',
        'zarm_r3_joint',
        'zarm_r4_joint',
        'zarm_r5_joint',
        'zarm_r6_joint',
        'zarm_r7_joint',
    ],
    preserve_order=True
)

KUAVO_S52_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/kuavo_s52/urdf/biped_s52.urdf",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.925),
        joint_pos={
            "leg_[l,r]1_joint": 0.0,
            "leg_[l,r]2_joint": 0.0,
            "leg_[l,r]3_joint": -0.4,
            "leg_[l,r]4_joint": 0.69,
            "leg_[l,r]5_joint": -0.33,
            "leg_[l,r]6_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zarm_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg_KuavoS52(
            joint_names_expr=[
                "leg_[l,r]1_joint",
                "leg_[l,r]2_joint",
                "leg_[l,r]3_joint",
                "leg_[l,r]4_joint",
                "leg_[l,r]5_joint",
                "leg_[l,r]6_joint",
                "waist_yaw_joint",
                "zarm_[l,r]1_joint",
                "zarm_[l,r]2_joint",
                "zarm_[l,r]3_joint",
                "zarm_[l,r]4_joint",
                "zarm_[l,r]5_joint",
                "zarm_[l,r]6_joint",
                "zarm_[l,r]7_joint"
            ],
            effort_limit_sim={
                "leg_[l,r]1_joint": 100.0,
                "leg_[l,r]2_joint": 56.8,
                "leg_[l,r]3_joint": 105.6,
                "leg_[l,r]4_joint": 250.0,
                "leg_[l,r]5_joint": 45.6,
                "leg_[l,r]6_joint": 45.6,
                "waist_yaw_joint": 81.6,
                "zarm_[l,r]1_joint": 52.8,
                "zarm_[l,r]2_joint": 20.0,
                "zarm_[l,r]3_joint": 20.0,
                "zarm_[l,r]4_joint": 60.0,
                "zarm_[l,r]5_joint": 20.0,
                "zarm_[l,r]6_joint": 20.0,
                "zarm_[l,r]7_joint": 20.0,
            },
            velocity_limit_sim={
                "leg_[l,r]1_joint": 15.0,
                "leg_[l,r]2_joint": 15.0,
                "leg_[l,r]3_joint": 15.0,
                "leg_[l,r]4_joint": 11.0,
                "leg_[l,r]5_joint": 15.0,
                "leg_[l,r]6_joint": 15.0,

                "waist_yaw_joint": 10.0,
                "zarm_[l,r]1_joint": 20.0,
                "zarm_[l,r]2_joint": 15.0,
                "zarm_[l,r]3_joint": 15.0,
                "zarm_[l,r]4_joint": 15.0,
                "zarm_[l,r]5_joint": 15.0,
                "zarm_[l,r]6_joint": 15.0,
                "zarm_[l,r]7_joint": 15.0,
            },
            stiffness={
                "leg_[l,r]1_joint": 45.0,
                "leg_[l,r]2_joint": 45.0,
                "leg_[l,r]3_joint": 65.0,
                "leg_[l,r]4_joint": 80.0,
                "leg_[l,r]5_joint": 40.0,
                "leg_[l,r]6_joint": 40.0,
                "waist_yaw_joint":  20.0,
                "zarm_[l,r]1_joint": 10.0,
                "zarm_[l,r]2_joint": 10.0,
                "zarm_[l,r]3_joint": 10.0,
                "zarm_[l,r]4_joint": 10.0,
                "zarm_[l,r]5_joint": 5.0,
                "zarm_[l,r]6_joint": 5.0,
                "zarm_[l,r]7_joint": 5.0,
            },
            damping={
                "leg_[l,r]1_joint": 6.0,
                "leg_[l,r]2_joint": 6.0,
                "leg_[l,r]3_joint": 6.0,
                "leg_[l,r]4_joint": 6.0,
                "leg_[l,r]5_joint": 7.5,
                "leg_[l,r]6_joint": 7.5,
                "waist_yaw_joint": 4.0,
                "zarm_[l,r]1_joint": 3.0,
                "zarm_[l,r]2_joint": 3.0,
                "zarm_[l,r]3_joint": 3.0,
                "zarm_[l,r]4_joint": 3.0,
                "zarm_[l,r]5_joint": 3.0,
                "zarm_[l,r]6_joint": 3.0,
                "zarm_[l,r]7_joint": 3.0,
            },
            armature={
                "leg_[l,r]1_joint": 0.05,
                "leg_[l,r]2_joint": 0.025,
                "leg_[l,r]3_joint": 0.025,
                "leg_[l,r]4_joint": 0.05,
                "leg_[l,r]5_joint": 0.05,
                "leg_[l,r]6_joint": 0.05,
                "waist_yaw_joint": 0.025,
                "zarm_[l,r]1_joint": 0.025,
                "zarm_[l,r]2_joint": 0.02,
                "zarm_[l,r]3_joint": 0.02,
                "zarm_[l,r]4_joint": 0.02,
                "zarm_[l,r]5_joint": 0.01,
                "zarm_[l,r]6_joint": 0.01,
                "zarm_[l,r]7_joint": 0.01,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "leg_[l,r]1_joint": 1.0,
                "leg_[l,r]2_joint": 0.5,
                "leg_[l,r]3_joint": 0.5,
                "leg_[l,r]4_joint": 1.0,
                "leg_[l,r]5_joint": 0.2,
                "leg_[l,r]6_joint": 0.2,
                "waist_yaw_joint": 0.2,
                "zarm_[l,r]1_joint": 0.5,
                "zarm_[l,r]2_joint": 0.3,
                "zarm_[l,r]3_joint": 0.2,
                "zarm_[l,r]4_joint": 0.3,
                "zarm_[l,r]5_joint": 0.1,
                "zarm_[l,r]6_joint": 0.1,
                "zarm_[l,r]7_joint": 0.1
            },
            activation_vel=0.1,
            friction_dynamic=0,
        ),
    },
)

# S52_ACTION_SCALE = {}
# for a in KUAVO_S52_CFG.actuators.values():
#     # e = a.effort_limit_sim
#     # s = a.stiffness
#     names = a.joint_names_expr
#     # if not isinstance(e, dict):
#     #     e = {n: e for n in names}
#     # if not isinstance(s, dict):
#     #     s = {n: s for n in names}
#     for n in names:
#         # if n in e and n in s and s[n]:
#             S52_ACTION_SCALE[n] = 0.25

S52_ACTION_SCALE = {}
for a in KUAVO_S52_CFG.actuators.values():
    e_cfg = a.effort_limit_sim
    s_cfg = a.stiffness
    name_patterns = a.joint_names_expr

    if not name_patterns:
        continue

    # 展开：用正则从已知关节名列表中过滤出属于该执行器的实际关节名
    candidate_joint_names = []
    joint_names_list = KUAVO_S52_PRESERVE_JOINT_ORDER_ASSET_CFG.joint_names or []
    for jn in joint_names_list:
        for pat in name_patterns:
            if re.fullmatch(pat, jn):
                candidate_joint_names.append(jn)
                break

    def _resolve_value(joint_name, cfg_value):
        # cfg_value 可能是标量或者以正则为键的 dict
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
        if e_val is None or s_val in (None, 0):
            continue
        S52_ACTION_SCALE[joint_name] = 0.25
