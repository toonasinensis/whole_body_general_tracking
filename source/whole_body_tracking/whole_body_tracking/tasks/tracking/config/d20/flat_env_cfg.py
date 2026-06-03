from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking import mdp
from whole_body_tracking.robots.d20_v2 import D20_V2_ACTION_SCALE, D20_V2_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

D20_V2_ANCHOR_BODY_NAME = "base_link"
D20_V2_BODY_NAMES = [
    "base_link",
    "waist_pitch_link",
    "left_hip_roll_link",
    "left_knee_pitch_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_pitch_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_wrist_roll_link",
]
D20_END_EFFECTOR_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
]
D20_UNDESIRED_CONTACT_BODY_PATTERN = (
    r"^(?!(?:left_ankle_roll_link|right_ankle_roll_link|"
    r"left_knee_pitch_link|right_knee_pitch_link|"
    r"left_elbow_pitch_link|right_elbow_pitch_link|"
    r"left_hand_link|right_hand_link|"
    r"left_hip_yaw_link|right_hip_yaw_link)$).+$"
)
VELOCITY_SMALL_RANGE = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (-0.1, 0.1),
    "roll": (-0.26, 0.26),
    "pitch": (-0.26, 0.26),
    "yaw": (-0.39, 0.39),
}


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.3),
            "dynamic_friction_range": (0.3, 1.3),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    waist_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.8, 1.5),
            "operation": "scale",
        },
    )

    add_waist_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link"),
            "mass_distribution_params": (0.8, 1.5),
            "operation": "scale",
        },
    )

    link_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="(left|right)_(hip_(pitch|roll|yaw)|knee_(aux|pitch)|ankle_(pitch|roll)|shoulder_(pitch|roll|yaw)|elbow_pitch|wrist_roll)_link",
            ),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )

    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="(left|right)_(hip_(pitch|roll|yaw)|knee_(aux|pitch)|ankle_(pitch|roll)|shoulder_(pitch|roll|yaw)|elbow_pitch|wrist_roll)_link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1, 3),
        params={"velocity_range": VELOCITY_SMALL_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.3,
        params={"command_name": "motion", "std": 0.3},
    )

    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.3,
        params={"command_name": "motion", "std": 0.4},
    )

    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )

    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )

    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 1.0},
    )

    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 3.14},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    # joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5e-3)

    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.010,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[D20_UNDESIRED_CONTACT_BODY_PATTERN],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.4},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 1.0},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.4,
            "body_names": D20_END_EFFECTOR_BODY_NAMES,
        },
    )


@configclass
class D20V2FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()
        self.rewards = RewardsCfg()
        self.terminations = TerminationsCfg()
        self.scene.robot = D20_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = D20_V2_ACTION_SCALE
        self.commands.motion.anchor_body_name = D20_V2_ANCHOR_BODY_NAME
        self.commands.motion.body_names = list(D20_V2_BODY_NAMES)
