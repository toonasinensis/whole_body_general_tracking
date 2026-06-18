from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.roban_s22 import RobanS22_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import ObservationsCfg, TrackingEnvCfg

ROBAN_S22_MOTION_ANCHOR_BODY_NAME = "base_link"
ROBAN_S22_MOTION_BODY_NAMES = [
    "base_link",
    "waist_yaw_link",
    "leg_l2_link",
    "leg_l4_link",
    "leg_l6_link",
    "leg_r2_link",
    "leg_r4_link",
    "leg_r6_link",
    "zarm_l2_link",
    "zarm_l4_link",
    "zarm_l5_link",
    "zarm_r2_link",
    "zarm_r4_link",
    "zarm_r5_link",
]


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
            "pos_distribution_params": (-0.03, 0.03),
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
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    add_waist_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    link_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[lr][1-6]_link|zarm_[lr][1-4]_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )

    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[lr][1-6]_link|zarm_[lr][1-4]_link"),
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
        weight=0.5,
        params={"command_name": "motion", "std": 0.3, "disable_on_delayed_termination": True},
    )

    motion_global_torso_pos_z = RewTerm(
        func=mdp.motion_global_torso_position_z_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3, "enable_on_delayed_termination": True},
    )
    
    motion_global_anchor_pos_z = RewTerm(
        func=mdp.motion_global_anchor_position_z_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )

    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3, "disable_on_delayed_termination": True},
    )

    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4, "disable_on_delayed_termination": True},
    )

    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0, "disable_on_delayed_termination": True},
    )

    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14, "disable_on_delayed_termination": True},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!leg_l6_link$)(?!leg_r6_link$)(?!zarm_l4_link$)(?!zarm_r4_link$).+$"],
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
            # NOTE use zarm_[l,r]5_link as the termination ee bodies ?
            "body_names": [
                "leg_l6_link",
                "leg_r6_link",
                "zarm_l5_link",
                "zarm_r5_link",
            ],
            "disable_on_delayed_termination_envs": True,
        },
    )





@configclass
class RobanS22FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()
        self.rewards = RewardsCfg()
        self.terminations = TerminationsCfg()
        self.scene.robot = RobanS22_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.25
        # Migration note:
        # In s17 URDF, torso/pelvis naming swaps compared with s14.
        # Use waist_yaw_link as the motion anchor to preserve previous "torso anchor" semantics.
        self.commands.motion.anchor_body_name = ROBAN_S22_MOTION_ANCHOR_BODY_NAME
        self.commands.motion.body_names = list(ROBAN_S22_MOTION_BODY_NAMES)
        self.commands.motion.smpl_file_path = None
        self.commands.motion.motion_sampling_start_frame = 5
        self.commands.motion.adaptive_sample_rewind_min_bins = 0
        self.commands.motion.adaptive_sample_rewind_bins = 0
        self.commands.motion.pose_range_init_mode = "lying"
        self.commands.motion.pose_range_lying_height_range = (0.25, 0.45)
        self.observations.smpl_cmd_mf = None
        self.events.delayed_termination = EventTerm(
            func=mdp.install_delayed_termination,
            mode="startup",
            params={"delay_reset_env_ratio": 1.0, "max_delay_steps": 250},
        )



@configclass
class RobanS22FlatAMPEnvCfg(RobanS22FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = ObservationsCfg.AmpCfg()
        for term in (
            self.observations.amp.body_pos_b,
            self.observations.amp.body_ori_b,
            self.observations.amp.body_lin_vel_b,
            self.observations.amp.body_ang_vel_b,
        ):
            term.params["asset_name"] = "robot"
            term.params["anchor_body_name"] = ROBAN_S22_MOTION_ANCHOR_BODY_NAME
            term.params["body_names"] = tuple(ROBAN_S22_MOTION_BODY_NAMES)
