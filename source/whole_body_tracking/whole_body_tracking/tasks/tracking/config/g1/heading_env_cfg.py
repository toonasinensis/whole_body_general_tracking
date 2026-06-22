from __future__ import annotations

import math
from copy import deepcopy

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking import mdp
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1_AMP_ANCHOR_BODY_NAME, G1_AMP_BODY_NAMES
from whole_body_tracking.tasks.tracking.tracking_env_cfg import ObservationsCfg, TrackingEnvCfg


@configclass
class G1HeadingCommandsCfg:
    heading = mdp.HeadingCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        debug_vis=True,
        target_speed=0.3,
        provider="motion_reset_velocity",
        facing_mode="same_as_heading",
        heading_range=(-0.25, 0.25),
        motion_reset_min_speed=0.2,
    )


@configclass
class G1HeadingObservationsCfg(ObservationsCfg):
    @configclass
    class HeadingPropCfg(ObservationsCfg.PropCfg):
        phase = ObsTerm(func=mdp.heading_phase, params={"period": 0.6, "command_name": "heading"})

        def __post_init__(self):
            self.history_length = 4
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HeadingCmdCfg(ObsGroup):
        heading_command = ObsTerm(
            func=mdp.heading_command,
            params={"command_name": "heading"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HeadingCriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    critic: HeadingCriticCfg = HeadingCriticCfg()
    heading_cmd: HeadingCmdCfg = HeadingCmdCfg()
    prop: HeadingPropCfg = HeadingPropCfg()
    amp: ObservationsCfg.AmpCfg | None = None
    rbt_cmd_mf: None = None
    zrbt_cmd_mf: None = None
    smpl_cmd_mf: None = None


@configclass
class G1HeadingRewardsCfg:
    heading = RewTerm(
        func=mdp.heading_reward,
        weight=0.5,
        params={
            "command_name": "heading",
            "asset_cfg": SceneEntityCfg("robot"),
            "target_speed": None,
            "alpha": 4.0,
            "velocity_weight": 0.7,
            "facing_weight": 0.3,
            "min_root_height": 0.70,
            "target_root_height": 0.76,
            "height_gate_std": 0.04,
            "upright_gate_std": 0.08,
            "disable_on_delayed_termination": True,
        },
    )
    heading_base_lin_vel = RewTerm(
        func=mdp.heading_base_linear_velocity_exp,
        weight=2.0,
        params={
            "command_name": "heading",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.5,
            "z_weight": 2.0,
            "target_speed": None,
            "min_root_height": 0.70,
            "target_root_height": 0.76,
            "height_gate_std": 0.05,
            "upright_gate_std": 0.10,
        },
    )
    root_height_recovery = RewTerm(
        func=mdp.root_height_recovery_exp,
        weight=12.0,
        params={
            "std": 0.08,
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.76,
            "active_only_on_delayed_termination": False,
            "active_scale": 1.0,
        },
    )
    low_root_height = RewTerm(
        func=mdp.root_height_below_target_l2,
        weight=-500.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.70,
        },
    )
    root_lin_vel_z = RewTerm(
        func=mdp.root_lin_vel_z_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    upright_orientation = RewTerm(
        func=mdp.upright_orientation_exp,
        weight=8.0,
        params={
            "std": 0.12,
            "asset_cfg": SceneEntityCfg("robot"),
            "disable_on_delayed_termination": True,
        },
    )
    upright_l2 = RewTerm(
        func=mdp.projected_gravity_xy_l2,
        weight=-24.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torso_upright_l2 = RewTerm(
        func=mdp.body_projected_gravity_xy_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    default_joint_pose = RewTerm(
        func=mdp.default_joint_position_exp,
        weight=0.15,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    r".*_hip_pitch_joint",
                    r".*_hip_roll_joint",
                    r".*_hip_yaw_joint",
                    r".*_knee_joint",
                    r".*_ankle_pitch_joint",
                    r".*_ankle_roll_joint",
                    r"waist_.*_joint",
                ],
            ),
            "disable_on_delayed_termination": True,
        },
    )
    variable_joint_posture = RewTerm(
        func=mdp.heading_variable_joint_posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
            "command_name": "heading",
            "walking_threshold": 0.05,
            "std_standing": {".*": 0.05},
            "std_walking": {
                r".*_hip_pitch_joint": 0.5,
                r".*_hip_roll_joint": 0.15,
                r".*_hip_yaw_joint": 0.15,
                r".*_knee_joint": 0.5,
                r".*_ankle_pitch_joint": 0.18,
                r".*_ankle_roll_joint": 0.1,
                r"waist_yaw_joint": 0.15,
                r"waist_roll_joint": 0.1,
                r"waist_pitch_joint": 0.1,
                r".*_shoulder_pitch_joint": 0.2,
                r".*_shoulder_roll_joint": 0.15,
                r".*_shoulder_yaw_joint": 0.15,
                r".*_elbow_joint": 0.2,
                r".*_wrist_roll_joint": 0.2,
                r".*_wrist_pitch_joint": 0.2,
                r".*_wrist_yaw_joint": 0.2,
            },
        },
    )
    body_xy_ang_vel_stability = RewTerm(
        func=mdp.body_xy_ang_vel_stability_exp,
        weight=2.0,
        params={
            "std": math.pi,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "disable_on_delayed_termination": True,
        },
    )
    foot_gait = RewTerm(
        func=mdp.heading_feet_gait,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "period": 0.6,
            "offset": [0.0, 0.5],
            "stance_threshold": 0.56,
            "force_threshold": 1.0,
            "command_name": "heading",
            "command_threshold": 0.05,
        },
    )
    foot_slip = RewTerm(
        func=mdp.heading_feet_slip,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "force_threshold": 1.0,
            "command_name": "heading",
            "command_threshold": 0.05,
        },
    )
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.height_filtered_undesired_contacts,
        weight=-0.02,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
            "max_body_height": 0.5,
        },
    )


@configclass
class G1HeadingTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0), "asset_cfg": SceneEntityCfg("robot")},
    )
    bad_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("robot")},
    )
    illegal_contact = DoneTerm(
        func=mdp.height_filtered_illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
            "max_body_height": 0.5,
        },
    )


@configclass
class G1HeadingWalkAMPEnvCfg(TrackingEnvCfg):
    commands: G1HeadingCommandsCfg = G1HeadingCommandsCfg()
    observations: G1HeadingObservationsCfg = G1HeadingObservationsCfg()
    rewards: G1HeadingRewardsCfg = G1HeadingRewardsCfg()
    terminations: G1HeadingTerminationsCfg = G1HeadingTerminationsCfg()
    reset_mode: str = "motion"
    motion_reset_dir: str = "data/g1_amp/LAFAN_WALK_STABLE"
    motion_reset_root_min_height: float = 0.70

    def __post_init__(self):
        super().__post_init__()

        robot_cfg = deepcopy(G1_CYLINDER_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Match amp_test's G1 KNEES_BENT_KEYFRAME and keep action offsets
        # synchronized to this deterministic reset posture.
        robot_cfg.init_state.pos = (0.0, 0.0, 0.76)
        robot_cfg.init_state.joint_pos.update(
            {
                ".*_hip_pitch_joint": -0.312,
                ".*_knee_joint": 0.669,
                ".*_ankle_pitch_joint": -0.363,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_yaw_joint": 0.0,
                ".*_ankle_roll_joint": 0.0,
            }
        )
        if robot_cfg.spawn is not None and robot_cfg.spawn.articulation_props is not None:
            robot_cfg.spawn.articulation_props.enabled_self_collisions = False
        self.scene.robot = robot_cfg
        self.scene.env_spacing = 3.0
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        # Start this standalone heading task from a deterministic, stable
        # posture; domain randomization can be re-enabled after it walks.
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None

        if self.reset_mode == "default":
            self.events.reset_base = EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.25, 0.25),
                        "y": (-0.25, 0.25),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    },
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            )
            self.events.reset_robot_joints = EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.0, 0.0),
                    "velocity_range": (-0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                },
            )
            self.events.reset_joint_targets = EventTerm(
                func=mdp.reset_default_joint_position_targets,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
                    "action_name": "joint_pos",
                },
            )
        elif self.reset_mode == "motion":
            self.events.init_motion_state_reset = EventTerm(
                func=mdp.init_motion_state_reset,
                mode="startup",
                params={
                    "motion_dir": self.motion_reset_dir,
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
                    "root_body_name": "pelvis",
                    "root_min_height": self.motion_reset_root_min_height,
                },
            )
            self.events.reset_from_motion_state = EventTerm(
                func=mdp.reset_from_motion_state,
                mode="reset",
                params={
                    "motion_dir": self.motion_reset_dir,
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
                    "root_body_name": "pelvis",
                    "root_min_height": self.motion_reset_root_min_height,
                },
            )
            self.events.delayed_termination = None
        else:
            raise ValueError(f"Unsupported HeadingWalkAMP reset_mode: {self.reset_mode!r}")

        self.observations.amp = ObservationsCfg.AmpCfg()
        for term in (
            self.observations.amp.body_pos_b,
            self.observations.amp.body_ori_b,
            self.observations.amp.body_lin_vel_b,
            self.observations.amp.body_ang_vel_b,
        ):
            term.params["asset_name"] = "robot"
            term.params["anchor_body_name"] = G1_AMP_ANCHOR_BODY_NAME
            term.params["body_names"] = tuple(G1_AMP_BODY_NAMES)


@configclass
class G1HeadingWalkAMPModalEnvCfg(G1HeadingWalkAMPEnvCfg):
    reset_mode: str = "default"

    def __post_init__(self):
        self.commands.heading.provider = "random"
        super().__post_init__()
