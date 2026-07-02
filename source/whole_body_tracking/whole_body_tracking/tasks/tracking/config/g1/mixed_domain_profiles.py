from __future__ import annotations

import math

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking import mdp
from whole_body_tracking.tasks.tracking.config.domain_profile_applier import ProfileObsRouteCfg


@configclass
class EmptyProfileCommandsCfg:
    pass


@configclass
class EmptyProfileResetCfg:
    pass


@configclass
class G1MixedWbcRewardsCfg:
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3, "disable_on_delayed_termination": False},
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
        params={"command_name": "motion", "std": 0.3, "disable_on_delayed_termination": False},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4, "disable_on_delayed_termination": False},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0, "disable_on_delayed_termination": False},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14, "disable_on_delayed_termination": False},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "ratio": 5.0,
            "force_threshold": 1.0,
        },
    )


@configclass
class G1MixedWbcTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.250},
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
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
            "disable_on_delayed_termination_envs": False,
        },
    )
    anchor_pos_xy = DoneTerm(
        func=mdp.bad_anchor_pos_xyz,
        params={"command_name": "motion", "threshold": 0.4},
    )

    parkour_reach = DoneTerm(
        func=mdp.parkour_reach_timeout,
        time_out=True,
        params={
            "command_name": "motion",
            "distance_threshold": 0.10,
            "end_margin_steps": 3,
        },
    )


@configclass
class G1MixedWbcTrackingProfileCfg:
    commands_cfg = EmptyProfileCommandsCfg()
    reset_cfg = EmptyProfileResetCfg()
    rewards_cfg = G1MixedWbcRewardsCfg()
    terminations_cfg = G1MixedWbcTerminationsCfg()
    obs_route_cfg = ProfileObsRouteCfg(velcommand_source="motion", aux_mask_enabled=True, amp_mask_enabled=True)


@configclass
class G1FlatVelocityCommandsCfg:
    base_velocity = velocity_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=velocity_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.50, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class G1MixedVelocityResetCfg:
    reset_base = EventTerm(
        func=velocity_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.05, 0.1), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class G1MixedVelocityRewardsCfg:
    termination_penalty = RewTerm(func=velocity_mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=velocity_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.50,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=velocity_mdp.track_ang_vel_z_world_exp,
        weight=1.50,
        params={"command_name": "base_velocity", "std": 0.7},
    )
    track_root_height = RewTerm(
        func=mdp.root_height_recovery_exp,
        weight=1.0,
        params={"std": 0.3, "active_only_on_delayed_termination": False},
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    )
    pose = RewTerm(
        func=mdp.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
            "command_name": "base_velocity",
            "std_standing": {".*": 0.05},
            "std_walking": {
                r".*hip_pitch.*": 0.5,
                r".*hip_roll.*": 0.15,
                r".*hip_yaw.*": 0.15,
                r".*knee.*": 0.5,
                r".*ankle_pitch.*": 0.15,
                r".*ankle_roll.*": 0.1,
                r".*waist_.*": 0.1,
                r".*shoulder_.*": 0.1,
                r".*elbow.*": 0.1,
                r".*wrist.*": 0.1,
            },
            "std_running": {
                r".*hip_pitch.*": 0.5,
                r".*hip_roll.*": 0.25,
                r".*hip_yaw.*": 0.25,
                r".*knee.*": 0.5,
                r".*ankle_pitch.*": 0.25,
                r".*ankle_roll.*": 0.1,
                r".*waist_.*": 0.1,
                r".*shoulder_.*": 0.15,
                r".*elbow.*": 0.1,
                r".*wrist.*": 0.1,
            },
            "walking_threshold": 0.1,
            "running_threshold": 1.5,
        },
    )
    lin_vel_z_l2 = RewTerm(func=velocity_mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=velocity_mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=velocity_mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(
        func=velocity_mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    action_rate_l2 = RewTerm(func=velocity_mdp.action_rate_l2, weight=-0.005)
    feet_air_time = RewTerm(
        func=velocity_mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=velocity_mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    dof_pos_limits = RewTerm(
        func=velocity_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=velocity_mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )


@configclass
class G1MixedVelocityTerminationsCfg:
    time_out = DoneTerm(func=velocity_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=velocity_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    low_root_height = DoneTerm(
        func=mdp.root_height_below_desired,
        params={"asset_cfg": SceneEntityCfg("robot"), "margin": 0.3},
    )


@configclass
class G1MixedVelocityObsRouteCfg(ProfileObsRouteCfg):
    velcommand_source: str = "base_velocity"
    aux_mask_enabled: bool = False
    amp_mask_enabled: bool = True


@configclass
class G1MixedVelocityFlatProfileCfg:
    commands_cfg = G1FlatVelocityCommandsCfg()
    reset_cfg = G1MixedVelocityResetCfg()
    rewards_cfg = G1MixedVelocityRewardsCfg()
    terminations_cfg = G1MixedVelocityTerminationsCfg()
    obs_route_cfg = G1MixedVelocityObsRouteCfg()


@configclass
class G1ParkourVelocityCommandsCfg:
    base_velocity = velocity_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=velocity_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.50, 2.0),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class G1MixedVelocityTerrainProfileCfg:
    commands_cfg = G1ParkourVelocityCommandsCfg()
    reset_cfg = G1MixedVelocityResetCfg()
    rewards_cfg = G1MixedVelocityRewardsCfg()
    terminations_cfg = G1MixedVelocityTerminationsCfg()
    obs_route_cfg = G1MixedVelocityObsRouteCfg()


def build_default_mixed_profiles() -> dict[str, object]:
    return {
        "wbc_tracking": G1MixedWbcTrackingProfileCfg(),
        "velocity_flat": G1MixedVelocityFlatProfileCfg(),
        "velocity_terrain": G1MixedVelocityTerrainProfileCfg(),
    }
