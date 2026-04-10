""" This file defines the MDP """

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.robots.roban_s22 import PRESERVE_JOINT_ORDER_ASSET_CFG

import whole_body_tracking.tasks.fsqtrack.mdp as mdp


##
## Scene definition
##


VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain (flat plane for Sonic training)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=40.0,
        debug_vis=True,
    )


@configclass
class CommandsCfg:
    """Command specifications for the FSQ-Track MDP."""
    # SON uses multi-motion command with motion bins (MultiMotionCommand).
    motion = mdp.MultiMotionCommandCfg(
        asset_name="robot",
        # Dataset configuration (can be overridden in task-specific configs or via CLI)
        # Body configuration
        anchor_body_name="base_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.005, 0.005),
            "y": (-0.005, 0.005),
            "z": (-0.001, 0.001),
            "roll": (-0.01, 0.01),
            "pitch": (-0.01, 0.01),
            "yaw": (-0.02, 0.02),
        },
        velocity_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.02, 0.02),
            "roll": (-0.052, 0.052),
            "pitch": (-0.052, 0.052),
            "yaw": (-0.078, 0.078),
        },
        joint_position_range=(-0.1, 0.1),
        adaptive_uniform_ratio=0.1,
        adaptive_cap=5,
        adaptive_alpha=0.001,
    )


@configclass
class ActionsCfg:
    """Action specifications for the FSQ-Track MDP."""

    joint_pos = mdp.JointPositionResidualsActionCfg(
        asset_name="robot",
        joint_names=".*",
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the FSQ-Track MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group (actor inputs)."""

        # Command window (multi-frame joint pos/vel from MultiMotionCommand)
        command = ObsTerm(func=mdp.sonic_commands, params={"command_name": "motion"})
        # Proprioception
        projected_gravity = ObsTerm(func=mdp.sonic_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.sonic_base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.sonic_joint_pos, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.sonic_joint_vel, noise=Unoise(n_min=-0.6, n_max=0.6))
        actions = ObsTerm(func=mdp.sonic_last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Observations for critic group (privileged inputs)."""

        # Command window (multi-frame joint pos/vel)
        command = ObsTerm(func=mdp.sonic_commands, params={"command_name": "motion"})
        # Proprioception
        projected_gravity = ObsTerm(func=mdp.sonic_projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.sonic_base_ang_vel)
        joint_pos = ObsTerm(func=mdp.sonic_joint_pos)
        joint_vel = ObsTerm(func=mdp.sonic_joint_vel)
        actions = ObsTerm(func=mdp.sonic_last_action)
        # Relative body pose wrt anchor
        body_pos_b = ObsTerm(func=mdp.sonic_body_pos_b, params={"command_name": "motion"})
        body_ori_b = ObsTerm(func=mdp.sonic_body_ori_b, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    """Reward terms for the FSQ-Track motion-tracking MDP."""

    # motion_global_anchor_pos = RewTerm(
    #     func=mdp.motion_global_anchor_position_error_exp,
    #     weight=0.3,
    #     params={"command_name": "motion", "std": 0.3},
    # )  # Optional, not used by default.

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

    # Action & joint penalties
    action_rate_l2 = RewTerm(func=mdp.sonic_action_rate_l2, weight=-1e-1)
    # joint limits / vel limits / undesired contacts (reuse generic MDP utilities)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-5.0,
        params={"soft_ratio": 1, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!leg_l6_link$)(?!leg_r6_link$).+$"],
            ),
            "threshold": 1.0,
        },
    )

    #######################################
    # reward functions for kuavo_s52 task #
    #######################################
    motion_knee_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.8,
        params={"command_name": "motion", "std": 0.04, "body_names": ["leg_l4_link", "leg_r4_link"]},
    )
    motion_knee_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.2,
        params={"command_name": "motion", "std": 1.0, "body_names": ["leg_l4_link", "leg_r4_link"]},
    )
    motion_feet_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 1.0, "body_names": ["leg_l6_link", "leg_r6_link"]},
    )
    motion_feet_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.2,
        params={"command_name": "motion", "std": 3.14, "body_names": ["leg_l6_link", "leg_r6_link"]},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the FSQ-Track MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )


@configclass
class EventCfg:
    """Configuration for events (domain randomization, disturbances)."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.3),
            "dynamic_friction_range": (0.3, 1.3),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG,
            "pos_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )

    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.08, 0.08), "z": (-0.08, 0.08)},
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.04, 0.04)},
        },
    )

    add_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "mass_distribution_params": (0.8, 1.5),
            "operation": "scale",
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

    link_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[lr][1-6]_link|zarm_[lr][1-7]_link"),
            "com_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (-0.04, 0.04)},
        },
    )

    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[lr][1-6]_link|zarm_[lr][1-7]_link"),
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
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "force_range": {
                "x": (-1200.0, 1200.0),
                "y": (-1200.0, 1200.0),
                "z": (-700.0, 700.0),
            },  # force = mass * dv / dt
            "torque_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
            "probability": 0.001,  # Expect step = 1 / probability
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the FSQ-Track MDP (unused for now)."""
    pass


#############################
# Environment configuration #
#############################
@configclass
class FsqtrackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Roban multi-motion tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
