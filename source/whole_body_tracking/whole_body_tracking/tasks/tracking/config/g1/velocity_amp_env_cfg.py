from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking import mdp as tracking_mdp
from whole_body_tracking.tasks.tracking.tracking_env_cfg import ObservationsCfg as TrackingObservationsCfg

G1_AMP_ANCHOR_BODY_NAME = "pelvis"
G1_AMP_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


@configclass
class G1VelocityFlatAMPSceneCfg(InteractiveSceneCfg):
    """Local copy of IsaacLab's velocity scene, with the project G1 plugged in."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class G1VelocityFlatAMPCommandsCfg:
    base_velocity = velocity_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=velocity_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class G1VelocityFlatAMPActionsCfg:
    joint_pos = velocity_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class G1VelocityFlatAMPObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=velocity_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=velocity_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=velocity_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=velocity_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=velocity_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=velocity_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=velocity_mdp.last_action)
        height_scan = ObsTerm(
            func=velocity_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PropCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.velocity_commands = None
            self.height_scan = None

    @configclass
    class BaseVelocityCommandCfg(ObsGroup):
        velocity_commands = ObsTerm(func=velocity_mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    prop: PropCfg = PropCfg()
    base_velocity_cmd: BaseVelocityCommandCfg = BaseVelocityCommandCfg()
    amp: TrackingObservationsCfg.AmpCfg | None = None


@configclass
class G1VelocityFlatAMPEventCfg:
    physics_material = EventTerm(
        func=velocity_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=velocity_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=velocity_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    base_external_force_torque = EventTerm(
        func=velocity_mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=velocity_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=velocity_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=velocity_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class G1VelocityFlatAMPRewardsCfg:
    termination_penalty = RewTerm(func=velocity_mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=velocity_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=velocity_mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_root_height = RewTerm(
        func=tracking_mdp.root_height_recovery_exp,
        weight=1.0,
        params={"std": 0.3, "active_only_on_delayed_termination": False},
    )
    body_orientation_l2 = RewTerm(
        func=tracking_mdp.body_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    )
    pose = RewTerm(
        func=tracking_mdp.variable_posture,
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
                r".*waist_yaw.*": 0.15,
                r".*waist_roll.*": 0.1,
                r".*waist_pitch.*": 0.1,
                r".*shoulder_pitch.*": 0.15,
                r".*shoulder_roll.*": 0.1,
                r".*shoulder_yaw.*": 0.1,
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
                r".*waist_yaw.*": 0.25,
                r".*waist_roll.*": 0.1,
                r".*waist_pitch.*": 0.1,
                r".*shoulder_pitch.*": 0.25,
                r".*shoulder_roll.*": 0.1,
                r".*shoulder_yaw.*": 0.1,
                r".*elbow.*": 0.1,
                r".*wrist.*": 0.1,
            },
            "walking_threshold": 0.1,
            "running_threshold": 1.5,
        },
    )
    lin_vel_z_l2 = RewTerm(func=velocity_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=velocity_mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=velocity_mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=velocity_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=velocity_mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=velocity_mdp.feet_air_time_positive_biped,
        weight=0.25,
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
    undesired_contacts = RewTerm(
        func=velocity_mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
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
    # joint_deviation_arms = RewTerm(
    #     func=velocity_mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_pitch_joint",
    #                 ".*_shoulder_roll_joint",
    #                 ".*_shoulder_yaw_joint",
    #                 ".*_elbow_joint",
    #                 ".*_wrist_.*",
    #             ],
    #         )
    #     },
    # )
    # joint_deviation_torso = RewTerm(
    #     func=velocity_mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot",
    #                                         joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])},
    # )


@configclass
class G1VelocityFlatAMPTerminationsCfg:
    time_out = DoneTerm(func=velocity_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=velocity_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )


@configclass
class G1VelocityFlatAMPCurriculumCfg:
    terrain_levels = CurrTerm(func=velocity_mdp.terrain_levels_vel)


@configclass
class G1VelocityFlatAMPModalEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1VelocityFlatAMPSceneCfg = G1VelocityFlatAMPSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1VelocityFlatAMPObservationsCfg = G1VelocityFlatAMPObservationsCfg()
    actions: G1VelocityFlatAMPActionsCfg = G1VelocityFlatAMPActionsCfg()
    commands: G1VelocityFlatAMPCommandsCfg = G1VelocityFlatAMPCommandsCfg()
    rewards: G1VelocityFlatAMPRewardsCfg = G1VelocityFlatAMPRewardsCfg()
    terminations: G1VelocityFlatAMPTerminationsCfg = G1VelocityFlatAMPTerminationsCfg()
    events: G1VelocityFlatAMPEventCfg = G1VelocityFlatAMPEventCfg()
    curriculum: G1VelocityFlatAMPCurriculumCfg = G1VelocityFlatAMPCurriculumCfg()

    def __post_init__(self):
        self._configure_common_velocity_env()
        self._configure_project_g1()
        self._configure_flat_velocity_overrides()
        self._configure_amp_observations()

    def _configure_common_velocity_env(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        elif self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

    def _configure_project_g1(self):
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"

        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        if hasattr(self.rewards, "flat_orientation_l2"):
            self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

    def _configure_flat_velocity_overrides(self):
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

    def _configure_amp_observations(self):
        self.observations.amp = TrackingObservationsCfg.AmpCfg()
        for term in (
            self.observations.amp.body_pos_b,
            self.observations.amp.body_ori_b,
            self.observations.amp.body_lin_vel_b,
            self.observations.amp.body_ang_vel_b,
        ):
            term.params["asset_name"] = "robot"
            term.params["anchor_body_name"] = G1_AMP_ANCHOR_BODY_NAME
            term.params["body_names"] = tuple(G1_AMP_BODY_NAMES)
