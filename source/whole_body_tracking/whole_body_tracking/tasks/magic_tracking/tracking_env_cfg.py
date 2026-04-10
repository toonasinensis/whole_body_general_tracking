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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.magic_tracking.mdp as mdp
from whole_body_tracking.robots.roban_s22 import PRESERVE_JOINT_ORDER_ASSET_CFG

##
# Scene definition
##
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

VELOCITY_SMALL_RANGE = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (-0.1, 0.1),
    "roll": (-0.26, 0.26),
    "pitch": (-0.26, 0.26),
    "yaw": (-0.39, 0.39),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            restitution=1.0,
            static_friction=1.0,
            dynamic_friction=1.0,
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
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=40.0, debug_vis=True
    )


##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionResidualsActionCfg(
        asset_name="robot",
        joint_names=".*",
        preserve_order=True,
        scale=0.5,
        use_default_offset=True
    )
    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=[".*"],
    #     preserve_order=True,
    #     use_default_offset=True
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_target_pos = ObsTerm(func=mdp.motion_target_pos, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.6, n_max=0.6))
        actions = ObsTerm(func=mdp.last_action)        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_target_pos = ObsTerm(func=mdp.motion_target_pos, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


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
            "asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG,
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
        params={"velocity_range":VELOCITY_SMALL_RANGE},
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": {
                "x": (-600.0, 600.0),
                "y": (-600.0, 600.0),
                "z": (-350.0, 350.0),
            },  # force = mass * dv / dt
            "torque_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
            "probability": 0.001,  # Expect step = 1 / probability
        },
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
    
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-5.0,
        params={"soft_ratio": 1, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!leg_l6_link$)(?!leg_r6_link$).+$"],
            ),
            "threshold": 1.0,
        },
    )

    motion_hand_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.04, "body_names": ["zarm_l4_link","zarm_r4_link"]},
    )

    motion_hand_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 1.0, "body_names": ["zarm_l4_link","zarm_r4_link"]},
    )
    
    # L4和R4位置跟踪（最重要的奖励项）
    motion_knee_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.8,
        params={"command_name": "motion", "std": 0.04, "body_names": ["leg_l4_link","leg_r4_link"]},
    )

    motion_knee_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.2,
        params={"command_name": "motion", "std": 1.0, "body_names": ["leg_l4_link","leg_r4_link"]},
    )

    motion_knee_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.8,
        params={"command_name": "motion", "std": 0.4, "body_names": ["leg_l4_link","leg_r4_link"]},
    )

    motion_knee_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.2,
        params={"command_name": "motion", "std": 3.14, "body_names": ["leg_l4_link","leg_r4_link"]},
    )

    motion_feet_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 1.0, "body_names": ["leg_l6_link","leg_r6_link"]},
    )
    
    feet_contact_forces = RewTerm(
        func=mdp.feet_contact_forces,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["leg_l6_link", "leg_r6_link"],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["leg_l6_link", "leg_r6_link"],
            ),
            "force_velocity_threshold": 100.0,  # 接触力阈值（N），超过此值开始惩罚
        },
    )

    feet_slide_vel = RewTerm(
        func=mdp.body_slide_vel,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_[l,r]6_link"]),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg_[l,r]6_link"]),
            "contact_threshold": 50.0
        },
    )

    # 屏蔽脚部接触惩罚（当该抬腿时）
    # penalize_feet_contact_when_should_lift = RewTerm(
    #     func=mdp.penalize_feet_contact_when_should_lift,
    #     weight=-2.0,  # 惩罚权重，从-2.0增加到-3.0以增强惩罚效果
    #     params={
    #         "command_name": "motion",
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["leg_l6_link", "leg_r6_link"]
    #         ),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["leg_l6_link", "leg_r6_link"]
    #         ),
    #         "body_names": ["leg_l6_link", "leg_r6_link"],  # 监控的脚部
    #         "lift_height_threshold": 0.035,  # 动捕数据中脚高度超过5cm认为应该离地
    #         "contact_threshold": 10.0,  # 接触力阈值（N）
    #         "penalty_scale": 1.0,  # 惩罚缩放因子，从1.0增加到1.5以增强惩罚
    #     },
    # )
    
    # 屏蔽保持脚底平整的奖励
    # 奖励脚底保持水平（平坦），但权重较低，因为位置跟踪更重要
    # 参数分析：
    # - weight=1.0: 降低权重，确保位置跟踪优先（motion_feet_pos权重为5.0）
    # - std=0.3: 约17度，对于脚底倾斜角度来说稍微宽松但合理
    #   考虑到跳舞时脚底可能不是完全平的，0.3是一个合理的起点
    #   如果希望更严格，可以减小到0.2（约11度）
    # - only_when_contact=True: 只在接触时给奖励，避免在摆动阶段干扰
    # feet_flat_reward = RewTerm(
    #     func=mdp.feet_flat_orientation_reward,
    #     weight=0.5,  # 降低权重，位置跟踪更重要
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["leg_l6_link", "leg_r6_link"]  # 左右脚底
    #         ),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["leg_l6_link", "leg_r6_link"]  # 左右脚底
    #         ),
    #         "contact_threshold": 10.0,  # 判断是否接触地面的力阈值（N）
    #         "std": 0.3,  # 标准差（弧度，约17度），倾斜超过此值奖励快速衰减
    #         "only_when_contact": True,  # 只在脚接触地面时给奖励
    #     },
    # )
    
    # 重心控制在落脚点的奖励函数
    # com_balance_reward = RewTerm(
    #     func=mdp.com_balance_when_stand,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["leg_l6_link", "leg_r6_link"]  # 左右脚底
    #         ),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["leg_l6_link", "leg_r6_link"]  # 左右脚底
    #         ),
    #         "std": 0.05,  # 标准差（米），质心偏移超过此值奖励快速衰减
    #     },
    # )
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True
    )

    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )

    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "leg_l6_link",
                "leg_r6_link",
                "zarm_l4_link",
                "zarm_r4_link",
            ],
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    update_push_with_entropy_and_eposidelength = CurrTerm(
        func=mdp.update_push_with_entropy_and_eposidelength,
        params={
            "entropy_threshold": 0.9,
            "target_episode_length": 480.0,
        },
    )


##
# Environment configuration
##
@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

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
