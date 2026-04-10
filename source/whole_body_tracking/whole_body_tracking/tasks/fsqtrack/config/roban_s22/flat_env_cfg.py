"""This file integrates the MDP process and the PPO algorithms"""

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.robots.roban_s22 import (
    PRESERVE_JOINT_ORDER_ASSET_CFG,
    RobanS22_ACTION_SCALE,
    RobanS22_CYLINDER_CFG,
)

from whole_body_tracking.tasks.fsqtrack.fsqtrack_env_cfg_roban import FsqtrackEnvCfg
from whole_body_tracking.tasks.fsqtrack.config.roban_s22.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE


@configclass
class RobanFlatEnvCfg(FsqtrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Ensure action/obs joint ordering matches Roban S2.2 URDF.
        self.actions.joint_pos.joint_names = PRESERVE_JOINT_ORDER_ASSET_CFG.joint_names

        self.observations.policy.joint_pos.params = {"asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG}
        self.observations.policy.joint_vel.params = {"asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG}
        self.observations.critic.joint_pos.params = {"asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG}
        self.observations.critic.joint_vel.params = {"asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG}

        self.scene.robot = RobanS22_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = RobanS22_ACTION_SCALE
        # MultiMotionCommand configuration overrides
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
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
            "zarm_r2_link",
            "zarm_r4_link",
        ]

        # Domain randomization configs in base FSQTrack env default to Kuavo naming.
        # Override the key Roban-specific ones here.
        self.events.add_joint_default_pos.params["asset_cfg"] = PRESERVE_JOINT_ORDER_ASSET_CFG
        self.events.torso_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_yaw_link")
        self.events.add_torso_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_yaw_link")
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_yaw_link")


@configclass
class RobanFlatWoStateEstimationEnvCfg(RobanFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class RobanFlatLowFreqEnvCfg(RobanFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
