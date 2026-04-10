from isaaclab.utils import configclass

from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.robots.roban_s2_2 import RobanS2_2_ACTION_SCALE, RobanS2_2_CYLINDER_CFG
from whole_body_tracking.tasks.roban_tracking.config.roban_s2_2.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.roban_tracking.tracking_env_cfg import TrackingEnvCfg


from whole_body_tracking.robots.roban_s2_2 import PRESERVE_JOINT_ORDER_ASSET_CFG
@configclass
class RobanS2_2FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.joint_names = PRESERVE_JOINT_ORDER_ASSET_CFG.joint_names

        self.observations.policy.joint_pos.params = {"asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG}
        self.observations.policy.joint_vel.params = {"asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG}
        self.observations.critic.joint_pos.params = {"asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG}
        self.observations.critic.joint_vel.params = {"asset_cfg": PRESERVE_JOINT_ORDER_ASSET_CFG}

        self.scene.robot = RobanS2_2_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = RobanS2_2_ACTION_SCALE
        # Migration note:
        # In s17 URDF, torso/pelvis naming swaps compared with s14.
        # Use waist_yaw_link as the motion anchor to preserve previous "torso anchor" semantics.
        self.commands.motion.anchor_body = "base_link"
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

        # Override event asset_cfg to use s2.2 joint order
        self.events.add_joint_default_pos.params["asset_cfg"] = PRESERVE_JOINT_ORDER_ASSET_CFG



@configclass
class RobanS2_2FlatWoStateEstimationEnvCfg(RobanS2_2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class RobanS2_2FlatLowFreqEnvCfg(RobanS2_2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
