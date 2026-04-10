from isaaclab.utils import configclass

from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.robots.kuavo_s52 import KUAVO_S52_CFG , S52_ACTION_SCALE
from whole_body_tracking.tasks.tracking.config.kuavo_s52.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg_s52 import TrackingEnvCfgS52



from whole_body_tracking.robots.kuavo_s52 import KUAVO_S52_PRESERVE_JOINT_ORDER_ASSET_CFG
@configclass
class KuavoS52FlatEnvCfg(TrackingEnvCfgS52):
    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.joint_names = KUAVO_S52_PRESERVE_JOINT_ORDER_ASSET_CFG.joint_names

        self.observations.policy.joint_pos.params = {"asset_cfg": KUAVO_S52_PRESERVE_JOINT_ORDER_ASSET_CFG}
        self.observations.policy.joint_vel.params = {"asset_cfg": KUAVO_S52_PRESERVE_JOINT_ORDER_ASSET_CFG}
        self.observations.critic.joint_pos.params = {"asset_cfg": KUAVO_S52_PRESERVE_JOINT_ORDER_ASSET_CFG}
        self.observations.critic.joint_vel.params = {"asset_cfg": KUAVO_S52_PRESERVE_JOINT_ORDER_ASSET_CFG}

        self.scene.robot = KUAVO_S52_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = S52_ACTION_SCALE
        self.commands.motion.anchor_body = "base_link"
        self.commands.motion.body_names = [
            "base_link",

            "waist_yaw",

            "leg_l2_link",
            "leg_l4_link",
            "leg_l6_link",

            "leg_r2_link",
            "leg_r4_link",
            "leg_r6_link",

            "zarm_l2_link",
            "zarm_l4_link",
            "zarm_l7_link",

            "zarm_r2_link",
            "zarm_r4_link",
            "zarm_r7_link"
        ]


@configclass
class KuavoS52FlatWoStateEstimationEnvCfg(KuavoS52FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class KuavoS52FlatLowFreqEnvCfg(KuavoS52FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
