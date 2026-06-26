from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.body_presets import (
    G1_AMP_ANCHOR_BODY_NAME,
    G1_AMP_BODY_NAMES,
    G1_TRACK_ANCHOR_BODY_NAME,
    G1_TRACK_BODY_NAMES,
)
from whole_body_tracking.tasks.tracking import mdp
from whole_body_tracking.tasks.tracking.mdp.tracking_env_cfg import ObservationsCfg, TrackingEnvCfg


###
# EnvCfg
###
@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = G1_TRACK_ANCHOR_BODY_NAME
        self.commands.motion.body_names = G1_TRACK_BODY_NAMES
        self.commands.motion.motion_sampling_start_frame = 5
        self.commands.motion.adaptive_sample_rewind_min_bins = 0
        self.commands.motion.adaptive_sample_rewind_bins = 1
        self.commands.motion.envs_classes_ratio = {"lying": 0.0, "range": 1.0}
        self.commands.motion.pose_range_lying_height_range = (0.25, 0.45)
        # replace Termination Manager
        self.events.delayed_termination = EventTerm(
            func=mdp.install_delayed_termination,
            mode="startup",
            params={"delay_reset_env_ratio": 1.0, "max_delay_steps": 250},
        )
        
        # # Assist the pose-range recovery curriculum early on, then fade with the global assisted timeout rate.
        # self.events.fallen_upward_assist = EventTerm(
        #     func=mdp.assist_fallen_robots_with_upward_force,
        #     mode="interval",
        #     interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        #         "command_name": "motion",
        #         "force": 250.0,
        #         "force_mode": "permanent",
        #         "min_force_scale": 0.0,
        #         "z_error_threshold": 0.15,
        #         "max_height_above_reference": 0.05,
        #         "max_upward_velocity": 1.0,
        #         "gravity_z_threshold": 0.8,
        #         "debug_steps": 300,
        #         "debug_interval_steps": 20,
        #         "debug_env_id": 0,
        #     },
        # )


@configclass
class G1FlatAMPEnvCfg(G1FlatEnvCfg):
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
            term.params["anchor_body_name"] = G1_AMP_ANCHOR_BODY_NAME
            term.params["body_names"] = tuple(G1_AMP_BODY_NAMES)

        self.commands.motion.envs_classes_ratio = {"lying": 0.3, "range": 0.7}

        # replace Termination Manager
        self.events.delayed_termination = EventTerm(
            func=mdp.install_delayed_termination,
            mode="startup",
            params={"delay_reset_env_ratio": 1.0, "max_delay_steps": 250},
        )
        
        # Assist the pose-range recovery curriculum early on, then fade with the global assisted timeout rate.
        self.events.fallen_upward_assist = EventTerm(
            func=mdp.assist_fallen_robots_with_upward_force,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "command_name": "motion",
                "force": 250.0,
                "force_mode": "permanent",
                "min_force_scale": 0.0,
                "z_error_threshold": 0.15,
                "max_height_above_reference": 0.05,
                "max_upward_velocity": 1.0,
                "gravity_z_threshold": 0.8,
                "debug_steps": 300,
                "debug_interval_steps": 20,
                "debug_env_id": 0,
            },
        )
