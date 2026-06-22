from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking import mdp
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1_AMP_ANCHOR_BODY_NAME
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from whole_body_tracking.terrains.mixed_domains import build_mixed_terrain_cfg


@configclass
class G1MixedFlatMeshEnvCfg(TrackingEnvCfg):
    pairs_jsonl: str = "data/omniretarget/g1_terrain/pairs.jsonl"
    flat_motion_file: str = "data"
    flat_dataset_txt: str = "data/tracking_npz_data/lafan_named.txt"
    flat_env_ratio: float = 0.25
    pair_limit: int | None = None
    terrain_size: tuple[float, float] = (16.0, 16.0)
    terrain_ground_thickness: float = 0.01
    terrain_ground_z: float = 0.0
    domain_separator_cell_count: int = 32
    strict_pairing: bool = True

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.configure_domains()
        self.actions.joint_pos.scale = G1_ACTION_SCALE

    def make_domain_cfgs(self):
        flat_ratio = max(0.0, min(1.0, float(self.flat_env_ratio)))
        flat_cells = max(1, int(round(float(self.scene.num_envs) * flat_ratio)))
        return [
            mdp.FlatMotionDomainCfg(
                name="flat_lafan",
                env_ratio=flat_ratio,
                motion_file=self.flat_motion_file,
                dataset_txt=self.flat_dataset_txt,
                terrain_cell_count=flat_cells,
            ),
            mdp.PairedMeshMotionDomainCfg(
                name="omniretarget_g1_terrain",
                env_ratio=max(0.0, 1.0 - flat_ratio),
                pairs_jsonl=self.pairs_jsonl,
                pair_limit=self.pair_limit,
            ),
        ]

    def configure_domains(self):
        domains = self.make_domain_cfgs()
        terrain_generator, _ = build_mixed_terrain_cfg(
            domains,
            terrain_size=self.terrain_size,
            ground_thickness=self.terrain_ground_thickness,
            ground_z=self.terrain_ground_z,
            domain_separator_cell_count=self.domain_separator_cell_count,
        )

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_generator,
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                project_uvw=True,
            ),
        )
        self.commands.motion = mdp.DomainMotionCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            max_motion_num=-1,
            motion_file=self.flat_motion_file,
            dataset_txt=None,
            smpl_file_path=None,
            domains=domains,
            domain_separator_cell_count=self.domain_separator_cell_count,
            anchor_body_name=G1_AMP_ANCHOR_BODY_NAME,
            body_names=[
                "pelvis",
                "left_hip_roll_link",
                "left_knee_link",
                "left_ankle_roll_link",
                "right_hip_roll_link",
                "right_knee_link",
                "right_ankle_roll_link",
                "torso_link",
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_yaw_link",
            ],
            pose_range={
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.1, 0.2),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            pose_range_env_ratio=1.0,
            pose_range_init_mode="range",
            velocity_range={
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
            joint_position_range=(-0.0, 0.0),
            motion_sampling_start_frame=5,
            adaptive_sample_rewind_min_bins=1,
            adaptive_sample_rewind_bins=2,
            debug_domain_log_count=16,
        )
