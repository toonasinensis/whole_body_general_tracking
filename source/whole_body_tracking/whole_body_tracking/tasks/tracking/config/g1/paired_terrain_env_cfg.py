from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking import mdp
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1_AMP_ANCHOR_BODY_NAME
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from whole_body_tracking.terrains.paired_manifest import (
    DEFAULT_OMNIRETARGET_G1_PAIRS_JSONL,
    TerrainMotionPairManifest,
    build_paired_terrain_cfg,
    resolve_repo_path,
)


@configclass
class TerrainHeightScanCfg(ObsGroup):
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class G1PairedTerrainEnvCfg(TrackingEnvCfg):
    pairs_jsonl: str = DEFAULT_OMNIRETARGET_G1_PAIRS_JSONL
    terrain_size: tuple[float, float] = (6.0, 6.0)
    terrain_ground_thickness: float = 0.01
    terrain_ground_z: float = 0.0
    strict_pairing: bool = True
    pair_limit: int | None = None

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.configure_pairs()
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.80, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.observations.terrain = TerrainHeightScanCfg()
        self.rewards.motion_global_anchor_pos.params["disable_on_delayed_termination"] = False
        self.rewards.motion_body_pos.params["disable_on_delayed_termination"] = False
        self.rewards.motion_body_ori.params["disable_on_delayed_termination"] = False
        self.rewards.motion_body_lin_vel.params["disable_on_delayed_termination"] = False
        self.rewards.motion_body_ang_vel.params["disable_on_delayed_termination"] = False
        self.terminations.ee_body_pos.params["disable_on_delayed_termination_envs"] = False

        self.actions.joint_pos.scale = G1_ACTION_SCALE

    def configure_pairs(self):
        manifest = TerrainMotionPairManifest.from_jsonl(self.pairs_jsonl, limit=self.pair_limit)
        num_cols = int(math.ceil(math.sqrt(manifest.count)))
        num_rows = int(math.ceil(manifest.count / num_cols))
        terrain_generator = build_paired_terrain_cfg(
            self.pairs_jsonl,
            num_rows=num_rows,
            num_cols=num_cols,
            terrain_size=self.terrain_size,
            add_ground_plane=True,
            ground_thickness=self.terrain_ground_thickness,
            ground_z=self.terrain_ground_z,
            limit=self.pair_limit,
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
        self.commands.motion = mdp.PairedTerrainMotionCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            max_motion_num=-1,
            motion_file=str(manifest_motion_root(manifest)),
            dataset_txt=str(manifest_dataset_txt(self.pairs_jsonl, manifest)),
            preserve_dataset_order=True,
            pairs_jsonl=self.pairs_jsonl,
            pair_limit=self.pair_limit,
            terrain_num_rows=num_rows,
            terrain_num_cols=num_cols,
            strict_pairing=self.strict_pairing,
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
        )


def manifest_motion_root(manifest: TerrainMotionPairManifest) -> Path:
    return Path(os.path.commonpath([str(Path(path).parent) for path in manifest.motion_paths]))


def manifest_dataset_txt(pairs_jsonl: str, manifest: TerrainMotionPairManifest) -> Path:
    output = resolve_repo_path(pairs_jsonl).with_name("paired_tracking_dataset.txt")
    if output.is_file():
        return output
    root = manifest_motion_root(manifest)
    lines = []
    for motion_path in manifest.motion_paths:
        motion = Path(motion_path)
        try:
            lines.append(str(motion.relative_to(root)))
        except ValueError:
            lines.append(str(motion))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
