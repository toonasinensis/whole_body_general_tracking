from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking import mdp
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1_AMP_ANCHOR_BODY_NAME, G1_AMP_BODY_NAMES
from whole_body_tracking.tasks.tracking.tracking_env_cfg import ObservationsCfg as TrackingObservationsCfg
from whole_body_tracking.terrains.mixed_domains import build_mixed_terrain_cfg

from ..domain_profile_applier import apply_domain_profiles, collect_domain_profile_routes
from .mixed_domain_profiles import build_default_mixed_profiles
from .paired_terrain_env_cfg import G1PairedTerrainEnvCfg


@configclass
class VelCommandTaskCfg(ObsGroup):
    velocity_command = ObsTerm(
        func=mdp.mixed_velocity_command,
        params={
            "motion_command_name": "motion",
            "velocity_command_name": "base_velocity",
            "velocity_domain_names": ("flat_velocity",),
        },
    )

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class AuxMaskCfg(ObsGroup):
    aux_mask = ObsTerm(
        func=mdp.domain_aux_mask,
        params={
            "command_name": "motion",
            "aux_domain_names": ("flat_wbc", "omniretarget_g1_terrain"),
        },
    )

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class VelTaskMaskCfg(ObsGroup):
    vel_task_mask = ObsTerm(
        func=mdp.domain_aux_mask,
        params={
            "command_name": "motion",
            "aux_domain_names": ("flat_velocity",),
        },
    )

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class AmpMaskCfg(ObsGroup):
    amp_mask = ObsTerm(
        func=mdp.domain_aux_mask,
        params={
            "command_name": "motion",
            "aux_domain_names": ("flat_velocity",),
        },
    )

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class AmpHeightScanCfg(TrackingObservationsCfg.AmpCfg):
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.00, n_max=0.00),
        clip=(-2.0, 2.0),
    )


@configclass
class G1MixedFlatMeshEnvCfg(G1PairedTerrainEnvCfg):
    pairs_jsonl: str = "data/omniretarget/g1_terrain/pairs.jsonl"
    flat_motion_file: str = "data"
    flat_dataset_txt: str = "data/tracking_npz_data/lafan_named.txt"
    flat_wbc_env_ratio: float = 0.20
    flat_velocity_env_ratio: float = 0.20
    velocity_terrain_env_ratio: float = 0.20
    mesh_env_ratio: float = 0.40
    velocity_terrain_cell_count: int = 8
    velocity_terrain_profile: str = "velocity_runway_steps"
    pair_limit: int | None = None
    terrain_size: tuple[float, float] = (24.0, 8.0)
    terrain_ground_thickness: float = 0.01
    terrain_ground_z: float = 0.0
    domain_separator_cell_count: int = 32
    strict_pairing: bool = True
    amp_use_height_scan: bool = True

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self._apply_env_overrides()
        self.configure_domains()
        self.actions.joint_pos.scale = G1_ACTION_SCALE

    def _apply_env_overrides(self):
        pairs_jsonl_override = os.getenv("WBT_PAIRS_JSONL")
        if pairs_jsonl_override:
            self.pairs_jsonl = pairs_jsonl_override
        flat_dataset_override = os.getenv("WBT_FLAT_DATASET_TXT")
        if flat_dataset_override:
            self.flat_dataset_txt = flat_dataset_override
        flat_env_ratio_override = os.getenv("WBT_FLAT_ENV_RATIO")
        flat_wbc_ratio_override = os.getenv("WBT_FLAT_WBC_ENV_RATIO")
        flat_velocity_ratio_override = os.getenv("WBT_FLAT_VELOCITY_ENV_RATIO")
        velocity_terrain_ratio_override = os.getenv("WBT_VELOCITY_TERRAIN_ENV_RATIO")
        mesh_ratio_override = os.getenv("WBT_MESH_ENV_RATIO")
        if flat_env_ratio_override and not (
            flat_wbc_ratio_override or flat_velocity_ratio_override or velocity_terrain_ratio_override
        ):
            flat_total = float(flat_env_ratio_override)
            self.flat_wbc_env_ratio = flat_total / 3.0
            self.flat_velocity_env_ratio = flat_total / 3.0
            self.velocity_terrain_env_ratio = flat_total / 3.0
            self.mesh_env_ratio = max(0.0, 1.0 - flat_total)
        if flat_wbc_ratio_override:
            self.flat_wbc_env_ratio = float(flat_wbc_ratio_override)
        if flat_velocity_ratio_override:
            self.flat_velocity_env_ratio = float(flat_velocity_ratio_override)
        if velocity_terrain_ratio_override:
            self.velocity_terrain_env_ratio = float(velocity_terrain_ratio_override)
        if mesh_ratio_override:
            self.mesh_env_ratio = float(mesh_ratio_override)
        velocity_terrain_cell_count_override = os.getenv("WBT_VELOCITY_TERRAIN_CELL_COUNT")
        if velocity_terrain_cell_count_override:
            self.velocity_terrain_cell_count = int(velocity_terrain_cell_count_override)
        velocity_terrain_profile_override = os.getenv("WBT_VELOCITY_TERRAIN_PROFILE")
        if velocity_terrain_profile_override:
            self.velocity_terrain_profile = velocity_terrain_profile_override
        separator_override = os.getenv("WBT_DOMAIN_SEPARATOR_CELL_COUNT")
        if separator_override:
            self.domain_separator_cell_count = int(separator_override)

    def _normalized_domain_ratios(self) -> tuple[float, float, float, float]:
        ratios = (
            max(0.0, float(self.flat_wbc_env_ratio)),
            max(0.0, float(self.flat_velocity_env_ratio)),
            max(0.0, float(self.velocity_terrain_env_ratio)),
            max(0.0, float(self.mesh_env_ratio)),
        )
        total = sum(ratios)
        if total <= 0.0:
            return (0.20, 0.20, 0.20, 0.40)
        return tuple(ratio / total for ratio in ratios)

    def make_domain_cfgs(self):
        flat_wbc_ratio, flat_velocity_ratio, velocity_terrain_ratio, mesh_ratio = self._normalized_domain_ratios()
        domains = []
        if flat_wbc_ratio > 0.0:
            flat_wbc_cells = max(1, int(round(float(self.scene.num_envs) * flat_wbc_ratio)))
            domains.append(
                mdp.FlatMotionDomainCfg(
                    name="flat_wbc",
                    env_ratio=flat_wbc_ratio,
                    motion_file=self.flat_motion_file,
                    dataset_txt=self.flat_dataset_txt,
                    terrain_cell_count=flat_wbc_cells,
                    task_profile="wbc_tracking",
                )
            )
        if flat_velocity_ratio > 0.0:
            flat_velocity_cells = max(1, int(round(float(self.scene.num_envs) * flat_velocity_ratio)))
            domains.append(
                mdp.FlatMotionDomainCfg(
                    name="flat_velocity",
                    env_ratio=flat_velocity_ratio,
                    motion_file=self.flat_motion_file,
                    dataset_txt=self.flat_dataset_txt,
                    terrain_cell_count=flat_velocity_cells,
                    task_profile="velocity_flat",
                )
            )
        if velocity_terrain_ratio > 0.0:
            domains.append(
                mdp.ProceduralVelocityTerrainDomainCfg(
                    name="velocity_terrain",
                    env_ratio=velocity_terrain_ratio,
                    motion_file=self.flat_motion_file,
                    dataset_txt=self.flat_dataset_txt,
                    terrain_cell_count=self.velocity_terrain_cell_count,
                    terrain_profile=self.velocity_terrain_profile,
                    curriculum=True,
                    task_profile="velocity_terrain",
                )
            )
        if mesh_ratio > 0.0:
            domains.append(
                mdp.PairedMeshMotionDomainCfg(
                    name="omniretarget_g1_terrain",
                    env_ratio=mesh_ratio,
                    pairs_jsonl=self.pairs_jsonl,
                    pair_limit=self.pair_limit,
                    task_profile="wbc_tracking",
                )
            )
        return domains

    def configure_domains(self):
        domains = self.make_domain_cfgs()
        profiles = build_default_mixed_profiles()
        routes = collect_domain_profile_routes(domains, profiles)
        if os.getenv("WBT_DEBUG_VELCOMMAND", "0") not in ("", "0", "false", "False"):
            print(
                "[WBT_DEBUG_VELCOMMAND_CFG] domains:", tuple((domain.name, domain.task_profile) for domain in domains)
            )
            print("[WBT_DEBUG_VELCOMMAND_CFG] velocity_domain_names:", routes.velocity_domain_names)
            print("[WBT_DEBUG_VELCOMMAND_CFG] vel_task_mask active domains:", routes.velocity_domain_names)
            print("[WBT_DEBUG_VELCOMMAND_CFG] aux_mask_domain_names:", routes.aux_mask_domain_names)
            print("[WBT_DEBUG_VELCOMMAND_CFG] amp_mask_domain_names:", routes.amp_mask_domain_names)
        terrain_generator, _ = build_mixed_terrain_cfg(
            domains,
            terrain_size=self.terrain_size,
            ground_thickness=self.terrain_ground_thickness,
            ground_z=self.terrain_ground_z,
            domain_separator_cell_count=self.domain_separator_cell_count,
        )
        terrain_generator.curriculum = any(bool(getattr(domain, "curriculum", False)) for domain in domains)

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

        self.observations.velcommand = VelCommandTaskCfg()
        self.observations.velcommand.velocity_command.params["velocity_domain_names"] = routes.velocity_domain_names
        self.observations.vel_task_mask = VelTaskMaskCfg()
        self.observations.vel_task_mask.vel_task_mask.params["aux_domain_names"] = routes.velocity_domain_names
        self.observations.aux_mask = AuxMaskCfg()
        self.observations.aux_mask.aux_mask.params["aux_domain_names"] = routes.aux_mask_domain_names
        self.observations.amp_mask = AmpMaskCfg()
        self.observations.amp_mask.amp_mask.params["aux_domain_names"] = routes.amp_mask_domain_names
        self._configure_amp_observations()

        apply_domain_profiles(self, domains, profiles)

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
            profile_reset_cfgs=routes.profile_reset_cfgs,
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
                "z": (0.05, 0.1),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            pose_range_env_ratio=1.0,
            pose_range_init_mode="range",
            velocity_range={
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            joint_position_range=(-0.0, 0.0),
            motion_sampling_start_frame=5,
            adaptive_sample_rewind_min_bins=1,
            adaptive_sample_rewind_bins=2,
            debug_domain_log_count=16,
            terminal_guidance_velocity_enabled=True,
            debug_body_pose=False,
            debug_anchor_speed=True,
        )

    def _configure_amp_observations(self):
        self.observations.amp = AmpHeightScanCfg() if self.amp_use_height_scan else TrackingObservationsCfg.AmpCfg()
        for term in (
            self.observations.amp.body_pos_b,
            self.observations.amp.body_ori_b,
            self.observations.amp.body_lin_vel_b,
            self.observations.amp.body_ang_vel_b,
        ):
            term.params["asset_name"] = "robot"
            term.params["anchor_body_name"] = G1_AMP_ANCHOR_BODY_NAME
            term.params["body_names"] = tuple(G1_AMP_BODY_NAMES)
