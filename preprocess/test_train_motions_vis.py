"""
Visualize motion NPZs in IsaacLab, similar to `preprocess/test_mimic_motions.py`,
but load motions from a given root directory and optional motion list txt
like `scripts/qq_rsl_rl/train.py` (`--motion_file`, `--motion_file_txt`).

python preprocess/test_train_motions_vis.py \
  --robot_cfg roban \
  --motion_file data/roban_motions \
  --motion_file_txt data/roban_motions_list/motions_main_kept.txt \
  --num_envs 512 \
  --max_motion_num 4000

"""

from __future__ import annotations

import argparse
import os
import time

import torch

from isaaclab.app import AppLauncher


# --------------------------------------------------------------------------------------
# Robot presets (copied from preprocess/test_mimic_motions.py for consistent visualization)
# --------------------------------------------------------------------------------------
G1_ANCHOR_BODY_NAME = "pelvis"
G1_BODY_NAMES = [
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
]

ROBAN_ANCHOR_BODY_NAME = "waist_yaw_link"
ROBAN_BODY_NAMES = [
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
    "zarm_l5_link",
    "zarm_r2_link",
    "zarm_r4_link",
    "zarm_r5_link",
]

ROBOT_PRESETS = {
    "g1": {"anchor_body_name": G1_ANCHOR_BODY_NAME, "body_names": G1_BODY_NAMES},
    "roban": {"anchor_body_name": ROBAN_ANCHOR_BODY_NAME, "body_names": ROBAN_BODY_NAMES},
}


parser = argparse.ArgumentParser(description="Visualize motion NPZs in IsaacLab.")
parser.add_argument("--robot_cfg", type=str, default="roban", choices=["g1", "roban"], help="Robot preset.")

# Motion selection (same naming as qq_rsl_rl/train.py)
parser.add_argument(
    "--motion_file",
    type=str,
    required=True,
    help="Root directory containing motion .npz files (recursively searched).",
)
parser.add_argument(
    "--motion_file_txt",
    type=str,
    default=None,
    help="Optional txt file listing relative .npz paths under --motion_file (one per line).",
)
parser.add_argument(
    "--max_motion_num",
    type=int,
    default=200,
    help="Max number of motions to load (use -1 to load all).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=200,
    help="Number of environments to spawn for visualization (each env replays one motion, modulo).",
)
parser.add_argument("--env_spacing", type=float, default=2.0, help="Environment spacing for visualization.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.robots.roban_s22 import RobanS22_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionCommandCfg, MotionLoader
from whole_body_tracking.terrains.config import GRAVEL_TERRAINS_CFG  # noqa: F401


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.1, 0.1, 0.1),
            roughness=1.0,
        ),
        debug_vis=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = RobanS22_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    preset = ROBOT_PRESETS[args_cli.robot_cfg]
    body_names = list(preset["body_names"])
    anchor_body_name = preset["anchor_body_name"]
    motion_anchor_body_index = body_names.index(anchor_body_name)
    robot_body_indexes = robot.find_bodies(body_names, preserve_order=True)[0]

    # Motion loader config: root dir + optional dataset list.
    # IMPORTANT: MotionLoader._find_npz_files uses string concatenation when dataset_txt is provided,
    # so keep motion_file as a string path (not a Path).
    motion_root = os.path.abspath(args_cli.motion_file)
    dataset_txt = os.path.abspath(args_cli.motion_file_txt) if args_cli.motion_file_txt else None

    cfg = MotionCommandCfg(
        anchor_body_name=anchor_body_name,
        motion_file=motion_root,
        body_names=body_names,
        asset_name="robot",
        max_motion_num=args_cli.max_motion_num,
        dataset_txt=dataset_txt,
        resampling_time_range=(1.0e9, 1.0e9),
        resample_interval=1000 * 24,
        debug_vis=False,
        pose_range={
            "x": (-0.00, 0.00),
            "y": (-0.00, 0.00),
            "z": (0.1, 0.15),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
        velocity_range={
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
        joint_position_range=(-0.0, 0.0),
    )

    motion = MotionLoader(
        cfg=cfg,
        body_indexes=robot_body_indexes,
        motion_anchor_body_index=motion_anchor_body_index,
        device=sim.device,
    )
    motion.resample_motionloader(sim.device)

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    num_motions = len(motion.time_step_start_idx)
    if num_motions == 0:
        raise RuntimeError(f"No motions loaded from: {motion_root} (txt={dataset_txt})")

    motion_ids = torch.arange(scene.num_envs, dtype=torch.long, device=sim.device) % num_motions
    for i in range(scene.num_envs):
        time_steps[i] = motion.time_step_start_idx[motion_ids[i]]

    print(f"[INFO] motion_root={motion_root}")
    print(f"[INFO] dataset_txt={dataset_txt}")
    print(f"[INFO] loaded motions={num_motions}  num_envs={scene.num_envs}")

    def _root_from_body_tensor(x: torch.Tensor) -> torch.Tensor:
        """Return root (body0) slice from motion tensors.

        Supports shapes:
        - [T, B, D] -> returns [E, D] at timestep indices
        - [T, D]    -> returns [E, D] at timestep indices
        """
        xt = x[time_steps]
        if xt.ndim == 3:
            return xt[:, 0]
        return xt

    while simulation_app.is_running():
        t0 = time.time()
        time_steps = time_steps + 1

        # Reset envs that reach the end of their motion.
        mask = (time_steps.unsqueeze(1) == motion.time_step_end_idx.unsqueeze(0) - 1).any(dim=1)
        for i in range(scene.num_envs):
            if mask[i]:
                time_steps[i] = motion.time_step_start_idx[motion_ids[i]]

        root_states = robot.data.default_root_state.clone()
        base_pos = motion.body_pos_w[time_steps][:, 0].clone()
        terrain_origins = scene.terrain.terrain_origins.reshape(-1, 3)
        terrain_ids = motion_ids % terrain_origins.shape[0]
        base_pos += terrain_origins[terrain_ids]

        root_states[:, :3] = base_pos
        root_states[:, 3:7] = _root_from_body_tensor(motion.body_quat_w)
        # For visualization we don't need exact root velocities; also motion tensors may be [T,3] not [T,1,3].
        root_states[:, 7:13] = 0.0
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()

        sim.render()
        scene.update(sim_dt)

        # keep ~50 FPS
        dt = time.time() - t0
        if dt < 0.02:
            time.sleep(0.02 - dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    robot_cfg_map = {"g1": G1_CYLINDER_CFG, "roban": RobanS22_CYLINDER_CFG}
    scene_cfg.robot = robot_cfg_map[args_cli.robot_cfg].replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()

