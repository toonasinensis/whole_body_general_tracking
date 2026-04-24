"""
Replay a tab-separated deploy motion CSV in IsaacLab at a fixed FPS.

The deploy CSV is expected to have columns:
- body_pos_x, body_pos_y, body_pos_z
- body_quat_w, body_quat_x, body_quat_y, body_quat_z   (wxyz)
- joint_pos00..joint_posNN
- joint_vel_00..joint_vel_NN

Notes
-----
- Joint columns are assumed to already be in IsaacLab's default joint order, so we do NOT reorder them.
- Root linear/angular velocities are not present in the deploy CSV. We write zeros for them.

Example
-------
python preprocess/test_deploy_csv_isaaclab.py \
  --motion data/roban_deploy_motions/body_check_001__A160.csv \
  --fps 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


def _load_motion_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load tab-separated motion CSV; returns (data, header)."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not text:
        raise ValueError(f"Empty file: {path}")
    header = text[0].split("\t")
    rows = [[float(x) for x in line.split("\t")] for line in text[1:] if line.strip()]
    if not rows:
        raise ValueError(f"No data rows in: {path}")
    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] != len(header):
        raise ValueError(f"Column count mismatch: header {len(header)} vs row {data.shape[1]}")
    return data, header


def _require_cols(col: dict[str, int], names: list[str]) -> None:
    missing = [n for n in names if n not in col]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


# -----------------------------------------------------------------------------
# Args / AppLauncher
# -----------------------------------------------------------------------------
def _find_repo_root(start: Path) -> Path:
    """Walk upwards until we find the repo root (contains `source/` and `data/`)."""
    for p in [start, *start.parents]:
        if (p / "source").is_dir() and (p / "data").is_dir():
            return p
    return start.parent


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())

parser = argparse.ArgumentParser(description="Replay deploy motion CSV in IsaacLab.")
parser.add_argument(
    "--motion",
    type=str,
    default=str(_REPO_ROOT / "data/roban_deploy_motions/body_check_001__A160.csv"),
    help="Tab-separated motion CSV (root pose + joint_posNN + joint_vel_NN).",
)
parser.add_argument("--fps", type=float, default=50.0, help="Playback rate (Hz).")
parser.add_argument("--loop", action="store_true", default=True, help="Loop the motion when reaching the end.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Safe imports after SimulationApp is running
# -----------------------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.roban_s22 import RobanS22_CYLINDER_CFG
from whole_body_tracking.terrains.config import GRAVEL_TERRAINS_CFG  # noqa: F401


@configclass
class ReplayDeployCsvSceneCfg(InteractiveSceneCfg):
    """Scene for replaying a deploy CSV motion."""

    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1), roughness=1.0),
        debug_vis=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = RobanS22_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: SimulationContext, scene: InteractiveScene, motion: np.ndarray, header: list[str], fps: float) -> None:
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    frame_dt = 1.0 / float(fps)

    col = {name: i for i, name in enumerate(header)}
    _require_cols(
        col,
        [
            "body_pos_x",
            "body_pos_y",
            "body_pos_z",
            "body_quat_w",
            "body_quat_x",
            "body_quat_y",
            "body_quat_z",
        ],
    )

    # Infer joint count from CSV and validate required joint columns exist.
    csv_joint_pos = sorted([n for n in header if n.startswith("joint_pos")])
    csv_joint_vel = sorted([n for n in header if n.startswith("joint_vel_")])
    if len(csv_joint_pos) == 0 or len(csv_joint_pos) != len(csv_joint_vel):
        raise ValueError(f"Bad joint columns: joint_pos={len(csv_joint_pos)} joint_vel={len(csv_joint_vel)}")
    n_joints = len(csv_joint_pos)
    for k in range(n_joints):
        _require_cols(col, [f"joint_pos{k:02d}", f"joint_vel_{k:02d}"])

    # IsaacLab joint count sanity check (best-effort).
    lab_joint_names = list(getattr(robot, "joint_names", []))
    if lab_joint_names and len(lab_joint_names) != n_joints:
        print(
            f"[WARN] CSV joint count={n_joints} but IsaacLab robot joint_names={len(lab_joint_names)}. "
            "Will still attempt to write joint state as-is."
        )

    device = sim.device
    num_envs = int(scene.num_envs)
    if num_envs != 1:
        raise ValueError("This replay script expects num_envs == 1.")

    joint_pos = torch.zeros((num_envs, n_joints), device=device, dtype=torch.float32)
    joint_vel = torch.zeros((num_envs, n_joints), device=device, dtype=torch.float32)

    frame = 0
    n_frames = int(motion.shape[0])

    while simulation_app.is_running():
        t0 = time.time()

        row = motion[frame]
        root_states = robot.data.default_root_state.clone()

        # Root pose
        root_states[:, 0] = float(row[col["body_pos_x"]])
        root_states[:, 1] = float(row[col["body_pos_y"]])
        root_states[:, 2] = float(row[col["body_pos_z"]])
        # Quaternion (wxyz), consistent with the deploy CSV and MotionLoader conventions.
        root_states[:, 3] = float(row[col["body_quat_w"]])
        root_states[:, 4] = float(row[col["body_quat_x"]])
        root_states[:, 5] = float(row[col["body_quat_y"]])
        root_states[:, 6] = float(row[col["body_quat_z"]])

        # Root velocities: not available in deploy CSV.
        root_states[:, 7:13] = 0.0

        # Joint state (no reordering).
        for k in range(n_joints):
            joint_pos[0, k] = float(row[col[f"joint_pos{k:02d}"]])
            joint_vel[0, k] = float(row[col[f"joint_vel_{k:02d}"]])

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()

        sim.render()
        scene.update(sim_dt)

        frame += 1
        if frame >= n_frames:
            if args_cli.loop:
                frame = 0
            else:
                break

        elapsed = time.time() - t0
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)


def main() -> None:
    motion_path = Path(args_cli.motion)
    if not motion_path.is_absolute():
        motion_path = _REPO_ROOT / motion_path
    motion, header = _load_motion_csv(motion_path)
    print(f"[INFO] Loaded motion: {motion_path} frames={motion.shape[0]} cols={motion.shape[1]}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(args_cli.fps)
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayDeployCsvSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    run_simulator(sim, scene, motion, header, args_cli.fps)


if __name__ == "__main__":
    main()
    simulation_app.close()

