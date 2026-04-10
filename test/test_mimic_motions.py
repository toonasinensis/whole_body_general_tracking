"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python test/test_mimic_motions.py --motion_file 
    python test/test_mimic_motions.py --motion_dir  assets/roban_motions  --num_envs 4


TODO add replay for kuavo  s52
TODO add replay for unitree g1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path
import numpy as np
import torch

from isaaclab.app import AppLauncher
g1 = False

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, required=False, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion npz (overrides registry)")
parser.add_argument(
    "--motion_dir",
    type=str,
    default=None,
    help="Directory containing multiple .npz motion files to replay in parallel (overrides --motion_file/--registry_name).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of envs to spawn for replay. If --motion_dir is set and --num_envs is omitted, defaults to number of .npz files.",
)
parser.add_argument(
    "--shuffle",
    action="store_true",
    default=False,
    help="Shuffle motion files when using --motion_dir.",
)
parser.add_argument(
    "--follow_camera",
    action="store_true",
    default=False,
    help="If set, camera follows env[0] root each frame. If not set, viewer stays interactive.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.robots.roban_s22 import RobanS22_CYLINDER_CFG, PRESERVE_JOINT_ORDER_ASSET_CFG
from whole_body_tracking.tasks.magic_tracking.mdp import MotionLoader


def _resolve_motion_files() -> list[str]:
    """Resolve motion sources into a list of file paths.

    Priority:
    1) --motion_dir (all *.npz in the directory)
    2) --motion_file
    3) --registry_name (wandb artifact -> motion.npz)
    """
    if args_cli.motion_dir is not None:
        motion_dir = Path(args_cli.motion_dir).expanduser().resolve()
        if not motion_dir.is_dir():
            raise FileNotFoundError(f"--motion_dir is not a directory: {motion_dir}")
        paths = sorted(str(p) for p in motion_dir.glob("*.npz"))
        if len(paths) == 0:
            raise FileNotFoundError(f"No .npz files found in: {motion_dir}")
        if args_cli.shuffle:
            import random

            random.shuffle(paths)
        return paths

    if args_cli.motion_file is not None:
        motion_file = str(Path(args_cli.motion_file).expanduser().resolve())
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"--motion_file not found: {motion_file}")
        return [motion_file]

    if args_cli.registry_name is None:
        raise ValueError("Please provide either --motion_dir, --motion_file, or --registry_name")

    registry_name = args_cli.registry_name
    if ":" not in registry_name:
        registry_name += ":latest"

    import pathlib
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    return [motion_file]


def _compute_reorder_indices(npz_joint_order: list[str], isaac_joint_names: list[str]) -> list[int]:
    """Compute indices to map NPZ joint order -> Isaac joint order."""
    reorder_indices: list[int] = []
    for isaac_joint in isaac_joint_names:
        try:
            reorder_indices.append(npz_joint_order.index(isaac_joint))
        except ValueError as e:
            raise ValueError(f"Joint '{isaac_joint}' not found in NPZ joint order") from e
    return reorder_indices


def reorder_joint_data_for_isaac(joint_data: torch.Tensor, npz_joint_order: list[str], isaac_joint_names: list[str]) -> torch.Tensor:
    """
    将NPZ文件中的关节数据重排序为IsaacLab期望的顺序
    
    Args:
        joint_data: NPZ文件中的关节数据 (按自定义顺序排列)
        npz_joint_order: NPZ文件中使用的关节顺序
        isaac_joint_names: IsaacLab机器人的关节顺序
    
    Returns:
        重排序后的关节数据 (按IsaacLab顺序排列)
    """
    # 创建从NPZ顺序到IsaacLab顺序的映射
    reorder_indices = _compute_reorder_indices(npz_joint_order, isaac_joint_names)
    
    # 应用重排序
    if joint_data.dim() == 1:
        # 一维数据: (num_joints,)
        return joint_data[reorder_indices]
    else:
        # 二维数据: (num_envs, num_joints)
        return joint_data[:, reorder_indices]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = RobanS22_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if g1:
        robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion_files: list[str]):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # assign motion file per env (repeat/cycle if needed)
    motion_files_per_env = [motion_files[i % len(motion_files)] for i in range(scene.num_envs)]
    print("[INFO] Replaying motions:")
    for i, p in enumerate(motion_files_per_env):
        print(f"  env[{i}]: {p}")

    motions: list[MotionLoader] = [
        MotionLoader(p, torch.tensor([0], dtype=torch.long, device=sim.device), sim.device) for p in motion_files_per_env
    ]
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    
    # NPZ文件中使用的关节顺序 (深度优先)
    npz_joint_order = PRESERVE_JOINT_ORDER_ASSET_CFG.joint_names

    # Pre-compute reorder indices once (assumes all motions share the same joint order)
    reorder_indices = _compute_reorder_indices(npz_joint_order, robot.joint_names)

    # Simulation loop
    while simulation_app.is_running():
        # advance and wrap per env (motions can have different lengths)
        time_steps += 1
        for env_id, motion in enumerate(motions):
            if time_steps[env_id] >= motion.time_step_total:
                time_steps[env_id] = 0

        # build batched root + joint states
        root_states = robot.data.default_root_state.clone()

        root_pos = []
        root_quat = []
        root_lin = []
        root_ang = []
        joint_pos = []
        joint_vel = []

        for env_id, motion in enumerate(motions):
            ts = int(time_steps[env_id].item())
            # root body assumed at index 0 in motion buffers
            root_pos.append(motion.body_pos_w[ts][0])
            root_quat.append(motion.body_quat_w[ts][0])
            root_lin.append(motion.body_lin_vel_w[ts][0])
            root_ang.append(motion.body_ang_vel_w[ts][0])

            # reorder joint arrays to match robot.joint_names
            jp = motion.joint_pos[ts]
            jv = motion.joint_vel[ts]
            joint_pos.append(jp[reorder_indices])
            joint_vel.append(jv[reorder_indices])

        # root_pos is [num_envs, 3]; env_origins is [num_envs, 3]
        # (do NOT add an extra dim, otherwise it broadcasts to [N, N, 3])
        root_pos = torch.stack(root_pos, dim=0) + scene.env_origins
        root_quat = torch.stack(root_quat, dim=0)
        root_lin = torch.stack(root_lin, dim=0)
        root_ang = torch.stack(root_ang, dim=0)
        joint_pos = torch.stack(joint_pos, dim=0)
        joint_vel = torch.stack(joint_vel, dim=0)

        root_states[:, :3] = root_pos
        root_states[:, 3:7] = root_quat
        root_states[:, 7:10] = root_lin
        root_states[:, 10:] = root_ang

        robot.write_root_state_to_sim(root_states)

        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        # Keep viewer interactive by default. Following the robot every frame overrides manual control.
        if args_cli.follow_camera:
            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    motion_files = _resolve_motion_files()
    # determine number of envs
    if args_cli.num_envs is None:
        num_envs = len(motion_files)
    else:
        num_envs = int(args_cli.num_envs)
        if num_envs <= 0:
            raise ValueError("--num_envs must be > 0")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene, motion_files)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
