"""Replay a converted Roban npz motion in Isaac Sim."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted Roban motions.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to a local motion npz file.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry artifact.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if (args_cli.motion_file is None) == (args_cli.registry_name is None):
    parser.error("Specify exactly one of --motion_file or --registry_name.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from whole_body_tracking.robots.roban_s22 import RobanS22_CYLINDER_CFG


ROBAN_JOINT_NAMES = [
    "waist_yaw_joint",
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
]


@configclass
class ReplayRobanSceneCfg(InteractiveSceneCfg):
    """Configuration for a Roban replay scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=750.0),
    )
    robot: ArticulationCfg = RobanS22_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class NpzMotion:
    """Small NPZ motion reader for replay visualization."""

    def __init__(self, motion_file: str, device: torch.device):
        raw = np.load(motion_file, allow_pickle=True)
        required_keys = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        missing = [key for key in required_keys if key not in raw]
        if missing:
            raise KeyError(f"{motion_file} is missing required keys: {missing}")

        self.joint_pos = torch.as_tensor(raw["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.as_tensor(raw["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w = torch.as_tensor(raw["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w = torch.as_tensor(raw["body_quat_w"], dtype=torch.float32, device=device)
        self.body_lin_vel_w = torch.as_tensor(raw["body_lin_vel_w"], dtype=torch.float32, device=device)
        self.body_ang_vel_w = torch.as_tensor(raw["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.joint_names = self._read_names(raw, ("joint_names", "motion_joint_names", "robot_joint_names"))
        self.time_step_total = int(self.joint_pos.shape[0])
        self.fps = float(np.asarray(raw["fps"]).reshape(-1)[0]) if "fps" in raw else 50.0
        print(
            f"[INFO]: Loaded motion: {motion_file}, frames={self.time_step_total}, "
            f"fps={self.fps}, joints={self.joint_pos.shape[1]}, bodies={self.body_pos_w.shape[1]}"
        )

    @staticmethod
    def _read_names(raw: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> list[str] | None:
        for key in keys:
            if key not in raw:
                continue
            value = raw[key]
            if value.shape == () and value.dtype == object:
                value = value.item()
            return [str(item) for item in np.asarray(value).reshape(-1).tolist()]
        return None


def resolve_motion_file() -> str:
    if args_cli.motion_file is not None:
        return args_cli.motion_file

    registry_name = args_cli.registry_name
    if ":" not in registry_name:
        registry_name += ":latest"

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    return str(Path(artifact.download()) / "motion.npz")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion = NpzMotion(resolve_motion_file(), sim.device)
    if motion.body_pos_w.shape[1] < 1:
        raise ValueError("Motion body_pos_w must contain at least one body for root replay.")

    motion_joint_names = motion.joint_names or ROBAN_JOINT_NAMES
    if len(motion_joint_names) != motion.joint_pos.shape[1]:
        raise ValueError(
            f"Motion joint_names length {len(motion_joint_names)} does not match "
            f"joint_pos dim {motion.joint_pos.shape[1]}."
        )
    joint_ids = robot.find_joints(motion_joint_names, preserve_order=True)[0]

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps], joint_ids=joint_ids)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayRobanSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
