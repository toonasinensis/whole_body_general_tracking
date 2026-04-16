"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

# Usage
    python preprocess/soma_npz_to_mimic_npz.py \
    --input_dir "data/roban_soma_motions" \
    --pattern "*.npz" \
    --output_fps 50 \
    --num_envs 64 \
    --headless \
    --device cuda:0 \
    --overwrite \
    --no_wandb \
    --output_dir "./data/roban_motions"
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, default=None, help="Path to a single input SOMA npz file.")
parser.add_argument(
    "--input_dir",
    type=str,
    default=None,
    help="Directory containing multiple SOMA npz files to convert in parallel.",
)
parser.add_argument(
    "--pattern",
    type=str,
    default="*.npz",
    help="Glob pattern within --input_dir (default: *.npz).",
)
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, default=None, help="Name of the output motion (single-file mode only).")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
# New: allow disabling wandb upload and choose save path
parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging and registry upload.")
parser.add_argument("--save_to", type=str, default=None, help="Path to save the generated npz (single-file mode).")
parser.add_argument("--robot", type=str, help="robot name")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory for batch conversion (--input_dir). If omitted, uses <input_dir>_tracking_npz_fps<output_fps>.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of parallel envs for batch conversion. Default: min(#files, 64).",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help="Overwrite existing output files in batch mode.",
)
parser.add_argument(
    "--follow_camera",
    action="store_true",
    default=False,
    help="Follow env[0] root each frame (interactive debug). Off by default for speed.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.robots.roban_s22 import RobanS22_CYLINDER_CFG
@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")    
    robot: ArticulationCfg = RobanS22_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# load npz file and interpolate to the output fps
class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        expected_dof_dim: int | None = None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.expected_dof_dim = expected_dof_dim
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from a CSV file or a SOMA-style NPZ file.

        Supported formats:
        - CSV: columns [base_pos(3), base_quat_xyzw(4), dof_pos(...)]
        - NPZ (SOMA): keys {'data','fps'} where data has same column layout as CSV.
        """
        if str(self.motion_file).lower().endswith(".npz"):
            data = np.load(self.motion_file, allow_pickle=True)
            if "data" not in data.files:
                raise ValueError(f"NPZ input must contain key 'data': {self.motion_file}")
            motion_np = np.asarray(data["data"])
            # override input fps from file if present
            if "fps" in data.files:
                fps_arr = np.asarray(data["fps"]).reshape(-1)
                if fps_arr.size > 0:
                    self.input_fps = float(fps_arr[0])
                    self.input_dt = 1.0 / float(self.input_fps)
            if self.frame_range is not None:
                # frame_range is 1-indexed inclusive in CLI
                start = self.frame_range[0] - 1
                end = self.frame_range[1]
                motion_np = motion_np[start:end]
            motion = torch.from_numpy(motion_np)
        else:
            # CSV
            if self.frame_range is None:
                motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
            else:
                motion = torch.from_numpy(
                    np.loadtxt(
                        self.motion_file,
                        delimiter=",",
                        skiprows=self.frame_range[0] - 1,
                        max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                    )
                )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        # Roban training format uses 21 controllable joints (no head joints).
        # Some SOMA exports include 2 additional head joints at the end (total 23).
        expected_dof_dim = int(self.expected_dof_dim or 0)
        if expected_dof_dim > 0 and self.motion_dof_poss_input.shape[1] == expected_dof_dim + 2:
            self.motion_dof_poss_input = self.motion_dof_poss_input[:, :expected_dof_dim]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        bool,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def _save_log_npz(path: str, log: dict):
    for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        log[k] = np.stack(log[k], axis=0)
    np.savez(path, **log)


def run_simulator_batch(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    joint_names: list[str],
    input_files: list[str],
    output_dir: str,
    input_root_dir: str,
):
    """Convert many SOMA npz files in parallel envs.

    We assign one motion per env, step them together, and write one output NPZ per input.
    """
    os.makedirs(output_dir, exist_ok=True)
    robot: Articulation = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    def make_motion(path: str) -> MotionLoader:
        return MotionLoader(
            motion_file=path,
            input_fps=args_cli.input_fps,
            output_fps=args_cli.output_fps,
            device=sim.device,
            frame_range=args_cli.frame_range,
            expected_dof_dim=len(joint_names),
        )

    # work queue
    queue = list(input_files)
    num_envs = scene.num_envs

    # per-env state
    motions: list[MotionLoader | None] = [None] * num_envs
    cur_path: list[str | None] = [None] * num_envs
    logs: list[dict] = [
        {
            "fps": [args_cli.output_fps],
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
        }
        for _ in range(num_envs)
    ]

    input_root_dir_abs = os.path.abspath(os.path.expanduser(input_root_dir))

    def _output_path_for_src(src_path: str) -> str:
        """Mirror src relative path under output_dir."""
        src_abs = os.path.abspath(os.path.expanduser(src_path))
        rel = os.path.relpath(src_abs, start=input_root_dir_abs)
        # Safety: if src is outside root for any reason, fall back to basename.
        if rel.startswith(".."):
            rel = os.path.basename(src_abs)
        out_path = os.path.join(output_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        return out_path

    def pop_next_for_env(env_id: int):
        nonlocal queue
        while True:
            if len(queue) == 0:
                motions[env_id] = None
                cur_path[env_id] = None
                return
            p = queue.pop(0)
            out_path = _output_path_for_src(p)
            if (not args_cli.overwrite) and os.path.exists(out_path):
                # skip existing
                continue
            motions[env_id] = make_motion(p)
            cur_path[env_id] = p
            # reset log buffers
            logs[env_id] = {
                "fps": [args_cli.output_fps],
                "joint_pos": [],
                "joint_vel": [],
                "body_pos_w": [],
                "body_quat_w": [],
                "body_lin_vel_w": [],
                "body_ang_vel_w": [],
            }
            return

    # initial fill
    for env_id in range(num_envs):
        pop_next_for_env(env_id)

    sim_dt = sim.get_physics_dt()

    while simulation_app.is_running():
        # stop when all envs are idle and queue empty
        if all(m is None for m in motions) and len(queue) == 0:
            break

        root_states = robot.data.default_root_state.clone()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        reset_flags = [False] * num_envs

        # step each env motion once
        for env_id, motion in enumerate(motions):
            if motion is None:
                continue

            (state, reset_flag) = motion.get_next_state()
            reset_flags[env_id] = bool(reset_flag)
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ) = state

            # write root state slice
            root_states[env_id : env_id + 1, :3] = motion_base_pos
            root_states[env_id : env_id + 1, :2] += scene.env_origins[env_id : env_id + 1, :2]
            root_states[env_id : env_id + 1, 3:7] = motion_base_rot
            root_states[env_id : env_id + 1, 7:10] = motion_base_lin_vel
            root_states[env_id : env_id + 1, 10:] = motion_base_ang_vel

            # write joint state slice
            if motion_dof_pos.shape[1] == len(robot_joint_indexes):
                joint_pos[env_id : env_id + 1, robot_joint_indexes] = motion_dof_pos
                joint_vel[env_id : env_id + 1, robot_joint_indexes] = motion_dof_vel
            elif motion_dof_pos.shape[1] == len(robot.joint_names):
                joint_pos[env_id : env_id + 1, :] = motion_dof_pos
                joint_vel[env_id : env_id + 1, :] = motion_dof_vel
            else:
                raise ValueError(
                    f"Unexpected dof dim in motion '{cur_path[env_id]}': got {motion_dof_pos.shape[1]}. "
                    f"Expected {len(robot_joint_indexes)} (subset) or {len(robot.joint_names)} (full)."
                )

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim_dt)

        if args_cli.follow_camera:
            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        # append logs for active envs (in IsaacLab default joint order)
        for env_id, motion in enumerate(motions):
            if motion is None:
                continue
            logs[env_id]["joint_pos"].append(robot.data.joint_pos[env_id, :].cpu().numpy().copy())
            logs[env_id]["joint_vel"].append(robot.data.joint_vel[env_id, :].cpu().numpy().copy())
            logs[env_id]["body_pos_w"].append(robot.data.body_pos_w[env_id, :].cpu().numpy().copy())
            logs[env_id]["body_quat_w"].append(robot.data.body_quat_w[env_id, :].cpu().numpy().copy())
            logs[env_id]["body_lin_vel_w"].append(robot.data.body_lin_vel_w[env_id, :].cpu().numpy().copy())
            logs[env_id]["body_ang_vel_w"].append(robot.data.body_ang_vel_w[env_id, :].cpu().numpy().copy())

            # if this motion ended, save it and load the next one for this env
            if reset_flags[env_id]:
                src_path = cur_path[env_id]
                if src_path is not None:
                    out_path = _output_path_for_src(src_path)
                    _save_log_npz(out_path, logs[env_id])
                    print(f"[INFO] Saved: {out_path}")
                pop_next_for_env(env_id)


def run_simulator_single(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Single-file conversion (backward compatible).
    joint names: the order of the joints in the src motion files (soma bones npz files to be converted)

    """
    if args_cli.input_file is None:
        raise ValueError("--input_file is required in single-file mode")
    if args_cli.save_to is None:
        raise ValueError("--save_to is required in single-file mode")

    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        expected_dof_dim=len(joint_names),
    )

    robot: Articulation = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        (state, reset_flag) = motion.get_next_state()
        (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
        ) = state

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        if motion_dof_pos.shape[1] == len(robot_joint_indexes):
            joint_pos[:, robot_joint_indexes] = motion_dof_pos
            joint_vel[:, robot_joint_indexes] = motion_dof_vel
        elif motion_dof_pos.shape[1] == len(robot.joint_names):
            joint_pos[:, :] = motion_dof_pos
            joint_vel[:, :] = motion_dof_vel
        else:
            raise ValueError(
                f"Unexpected dof dim in motion: got {motion_dof_pos.shape[1]}. "
                f"Expected {len(robot_joint_indexes)} (subset) or {len(robot.joint_names)} (full)."
            )
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim_dt)

        if args_cli.follow_camera:
            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        # Save joints in IsaacLab's default joint order.
        # rjp = robot.data.joint_pos[0, :]
        # rjv = robot.data.joint_vel[0, :]
        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag:
            _save_log_npz(args_cli.save_to, log)
            print(f"[INFO]: Motion saved to: {args_cli.save_to}")
            break


def main():
    """Main function."""
    # Resolve inputs
    batch_files: list[str] | None = None
    output_dir: str | None = None
    if args_cli.input_dir is not None:
        in_dir = os.path.abspath(os.path.expanduser(args_cli.input_dir))
        if not os.path.isdir(in_dir):
            raise FileNotFoundError(f"--input_dir not found: {in_dir}")
        import glob
        # By default, search recursively since many SOMA exports are nested in subfolders.
        pattern = args_cli.pattern
        if "**" in pattern:
            search_pattern = os.path.join(in_dir, pattern)
            batch_files = sorted(glob.glob(search_pattern, recursive=True))
        else:
            # always include recursive search
            search_pattern = os.path.join(in_dir, "**", pattern)
            batch_files = sorted(glob.glob(search_pattern, recursive=True))
        if len(batch_files) == 0:
            raise FileNotFoundError(f"No files matched in {in_dir} with pattern {pattern} (recursive)")
        if args_cli.output_dir is None:
            output_dir = f"{in_dir}_tracking_npz_fps{args_cli.output_fps}"
        else:
            output_dir = os.path.abspath(os.path.expanduser(args_cli.output_dir))

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    if batch_files is None:
        num_envs = 1
    else:
        if args_cli.num_envs is None:
            num_envs = min(len(batch_files), 64)
        else:
            num_envs = int(args_cli.num_envs)
        num_envs = max(1, num_envs)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    joint_names = [ # The joints order in the src motion files (soma bones npz files to be converted)
            # "left_hip_pitch_joint",
            # "left_hip_roll_joint",
            # "left_hip_yaw_joint",
            # "left_knee_joint",
            # "left_ankle_pitch_joint",
            # "left_ankle_roll_joint",
            # "right_hip_pitch_joint",
            # "right_hip_roll_joint",
            # "right_hip_yaw_joint",
            # "right_knee_joint",
            # "right_ankle_pitch_joint",
            # "right_ankle_roll_joint",
            # "waist_yaw_joint",
            # "waist_roll_joint",
            # "waist_pitch_joint",
            # "left_shoulder_pitch_joint",
            # "left_shoulder_roll_joint",
            # "left_shoulder_yaw_joint",
            # "left_elbow_joint",
            # "left_wrist_roll_joint",
            # "left_wrist_pitch_joint",
            # "left_wrist_yaw_joint",
            # "right_shoulder_pitch_joint",
            # "right_shoulder_roll_joint",
            # "right_shoulder_yaw_joint",
            # "right_elbow_joint",
            # "right_wrist_roll_joint",
            # "right_wrist_pitch_joint",
            # "right_wrist_yaw_joint",
            
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

    # Run conversion
    if batch_files is None:
        run_simulator_single(sim, scene, joint_names=joint_names)
    else:
        assert output_dir is not None
        run_simulator_batch(
            sim,
            scene,
            joint_names=joint_names,
            input_files=batch_files,
            output_dir=output_dir,
            input_root_dir=in_dir,
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
