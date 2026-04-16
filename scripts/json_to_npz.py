"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import numpy as np
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, required=False, help="The path to the input motion csv file.")
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
parser.add_argument("--output_name", type=str, required=False, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


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

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _pad_list(tensor_list, max_frame):
    """
    把 list[tensor(num_frame, D)] pad 成 (num_motion, max_frame, D)
    """
    num_motion = len(tensor_list)
    D = tensor_list[0].shape[-1]
    padded = torch.zeros(num_motion, max_frame, D, device=tensor_list[0].device)
    for i, t in enumerate(tensor_list):
        L = t.shape[0]
        padded[i, :L] = t
    return padded


import time


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        output_fps: int,
        device: torch.device,
    ):
        self.motion_file = motion_file
        self.output_fps = output_fps
        self.output_dt = 1.0 / self.output_fps
        self.device = device
        self._load_motion()
        self.lerp_bp = []
        self.lerp_bq = []
        self.lerp_bv = []
        self.lerp_bw = []
        self.lerp_dp = []
        self.lerp_dv = []
        self.lerpfn = []

        for i in range(self.motion_num):
            # import ipdb;ipdb.set_trace()
            st = time.time()
            motion_base_poss, motion_base_rots, motion_dof_poss = self._interpolate_motion(
                self.duration_list[i],
                self.output_dt,
                self.body_pos_w_list[i][0],
                self.body_quat_w_list[i][0],
                self.joint_pos_list[i][0],
            )
            motion_base_lin_vels, motion_base_ang_vels, motion_dof_vels = self._compute_velocities(
                motion_base_poss, motion_base_rots, motion_dof_poss, self.output_dt
            )
            frame_out_num = motion_base_poss.shape[0]
            self.lerp_bp.append(motion_base_poss)
            self.lerp_bq.append(motion_base_rots)
            self.lerp_bv.append(motion_base_lin_vels)
            self.lerp_bw.append(motion_base_ang_vels)
            self.lerp_dp.append(motion_dof_poss)
            self.lerp_dv.append(motion_dof_vels)
            self.lerpfn.append(frame_out_num)
            et = time.time()
            print(
                f"motion {i} processed, duration: {self.duration_list[i]:.2f} sec, input frames:"
                f" {self.input_frames_list[i]}, output frames: {frame_out_num}, time cost: {et-st:.2f} sec"
            )

        self.lerpfn = torch.tensor(self.lerpfn, device=self.device)

        # for i range(self.motion_num):
        self.lerp_max_frame_num = max(self.lerpfn)
        self.lerp_bp = _pad_list(self.lerp_bp, self.lerp_max_frame_num)
        self.lerp_bq = _pad_list(self.lerp_bq, self.lerp_max_frame_num)
        self.lerp_bv = _pad_list(self.lerp_bv, self.lerp_max_frame_num)
        self.lerp_bw = _pad_list(self.lerp_bw, self.lerp_max_frame_num)
        self.lerp_dp = _pad_list(self.lerp_dp, self.lerp_max_frame_num)
        self.lerp_dv = _pad_list(self.lerp_dv, self.lerp_max_frame_num)
        self.current_idx = torch.zeros_like(self.lerpfn, device=self.device)
        self.finish_flag = torch.zeros_like(self.lerpfn, device=self.device, dtype=torch.bool)

    def _load_motion(self):
        """Loads the motion from the csv file."""
        ext = ".csv"
        # import ipdb;ipdb.set_trace()
        if ext == ".csv":
            self.motion_names = []
            for root, _, files in os.walk(self.motion_file):
                for f in files:
                    if f.endswith(".csv"):
                        rel_path = os.path.relpath(os.path.join(root, f), self.motion_file)
                        self.motion_names.append(rel_path)

            self.joint_pos_list = []
            self.body_pos_w_list = []
            self.body_quat_w_list = []
            self.fps_list = []
            data_list = []
            self.duration_list = []
            self.input_frames_list = []

            csv_file_paths = [os.path.join(self.motion_file, f) for f in self.motion_names]

            for f_path in csv_file_paths:
                motion = torch.from_numpy(np.loadtxt(f_path, delimiter=","))
                self.data_joint_order = [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ]
                data_list.append(motion)

            for data in data_list:
                data = data.to(torch.float32).to(self.device)

                motion_base_rots_input = data[:, 3:7]

                j_pos = data[:, 7:].to(self.device)  # (num_frame, dof)
                b_pos = data[:, :3]  # (num_frame, 3)
                b_quat = motion_base_rots_input[:, [3, 0, 1, 2]]  # (num_frame, 4)
                fps = float(args_cli.input_fps)
                num_frame, dof = j_pos.shape

                self.joint_pos_list.append(j_pos.unsqueeze(0))  # (1, max_frame, dof)
                self.body_pos_w_list.append(b_pos.unsqueeze(0))  # (1, max_frame, dof)
                self.body_quat_w_list.append(b_quat.unsqueeze(0))  # (1, max_frame, dof)
                self.fps_list.append(fps)
                self.duration_list.append((num_frame - 1) / float(fps))
                self.input_frames_list.append(num_frame)
                self.motion_num = len(data_list)
            print(
                f"Motion loaded ({self.motion_file}), duration: {self.duration_list} sec, frames:"
                f" {self.input_frames_list}"
            )

            pass
        elif ext == ".json":
            self.motion_names = []
            for root, _, files in os.walk(self.motion_file):
                for f in files:
                    if f.endswith(".json"):
                        rel_path = os.path.relpath(os.path.join(root, f), self.motion_file)
                        self.motion_names.append(rel_path)

            # self.motion_names = [f for f in os.listdir(self.motion_file) if f.endswith(".json")]
            # self.motion_names.sort()  # 可选，保证顺序
            # self.motion_names = npz_files

            self.joint_pos_list = []
            self.body_pos_w_list = []
            self.body_quat_w_list = []
            self.fps_list = []
            data_list = []
            self.duration_list = []
            self.input_frames_list = []

            npz_file_paths = [os.path.join(self.motion_file, f) for f in self.motion_names]
            # import ipdb;ipdb.set_trace()
            for f_path in npz_file_paths:
                with open(f_path) as f:
                    data = json.load(f)
                    self.data_joint_order = data["data_joint_names"]
                    data_list.append(data)

            max_frame = max(len(data["dof_pos"]) for data in data_list)

            for data in data_list:
                j_pos = torch.tensor(data["dof_pos"], dtype=torch.float32, device=self.device)  # (num_frame, dof)
                b_pos = torch.tensor(data["root_trans"], dtype=torch.float32, device=self.device)  # (num_frame, 3)
                b_quat = torch.tensor(data["root_wxyz"], dtype=torch.float32, device=self.device)  # (num_frame, 4)
                fps = float(data["fps"])
                num_frame, dof = j_pos.shape

                self.joint_pos_list.append(j_pos.unsqueeze(0))  # (1, max_frame, dof)
                self.body_pos_w_list.append(b_pos.unsqueeze(0))  # (1, max_frame, dof)
                self.body_quat_w_list.append(b_quat.unsqueeze(0))  # (1, max_frame, dof)
                self.fps_list.append(fps)
                self.duration_list.append((num_frame - 1) / float(fps))
                self.input_frames_list.append(num_frame)
                self.motion_num = len(data_list)
            print(
                f"Motion loaded ({self.motion_file}), duration: {self.duration_list} sec, frames:"
                f" {self.input_frames_list}"
            )

        elif ext == ".npz":
            self.motion_names = []
            for root, _, files in os.walk(self.motion_file):
                for f in files:
                    if f.endswith(".npz"):
                        rel_path = os.path.relpath(os.path.join(root, f), self.motion_file)
                        self.motion_names.append(rel_path)

            # self.motion_names = [f for f in os.listdir(self.motion_file) if f.endswith(".npz")]
            # self.motion_names.sort()  # 可选，保证顺序
            # self.motion_names = npz_files

            self.joint_pos_list = []
            self.body_pos_w_list = []
            self.body_quat_w_list = []
            self.fps_list = []
            data_list = []
            self.duration_list = []
            self.input_frames_list = []

            npz_file_paths = [os.path.join(self.motion_file, f) for f in self.motion_names]
            # import ipdb;ipdb.set_trace()
            for f_path in npz_file_paths:
                data = np.load(f_path, allow_pickle=False)
                self.data_joint_order = [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ]
                data_list.append(data)
            # import ipdb;ipdb.set_trace()
            max_frame = max(data["qpos"].shape[0] for data in data_list)  # noqa: F841

            for data in data_list:
                j_pos = torch.tensor(data["qpos"][:, 7:], dtype=torch.float32, device=self.device)  # (num_frame, dof)
                b_pos = torch.tensor(data["qpos"][:, 4:7], dtype=torch.float32, device=self.device)  # (num_frame, 3)
                b_quat = torch.tensor(data["qpos"][:, 0:4], dtype=torch.float32, device=self.device)  # (num_frame, 4)
                fps = float(data["fps"])
                num_frame, dof = j_pos.shape

                self.joint_pos_list.append(j_pos.unsqueeze(0))  # (1, max_frame, dof)
                self.body_pos_w_list.append(b_pos.unsqueeze(0))  # (1, max_frame, dof)
                self.body_quat_w_list.append(b_quat.unsqueeze(0))  # (1, max_frame, dof)
                self.fps_list.append(fps)
                self.duration_list.append((num_frame - 1) / float(fps))
                self.input_frames_list.append(num_frame)
                self.motion_num = len(data_list)
            print(
                f"Motion loaded ({self.motion_file}), duration: {self.duration_list} sec, frames:"
                f" {self.input_frames_list}"
            )

        # print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(
        self, duration, output_dt, motion_base_poss_input, motion_base_rots_input, motion_dof_poss_input
    ):
        """Interpolates the motion to the output fps."""
        duration = float(duration)
        times = torch.arange(0, duration, output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times, duration, motion_dof_poss_input.shape[0])
        # import ipdb;ipdb.set_trace()

        motion_base_poss = self._lerp(
            motion_base_poss_input[index_0],
            motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        motion_base_rots = self._slerp(
            motion_base_rots_input[index_0],
            motion_base_rots_input[index_1],
            blend,
        )
        motion_dof_poss = self._lerp(
            motion_dof_poss_input[index_0],
            motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

        return motion_base_poss, motion_base_rots, motion_dof_poss
        # print(
        #     f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
        #     f" {self.output_frames}, output fps: {self.output_fps}"
        # )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor, duration: float, input_frames: int) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        # import ipdb;ipdb.set_trace()

        phase = times / duration
        index_0 = (phase * (input_frames - 1)).floor().long()
        index_1 = torch.clamp(index_0 + 1, max=input_frames - 1)
        blend = phase * (input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self, motion_base_poss, motion_base_rots, motion_dof_poss, dt):
        """Computes the velocities of the motion."""
        motion_base_lin_vels = torch.gradient(motion_base_poss, spacing=dt, dim=0)[0]
        motion_dof_vels = torch.gradient(motion_dof_poss, spacing=dt, dim=0)[0]
        motion_base_ang_vels = self._so3_derivative(motion_base_rots, dt)
        return motion_base_lin_vels, motion_base_ang_vels, motion_dof_vels

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
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        # import ipdb;ipdb.set_trace()
        motion_idx = torch.arange(self.motion_num, device=self.lerp_bp.device)  # [0,1,2,...]
        # print("self.lerp_bp",self.lerp_bp.shape)
        max_idx = torch.argmax(self.lerpfn)
        # print(max_idx)  # 输出最大值的索引
        # print("self.current_idx",self.current_idx[max_idx])
        state = (
            self.lerp_bp[motion_idx, self.current_idx],
            self.lerp_bq[motion_idx, self.current_idx],
            self.lerp_bv[motion_idx, self.current_idx],
            self.lerp_bw[motion_idx, self.current_idx],
            self.lerp_dp[motion_idx, self.current_idx],
            self.lerp_dv[motion_idx, self.current_idx],
        )

        self.current_idx += 1
        reset_flag = torch.zeros(self.motion_num, dtype=torch.bool, device=self.device)

        max_idx = torch.argmax(self.lerpfn)  # 获取最大值的索引

        print(
            "仿真进度： ", float(self.current_idx[max_idx] / self.lerpfn[max_idx]), "max length :", self.lerpfn[max_idx]
        )
        mask = self.current_idx >= self.lerpfn
        self.current_idx[mask] = 0
        reset_flag = mask
        self.finish_flag = self.finish_flag | reset_flag
        return state, reset_flag, self.finish_flag


from tqdm import tqdm


def run_simulator(
    motion: MotionLoader, sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]
):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------

    log = {
        "fps": [motion.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    # log_all = {"name": str, "log": log}
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
            finish_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        # root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        # pos_lookat = root_states[0, :3].cpu().numpy()
        # sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        _mask = ~finish_flag  # noqa: F841

        motion_log = {
            "joint_pos": robot.data.joint_pos[:].cpu().numpy().copy(),  # (num_motion, num_frame, dof)
            "joint_vel": robot.data.joint_vel[:].cpu().numpy().copy(),
            "body_pos_w": robot.data.body_pos_w[:].cpu().numpy().copy(),
            "body_quat_w": robot.data.body_quat_w[:].cpu().numpy().copy(),
            "body_lin_vel_w": robot.data.body_lin_vel_w[:].cpu().numpy().copy(),
            "body_ang_vel_w": robot.data.body_ang_vel_w[:].cpu().numpy().copy(),
        }

        # append 到 log_all 对应的字段
        if not torch.all(finish_flag):
            for k in motion_log.keys():
                log[k].append(motion_log[k])

        # 只创建一次保存目录
        parent_dir = os.path.dirname(motion.motion_file)
        # last_dir   = os.path.basename(motion.motion_file)
        # save_dir   = os.path.join(parent_dir, "tracking_npz_" + last_dir)
        save_dir = os.path.join(parent_dir, "tracking_npz_data")
        os.makedirs(save_dir, exist_ok=True)

        def sanitize_filename(fname: str) -> str:
            return os.path.splitext(fname)[0].replace("/", "^").replace("\\", "^")

        keys = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )

        if torch.all(finish_flag) and not file_saved:
            file_saved = True
            for i in tqdm(range(motion.motion_num), desc="Saving motions"):
                motion_file2_save = {"fps": [motion.output_fps]}
                for k in keys:
                    # 每次只收集当前 motion 的帧，避免 stack 全部帧
                    # log[k] 是 list，长度 = num_frame
                    frames_list = [frame[i].astype(np.float32) for frame in log[k]]  # 按帧收集
                    num_frames = int(motion.lerpfn[i].item())
                    motion_file2_save[k] = np.stack(frames_list[:num_frames], axis=0)  # shape: (num_frames, dof)

                # 构造文件名
                fname = motion.motion_names[i]
                name_no_ext = sanitize_filename(fname)
                save_path = os.path.join(save_dir, name_no_ext + ".npz")

                np.savez_compressed(save_path, **motion_file2_save)

            # os.makedirs(COLLECTION, exist_ok=True)   # 确保目录存在
            # np.savez(motion.motion_file, **log)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps

    sim = SimulationContext(sim_cfg)
    # Design scene
    motion = MotionLoader(
        # motion_file="/home/ubuntu/mgg_worspace/project/mgg_wbc/whole_body_parkour/to_real_data/getup2_test/json", #[edit]
        # motion_file="/home/ubuntu/mgg_worspace/project/dataset/amass_cr1s/cr1s/retargeted_data", #[edit]
        # motion_file="/home/ubuntu/mgg_worspace/project/mgg_imma/output/cr1s/lafan_walk_things", #[edit]
        motion_file="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/g1_lafan",
        # motion_file="/home/ubuntu/mgg_worspace/project/mgg_wbc/whole_body_parkour/to_real_data/run_test/json", #[edit]
        # input_fps=args_cli.input_fps,
        output_fps=50,
        device=sim.device,
    )
    scene_cfg = ReplayMotionsSceneCfg(num_envs=motion.motion_num, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        motion,
        sim,
        scene,
        joint_names=motion.data_joint_order,
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
