"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/xxxx/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch
import os
from collections.abc import Sequence

from isaaclab.app import AppLauncher

# NPZ文件集合（在脚本中硬编码）
MOTION_FILES = [
    "assets/motions/Take_2025_11_09_10_12_14_b_copy_with_default25_trunk_interp.npz",
]

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, required=False, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion npz (overrides registry, if not provided, will use MOTION_FILES list)")

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
from isaaclab.utils.math import quat_rotate

##
# Pre-defined configs
##
import whole_body_tracking.robots.roban_s2 as roban_assets
from whole_body_tracking.robots.roban_s2 import RobanS2_CYLINDER_CFG
# from tasks.tracking.mdp import MotionLoader


def reorder_joint_data_for_isaac(joint_data: torch.Tensor, npz_joint_order: list[str], isaac_joint_names: list[str]) -> torch.Tensor:
    """
    将NPZ文件中的关节数据重排序为IsaacLab期望的顺序
    
    Args:
        joint_data: NPZ文件中的关节数据 (按自定义顺序排列)
        target_joint_order: NPZ文件中使用的关节顺序
        isaac_joint_names: IsaacLab机器人的关节顺序
    
    Returns:
        重排序后的关节数据 (按IsaacLab顺序排列)
    """
    # 创建从NPZ顺序到IsaacLab顺序的映射
    reorder_indices = []
    
    for isaac_joint in isaac_joint_names:
        if isaac_joint in npz_joint_order:
            npz_index = npz_joint_order.index(isaac_joint)
            reorder_indices.append(npz_index)
        else:
            raise ValueError(f"关节 '{isaac_joint}' 在NPZ文件的关节顺序中未找到")
    
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

    robot: ArticulationCfg = RobanS2_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Determine motion files: prefer --motion_file, otherwise use MOTION_FILES list, or WandB registry
    if args_cli.motion_file is not None:
        motion_files = [args_cli.motion_file]
    elif args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        import pathlib
        import wandb
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_files = [str(pathlib.Path(artifact.download()) / "motion.npz")]
    else:
        # 使用脚本中定义的MOTION_FILES集合
        motion_files = MOTION_FILES
    
    # 确保所有文件路径都是绝对路径（相对于脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)  # scripts的父目录是workspace root
    motion_files = [
        os.path.abspath(os.path.join(workspace_root, f)) if not os.path.isabs(f) else f 
        for f in motion_files
    ]
    
    # 验证文件存在
    for f in motion_files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Motion file not found: {f}")
    
    print(f"\n{'='*60}")
    print(f"总共找到 {len(motion_files)} 个NPZ文件，将循环播放")
    print(f"{'='*60}\n")
    
    npz_joint_order = roban_assets.PRESERVE_JOINT_ORDER_ASSET_CFG.joint_names

    # 相机自动跟随标志
    camera_follow = False
    initial_camera_set = False
    camera_distance = 3.0
    camera_height = 1.2
    camera_lookahead = 0.5

    def get_camera_pose(root_state: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        pos = root_state[:3]
        quat = root_state[3:7]
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=root_state.device).unsqueeze(0)
        forward = quat_rotate(quat.unsqueeze(0), forward_local).squeeze(0)
        forward = forward / (torch.linalg.norm(forward) + 1e-8)
        pos_np = pos.cpu().numpy()
        forward_np = forward.cpu().numpy()
        camera_pos = pos_np - forward_np * camera_distance
        # camera_pos[2] += camera_height
        camera_pos[1] = 1.0
        look_at = pos_np + forward_np * camera_lookahead
        return camera_pos, look_at

    # 循环播放所有NPZ文件
    current_file_index = 0
    motion = None
    time_steps = None
    
    cam_pos = [1,1,1]
    cam_look = [0,0,0]
    sim.set_camera_view(cam_pos, cam_look)
    initial_camera_set = True
    while simulation_app.is_running():
        # 如果当前motion文件播放完毕，切换到下一个文件
        if motion is None or (time_steps is not None and time_steps[0].item() >= motion.time_step_total - 1):
            # 加载下一个motion文件
            motion_file = motion_files[current_file_index]
            
            print(f"\n{'='*60}")
            print(f"[{current_file_index + 1}/{len(motion_files)}] 正在加载NPZ文件:")
            print(f"  {os.path.basename(motion_file)}")
            print(f"  完整路径: {motion_file}")
            print(f"{'='*60}\n")
            
            motion = MotionLoader(
                motion_file,
                torch.tensor([0], dtype=torch.long, device=sim.device),
                sim.device,
            )
            time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
            initial_camera_set = False  # 重置相机设置，为新文件设置初始视角
            
            # 移动到下一个文件索引（循环）
            current_file_index = (current_file_index + 1) % len(motion_files)

        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        
        # 重排序关节数据：从NPZ顺序转换为IsaacLab顺序
        npz_joint_pos = motion.joint_pos[time_steps]
        npz_joint_vel = motion.joint_vel[time_steps]
        
        isaac_joint_pos = reorder_joint_data_for_isaac(npz_joint_pos, npz_joint_order, robot.joint_names)
        isaac_joint_vel = reorder_joint_data_for_isaac(npz_joint_vel, npz_joint_order, robot.joint_names)
        
        robot.write_joint_state_to_sim(isaac_joint_pos, isaac_joint_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        # if not initial_camera_set or camera_follow:
        #     cam_pos, cam_look = get_camera_pose(root_states[0])
        #     sim.set_camera_view(cam_pos, cam_look)
        #     initial_camera_set = True


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
