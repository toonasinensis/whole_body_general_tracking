"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
# parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")

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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

##
# Pre-defined configs
##
from whole_body_tracking.tasks.tracking.mdp import MotionCommandCfg, MotionLoader
from whole_body_tracking.terrains.config import GRAVEL_TERRAINS_CFG, STL_PLATFORM_TERRAINS_CFG  # noqa: F401

VELOCITY_BORN_RANGE = {
    "x": (-0.0, 0.0),
    "y": (-0.0, 0.0),
    "z": (-0.0, 0.0),
    "roll": (-0.0, 0.0),
    "pitch": (-0.0, 0.0),
    "yaw": (-0.0, 0.0),
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane",
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
            # emissive_color = (0.0, 0.0, 0.0),
            roughness=1.0,
            # metallic = 0.0,
            # opacity = 0.0,
        ),
        debug_vis=True,
    )
    # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


num_motion = 200


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # registry_name = args_cli.registry_name
    # if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
    #     registry_name += ":latest"
    # import pathlib

    # import wandb

    # api = wandb.Api()
    # artifact = api.artifact(registry_name)
    # motion_file = "/home/ubuntu/mgg_worspace/project/dataset/amass_cr1s/cr1s/hard_tracking_npz" # TODO
    motion_file = "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/tracking_npz_data/lafan"

    cfg = MotionCommandCfg(
        anchor_body_name="pelvis",
        motion_file=motion_file,
        body_names=[
            "pelvis",  # 0
            "left_hip_roll_link",  # 1
            "left_knee_link",  # 2
            "left_ankle_roll_link",  # 3
            "right_hip_roll_link",  # 4
            "right_knee_link",  # 5
            "right_ankle_roll_link",  # 6
            "torso_link",  # 7
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
        asset_name="robot",
        max_motion_num=num_motion,
        resampling_time_range=(1.0e9, 1.0e9),  # isaaclab自带的不想用
        resample_interval=1000 * 24,  # 真实interval, step次数
        debug_vis=False,
        pose_range={
            "x": (-0.00, 0.00),
            "y": (-0.00, 0.00),
            "z": (0.1, 0.15),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
        velocity_range=VELOCITY_BORN_RANGE,
        joint_position_range=(-0.0, 0.0),
    )

    # motion_file = "/home/ubuntu/mgg_worspace/project/mgg_wbc/whole_body_tracking/to_real_data/getup2_test/tracking_npz_data"

    motion = MotionLoader(
        cfg,
        12345,  # 占位
        sim.device,
    )
    motion.resample_motionloader(sim.device)
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # 打印可用动作信息
    for idx_print, file_name in enumerate(motion.file_names):
        print(f"Motion {idx_print}: {file_name}")

    # 为每个 env 分配 motion：env i -> motion (i % num_motions)
    num_motions = len(motion.time_step_start_idx)
    motion_ids = torch.arange(scene.num_envs, dtype=torch.long, device=sim.device) % num_motions
    for i in range(scene.num_envs):
        time_steps[i] = motion.time_step_start_idx[motion_ids[i]]

    import time

    while simulation_app.is_running():
        time_start = time.time()
        time_steps = time_steps + 1

        # 检查每个 env 是否达到了其对应动作的结束帧
        mask = (time_steps.unsqueeze(1) == motion.time_step_end_idx.unsqueeze(0) - 1).any(dim=1)

        for i in range(scene.num_envs):
            if not mask[i]:
                continue
            time_steps[i] = motion.time_step_start_idx[motion_ids[i]]

        root_states = robot.data.default_root_state.clone()

        # 获取 motion 中的 base pos（动作录制时的原始位置）
        base_pos = motion._body_pos_w[time_steps][:, 0].clone()  # [num_envs, 3]
        # import ipdb; ipdb.set_trace()
        # 根据 motion_id 添加位置偏移，实现不同动作在不同地形行
        # terrain_origins 形状通常为 [num_levels, num_rows, 3]，先展平再按 motion_id 映射
        terrain_origins = scene.terrain.terrain_origins.reshape(-1, 3)
        terrain_ids = motion_ids % terrain_origins.shape[0]  # prevent out of index
        base_pos += terrain_origins[terrain_ids]
        # base_pos[:, 1] += motion_id_offset * row_spacing  - ( row_spacing * (STL_PLATFORM_TERRAINS_CFG.num_cols - 1)) / 2  # 居中排列

        root_states[:, :3] = base_pos
        root_states[:, 3:7] = motion._body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion._body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion._body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)
        time_end = time.time()
        duration = time_end - time_start
        if duration < 0.02:
            time.sleep(0.02 - duration)
        # pos_lookat = root_states[0, :3].cpu().numpy()
        # sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=num_motion, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
