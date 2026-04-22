"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.
The aims of this script is to
1. The joints order in the motion files shall be aligned with the joints order in the isaaclab robot (check using joints)
2. The bodies order in the motion files shall be aligned with the bodies order in the isaaclab robot (check using body links error)
3. Visualize the motions in the isaaclab to show their feasibility

python preprocess/test_mimic_motions.py --robot_cfg g1 \
    --dataset_txt "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/all_top_files.txt" \
    --motion_dir "/home/thl/Documents/g1-mimic-npz"

"""

import argparse
import torch

from isaaclab.app import AppLauncher

# region robot configs
G1_MOTION_FILE = "/home/thl/Documents/g1-mimic-npz"
G1_MOTION_DATASET_TXT = "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/all_top_files.txt"
G1_ANCHOR_BODY_NAMES = "pelvis"
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

ROBAN_MOTION_FILE = "data/roban_motions/210531"
ROBAN_ANCHOR_BODY_NAMES = "waist_yaw_link"
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
    "g1": {
        "anchor_body_name": G1_ANCHOR_BODY_NAMES,
        "body_names": G1_BODY_NAMES,
        "default_motion_file": G1_MOTION_FILE,
        "default_motion_dataset_txt": G1_MOTION_DATASET_TXT,
    },
    "roban": {
        "anchor_body_name": ROBAN_ANCHOR_BODY_NAMES,
        "body_names": ROBAN_BODY_NAMES,
        "default_motion_file": ROBAN_MOTION_FILE,
        "default_motion_dataset_txt": "/home/thl/Documents/roban-mimic-npz",
    },
}
# endregion robot configs


# region add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")

parser.add_argument(
    "--robot_cfg",
    type=str,
    default="g1",
    choices=["g1", "roban"],
    help="Robot preset: g1 or roban.",
)
# 新增 motion_dir 和 dataset_txt 参数
parser.add_argument(
    "--motion_dir",
    type=str,
    default=None,
    help="根目录，txt 里的路径为相对路径时拼接用。默认用 preset 的 default_motion_file。",
)
parser.add_argument(
    "--dataset_txt",
    type=str,
    nargs="+",
    default=None,
    help="一个或多个 txt 文件，每行一个 npz 路径（相对 motion_dir 或绝对路径）",
)
# endregion add argparse arguments


# append AppLauncher cli args
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

##
# Pre-defined configs
##
from whole_body_tracking.tasks.tracking.mdp import MotionCommandCfg, MotionLoader
from whole_body_tracking.terrains.config import GRAVEL_TERRAINS_CFG  # noqa: F401


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

    from pathlib import Path

    preset = ROBOT_PRESETS[args_cli.robot_cfg]
    body_names = list(preset["body_names"])
    if len(body_names) == 0:
        raise ValueError("--body-names must contain at least one body name")
    anchor_body_name = preset["anchor_body_name"]
    if anchor_body_name not in body_names:
        raise ValueError(f"anchor_body_name '{anchor_body_name}' must be in body_names preset")
    motion_anchor_body_index = body_names.index(anchor_body_name)
    # Build body index mapping: motion body_names -> IsaacLab body indices.
    robot_body_indexes = robot.find_bodies(body_names, preserve_order=True)[0]

    # 处理 motion 文件列表
    motion_dir = args_cli.motion_dir or preset["default_motion_file"]
    motion_dir = str(Path(motion_dir).expanduser().resolve())
    dataset_txt = args_cli.dataset_txt or preset.get("default_motion_dataset_txt")
    if isinstance(dataset_txt, list) and len(dataset_txt) == 1:
        dataset_txt = dataset_txt[0]

    cfg = MotionCommandCfg(
        anchor_body_name=anchor_body_name,
        motion_file=motion_dir,
        dataset_txt=dataset_txt,
        body_names=body_names,
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

    # region Print joint/body order as defined by IsaacLab.
    print("\n[IsaacLab] joint_names (order):")
    for i, n in enumerate(getattr(robot, "joint_names", [])):
        print(f"  {i:03d}: {n}")
    print("\n[IsaacLab] body_names (order):")
    for i, n in enumerate(getattr(robot, "body_names", [])):
        print(f"  {i:03d}: {n}")
    root_name = getattr(robot, "root_name", None)
    if root_name is None and getattr(robot, "body_names", None):
        root_name = robot.body_names[0]
    print(f"\n[IsaacLab] root body name: {root_name}\n")
    # Print controllable (actuated) joint names in IsaacLab joint order.
    actuated_joint_ids: set[int] = set()
    actuators = getattr(robot, "actuators", None)
    if isinstance(actuators, dict):
        for _act_name, act in actuators.items():
            # Most actuator classes expose joint_ids (tensor/list) or joint_names.
            jids = getattr(act, "joint_ids", None)
            if jids is not None:
                try:
                    if isinstance(jids, torch.Tensor):
                        jids = jids.detach().cpu().tolist()
                    actuated_joint_ids.update(int(x) for x in jids)
                    continue
                except Exception:
                    pass
            jnames = getattr(act, "joint_names", None)
            if jnames is not None:
                try:
                    lab_names = list(getattr(robot, "joint_names", []))
                    name_to_id = {n: i for i, n in enumerate(lab_names)}
                    actuated_joint_ids.update(int(name_to_id[n]) for n in jnames if n in name_to_id)
                except Exception:
                    pass

    lab_joint_names = list(getattr(robot, "joint_names", []))
    controllable_ids_sorted = [i for i in range(len(lab_joint_names)) if i in actuated_joint_ids]
    print("[IsaacLab] controllable joint_names (actuated, order):")
    for i in controllable_ids_sorted:
        print(f"  {i:03d}: {lab_joint_names[i]}")
    print(f"[IsaacLab] controllable joint count: {len(controllable_ids_sorted)}\n")
    # endregion Print joint/body order as defined by IsaacLab.

    # 打印可用动作信息
    # for idx_print, file_name in enumerate(motion.file_names):
    #     print(f"Motion {idx_print}: {file_name}")

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
        base_pos = motion.body_pos_w[time_steps][:, 0].clone()  # [num_envs, 3]
        # import ipdb; ipdb.set_trace()
        # 根据 motion_id 添加位置偏移，实现不同动作在不同地形行
        # terrain_origins 形状通常为 [num_levels, num_rows, 3]，先展平再按 motion_id 映射
        terrain_origins = scene.terrain.terrain_origins.reshape(-1, 3)
        terrain_ids = motion_ids % terrain_origins.shape[0]  # prevent out of index
        base_pos += terrain_origins[terrain_ids]
        # base_pos[:, 1] += motion_id_offset * row_spacing  - ( row_spacing * (STL_PLATFORM_TERRAINS_CFG.num_cols - 1)) / 2  # 居中排列
        root_states[:, :3] = base_pos
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()

        # Validate body pos/quat match motion (in IsaacLab body order corresponding to `body_names`).
        # Note: motion.body_* are stored in motion file order (given by preset `body_names`).
        robot_body_quat_w = robot.data.body_quat_w[:, robot_body_indexes]
        # Motion tensors need env origin added (same convention as in tracking command implementation).
        motion_body_quat_w = motion.body_quat_w[time_steps]
        # Quaternion angular error (radians): 2 * acos(|dot(q1,q2)|)
        dot = torch.sum(robot_body_quat_w * motion_body_quat_w, dim=-1).abs().clamp(0.0, 1.0)
        ang_err = 2.0 * torch.acos(dot)
        # Print a short summary occasionally (env0 only)
        if int(time_steps[0].item()) % 200 == 0:
            print(
                f"[t={int(time_steps[0].item())}] "
                f"ang_err(env0) mean={ang_err[0].mean().item():.6f} max={ang_err[0].max().item():.6f}"
            )

        sim.render()
        scene.update(sim_dt)
        time_end = time.time()
        duration = time_end - time_start
        if duration < 0.02:
            time.sleep(0.02 - duration)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=num_motion, env_spacing=2.0)
    robot_cfg_map = {
        "g1": G1_CYLINDER_CFG,
        "roban": RobanS22_CYLINDER_CFG,
    }
    scene_cfg.robot = robot_cfg_map[args_cli.robot_cfg].replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
