"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion npz (overrides wandb_path)")
parser.add_argument("--model_path", type=str, default=None, help="Direct path to model checkpoint file (.pt)")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
import copy

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
    else:
        # Allow local-only run: require --motion_file and a local checkpoint path via --load_run/--checkpoint
        if args_cli.motion_file is not None:
            env_cfg.commands.motion.motion_file = args_cli.motion_file
        
        # Use direct model path if provided, otherwise use the old logic
        if args_cli.model_path is not None:
            resume_path = args_cli.model_path
            print(f"[INFO]: Loading model checkpoint from direct path: {resume_path}")
        else:
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # -------------------------------------------------------------------------
    # Play mode: disable all training-only behaviours before env creation
    # -------------------------------------------------------------------------
    # 1. Disable randomisation/perturbation events selectively.
    #    add_joint_default_pos is KEPT (with zero noise) because it initialises
    #    default_joint_pos_nominal which is needed by rewards and the exporter.
    env_cfg.events.physics_material = None
    env_cfg.events.base_com = None
    env_cfg.events.waist_com = None
    env_cfg.events.link_com = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.add_waist_mass = None
    env_cfg.events.add_link_mass = None
    env_cfg.events.scale_actuator_gains = None
    env_cfg.events.scale_joint_parameters = None
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None
    # Keep add_joint_default_pos but zero the noise so it only saves the
    # nominal value without perturbing the default joint positions.
    env_cfg.events.add_joint_default_pos.params["pos_distribution_params"] = None

    # 2. Disable observation noise for every observation group.
    for group in vars(env_cfg.observations).values():
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False

    # 3. Zero out command randomisation (initial pose / velocity / joint offsets).
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    # Skip start-hold and end-hold so the motion plays straight through.
    env_cfg.commands.motion.start_hold_steps = 0
    env_cfg.commands.motion.end_hold_steps = 0

    # 4. Remove early-failure terminations; keep time_out as a safety net
    #    but set episode_length_s so long that it never fires in practice.
    env_cfg.terminations.anchor_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.ee_body_pos = None
    env_cfg.episode_length_s = 30
    # -------------------------------------------------------------------------

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # -------------------------------------------------------------------------
    # Play mode: force every reset to restart the motion from frame 0.
    # The motion command's _adaptive_sampling normally picks a random start
    # frame; we override _resample_command so that after the original robot-
    # state reset (which now applies zero noise because we cleared the ranges
    # above), all env_ids are snapped back to time_step = 0.
    # -------------------------------------------------------------------------
    _raw_env = env.unwrapped
    if hasattr(_raw_env, "command_manager"):
        _motion_cmd = _raw_env.command_manager.get_term("motion")
        _orig_resample = _motion_cmd._resample_command

        def _play_resample(env_ids):
            if len(env_ids) == 0:
                return
            # Run original logic to reset the robot pose/state at frame 0
            # (all randomisation ranges are zero, so no noise is applied).
            _orig_resample(env_ids)
            # Override the start frame: always begin from the first frame.
            _motion_cmd.time_steps[env_ids] = 0
            _motion_cmd.start_time[env_ids] = 0  # skip start-hold, enter motion phase immediately
            _motion_cmd.out_time[env_ids] = 0

        _motion_cmd._resample_command = _play_resample
    # -------------------------------------------------------------------------

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            print("obs shape:", obs.shape)
            print("obs[0] =", obs[0])       # 第0个env的观测
            print("actions =", actions[0])  # 第0个env的动作输出

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
