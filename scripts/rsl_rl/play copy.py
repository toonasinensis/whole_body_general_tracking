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
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--dataset_txt", type=str, default=None, help="Path to the motion dataset_txt.")
parser.add_argument("--smpl_file_path", type=str, default=None, help="Path to the SMPL file directory.")
parser.add_argument(
    "--encoder_mode",
    type=str,
    default=None,
    choices=["split", "robot", "latent", "g1", "smpl", "encoder_g1", "encoder_smpl"],
    help=(
        "Force MyModel encoder routing during play. 'split' keeps the original batch split, "
        "'robot' uses encoder_g1 for all envs, and 'latent' uses encoder_smpl for all envs."
    ),
)
parser.add_argument(
    "--fsq_sample_mode",
    type=str,
    default=None,
    choices=["encode", "normal", "none", "random", "shuffle"],
    help=(
        "Override MyModel FSQ latent source during play. 'encode' keeps encoder->FSQ, "
        "'random' samples legal FSQ codes, and 'shuffle' permutes quantized latents across the batch."
    ),
)

parser.add_argument("--resume_path", type=str, default=None, help="Path to the model file.")
parser.add_argument(
    "--tb_log_dir",
    type=str,
    default=None,
    help="TensorBoard output directory used when --logger=tensorboard. Defaults to <checkpoint_dir>/tensorboard/play.",
)
parser.add_argument(
    "--tb_log_interval",
    type=int,
    default=1,
    help="Log command metrics to TensorBoard every N env steps when --logger=tensorboard.",
)
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export the loaded policy to ONNX.")
parser.add_argument(
    "--onnx_dir",
    type=str,
    default=None,
    help="Output directory for exported ONNX. Defaults to <checkpoint_dir>/exported.",
)
parser.add_argument("--onnx_filename", type=str, default="policy.onnx", help="Filename for exported ONNX.")
parser.add_argument("--export_only", action="store_true", default=False, help="Exit after exporting ONNX.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = False  # True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

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
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import (  # noqa: F401
    export_grouped_motion_policy_as_onnx,
    resolve_policy_observation_groups,
)


def _set_backbone_encoder_mode(policy, encoder_mode: str) -> None:
    """Set MyModel encoder routing without adding backbone-specific API to ActorModel."""
    backbone = getattr(policy, "backbone", policy)
    if not hasattr(backbone, "set_encoder_mode"):
        raise AttributeError(f"Loaded policy backbone type {type(backbone).__name__} does not support --encoder_mode.")
    backbone.set_encoder_mode(encoder_mode)
    active_mode = backbone.get_encoder_mode() if hasattr(backbone, "get_encoder_mode") else encoder_mode
    print(f"[INFO]: MyModel encoder mode set to: {active_mode}")


def _try_set_backbone_encoder_mode(policy, encoder_mode: str) -> bool:
    backbone = getattr(policy, "backbone", policy)
    if not hasattr(backbone, "set_encoder_mode"):
        return False
    _set_backbone_encoder_mode(policy, encoder_mode)
    return True


def _get_backbone_encoder_mode(policy, fallback: str | None = None) -> str:
    backbone = getattr(policy, "backbone", policy)
    if hasattr(backbone, "get_encoder_mode"):
        return backbone.get_encoder_mode()
    return fallback or "split"


def _set_backbone_fsq_sample_mode(policy, fsq_sample_mode: str) -> None:
    """Set MyModel FSQ sampling without adding backbone-specific API to ActorModel."""
    backbone = getattr(policy, "backbone", policy)
    if not hasattr(backbone, "set_fsq_sample_mode"):
        raise AttributeError(
            f"Loaded policy backbone type {type(backbone).__name__} does not support --fsq_sample_mode."
        )
    backbone.set_fsq_sample_mode(fsq_sample_mode)
    active_mode = backbone.get_fsq_sample_mode() if hasattr(backbone, "get_fsq_sample_mode") else fsq_sample_mode
    print(f"[INFO]: MyModel FSQ sample mode set to: {active_mode}")


def _get_backbone_fsq_sample_mode(policy, fallback: str | None = None) -> str:
    backbone = getattr(policy, "backbone", policy)
    if hasattr(backbone, "get_fsq_sample_mode"):
        return backbone.get_fsq_sample_mode()
    return fallback or "encode"


def _to_serializable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _first_env_value(value):
    if torch.is_tensor(value) and value.ndim > 1:
        return value[0]
    return value


def _maybe_first_env_list(value, env_id: int = 0):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim > 1:
            value = value[env_id]
        return value.tolist()
    return value


def _collect_observation_terms_metadata(base_env, observation_groups: list[str]) -> dict:
    obs_manager = base_env.observation_manager
    active_terms = obs_manager.active_terms
    term_dims = obs_manager.group_obs_term_dim
    concatenate = obs_manager.group_obs_concatenate
    term_cfgs = getattr(obs_manager, "_group_obs_term_cfgs", {})

    out = {}
    for group_name in observation_groups:
        names = list(active_terms.get(group_name, []))
        dims = list(term_dims.get(group_name, []))
        cfgs = list(term_cfgs.get(group_name, []))
        terms = []
        for idx, name in enumerate(names):
            shape = list(dims[idx]) if idx < len(dims) else []
            cfg = cfgs[idx] if idx < len(cfgs) else None
            history_length = int(getattr(cfg, "history_length", 0)) if cfg is not None else 0
            flatten_history_dim = bool(getattr(cfg, "flatten_history_dim", True)) if cfg is not None else True
            base_shape = list(shape)
            if history_length > 0:
                if flatten_history_dim and len(shape) == 1 and shape[0] % history_length == 0:
                    base_shape = [shape[0] // history_length]
                elif not flatten_history_dim and len(shape) >= 1 and shape[0] == history_length:
                    base_shape = shape[1:]
            terms.append(
                {
                    "name": name,
                    "shape": shape,
                    "base_shape": base_shape,
                    "history_length": history_length,
                    "flatten_history_dim": flatten_history_dim,
                }
            )
        out[group_name] = {
            "concatenate_terms": bool(concatenate.get(group_name, True)),
            "terms": terms,
        }
    return out


def _collect_onnx_metadata(vec_env, base_env, policy, encoder_mode: str, fsq_sample_mode: str) -> dict:
    action_term = base_env.action_manager.get_term("joint_pos")
    robot = base_env.scene["robot"]
    obs = vec_env.get_observations()
    observation_groups = resolve_policy_observation_groups(policy, obs)
    scale = _first_env_value(getattr(action_term, "_scale", 1.0))
    offset = _first_env_value(getattr(action_term, "_offset", 0.0))
    default_joint_pos = robot.data.default_joint_pos[0]
    default_joint_vel = getattr(robot.data, "default_joint_vel", torch.zeros_like(robot.data.joint_vel))[0]
    joint_stiffness = _maybe_first_env_list(getattr(robot.data, "joint_stiffness", []))
    joint_damping = _maybe_first_env_list(getattr(robot.data, "joint_damping", []))
    return {
        "encoder_mode": encoder_mode,
        "fsq_sample_mode": fsq_sample_mode,
        "input_names": observation_groups,
        "input_groups": observation_groups,
        "observation_groups": observation_groups,
        "observation_shapes": {name: list(obs[name].shape[1:]) for name in observation_groups},
        "observation_terms": _collect_observation_terms_metadata(base_env, observation_groups),
        "action_joint_names": list(getattr(action_term, "_joint_names", robot.data.joint_names)),
        "robot_joint_names": list(robot.data.joint_names),
        "default_joint_pos": _to_serializable(default_joint_pos),
        "default_joint_vel": _to_serializable(default_joint_vel),
        "joint_stiffness": joint_stiffness,
        "joint_damping": joint_damping,
        "action_scale": _to_serializable(scale),
        "action_offset": _to_serializable(offset),
        "motion_body_names": list(base_env.cfg.commands.motion.body_names),
        "anchor_body_name": base_env.cfg.commands.motion.anchor_body_name,
        "future_step_num": list(base_env.cfg.commands.motion.future_step_num),
        "decimation": int(base_env.cfg.decimation),
        "sim_dt": float(base_env.cfg.sim.dt),
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.terminations.ee_body_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.anchor_pos = None
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        pass
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        resume_path = args_cli.resume_path

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file
            print(
                f"[INFO]: Overriding motion file in the environment config with: {env_cfg.commands.motion.motion_file}"
            )

        if args_cli.dataset_txt is not None:
            print(f"[INFO]: Using motion file filter from CLI: {args_cli.dataset_txt}")
            env_cfg.commands.motion.dataset_txt = args_cli.dataset_txt
            print(
                "[INFO]: Overriding motion file filter in the environment config with:"
                f" {env_cfg.commands.motion.dataset_txt}"
            )

        if args_cli.smpl_file_path is not None:
            print(f"[INFO]: Using SMPL file from CLI: {args_cli.smpl_file_path}")
            env_cfg.commands.motion.smpl_file_path = args_cli.smpl_file_path
            print(
                f"[INFO]: Overriding SMPL file in the environment config with: {env_cfg.commands.motion.smpl_file_path}"
            )

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    env_cfg.episode_length_s = 9999
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

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    effective_encoder_mode = args_cli.encoder_mode or "robot"
    if args_cli.encoder_mode is not None:
        _set_backbone_encoder_mode(policy, effective_encoder_mode)
    elif args_cli.export_onnx:
        _try_set_backbone_encoder_mode(policy, effective_encoder_mode)
    effective_encoder_mode = _get_backbone_encoder_mode(policy, effective_encoder_mode)
    if args_cli.fsq_sample_mode is not None:
        _set_backbone_fsq_sample_mode(policy, args_cli.fsq_sample_mode)
    effective_fsq_sample_mode = _get_backbone_fsq_sample_mode(policy, args_cli.fsq_sample_mode)

    base_env = env.unwrapped
    motion_cmd = None
    if hasattr(base_env, "command_manager"):
        try:
            motion_cmd = base_env.command_manager.get_term("motion")
        except Exception:
            motion_cmd = None

    tb_writer = None
    tb_log_interval = max(1, args_cli.tb_log_interval)
    if args_cli.logger == "tensorboard":
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as err:
            raise ImportError("TensorBoard logging requested, but 'tensorboard' is not installed.") from err

        tb_log_dir = args_cli.tb_log_dir or os.path.join(log_dir, "tensorboard", "play")
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"[INFO] TensorBoard logging enabled. log_dir: {tb_log_dir}")
        if motion_cmd is None:
            print("[WARN] Could not find command term 'motion'; command metrics will not be logged.")

    # export policy to onnx/jit
    export_model_dir = args_cli.onnx_dir or os.path.join(os.path.dirname(resume_path), "exported")
    if args_cli.export_onnx:
        onnx_path = os.path.join(export_model_dir, args_cli.onnx_filename)
        print(f"[INFO]: Exporting ONNX policy to: {onnx_path}")
        onnx_path = export_grouped_motion_policy_as_onnx(
            env,
            policy,
            path=export_model_dir,
            filename=args_cli.onnx_filename,
            metadata=_collect_onnx_metadata(
                env, env.unwrapped, policy, effective_encoder_mode, effective_fsq_sample_mode
            ),
        )
        print(f"[INFO]: Exported ONNX policy: {onnx_path}")
        if args_cli.export_only:
            env.close()
            return

    # reset environment
    # import ipdb; ipdb.set_trace()
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            if isinstance(actions, dict) and "actions" in actions:
                actions = actions["actions"]
            obs, _, _, _ = env.step(actions)

        timestep += 1
        if tb_writer is not None and motion_cmd is not None and timestep % tb_log_interval == 0:
            for key, value in motion_cmd.metrics.items():
                if torch.is_tensor(value):
                    scalar = float(value.mean().item())
                else:
                    scalar = float(value)
                tb_writer.add_scalar(f"command/{key}", scalar, global_step=timestep)

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
