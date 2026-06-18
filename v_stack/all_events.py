from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

# Isaac Lab 2.3 on Isaac Sim 4.5 exposes quat_rotate_inverse instead of quat_apply_inverse.
if not hasattr(math_utils, "quat_apply_inverse"):
    math_utils.quat_apply_inverse = math_utils.quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def assist_fallen_robots_with_upward_force(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    command_name: str = "motion",
    force: float = 1250.0,
    force_mode: Literal["instantaneous", "permanent"] = "permanent",
    min_force_scale: float = 0.0,
    z_error_threshold: float = 0.15,
    max_height_above_reference: float = 0.05,
    max_upward_velocity: float = 1.0,
    gravity_z_threshold: float = 0.8,
    use_motion_pose_range_mask: bool = True,
    log_metrics: bool = False,
    debug_steps: int = 0,
    debug_interval_steps: int = 20,
    debug_env_id: int = 0,
) -> None:
    """Apply an upward recovery assist force and decay it with the global timeout average."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    if env_ids.numel() == 0 or force <= 0.0:
        return

    command = env.command_manager.get_term(command_name)
    active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=asset.device)
    if use_motion_pose_range_mask:
        pose_mask = getattr(env, "_motion_pose_range_env_mask", None)
        if pose_mask is None:
            return
        active_mask = pose_mask.to(device=asset.device, dtype=torch.bool)

    active_env_ids = env_ids[active_mask[env_ids]]
    if active_env_ids.numel() == 0:
        return

    # Use a cumulative timeout average over the assisted env subset as the curriculum signal.
    # Per-episode progress makes the force jump back to full strength after every reset.
    stats = getattr(env, "_fallen_upward_assist_timeout_stats", None)
    if stats is None:
        stats = {
            "timeouts": torch.zeros((), dtype=torch.float32, device=asset.device),
            "resets": torch.zeros((), dtype=torch.float32, device=asset.device),
            "last_step": -1,
        }
        setattr(env, "_fallen_upward_assist_timeout_stats", stats)

    # Interval events can run every control step. Count resets once per env step, after termination/reset.
    current_step = int(getattr(env, "common_step_counter", 0))
    if stats["last_step"] != current_step:
        reset_buf = getattr(env, "reset_buf", None)
        timeout_buf = getattr(env, "reset_time_outs", None)
        if reset_buf is not None and timeout_buf is not None:
            assisted_reset_ids = torch.where(reset_buf.to(asset.device) & active_mask)[0]
            if assisted_reset_ids.numel() > 0:
                stats["resets"] += float(assisted_reset_ids.numel())
                stats["timeouts"] += timeout_buf[assisted_reset_ids].to(asset.device, dtype=torch.float32).sum()
        stats["last_step"] = current_step

    timeout_average = stats["timeouts"] / torch.clamp(stats["resets"], min=1.0)
    force_scale = torch.clamp(1.0 - timeout_average, min=float(min_force_scale), max=1.0)

    force_body_ids = asset_cfg.body_ids
    if isinstance(force_body_ids, slice):
        if asset_cfg.body_names is not None:
            force_body_ids, _ = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
        else:
            force_body_ids = [asset.body_names.index(command.cfg.anchor_body_name)]
    elif isinstance(force_body_ids, int):
        force_body_ids = [force_body_ids]
    else:
        force_body_ids = list(force_body_ids)
    if len(force_body_ids) != 1:
        raise ValueError(
            "assist_fallen_robots_with_upward_force expects exactly one force body. "
            f"Got body_ids={force_body_ids} from asset_cfg={asset_cfg}."
        )
    force_body_id = int(force_body_ids[0])
    force_body_name = asset.body_names[force_body_id]

    # Only push while the assisted robot is low or tilted, and stop as soon as it is
    # already above the reference height or moving upward quickly.  Without this
    # gate, a large permanent force keeps firing after overshoot because
    # abs(reference_z - robot_z) is also large when the robot is too high.
    anchor_body_id = asset.body_names.index(command.cfg.anchor_body_name)
    robot_anchor_pos = asset.data.body_pos_w[:, anchor_body_id]
    robot_anchor_quat = asset.data.body_quat_w[:, anchor_body_id]
    robot_anchor_lin_vel = asset.data.body_lin_vel_w[:, anchor_body_id]
    projected_gravity = math_utils.quat_apply_inverse(robot_anchor_quat, asset.data.GRAVITY_VEC_W)
    z_error = command.anchor_pos_w[:, 2] - robot_anchor_pos[:, 2]
    too_low = z_error > z_error_threshold
    tilted = projected_gravity[:, 2] > -gravity_z_threshold
    too_high = robot_anchor_pos[:, 2] > command.anchor_pos_w[:, 2] + max_height_above_reference
    rising_fast = robot_anchor_lin_vel[:, 2] > max_upward_velocity
    fallen = (too_low | tilted) & ~too_high & ~rising_fast
    assist_env_ids = active_env_ids[fallen[active_env_ids]]

    forces = torch.zeros((active_env_ids.numel(), 1, 3), device=asset.device)
    torques = torch.zeros_like(forces)
    if assist_env_ids.numel() > 0:
        active_lookup = torch.searchsorted(active_env_ids, assist_env_ids)
        forces[active_lookup, 0, 2] = float(force) * force_scale

    if force_mode == "instantaneous":
        # Applied for one physics substep only. With decimation=4, the average force is roughly force / 4.
        composer = asset.instantaneous_wrench_composer
        duty_cycle = 1.0 / max(1, int(getattr(env.cfg, "decimation", 1)))
    elif force_mode == "permanent":
        # Applied across the full control step. Reset first so global-to-local conversion uses the current link pose.
        composer = asset.permanent_wrench_composer
        composer.reset(active_env_ids)
        duty_cycle = 1.0
    else:
        raise ValueError(f"Unsupported force_mode={force_mode!r}. Expected 'instantaneous' or 'permanent'.")

    composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=[force_body_id],
        env_ids=active_env_ids,
        is_global=True,
    )
    if debug_steps > 0 and current_step <= debug_steps and current_step % max(1, debug_interval_steps) == 0:
        dbg_env = int(max(0, min(debug_env_id, env.num_envs - 1)))
        if active_mask[dbg_env]:
            force_local = composer.composed_force_as_torch[dbg_env, force_body_id].detach()
            link_quat = asset.data.body_link_quat_w[dbg_env, force_body_id]
            force_world = math_utils.quat_apply(link_quat, force_local)
            max_force = float((float(force) * force_scale).item())
            dbg_force_matches = torch.where(active_env_ids == dbg_env)[0]
            applied_force = 0.0
            if dbg_force_matches.numel() > 0:
                applied_force = float(forces[dbg_force_matches[0], 0, 2].item())
            print(
                "[fallen_upward_assist] "
                f"step={current_step} env={dbg_env} mode={force_mode} duty={duty_cycle:.3f} "
                f"force_body={force_body_name} "
                f"anchor_body={asset.body_names[anchor_body_id]} "
                f"fallen={bool(fallen[dbg_env].item())} "
                f"too_low={bool(too_low[dbg_env].item())} "
                f"tilted={bool(tilted[dbg_env].item())} "
                f"too_high={bool(too_high[dbg_env].item())} "
                f"rising_fast={bool(rising_fast[dbg_env].item())} "
                f"force_max_w={[0.0, 0.0, max_force]} "
                f"force_set_w={[0.0, 0.0, applied_force]} "
                f"force_avg_w={[0.0, 0.0, applied_force * duty_cycle]} "
                f"force_local={force_local.detach().cpu().tolist()} "
                f"force_world_from_local={force_world.detach().cpu().tolist()} "
                f"ref_z={float(command.anchor_pos_w[dbg_env, 2].item()):.4f} "
                f"pelvis_z={float(robot_anchor_pos[dbg_env, 2].item()):.4f} "
                f"z_error={float(z_error[dbg_env].item()):.4f} "
                f"pelvis_vz={float(robot_anchor_lin_vel[dbg_env, 2].item()):.4f} "
                f"projected_gravity_z={float(projected_gravity[dbg_env, 2].item()):.4f} "
                f"timeout_avg={float(timeout_average.item()):.4f}"
            )
    if log_metrics:
        env.extras.setdefault("log", {})
        env.extras["log"]["fallen_upward_assist/force_mean"] = forces[..., 2].mean().item()
        env.extras["log"]["fallen_upward_assist/env_ratio"] = assist_env_ids.numel() / max(1, active_env_ids.numel())
        env.extras["log"]["fallen_upward_assist/timeout_average"] = timeout_average.item()
        env.extras["log"]["fallen_upward_assist/force_scale"] = force_scale.item()
