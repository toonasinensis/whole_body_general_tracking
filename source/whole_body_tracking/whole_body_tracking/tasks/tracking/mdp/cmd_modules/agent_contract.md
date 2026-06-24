# Command Modules Agent Contract

`cmd_modules` owns the motion command runtime. Its top-level public API is the
`__all__` list in `cmd_modules/__init__.py`.

## Public Exports

- `MotionCommand`, `MotionCommandCfg`, `MotionLoader`
- `MotionDataSource`, `MotionSelection`
- `MotionReferenceCache`
- `MotionCommandResetter`
- `MotionCommandTimeline`
- `MotionCommandDebugVisualizer`
- `AdaptiveMotionSampler`
- `MotionSelectionPolicy`, `FixedEvalMotionSelectionPolicy`,
  `AdaptiveMotionSelectionPolicy`, `UnsupportedMotionSelectionPolicy`,
  `create_motion_selection_policy`
- Sampling helpers: `validate_motion_local_frame_bounds`,
  `compute_motion_sample_bin_counts`, `accumulate_rewinded_motion_sample_bin_counts`,
  `sample_motion_local_times_from_bins`, `sample_rewinded_motion_local_times`

These exports are available through both `mdp.cmd_modules` and the package facade
`mdp`.

## `MotionCommand`

`MotionCommand` is the IsaacLab `CommandTerm` orchestrator. It wires together motion
source loading, timeline selection, reset, reference cache refresh, metrics, and
debug visualization.

Public command-facing properties read by observations, rewards, terminations, and
events:

- Current reference: `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`,
  `body_lin_vel_w`, `body_ang_vel_w`, `anchor_pos_w`, `anchor_quat_w`,
  `anchor_lin_vel_w`, `anchor_ang_vel_w`, `anchor_pos_z`,
  `anchor_project_gravity`, `anchor_6d_rotation`.
- Future reference: `num_future_frames`, `expanded_future_motion_ids`, `global_future_steps`,
  `future_time_steps_init`, `anchor_pos_w_future`, `anchor_quat_w_future`,
  `joint_pos_future`, `joint_vel_future`, `joint_vel_multi_future`.
- Timeline metadata: `global_time_steps`, `global_start_steps`,
  `motion_num_steps`, `motion_ids_from_timestamps()`.
- SMPL reference: `has_smpl_data`, `smpl_joints`, `smpl_transl`, `smpl_poses`,
  `smpl_poses_future`, `smpl_joints_future`, `smpl_transl_future`,
  `smpl_global_position`, `smpl_global_position_future`,
  `smpl_root_quat_w`, `smpl_root_quat_w_multi_future`,
  `smpl_root_quat_w_dif_l_multi_future`, `smpl_joints_local_multi_future`,
  `get_smpl_joints()`, `get_smpl_transl()`, `get_smpl_pose()`,
  `get_smpl_global_position()`.
- Robot state views: `robot_joint_pos`, `robot_joint_vel`, `robot_body_pos_w`,
  `robot_body_quat_w`, `robot_body_lin_vel_w`, `robot_body_ang_vel_w`,
  `robot_anchor_pos_w`, `robot_anchor_quat_w`, `robot_anchor_lin_vel_w`,
  `robot_anchor_ang_vel_w`.
- Aligned reference cache exposed for reward/termination: `body_pos_relative_w`,
  `body_quat_relative_w`.
- Policy command tensor: `command`.

Allowed side effects:

- Owns `self.metrics`.
- Resamples command timelines.
- Calls `MotionCommandResetter.apply()` during command resets/resamples.
- Refreshes `MotionReferenceCache`.
- Sets `env.envs_classes_mask` for recovery assist and delayed termination.
- Creates and drives debug visualizer when `cfg.debug_vis` is enabled.

Do not move simulator state writes outside `MotionCommandResetter`. Do not let
observation, reward, termination, or event modules mutate command timeline state.

## `MotionLoader`

`MotionLoader` wraps `UnifiedMotionLib` and is the motion data source used by the
command. It must satisfy `MotionDataSource`:

- `time_step_total`
- `time_step_start_idx`
- `time_step_end_idx`
- `motion_ids_from_timestamps(timestamps)`

It also exposes reference tensors consumed by `MotionCommand` properties. Its tensors
are indexed by global frame on dimension 0. It may reload/resample motion files via
`resample_motionloader(device)`.

Boundary:

- It may own motion tensors, file names, fps, body selection, and sample counters.
- It must not know about simulator reset, reward, termination, or event logic.

## `MotionCommandTimeline`

Owns per-env cursor state:

- `motion_ids`
- `local_time_steps`
- `motion_steps_len`
- `future_step_offsets`

Public methods:

- `num_future_frames`
- `global_time_steps(motion_source)`
- `global_start_steps(motion_source)`
- `motion_num_steps(motion_source)`
- `expanded_future_motion_ids()`
- `global_future_steps(motion_source)`
- `build_selection_from_global_timestamps(motion_source, timestamps)`
- `global_timestamps_from_sampled_bins(...)`
- `build_selection_from_motion_ids(motion_source, motion_ids)`
- `apply_selection(env_ids, selection)`
- `clear_timeline()`
- `step()`
- `expired_env_ids(max_future_step)`

Timeline may mutate only its own cursor state. It must not write simulator state,
load motion files, compute rewards, or update metrics.

## `MotionCommandResetter`

Owns robot state reset from selected reference tensors.

Public methods:

- `apply(env_ids, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, joint_pos,
  joint_vel, env_origins)`

Shared state:

- `envs_classes_mask`: dict of class-name to bool mask, built from
  `cfg.envs_classes_ratio`.

Reset behavior:

- `add_root_pose_randomization()` dispatches root pose randomization by env class.
- `range` envs receive xyz/rpy noise from `cfg.pose_range`.
- `lying` envs receive xy/yaw noise, sampled lying height, and randomized lying
  roll/pitch orientation.

Allowed side effects:

- Calls `robot.write_joint_state_to_sim()`.
- Calls `robot.write_root_state_to_sim()`.

Boundary:

- It must not choose motions, mutate timeline, reload data, refresh reference cache,
  update metrics, or create visualization markers.
- It currently assumes the selected reference root/anchor state is available in the
  tensors passed by `MotionCommand`; keep this assumption explicit when changing body
  ordering.

## `MotionReferenceCache`

Owns aligned reference body pose cache:

- `body_pos_relative_w`
- `body_quat_relative_w`

Public method:

- `refresh(anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w,
  robot_anchor_pos_w, robot_anchor_quat_w)`

The cache aligns reference XY/yaw to the current robot anchor while preserving the
reference height profile. It must not read `MotionCommand` directly or write simulator
state.

## Sampling and Selection

`MotionSelection` is the selection carrier:

- `motion_ids`
- `local_time_steps`
- `frame_end`

`AdaptiveMotionSampler` owns adaptive bin state and can update adaptive sampling
metrics. It returns `MotionSelection`; it must not call `timeline.apply_selection()`
or write simulator state.

`MotionSelectionPolicy` implementations hide policy differences:

- `FixedEvalMotionSelectionPolicy` maps envs to fixed motions for deterministic eval.
- `AdaptiveMotionSelectionPolicy` delegates to `AdaptiveMotionSampler`.
- `UnsupportedMotionSelectionPolicy` fails loudly for unsupported cfg combinations.

Any new sampling policy should implement `bind_motion_source()`, `select()`,
`step_post_update()`, and `counts_eval_cycles`.

## Debug Visualizer

`MotionCommandDebugVisualizer` owns marker creation and rendering only.

Public methods:

- `set_enabled(debug_vis)`
- `render()`

It may hold `VisualizationMarkers`. It must not mutate timeline, motion source,
simulator state, rewards, terminations, or metrics. Current implementation reads the
whole `MotionCommand`; if this is narrowed later, use a read-only provider protocol.
