# Command Modules Agent Contract

`cmd_modules` owns the motion-command runtime. It loads reference motions,
maintains per-env motion cursors, chooses new motion frames, writes reset states to
the simulator, exposes command features, and renders debug markers.

The public API is the `__all__` list in `cmd_modules/__init__.py`. These symbols
are also re-exported through the package facade `whole_body_tracking.tasks.tracking.mdp`.

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

## `MotionCommand`

`MotionCommand` is the IsaacLab `CommandTerm` orchestrator. It owns the lifecycle
of the runtime modules, but should keep the actual domain logic inside those
modules.

Owned runtime modules:

- `motion`: `MotionLoader`
- `timeline`: `MotionCommandTimeline`
- `reference_cache`: `MotionReferenceCache`
- `resetter`: `MotionCommandResetter`
- `selection_policy`: fixed-eval or adaptive selection policy
- `adaptive_sampler`: optional alias to `selection_policy.sampler`
- `debug_visualizer`: `MotionCommandDebugVisualizer`

Command-facing properties read by observations, rewards, terminations, events, and
debug visualization:

- Current reference: `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`,
  `body_lin_vel_w`, `body_ang_vel_w`, `anchor_pos_w`, `anchor_quat_w`,
  `anchor_lin_vel_w`, `anchor_ang_vel_w`, `anchor_pos_z`,
  `anchor_project_gravity`, `anchor_6d_rotation`.
- Future reference: `num_future_frames`, `expanded_future_motion_ids`,
  `global_future_steps`, `future_time_steps_init`, `anchor_pos_w_future`,
  `anchor_quat_w_future`, `joint_pos_future`, `joint_vel_future`,
  `joint_vel_multi_future`.
- Timeline views: `motion_ids`, `local_time_steps`, `global_time_steps`,
  `global_start_steps`, `motion_num_steps`, `motion_ids_from_timestamps()`.
- Aligned reference cache: `body_pos_relative_w`, `body_quat_relative_w`.
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
- Policy command tensor: `command`.

Allowed side effects:

- Initializes and owns `self.metrics`.
- Loads or reloads motion source data through `MotionLoader`.
- Calls `selection_policy.select()` and applies returned selections to `timeline`.
- Calls `MotionCommandResetter.apply()` during command resets/resamples.
- Refreshes `MotionReferenceCache`.
- Sets `env.envs_classes_mask` from `resetter.envs_classes_mask`.
- Drives `selection_policy.step_post_update()` after each command step.
- Creates and drives debug visualization when `cfg.debug_vis` is enabled.

Forbidden side effects:

- Do not put simulator root/joint state writes directly in `MotionCommand`; keep
  them in `MotionCommandResetter`.
- Do not let observation, reward, termination, event, or visualizer code mutate the
  timeline, sampler, motion source, or resetter state.

## `MotionLoader`

`MotionLoader` wraps `UnifiedMotionLib` and is the command's motion data source. It
must satisfy `MotionDataSource`:

- `time_step_total`
- `time_step_start_idx`
- `time_step_end_idx`
- `motion_num`
- `motion_ids_from_timestamps(timestamps)`

It also exposes reference tensors consumed by `MotionCommand` properties. These
tensors are indexed by global frame on dimension 0. It may reload motion files via
`resample_motionloader(device)`.

Boundary:

- It may own motion tensors, file names, fps, body selection, and source metadata.
- It must not know about simulator reset, reward, termination, event, or policy
  training logic.

## `MotionCommandTimeline`

`MotionCommandTimeline` owns per-env cursor state:

- `motion_ids`
- `local_time_steps`
- `motion_steps_len`
- `future_step_offsets`

Public methods and properties:

- `num_future_frames`
- `global_time_steps(motion_source)`
- `global_start_steps(motion_source)`
- `motion_num_steps(motion_source)`
- `expanded_future_motion_ids()`
- `global_future_steps(motion_source)`
- `build_selection_from_global_timestamps(motion_source, timestamps)`
- `build_selection_from_motion_ids(motion_source, motion_ids, local_time_steps=None)`
- `apply_selection(env_ids, selection)`
- `clear_timeline()`
- `step()`
- `expired_env_ids(max_future_step)`

Deprecated compatibility:

- `global_timestamps_from_sampled_bins(...)` is retained but not used by the
  current adaptive sampler. Prefer sampler-owned bin/frame conversion.

Boundary:

- Timeline may mutate only its own cursor tensors.
- It must not write simulator state, load motion files, update metrics, choose
  policy probabilities, or refresh reference cache.

## `MotionSelection`

`MotionSelection` is the carrier returned by selection policies and consumed by
`MotionCommandTimeline.apply_selection()`.

- `motion_ids`: selected motion id per requested env.
- `local_time_steps`: selected local frame per requested env.
- `frame_end`: selected motion length/end frame per requested env.

The carrier should remain a small data object. Do not add simulator state, metrics,
or motion tensors to it.

## `MotionCommandResetter`

`MotionCommandResetter` owns robot state reset from selected reference tensors.

Public method:

- `apply(env_ids, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w,
  joint_pos, joint_vel, env_origins)`

Owned/shared state:

- `envs_classes_mask`: dict of class-name to bool mask, built from
  `cfg.envs_classes_ratio`. `MotionCommand` exposes this dict on
  `env.envs_classes_mask` for event and termination modules.

Reset behavior:

- `range` envs receive xyz/rpy root pose noise from `cfg.pose_range`.
- `lying` envs receive xy/yaw pose noise, sampled height from
  `cfg.pose_range_lying_height_range`, and randomized lying roll/pitch orientation.
- All reset envs receive root velocity noise from `cfg.velocity_range`.
- Joint positions receive noise from `cfg.joint_position_range`.
- Joint position, joint velocity, and root angular velocity are clipped before
  writing to sim.

Allowed side effects:

- Calls `robot.write_joint_state_to_sim()`.
- Calls `robot.write_root_state_to_sim()`.

Boundary:

- It must not choose motions, mutate timeline, reload motion data, refresh
  reference cache, update metrics, or render markers.
- It currently assumes root/anchor state is in body index `0` of the tensors passed
  by `MotionCommand`; keep `MotionCommand` anchor assertions aligned with this.

Known follow-ups:

- Add an explicit no-pose-randomization class if `range` should not represent
  normal reset.
- Pass root/anchor index explicitly instead of relying on body order.
- Consider skipping zero-range sampling to preserve deterministic RNG streams when
  exact legacy behavior matters.

## `MotionReferenceCache`

`MotionReferenceCache` owns aligned reference body pose cache:

- `body_pos_relative_w`
- `body_quat_relative_w`

Public method:

- `refresh(anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w,
  robot_anchor_pos_w, robot_anchor_quat_w)`

Behavior:

- Aligns reference body pose to the robot anchor XY position and yaw.
- Preserves the reference height profile.
- Provides cache tensors used by rewards, terminations, metrics, observations, and
  debug visualization.

Boundary:

- It must not read `MotionCommand` directly.
- It must not write simulator state or metrics.

## Selection Policies

`MotionSelectionPolicy` implementations isolate selection strategy from
`MotionCommand`.

Unified interface:

- `bind_motion_source(motion_source, timeline, decimation, sim_dt)`
- `select(env_ids, motion_source, timeline, terminated, metrics,
  allow_failure_accounting=True)`
- `step_post_update()`
- `counts_eval_cycles`
- `fixed_eval_motion_ids`

`FixedEvalMotionSelectionPolicy`:

- Requires `num_envs == motion_source.motion_num`.
- Binds env `i` to motion `i`.
- Applies the initial fixed selection directly to the timeline during
  `bind_motion_source()`.
- Does not update state in `step_post_update()`.

`AdaptiveMotionSelectionPolicy`:

- Delegates all sampling state to `AdaptiveMotionSampler`.
- Calls `sampler.reset_for_motion_source()` on bind.
- Calls `sampler.sample_selection()` on select.
- Calls `sampler.step_post_update()` after each command step.

`UnsupportedMotionSelectionPolicy`:

- Fails loudly from `select()` for unsupported config combinations.

## `AdaptiveMotionSampler`

`AdaptiveMotionSampler` owns adaptive sampling state and returns `MotionSelection`.
It must not call `timeline.apply_selection()` or write simulator state.

Public methods:

- `reset_for_motion_source(motion_source, decimation, sim_dt)`
- `sample_selection(env_ids, motion_source, timeline, terminated, metrics,
  allow_failure_accounting=True)`
- `step_post_update()`

Current bin model:

- Motion-local bins are flattened into one global bin vector ordered by motion.
- `bin_start_idx` and `bin_end_idx` map flattened bin ids to global frame ranges.
- Failure counts are stored in `motion_failure_bin_counts`.
- Per-step pending counts are stored in `pending_motion_failure_bin_counts`.
- `step_post_update()` applies EMA using `cfg.adaptive_alpha`.

Sampling behavior:

- If `allow_failure_accounting=True`, terminated envs are recorded into pending
  failure bins based on the previous global frame (`timeline.global_time_steps() - 1`).
- Sampling probabilities are computed from EMA failure counts.
- Counts are first clipped by `failure_most_hard_cap_beta * mean`.
- Counts are smoothed along the flattened bin axis using `sampling_kernel`.
- Probability layers are mixed as
  `motion_ratio[0] * uniform + motion_ratio[1] * mid + motion_ratio[2] * top`.
- `mid` caps probabilities with `failure_cap_beta`; `top` caps probabilities with
  `failure_most_hard_cap_beta`.
- A frame is sampled inside the selected flattened bin, then shifted backward by a
  random frame offset in `[0, bin_frame_width]` and clamped to the selected motion's
  valid frame range.

Metrics written:

- `failures_max`, `failures_mean`, `failures_min`, `failures_max_over_uniform`
- `sampling_entropy`, `sampling_top1_prob_max`, `sampling_top1_prob_bin`,
  `sampling_top1_prob_mean`, `sampling_top1_prob_min`
- `prob_max_over_uniform`, `prob_uniform`, `num_concentrate_bins`

Known follow-ups:

- `adaptive_sample_rewind_min_bins` / `adaptive_sample_rewind_bins` are not wired
  into the current `_shift_frame_idx()` logic.
- Smoothing currently ignores motion boundaries because bins are flattened.
- `adaptive_uniform_ratio` exists in cfg but is not used by the current probability
  mixture; `motion_ratio[0]` controls the uniform component.
- Reintroduce adaptive-bin export if analysis artifacts are needed.
- Give sampler a dedicated config type instead of reading the whole command cfg.

## `MotionCommandDebugVisualizer`

`MotionCommandDebugVisualizer` owns marker creation and rendering only.

Public methods:

- `set_enabled(debug_vis)`
- `render()`

Boundary:

- It may read `MotionCommand` properties.
- It must not mutate timeline, motion source, resetter, sampler, simulator state,
  rewards, terminations, events, or metrics.
- If its dependency surface is narrowed later, use a read-only provider protocol.
