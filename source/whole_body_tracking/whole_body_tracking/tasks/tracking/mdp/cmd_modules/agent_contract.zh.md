# Command Modules 合约

`cmd_modules` 拥有 motion-command runtime。它负责加载参考动作、维护 per-env
motion cursor、选择新的 motion frame、把 reset 状态写入 simulator、暴露 command
features，并渲染 debug markers。

公开 API 以 `cmd_modules/__init__.py` 中的 `__all__` 为准。这些符号也会通过
`whole_body_tracking.tasks.tracking.mdp` 门面重新导出。

## 公开导出

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

`MotionCommand` 是 IsaacLab `CommandTerm` 编排者。它拥有 runtime 模块的生命周期，
但具体领域逻辑应保留在对应子模块中。

拥有的 runtime 模块：

- `motion`: `MotionLoader`
- `timeline`: `MotionCommandTimeline`
- `reference_cache`: `MotionReferenceCache`
- `resetter`: `MotionCommandResetter`
- `selection_policy`: fixed-eval 或 adaptive selection policy
- `adaptive_sampler`: 可选 alias，指向 `selection_policy.sampler`
- `debug_visualizer`: `MotionCommandDebugVisualizer`

Observation、reward、termination、event 和 debug visualization 会读取的
command-facing properties：

- 当前参考：`joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`,
  `body_lin_vel_w`, `body_ang_vel_w`, `anchor_pos_w`, `anchor_quat_w`,
  `anchor_lin_vel_w`, `anchor_ang_vel_w`, `anchor_pos_z`,
  `anchor_project_gravity`, `anchor_6d_rotation`。
- 未来参考：`num_future_frames`, `expanded_future_motion_ids`,
  `global_future_steps`, `future_time_steps_init`, `anchor_pos_w_future`,
  `anchor_quat_w_future`, `joint_pos_future`, `joint_vel_future`,
  `joint_vel_multi_future`。
- Timeline views：`motion_ids`, `local_time_steps`, `global_time_steps`,
  `global_start_steps`, `motion_num_steps`, `motion_ids_from_timestamps()`。
- 对齐后的 reference cache：`body_pos_relative_w`, `body_quat_relative_w`。
- SMPL reference：`has_smpl_data`, `smpl_joints`, `smpl_transl`, `smpl_poses`,
  `smpl_poses_future`, `smpl_joints_future`, `smpl_transl_future`,
  `smpl_global_position`, `smpl_global_position_future`,
  `smpl_root_quat_w`, `smpl_root_quat_w_multi_future`,
  `smpl_root_quat_w_dif_l_multi_future`, `smpl_joints_local_multi_future`,
  `get_smpl_joints()`, `get_smpl_transl()`, `get_smpl_pose()`,
  `get_smpl_global_position()`。
- Robot state views：`robot_joint_pos`, `robot_joint_vel`, `robot_body_pos_w`,
  `robot_body_quat_w`, `robot_body_lin_vel_w`, `robot_body_ang_vel_w`,
  `robot_anchor_pos_w`, `robot_anchor_quat_w`, `robot_anchor_lin_vel_w`,
  `robot_anchor_ang_vel_w`。
- Policy command tensor：`command`。

允许的副作用：

- 初始化并拥有 `self.metrics`。
- 通过 `MotionLoader` 加载或重新加载 motion source。
- 调用 `selection_policy.select()`，并把返回的 selection 应用到 `timeline`。
- 在 command reset/resample 时调用 `MotionCommandResetter.apply()`。
- 刷新 `MotionReferenceCache`。
- 从 `resetter.envs_classes_mask` 设置 `env.envs_classes_mask`。
- 每个 command step 后驱动 `selection_policy.step_post_update()`。
- 当 `cfg.debug_vis` 启用时创建并驱动 debug visualization。

禁止的副作用：

- 不要把 simulator root/joint state 写入直接放到 `MotionCommand` 中；这应保留在
  `MotionCommandResetter`。
- 不要让 observation、reward、termination、event 或 visualizer 代码修改 timeline、
  sampler、motion source 或 resetter 状态。

## `MotionLoader`

`MotionLoader` 包装 `UnifiedMotionLib`，是 command 的 motion data source。它必须
满足 `MotionDataSource`：

- `time_step_total`
- `time_step_start_idx`
- `time_step_end_idx`
- `motion_num`
- `motion_ids_from_timestamps(timestamps)`

它还暴露 `MotionCommand` properties 消费的 reference tensors。这些 tensors 在第 0
维按 global frame 索引。它可以通过 `resample_motionloader(device)` 重新加载 motion
files。

边界：

- 可以拥有 motion tensors、file names、fps、body selection 和 source metadata。
- 不得知道 simulator reset、reward、termination、event 或 policy training 逻辑。

## `MotionCommandTimeline`

`MotionCommandTimeline` 拥有 per-env cursor state：

- `motion_ids`
- `local_time_steps`
- `motion_steps_len`
- `future_step_offsets`

公开方法和属性：

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

兼容保留：

- `global_timestamps_from_sampled_bins(...)` 仍保留，但当前 adaptive sampler 不再使用。
  优先使用 sampler 自己的 bin/frame conversion。

边界：

- Timeline 只能修改自己的 cursor tensors。
- 不得写 simulator state、加载 motion files、更新 metrics、选择 policy probabilities
  或刷新 reference cache。

## `MotionSelection`

`MotionSelection` 是 selection policy 返回、并由
`MotionCommandTimeline.apply_selection()` 消费的数据载体。

- `motion_ids`：每个请求 env 选择的 motion id。
- `local_time_steps`：每个请求 env 选择的 local frame。
- `frame_end`：每个请求 env 对应 motion 的 length/end frame。

该对象应保持轻量。不要向其中加入 simulator state、metrics 或 motion tensors。

## `MotionCommandResetter`

`MotionCommandResetter` 拥有从选中 reference tensors 重置 robot state 的逻辑。

公开方法：

- `apply(env_ids, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w,
  joint_pos, joint_vel, env_origins)`

拥有/共享状态：

- `envs_classes_mask`：class-name 到 bool mask 的 dict，由
  `cfg.envs_classes_ratio` 构建。`MotionCommand` 会把该 dict 暴露到
  `env.envs_classes_mask`，供 event 和 termination 模块使用。

Reset 行为：

- `range` env 从 `cfg.pose_range` 接收 xyz/rpy root pose noise。
- `lying` env 接收 xy/yaw pose noise、从 `cfg.pose_range_lying_height_range` 采样的
  height，以及随机 lying roll/pitch orientation。
- 所有 reset env 都从 `cfg.velocity_range` 接收 root velocity noise。
- Joint positions 从 `cfg.joint_position_range` 接收噪声。
- 写入 simulator 前会 clip joint position、joint velocity 和 root angular velocity。

允许的副作用：

- 调用 `robot.write_joint_state_to_sim()`。
- 调用 `robot.write_root_state_to_sim()`。

边界：

- 不得选择 motions、修改 timeline、重新加载 motion data、刷新 reference cache、
  更新 metrics 或渲染 markers。
- 当前假设 root/anchor state 位于 `MotionCommand` 传入 tensor 的 body index `0`；
  `MotionCommand` 的 anchor asserts 必须与该假设保持一致。

留待改进：

- 如果 `range` 不应代表 normal reset，需要显式增加 no-pose-randomization class。
- 显式传入 root/anchor index，避免依赖 body 顺序。
- 当需要严格 legacy deterministic 行为时，考虑跳过 zero-range sampling 以保持 RNG
  stream 不变。

## `MotionReferenceCache`

`MotionReferenceCache` 拥有对齐后的 reference body pose cache：

- `body_pos_relative_w`
- `body_quat_relative_w`

公开方法：

- `refresh(anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w,
  robot_anchor_pos_w, robot_anchor_quat_w)`

行为：

- 将 reference body pose 对齐到 robot anchor XY position 和 yaw。
- 保留 reference height profile。
- 提供 reward、termination、metrics、observation 和 debug visualization 使用的 cache
  tensors。

边界：

- 不得直接读取 `MotionCommand`。
- 不得写 simulator state 或 metrics。

## Selection Policies

`MotionSelectionPolicy` 实现用于把 selection strategy 从 `MotionCommand` 中隔离出来。

统一接口：

- `bind_motion_source(motion_source, timeline, decimation, sim_dt)`
- `select(env_ids, motion_source, timeline, terminated, metrics,
  allow_failure_accounting=True)`
- `step_post_update()`
- `counts_eval_cycles`
- `fixed_eval_motion_ids`

`FixedEvalMotionSelectionPolicy`：

- 要求 `num_envs == motion_source.motion_num`。
- 将 env `i` 绑定到 motion `i`。
- 在 `bind_motion_source()` 时直接把初始 fixed selection 应用到 timeline。
- `step_post_update()` 不更新状态。

`AdaptiveMotionSelectionPolicy`：

- 将全部 sampling 状态委托给 `AdaptiveMotionSampler`。
- bind 时调用 `sampler.reset_for_motion_source()`。
- select 时调用 `sampler.sample_selection()`。
- 每个 command step 后调用 `sampler.step_post_update()`。

`UnsupportedMotionSelectionPolicy`：

- 对不支持的配置组合，在 `select()` 中显式报错。

## `AdaptiveMotionSampler`

`AdaptiveMotionSampler` 拥有 adaptive sampling 状态并返回 `MotionSelection`。它不得
调用 `timeline.apply_selection()`，也不得写 simulator state。

公开方法：

- `reset_for_motion_source(motion_source, decimation, sim_dt)`
- `sample_selection(env_ids, motion_source, timeline, terminated, metrics,
  allow_failure_accounting=True)`
- `step_post_update()`

当前 bin 模型：

- Motion-local bins 会按 motion 顺序 flatten 成一条 global bin vector。
- `bin_start_idx` 和 `bin_end_idx` 将 flattened bin id 映射到 global frame range。
- Failure counts 存在 `motion_failure_bin_counts`。
- Per-step pending counts 存在 `pending_motion_failure_bin_counts`。
- `step_post_update()` 使用 `cfg.adaptive_alpha` 做 EMA。

Sampling 行为：

- 如果 `allow_failure_accounting=True`，terminated env 会根据上一帧 global frame
  `timeline.global_time_steps() - 1` 记录到 pending failure bins。
- Sampling probabilities 从 EMA failure counts 计算。
- Counts 先按 `failure_most_hard_cap_beta * mean` 削峰。
- Counts 沿 flattened bin axis 使用 `sampling_kernel` 平滑。
- Probability layers 按
  `motion_ratio[0] * uniform + motion_ratio[1] * mid + motion_ratio[2] * top`
  混合。
- `mid` 使用 `failure_cap_beta` 削峰；`top` 使用
  `failure_most_hard_cap_beta` 削峰。
- 在选中的 flattened bin 内采样 frame，然后向前随机偏移 `[0, bin_frame_width]`
  个 frame，并 clamp 到所选 motion 的合法 frame range。

写入的 metrics：

- `failures_max`, `failures_mean`, `failures_min`, `failures_max_over_uniform`
- `sampling_entropy`, `sampling_top1_prob_max`, `sampling_top1_prob_bin`,
  `sampling_top1_prob_mean`, `sampling_top1_prob_min`
- `prob_max_over_uniform`, `prob_uniform`, `num_concentrate_bins`

留待改进：

- `adaptive_sample_rewind_min_bins` / `adaptive_sample_rewind_bins` 尚未接入当前
  `_shift_frame_idx()` 逻辑。
- 由于 bins 被 flatten，当前 smoothing 不考虑 motion 边界。
- `adaptive_uniform_ratio` 仍存在于 cfg 中，但当前 probability mixture 不使用；
  uniform 分量由 `motion_ratio[0]` 控制。
- 如需要分析 artifact，可重新引入 adaptive-bin export。
- 给 sampler 单独定义 config type，避免读取完整 command cfg。

## `MotionCommandDebugVisualizer`

`MotionCommandDebugVisualizer` 只拥有 marker 创建和渲染。

公开方法：

- `set_enabled(debug_vis)`
- `render()`

边界：

- 可以读取 `MotionCommand` properties。
- 不得修改 timeline、motion source、resetter、sampler、simulator state、rewards、
  terminations、events 或 metrics。
- 如果后续收窄依赖面，使用只读 provider protocol。
