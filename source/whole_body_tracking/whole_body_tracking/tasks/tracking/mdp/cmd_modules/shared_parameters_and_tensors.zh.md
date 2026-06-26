# Cmd Modules 共享参数和张量

本文档记录 `cmd_modules` 内部和跨模块共享的对象、参数和张量。目标是让
`MotionCommand` 作为编排者，而把 motion source、timeline、reset、reference cache、
selection policy、adaptive sampler 和 debug visualizer 的状态边界保持清晰。

## 核心运行时对象

| 对象/参数 | 拥有者 | 使用方 | 语义 |
|---|---|---|---|
| `cfg` | `MotionCommandCfg` / `MotionCommand` | command 子模块 | 运行模式、采样、reset randomization、可视化配置 |
| `motion` / `motion_source` | `MotionLoader` | `MotionCommand`, `MotionCommandTimeline`, selection policy, sampler | 参考动作数据源 |
| `timeline` | `MotionCommandTimeline` | `MotionCommand`, selection policy, sampler | 每个 env 当前 motion id 和 local frame |
| `reference_cache` | `MotionReferenceCache` | `MotionCommand`, reward, termination, visualizer | 对齐后的 reference body pose |
| `resetter` | `MotionCommandResetter` | `MotionCommand` | simulator reset 写入 |
| `selection_policy` | `create_motion_selection_policy()` | `MotionCommand` | fixed eval 或 adaptive selection |
| `adaptive_sampler` | `AdaptiveMotionSelectionPolicy` | policy, `MotionCommand` debug access | adaptive sampling 状态 |
| `metrics` | `MotionCommand` | `MotionCommand`, `AdaptiveMotionSampler` | reward/debug/adaptive sampling 指标 |
| `robot` | `MotionCommand` | `MotionCommand`, resetter, visualizer | 仿真机器人状态读写 |
| `env_origins` | env scene | `MotionCommand`, resetter, cache | reference motion 转 world frame 时加 origin |

## Interface Types

### `MotionDataSource`

`MotionDataSource` 是 timeline 和 sampler 对 motion 数据源的最小依赖协议。

| 字段/方法 | 类型 | 语义 |
|---|---|---|
| `time_step_total` | `int` | 拼接后 motion buffer 的总帧数 |
| `time_step_start_idx` | `Tensor[motion_num]` | 每个 motion 的全局起始帧 |
| `time_step_end_idx` | `Tensor[motion_num]` | 每个 motion 的全局结束帧 |
| `motion_num` | `int` | motion 数量 |
| `motion_ids_from_timestamps(timestamps)` | function | 全局帧到 motion id 的映射 |

### `MotionSelection`

`MotionSelection` 是 selection policy / sampler 传给 timeline 的选择结果。

| 字段 | 形状 | 语义 |
|---|---:|---|
| `motion_ids` | `[len(env_ids)]` | 为目标 env 选择的 motion id |
| `local_time_steps` | `[len(env_ids)]` | motion 内 local frame |
| `frame_end` | `[len(env_ids)]` | 当前 motion 的 local end/length |

`MotionCommandTimeline.apply_selection()` 是唯一应写入 timeline cursor 的入口。

## MotionCommand Public Views

`MotionCommand` 对外暴露大量只读语义的 property。调用方应把它们当作视图，不要
原地写入返回 tensor。

| 类别 | 属性 |
|---|---|
| Timeline views | `motion_ids`, `local_time_steps`, `global_time_steps`, `global_start_steps`, `motion_num_steps` |
| 当前参考 | `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, `anchor_pos_w`, `anchor_quat_w`, `anchor_lin_vel_w`, `anchor_ang_vel_w` |
| 当前派生特征 | `anchor_pos_z`, `anchor_project_gravity`, `anchor_6d_rotation`, `anchor_lin_vel_b`, `anchor_ang_vel_b`, `command` |
| 未来参考 | `anchor_pos_w_future`, `anchor_quat_w_future`, `joint_pos_future`, `joint_vel_future`, `joint_vel_multi_future`, `global_future_steps`, `expanded_future_motion_ids`, `num_future_frames` |
| Cache views | `body_pos_relative_w`, `body_quat_relative_w` |
| Robot views | `robot_joint_pos`, `robot_joint_vel`, `robot_body_pos_w`, `robot_body_quat_w`, `robot_body_lin_vel_w`, `robot_body_ang_vel_w`, `robot_anchor_pos_w`, `robot_anchor_quat_w`, `robot_anchor_lin_vel_w`, `robot_anchor_ang_vel_w` |
| SMPL views | `has_smpl_data`, `smpl_joints`, `smpl_transl`, `smpl_poses`, future SMPL properties, `get_smpl_*()` methods |

已移除的 legacy aliases：

- `MotionCommand` 不再把 sampler 内部状态如 `bin_failed_count`,
  `_current_bin_failed`, `kernel`, `success_motion` 暴露为 public properties。
- 外部如果需要 sampler 内部信息，应只通过 `command.adaptive_sampler` 做 debug/分析，
  不应在训练时修改。

## MotionLoader / Reference Tensors

`MotionLoader` 继承 `UnifiedMotionLib`，提供按 global frame 索引的 reference tensors。

| Tensor / 方法 | 当前访问 | Future 访问 | 语义 |
|---|---|---|---|
| `joint_pos` | `MotionCommand.joint_pos` | `joint_pos_future` | 参考关节位置 |
| `joint_vel` | `MotionCommand.joint_vel` | `joint_vel_future`, `joint_vel_multi_future` | 参考关节速度 |
| `body_pos_w` | `body_pos_w` | - | 参考 body 位置，当前帧加 `env_origins` |
| `body_quat_w` | `body_quat_w` | - | 参考 body 朝向 |
| `body_lin_vel_w` | `body_lin_vel_w` | - | 参考 body 线速度 |
| `body_ang_vel_w` | `body_ang_vel_w` | - | 参考 body 角速度 |
| `anchor_pos_w` | `anchor_pos_w` | `anchor_pos_w_future` | 参考 anchor 位置，当前/未来访问会加 `env_origins` |
| `anchor_quat_w` | `anchor_quat_w` | `anchor_quat_w_future` | 参考 anchor 朝向 |
| `anchor_lin_vel_w` | `anchor_lin_vel_w` | - | 参考 anchor 线速度 |
| `anchor_ang_vel_w` | `anchor_ang_vel_w` | - | 参考 anchor 角速度 |
| `get_smpl_*()` | SMPL properties | SMPL future properties | SMPL 查询 |

## MotionCommandTimeline

Timeline 拥有 per-env cursor state。

| Tensor | 形状 | 写入方 | 读取方 | 语义 |
|---|---:|---|---|---|
| `motion_ids` | `[num_envs]` | `apply_selection()`, `clear_timeline()` | command, sampler, timeline methods | 当前 motion |
| `local_time_steps` | `[num_envs]` | `apply_selection()`, `step()`, `clear_timeline()` | command, sampler, timeline methods | 当前 local frame |
| `motion_steps_len` | `[num_envs]` | `apply_selection()`, `clear_timeline()` | `expired_env_ids()` | 当前 motion 长度/end |
| `future_step_offsets` | `[num_future_frames]` | init only | future indexing | future reference offsets |

Timeline 只依赖 `MotionDataSource` 协议。它不应知道 reset、reward、termination、
metrics 或 adaptive probabilities。

## MotionReferenceCache

| Tensor | 形状 | 写入方 | 读取方 | 语义 |
|---|---:|---|---|---|
| `body_pos_relative_w` | `[num_envs, body_count, 3]` | `refresh()` | command metrics, reward, termination, visualizer | 对齐到 robot anchor XY/yaw 的 reference body 位置 |
| `body_quat_relative_w` | `[num_envs, body_count, 4]` | `refresh()` | command metrics, reward, visualizer | 对齐到 robot anchor yaw 的 reference body 朝向 |

`refresh()` 输入全部由 `MotionCommand` 显式传入：

- reference anchor/body pose
- robot anchor pose

Cache 不读取 `MotionCommand`，不写 simulator，不更新 metrics。

## MotionCommandResetter

Resetter 把 `MotionCommand` 选中的 reference state 写回 simulator。

### 持有依赖

| 字段 | 语义 |
|---|---|
| `cfg` | reset randomization 参数 |
| `robot` | simulator write target |
| `device` | tensor device |
| `envs_classes_mask` | 由 `cfg.envs_classes_ratio` 构建的 class-name -> bool mask |

### `apply()` 输入

| 输入 | 形状 | 来源 | 用途 |
|---|---:|---|---|
| `body_pos_w` | `[num_envs, body_count, 3]` | `MotionCommand.body_pos_w` | root position |
| `body_quat_w` | `[num_envs, body_count, 4]` | `MotionCommand.body_quat_w` | root orientation |
| `body_lin_vel_w` | `[num_envs, body_count, 3]` | `MotionCommand.body_lin_vel_w` | root linear velocity |
| `body_ang_vel_w` | `[num_envs, body_count, 3]` | `MotionCommand.body_ang_vel_w` | root angular velocity |
| `joint_pos` | `[num_envs, num_joints]` | `MotionCommand.joint_pos` | joint position |
| `joint_vel` | `[num_envs, num_joints]` | `MotionCommand.joint_vel` | joint velocity |
| `env_origins` | `[num_envs, 3]` | env scene | lying height/origin handling |

### Env Classes

| Class | 行为 |
|---|---|
| `range` | 从 `cfg.pose_range` 向 root pose 添加 xyz/rpy noise |
| `lying` | 添加 xy/yaw noise，设置 sampled lying height，并把 roll/pitch 随机为 lying orientations |

所有 reset env 都会接收 velocity noise 和 joint-position noise，不受 env class 限制。

### 副作用

- `robot.write_joint_state_to_sim()`
- `robot.write_root_state_to_sim()`

### 假设

`body_*[:, 0]` 被视为 root/anchor state。`MotionCommand.__init__()` assert robot 和
motion anchor body index 都为 `0`。

## MotionSelectionPolicy

| 方法/属性 | 输入 | 输出/副作用 |
|---|---|---|
| `bind_motion_source()` | `motion_source`, `timeline`, `decimation`, `sim_dt` | motion reload 后 reset policy/sampler |
| `select()` | `env_ids`, `motion_source`, `timeline`, `terminated`, `metrics`, `allow_failure_accounting` | 返回 `MotionSelection` |
| `step_post_update()` | none | command step 后更新 policy/sampler state |
| `counts_eval_cycles` | none | command 是否应统计 completed eval cycles |
| `fixed_eval_motion_ids` | policy owned | fixed eval assignment 或 `None` |

`FixedEvalMotionSelectionPolicy` 可以在 binding 时写 timeline，以初始化所有 env。
`AdaptiveMotionSelectionPolicy` 不应直接写 timeline；它把 selection 返回给
`MotionCommand`。

## AdaptiveMotionSampler

当前 sampler 使用 flatten motion-local bins：每个 motion 先按 `bin_frame_width` 切分，
再按 motion 顺序拼成一条全局 bin 向量。

### State

| Tensor / 参数 | 形状 | 写入方 | 语义 |
|---|---:|---|---|
| `bin_frame_width` | scalar int | `reset_for_motion_source()` | 约 1 秒对应的 frame 数 |
| `max_motion_bin_count` | scalar int | `reset_for_motion_source()` | 单个 motion 最大 bin 数 |
| `total_motion_bin_num` | scalar int | `reset_for_motion_source()` | flatten 后总 bin 数 |
| `bin_count_per_motion` | `[motion_num]` | `reset_for_motion_source()` | 每个 motion 的 bin 数 |
| `bin_start_idx` | `[total_motion_bin_num]` | `reset_for_motion_source()` | 每个 flatten bin 的全局起始帧 |
| `bin_end_idx` | `[total_motion_bin_num]` | `reset_for_motion_source()` | 每个 flatten bin 的全局结束帧 |
| `motion_failure_bin_counts` | `[total_motion_bin_num]` | `step_post_update()` | failure EMA |
| `pending_motion_failure_bin_counts` | `[total_motion_bin_num]` | `_record_failed_bins()`, `step_post_update()` | 当前 step 累计的 failure counts |
| `sampling_kernel` | `[adaptive_kernel_size]` | `__init__()` | smoothing kernel |

### Probability Pipeline

1. 使用上一帧 global frame `timeline.global_time_steps(motion_source) - 1`，把 failed
   env 记录到 `pending_motion_failure_bin_counts`。
2. `step_post_update()` 执行 EMA：
   `adaptive_alpha * pending + (1 - adaptive_alpha) * previous`。
3. `_compute_sampling_prob()` 使用 `failure_most_hard_cap_beta * mean` 对 failure
   counts 削峰。
4. 沿 flattened bin axis 平滑 clipped counts。当前不考虑 motion 边界。
5. 归一化得到 hard probabilities。
6. 用 `failure_cap_beta` cap 得到 `mid`；用 `failure_most_hard_cap_beta` cap 得到
   `top`。
7. 使用 `motion_ratio` 混合 `uniform`、`mid` 和 `top`。

### Sampling Pipeline

1. 用 `torch.multinomial` 抽取 flatten bin ids。
2. 在每个选中 bin 内采样 frame。
3. 将 sampled frame 向前随机偏移 `[0, bin_frame_width]`。
4. 将偏移后的 frame clamp 到所选 motion 的合法范围：
   `[motion_start + motion_sampling_start_frame, motion_end - max_future_step - 1]`。
5. 把 global frame 转成 `motion_id` 和 local frame。
6. 返回 `MotionSelection`。

### 写入的 Metrics

- Failure stats：`failures_max`, `failures_mean`, `failures_min`,
  `failures_max_over_uniform`
- Sampling stats：`sampling_entropy`, `sampling_top1_prob_max`,
  `sampling_top1_prob_bin`, `sampling_top1_prob_mean`, `sampling_top1_prob_min`,
  `prob_max_over_uniform`, `prob_uniform`, `num_concentrate_bins`

### 留待改进

- 将 `adaptive_sample_rewind_min_bins` 和 `adaptive_sample_rewind_bins` 接入当前
  frame-shift 逻辑，或从 cfg 移除。
- 判断 smoothing 是否应尊重 motion 边界。
- 判断 `adaptive_uniform_ratio` 应被移除还是真正使用。
- 如需要分析，恢复/export adaptive-bin artifacts。

## MotionCommandDebugVisualizer

Visualizer 读取 `MotionCommand` properties，并拥有 marker rendering。

读取：

- Robot anchor/body pose 和 velocity views。
- 当前和未来 reference anchor pose。
- 对齐后的 reference body cache。

它不得修改 command state、timeline、sampler、resetter、simulator state、reward、
termination、event state 或 metrics。

## 跨包共享状态

`MotionCommand` 设置：

```python
env.envs_classes_mask = self.resetter.envs_classes_mask
```

| 字段 | 写入方 | 读取方 | 语义 |
|---|---|---|---|
| `env.envs_classes_mask` | `MotionCommand` / `MotionCommandResetter` | `evt_modules`, `tmt_modules` | Env class masks；`["lying"]` 选择 recovery/delayed-termination envs |

## 边界风险和后续方向

1. `MotionCommand` 仍暴露很多 derived properties，是 observations、rewards、
   terminations 和 visualizer 的宽读取依赖。
2. Timeline/cache tensors 通过 command properties 暴露；调用方应视作只读。
3. `MotionCommandResetter` 依赖 body index `0` 是 root/anchor。
4. `env.envs_classes_mask` 是隐式跨包 env 属性；如果继续增长，应考虑 manager-owned
   provider。
5. `MotionCommandDebugVisualizer` 依赖完整 `MotionCommand`；窄的只读 provider 可降低
   耦合。
