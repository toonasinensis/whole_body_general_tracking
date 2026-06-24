# Cmd Modules Shared Parameters and Tensors

本文档统计 `cmd_modules` 内各模块之间共享的参数、对象和张量，用于明确当前架构中的模块边界和依赖关系。

当前整体结构是：`MotionCommand` 作为 IsaacLab `CommandTerm` 编排者，负责连接 `MotionLoader`、`MotionCommandTimeline`、`MotionReferenceCache`、`MotionCommandResetter`、`MotionSelectionPolicy` / `AdaptiveMotionSampler` 和 `MotionCommandDebugVisualizer`。多数共享数据通过显式参数传递，但也存在少量 tensor alias 暴露到 `MotionCommand` 的 public 字段上。

## 核心共享对象

| 共享对象/参数 | 拥有者 | 被谁使用 | 语义 |
|---|---|---|---|
| `cfg` | `MotionCommandCfg` / `MotionCommand` | 几乎所有子模块 | 运行模式、采样、reset randomization、可视化配置 |
| `motion` / `motion_source` | `MotionLoader` | `MotionCommandTimeline`, `MotionSelectionPolicy`, `AdaptiveMotionSampler`, `MotionCommand` | 参考动作数据源和全局帧索引 |
| `timeline` | `MotionCommandTimeline` | `MotionCommand`, `MotionSelectionPolicy`, `AdaptiveMotionSampler` | 每个 env 当前 motion id 和 local frame |
| `metrics` | `MotionCommand` | `MotionCommand`, `AdaptiveMotionSampler` | reward/debug/adaptive sampling 指标 |
| `robot` | `MotionCommand` | `MotionCommandResetter`, `MotionCommandDebugVisualizer`, `MotionCommand` properties | 仿真机器人状态读写 |
| `env_origins` | env scene | `MotionCommand`, `MotionCommandResetter`, `MotionReferenceCache` | reference motion 转 world frame 时加 origin |

## Interface

`interface.py` 定义两个跨模块协议/数据载体。

### `MotionDataSource`

`MotionDataSource` 是 motion 数据源最小协议。`MotionCommandTimeline` 和 `AdaptiveMotionSampler` 只依赖这些字段/方法，而不应该依赖 `UnifiedMotionLib` 的完整实现。

| 字段/方法 | 类型 | 使用方 | 语义 |
|---|---|---|---|
| `time_step_total` | `int` | `MotionCommandTimeline`, `AdaptiveMotionSampler` | 拼接后 motion buffer 的总帧数 |
| `time_step_start_idx` | `torch.Tensor[motion_num]` | `MotionCommandTimeline`, `AdaptiveMotionSampler`, sampling helpers | 每个 motion 的全局起始帧 |
| `time_step_end_idx` | `torch.Tensor[motion_num]` | `MotionCommandTimeline`, `AdaptiveMotionSampler`, sampling helpers | 每个 motion 的全局结束帧 |
| `motion_num` | `int` | fixed eval policy, sampler | motion 数量 |
| `motion_ids_from_timestamps(timestamps)` | function | timeline/sampler | 将全局帧映射到 motion id |

### `MotionSelection`

`MotionSelection` 是 selection policy / sampler 返回给 timeline 的选择结果。

| 字段 | 形状 | 语义 |
|---|---:|---|
| `motion_ids` | `[len(env_ids)]` | 为每个目标 env 选择的 motion id |
| `local_time_steps` | `[len(env_ids)]` | 每个 env 在该 motion 内的 local frame |
| `frame_end` | `[len(env_ids)]` | 每个 env 当前 motion 的 local 结束帧 |

`MotionCommandTimeline.apply_selection()` 会把这些字段写入 timeline 自己持有的 cursor tensor。

## MotionCommand

`MotionCommand` 是中心编排者，拥有或连接以下模块：

| 字段 | 来源/拥有者 | 说明 |
|---|---|---|
| `motion` | `MotionLoader` | motion 数据源 |
| `timeline` | `MotionCommandTimeline` | per-env motion cursor |
| `reference_cache` | `MotionReferenceCache` | 对齐后的 body reference cache |
| `resetter` | `MotionCommandResetter` | simulator state reset 写入 |
| `selection_policy` | `create_motion_selection_policy()` | fixed eval 或 adaptive sampling policy |
| `adaptive_sampler` | `selection_policy.sampler` 可选 | adaptive sampling 内部状态 |
| `debug_visualizer` | `MotionCommandDebugVisualizer` | debug marker 渲染 |

### Public Alias

以下字段是 `MotionCommand` 对子模块内部 tensor 的直接 alias，不是 copy。

| `MotionCommand` 字段 | 实际拥有者 | 形状 | 风险 |
|---|---|---:|---|
| `motion_ids` | `timeline.motion_ids` | `[num_envs]` | 外部写入会直接改变 timeline state |
| `local_time_steps` | `timeline.local_time_steps` | `[num_envs]` | 外部写入会直接改变 timeline state |
| `eval_cycle_count` | `MotionCommand` | `[num_envs]` | eval bookkeeping，由 command 直接拥有 |
| `motion_steps_len` | `timeline.motion_steps_len` | `[num_envs]` | 外部写入会影响 expired env 判断 |
| `body_pos_relative_w` | `reference_cache.body_pos_relative_w` | `[num_envs, body_count, 3]` | 外部写入会改变 reward/termination/debug 使用的 cache |
| `body_quat_relative_w` | `reference_cache.body_quat_relative_w` | `[num_envs, body_count, 4]` | 外部写入会改变 reward/debug 使用的 cache |
| `bin_failed_count` | `adaptive_sampler.bin_failed_count` | `[bin_count]` | legacy alias，外部写入会改变 sampler 状态 |
| `_current_bin_failed` | `adaptive_sampler._current_bin_failed` | `[bin_count]` | legacy alias，外部写入会改变 sampler 状态 |
| `kernel` | `adaptive_sampler.kernel` | `[adaptive_kernel_size]` | legacy alias |
| `success_motion` | `adaptive_sampler.success_motion` | `[motion_num]` | legacy alias |

## MotionLoader / Motion Source

`MotionLoader` 继承 `UnifiedMotionLib`，是 `MotionCommand` 当前使用的 motion 数据源。它除了满足 `MotionDataSource` 协议，还向 `MotionCommand` 暴露 reference tensor。

### Reference Tensor

这些 tensor 由 `MotionCommand` 通过 `global_time_steps` 或 `global_future_steps` 索引。

| Tensor / 方法 | 当前帧访问 | Future 访问 | 语义 |
|---|---|---|---|
| `joint_pos` | `joint_pos` | `joint_pos_future` | 参考关节位置 |
| `joint_vel` | `joint_vel` | `joint_vel_future`, `joint_vel_multi_future` | 参考关节速度 |
| `body_pos_w` | `body_pos_w` | - | 参考 body 位置，当前帧会加 `env_origins` |
| `body_quat_w` | `body_quat_w` | - | 参考 body 朝向 |
| `body_lin_vel_w` | `body_lin_vel_w` | - | 参考 body 线速度 |
| `body_ang_vel_w` | `body_ang_vel_w` | - | 参考 body 角速度 |
| `anchor_pos_w` | `anchor_pos_w` | `anchor_pos_w_future` | 参考 anchor 位置，会加 `env_origins` |
| `anchor_quat_w` | `anchor_quat_w` | `anchor_quat_w_future` | 参考 anchor 朝向 |
| `anchor_lin_vel_w` | `anchor_lin_vel_w` | - | 参考 anchor 线速度 |
| `anchor_ang_vel_w` | `anchor_ang_vel_w` | - | 参考 anchor 角速度 |
| `smpl_joints` | `smpl_joints` | `smpl_joints_future` | SMPL joint reference |
| `smpl_transl` | `smpl_transl` | `smpl_transl_future` | SMPL translation |
| `smpl_poses` | `smpl_poses` | `smpl_poses_future` | SMPL pose |
| `get_smpl_*()` | several properties | several future properties | motion-local SMPL 查询 |

## MotionCommandTimeline

`MotionCommandTimeline` 拥有 per-env cursor state。

| Tensor | 形状 | 写入方 | 读取方 | 语义 |
|---|---:|---|---|---|
| `motion_ids` | `[num_envs]` | `apply_selection()`, `invalidate()` | `MotionCommand`, timeline methods, sampler | 每个 env 当前 motion |
| `local_time_steps` | `[num_envs]` | `apply_selection()`, `step()`, `invalidate()` | `MotionCommand`, timeline methods, sampler | 每个 env 当前 local frame |
| `motion_steps_len` | `[num_envs]` | `apply_selection()`, `invalidate()` | `expired_env_ids()` | 每个 env 当前 motion 的 local end |
| `future_step_offsets` | `[num_future_frames]` | init only | `global_future_steps()` | future reference offsets |

### 输入依赖

`MotionCommandTimeline` 只通过 `MotionDataSource` 读取 motion 边界信息：

- `time_step_start_idx`
- `time_step_end_idx`
- `time_step_total`
- `motion_ids_from_timestamps()`

### 输出依赖

`MotionCommandTimeline` 输出两类结果：

- 派生索引：`global_time_steps`, `global_future_steps`, `expanded_future_motion_ids`, `motion_num_steps`
- selection carrier：`build_selection_from_global_timestamps()`, `build_selection_from_motion_ids()`

## MotionReferenceCache

`MotionReferenceCache` 拥有对齐后的 reference pose cache，用于 reward/termination/debug。

| Tensor | 形状 | 写入方 | 读取方 | 语义 |
|---|---:|---|---|---|
| `body_pos_relative_w` | `[num_envs, body_count, 3]` | `refresh()` | `MotionCommand.metrics`, reward, termination, visualizer | 对齐到当前机器人 anchor XY/yaw 的 reference body 位置 |
| `body_quat_relative_w` | `[num_envs, body_count, 4]` | `refresh()` | `MotionCommand.metrics`, reward, visualizer | 对齐到当前机器人 anchor yaw 的 reference body 朝向 |

### `refresh()` 输入

| 输入 | 来源 | 语义 |
|---|---|---|
| `anchor_pos_w` | `MotionCommand.anchor_pos_w` | 当前 reference anchor 位置 |
| `anchor_quat_w` | `MotionCommand.anchor_quat_w` | 当前 reference anchor 朝向 |
| `body_pos_w` | `MotionCommand.body_pos_w` | 当前 reference body 位置 |
| `body_quat_w` | `MotionCommand.body_quat_w` | 当前 reference body 朝向 |
| `robot_anchor_pos_w` | `MotionCommand.robot_anchor_pos_w` | 当前 robot anchor 位置 |
| `robot_anchor_quat_w` | `MotionCommand.robot_anchor_quat_w` | 当前 robot anchor 朝向 |

## MotionCommandResetter

`MotionCommandResetter` 负责把选中的 reference state 写回 simulator。它不负责选择 motion，不应该修改 timeline，也不应该刷新 cache。

### 持有依赖

| 字段 | 语义 |
|---|---|
| `cfg` | reset randomization 参数 |
| `robot` | simulator state 写入对象 |
| `device` | tensor device |
| `_pose_range_env_mask` | pose-range reset env mask cache |

### `apply()` 输入张量

| 输入 tensor | 形状 | 来源 | 用途 |
|---|---:|---|---|
| `body_pos_w` | `[num_envs, body_count, 3]` | `MotionCommand.body_pos_w` | root position reset |
| `body_quat_w` | `[num_envs, body_count, 4]` | `MotionCommand.body_quat_w` | root orientation reset |
| `body_lin_vel_w` | `[num_envs, body_count, 3]` | `MotionCommand.body_lin_vel_w` | root linear velocity reset |
| `body_ang_vel_w` | `[num_envs, body_count, 3]` | `MotionCommand.body_ang_vel_w` | root angular velocity reset |
| `joint_pos` | `[num_envs, num_joints]` | `MotionCommand.joint_pos` | joint position reset |
| `joint_vel` | `[num_envs, num_joints]` | `MotionCommand.joint_vel` | joint velocity reset |
| `env_origins` | `[num_envs, 3]` | env scene | lying mode height/origin handling |

### 输出副作用

- `robot.write_joint_state_to_sim()`
- `robot.write_root_state_to_sim()`

### 重要隐含假设

`MotionCommandResetter.apply()` 当前使用 `body_pos_w[:, 0]`、`body_quat_w[:, 0]`、`body_lin_vel_w[:, 0]`、`body_ang_vel_w[:, 0]` 作为 root/anchor state。因此 `cfg.body_names[0]` 必须是 robot anchor/root。`MotionCommand.__init__()` 目前也 assert `robot_anchor_body_index == 0` 且 `motion_anchor_body_index == 0`。

## MotionSelectionPolicy

`motion_selection.py` 定义 selection policy 抽象，用来隔离 fixed eval 和 adaptive sampling。

### 统一接口

| 方法/属性 | 共享输入 | 输出/副作用 |
|---|---|---|
| `bind_motion_source()` | `motion_source`, `timeline`, `decimation`, `sim_dt` | motion reload 后重置 policy 状态 |
| `select()` | `env_ids`, `motion_source`, `timeline`, `terminated`, `metrics`, `allow_failure_accounting` | 返回 `MotionSelection` |
| `step_post_update()` | `motion_source`, `command_step_count` | 每个 command step 后更新 policy/sampler 状态 |
| `counts_eval_cycles` | - | 是否由 `MotionCommand` 统计 eval cycle |
| `fixed_eval_motion_ids` | policy owned | fixed eval motion assignment |

### FixedEvalMotionSelectionPolicy

| Tensor | 形状 | 语义 |
|---|---:|---|
| `fixed_eval_motion_ids` | `[num_envs]` | env id 到 motion id 的固定映射 |

`bind_motion_source()` 会直接调用 `timeline.apply_selection()` 初始化所有 env。`eval_cycle_count` 由 `MotionCommand` 直接拥有并在 motion source 绑定后清零。

### AdaptiveMotionSelectionPolicy

`AdaptiveMotionSelectionPolicy` 只是 adapter，真正状态在 `AdaptiveMotionSampler` 中。它把 `select()` 转发给 `sampler.sample_selection()`，把 `step_post_update()` 转发给 `sampler.step_post_update()`。

## AdaptiveMotionSampler

`AdaptiveMotionSampler` 拥有 adaptive sampling 的内部状态。

| Tensor / 参数 | 形状 | 写入方 | 语义 |
|---|---:|---|---|
| `adaptive_bin_frame_width` | scalar int | `reset_for_motion_source()` | 约 1 秒 motion frame 对应的 bin 宽度 |
| `bin_count` | scalar int | `reset_for_motion_source()` | global failure bin 数 |
| `bin_failed_count` | `[bin_count]` | `step_post_update()` | global failure EMA |
| `_current_bin_failed` | `[bin_count]` | `sample_selection()` / `step_post_update()` | 当前累计 global failure |
| `motion_sample_bin_counts` | `[motion_num]` | `reset_for_motion_source()` | 每个 motion 可采样 local bin 数 |
| `max_motion_sample_bin_count` | scalar int | `reset_for_motion_source()` | motion-local bin matrix 的第二维 |
| `motion_bin_failed_count` | `[motion_num, max_motion_sample_bin_count]` | `step_post_update()` | motion-local failure EMA |
| `_current_motion_bin_failed` | `[motion_num, max_motion_sample_bin_count]` | `sample_selection()` / `step_post_update()` | 当前累计 motion-local failure |
| `kernel` | `[adaptive_kernel_size]` | `reset_for_motion_source()` | failure smoothing kernel |
| `success_motion` | `[motion_num]` | `reset_for_motion_source()` | eval/statistics 用 |

### `sample_selection()` 输入依赖

| 输入 | 来源 | 用途 |
|---|---|---|
| `env_ids` | `MotionCommand._resample_command()` | 需要 resample 的 env |
| `motion_source` | `MotionCommand.motion` | motion 边界和 timestamp 映射 |
| `timeline` | `MotionCommand.timeline` | 当前 global timestamp / 构造 selection |
| `terminated` | `env.termination_manager.terminated` | failure accounting |
| `metrics` | `MotionCommand.metrics` | 写 sampling/failure 指标 |
| `allow_failure_accounting` | `MotionCommand` | 控制是否把本次 terminated 计入 failure bins |

### `metrics` 写入

`AdaptiveMotionSampler` 会写入以下 metric tensor：

- `failures_max`
- `failures_mean`
- `failures_min`
- `failures_max_over_uniform`
- `sampling_entropy`
- `sampling_top1_prob_max`
- `sampling_top1_prob_bin`
- `sampling_top1_prob_mean`
- `sampling_top1_prob_min`
- `prob_max_over_uniform`
- `prob_uniform`
- `num_concentrate_bins`

这些 metrics 由 `MotionCommand` 初始化，但具体 sampling 相关数值由 `AdaptiveMotionSampler` 更新。

## MotionCommandDebugVisualizer

`MotionCommandDebugVisualizer` 持有完整 `MotionCommand` 对象，而不是窄接口。因此它是当前边界最宽的只读消费者。

### 读取的 `MotionCommand` 字段/属性

| 类别 | 字段/属性 |
|---|---|
| Robot anchor | `robot_anchor_pos_w`, `robot_anchor_quat_w`, `robot_anchor_lin_vel_w` |
| Reference anchor | `anchor_pos_w`, `anchor_quat_w`, `anchor_lin_vel_w` |
| Future reference | `anchor_pos_w_future`, `anchor_quat_w_future` |
| Robot bodies | `robot_body_pos_w`, `robot_body_quat_w` |
| Aligned reference bodies | `body_pos_relative_w`, `body_quat_relative_w` |

它只负责 marker 初始化、可见性和渲染，不应该修改 timeline、motion source、simulator state、reward、termination 或 metrics。

## Cross-Package Shared State

`MotionCommand` 会设置：

```python
env._motion_pose_range_env_mask = self.pose_range_env_mask
```

该字段跨出 `cmd_modules`，被 event / termination 相关模块读取，用于 recovery assist 和 delayed termination 等逻辑。

| 字段 | 写入方 | 读取方 | 语义 |
|---|---|---|---|
| `env._motion_pose_range_env_mask` | `MotionCommand` / `MotionCommandResetter.build_recovery_assist_env_mask()` | `evt_modules`, `tmt_modules` | 哪些 env 使用 pose-range reset/randomization/recovery assist |

## 主要边界风险

1. `MotionCommand` 是强中心模块：它既持有 motion/timeline/cache/resetter/policy，又向外暴露大量 derived properties。
2. `Timeline` state 通过 alias 暴露到 `MotionCommand`，外部写 `command.motion_ids` / `command.local_time_steps` 会直接改变 timeline。
3. `ReferenceCache` state 通过 alias 暴露到 `MotionCommand`，外部写 `command.body_pos_relative_w` / `command.body_quat_relative_w` 会直接改变 reward/termination/debug 使用的 cache。
4. `AdaptiveMotionSampler` 的部分内部状态通过 legacy alias 暴露到 `MotionCommand`，如 `bin_failed_count`、`kernel`、`success_motion`。
5. `MotionCommandResetter` 依赖 `body_names[0]` 是 anchor/root 的顺序假设。
6. `MotionCommandDebugVisualizer` 读取完整 `MotionCommand`，依赖面较宽，但当前应保持只读。
7. `env._motion_pose_range_env_mask` 是跨 package 的隐式共享状态，后续如果要收紧边界，可以考虑改为显式 provider 或 manager-owned state。

## 建议的边界收敛方向

1. 将 `MotionCommand` 上的 tensor alias 改成只读 property，避免外部意外修改子模块内部状态。
2. 把 reference motion 查询从 `MotionCommand` 中拆成 `ReferenceMotionAccessor`，专门负责 current/future frame indexing。
3. 把 observation/reward/debug 面向的派生特征拆成 `ReferenceFeatureView` 或 `MotionCommandFeatures`。
4. 给 `MotionCommandDebugVisualizer` 定义只读 provider protocol，替代持有完整 `MotionCommand`。
5. 将 `env._motion_pose_range_env_mask` 收敛为显式接口，减少跨模块隐式依赖。
6. 将 `MotionCommandResetter` 的 root/anchor index 作为显式参数传入，去掉对 `body_names[0]` 的隐式依赖。
