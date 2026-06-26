# Termination Modules 合约

`tmt_modules` 拥有 termination term 函数和 delayed termination 行为。公开 API 以
`tmt_modules/__init__.py` 中的 `__all__` 为准。

## 公开导出

Delayed termination：

- `DelayedTerminationManager`
- `install_delayed_termination`
- `tmt_cnd_disable_wrapper`

Wrapped termination terms：

- `bad_anchor_pos`
- `bad_anchor_pos_z_only`
- `bad_anchor_ori`
- `bad_motion_body_pos`
- `bad_motion_body_pos_z_only`

配置应使用 `tmt_modules` 或门面 `mdp` 中的 wrapped exports，而不是直接使用
`tmt_functions.py` 中的 raw functions。

## `DelayedTerminationManager`

`DelayedTerminationManager` 包装 IsaacLab `TerminationManager`，并为选中的 env
子集延迟 bad termination。

其他模块有意读取的公开属性：

- `delayed_termination_env_mask`：bool mask，表示哪些 env 可被 delay。
- `delayed_termination_active_mask`：bool mask，表示哪些 env 当前 termination 被
  delay window 抑制。

内部状态：

- `_delay_env_mask`
- `_delay_counters`
- `_max_delay_steps`

公开方法：

- `reset(env_ids=None)`
- `compute()`

`compute()` 调用 base termination computation，在 `_max_delay_steps` 内对 eligible
env 抑制 `_terminated_buf`，并返回 truncated-or-terminated dones。

## 安装合约

`install_delayed_termination(env, env_ids, delay_reset_env_ratio, max_delay_steps,
use_motion_pose_range_mask=True)` 是预期用于 startup mode 的 event function。

当 `use_motion_pose_range_mask=True` 时，它从 `env.envs_classes_mask["lying"]`
推导 delayed env。如果该 runtime mask 尚不可用，可以 fallback 到
`env.cfg.commands.motion.envs_classes_ratio["lying"]`，并构建与
`MotionCommandResetter` 相同的 contiguous prefix mask。该逻辑必须与
`MotionCommandResetter` 和 `cfg.envs_classes_ratio` 保持一致。

当 `use_motion_pose_range_mask=False` 时，它选择前
`int(env.num_envs * delay_reset_env_ratio)` 个 env。

如果启用且尚未安装，该函数会把 `env.termination_manager` 替换为
`DelayedTerminationManager`。

## Wrapper 合约

`tmt_cnd_disable_wrapper()` 包装 raw termination functions，并添加：

```python
disable_on_delayed_termination_envs: bool = False
```

启用时，wrapper 会在 `env.termination_manager.delayed_termination_env_mask` 为
true 的 env 上屏蔽 termination 结果。

这是面向配置的行为。环境配置中不要绕过 wrappers。

## Raw Function 所有权

`tmt_functions.py` 包含纯 termination predicates。它们可以读取：

- `env.command_manager.get_term(command_name)` 返回的 `MotionCommand`。
- `env.scene[asset_cfg.name]` 用于 gravity/orientation checks。

它们必须返回形状为 `(env.num_envs,)` 的 bool tensor。

禁止的副作用：

- Raw termination 函数不得修改 env、command state、simulator state、metrics、
  rewards 或 event state。
- 只有 `install_delayed_termination()` 可以替换 `env.termination_manager`。

## 对 `MotionCommand` 的依赖

Termination terms 依赖：

- `anchor_pos_w`, `anchor_quat_w`
- `robot_anchor_pos_w`, `robot_anchor_quat_w`
- `body_pos_relative_w`
- `robot_body_pos_w`
- `cfg.body_names`

如果这些 command property 的形状或语义变化，需要更新本文档。

## 留待改进

- 如果 env class ownership 从 `MotionCommandResetter` 迁移出去，需要同时更新
  `install_delayed_termination()` 和本文档。
- fallback path 重复了 contiguous-mask 构建逻辑；如果更多模块需要 env class masks，
  应优先引入共享 provider。
