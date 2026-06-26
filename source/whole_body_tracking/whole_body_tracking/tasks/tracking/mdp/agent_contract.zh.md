# MDP 包合约

本包是 tracking task 的公开 MDP 门面。任务配置默认应通过
`whole_body_tracking.tasks.tracking.mdp` 导入运行时函数和面向配置的类；只有在
确实需要子模块内部类型时，才直接导入子模块。

## 公开门面

`mdp/__init__.py` 重新导出：

- IsaacLab 内置 MDP 函数，来自 `isaaclab.envs.mdp`。
- Motion command API，来自 `cmd_modules`。
- Event API，来自 `evt_modules`。
- Observation API，来自 `obs_modules`。
- Reward API，来自 `rwd_modules`。
- Termination API，来自 `tmt_modules`。

这个门面有意保持较宽，以方便 IsaacLab 配置使用。真正的所有权边界由各
`*_modules/agent_contract.md` 文件定义。

## 模块所有权

- `cmd_modules`：负责参考动作加载、timeline 状态、motion selection、从参考姿态
  reset 机器人、对齐后的 reference cache、motion command 特征和 debug 可视化。
- `obs_modules`：负责 observation term 函数。它读取 env、asset 和 command 状态并
  返回 tensor，不修改 simulator 或 command 状态。
- `rwd_modules`：负责 reward term 函数和 delayed-termination reward masking。配置
  应使用 `rwd_modules` 中的 wrapped exports，而不是 raw functions。
- `tmt_modules`：负责 termination term 函数、delayed termination 安装和
  `DelayedTerminationManager`。
- `evt_modules`：负责 startup/interval event 函数，这些函数可以有意修改 env、
  robot defaults、PhysX 属性或外力 buffer。

## 跨模块规则

- 配置文件可以导入 `mdp` 门面符号。
- 子模块内部应优先使用所属 package 内的显式相对导入。
- Observation、reward、termination 和 event 函数可以通过
  `env.command_manager.get_term(command_name)` 读取 `MotionCommand`。
- 只有 `cmd_modules.motion_reset` 应在 command reset 时写 robot root/joint state。
- 只有 event 函数应执行 event-manager 侧效应，例如 randomization 或 assist force。
- Reward 和 termination 配置应从 `rwd_modules` 与 `tmt_modules` 的 wrapper 导入，
  以保持 delayed-termination flags 可用。

## 共享运行时状态

以下 env 属性是有意跨模块共享的：

- `env.envs_classes_mask`：由 `MotionCommand` 写入，由 delayed termination 和
  fallen upward assist 读取。`envs_classes_mask["lying"]` 选择 recovery/lying env。
  该 mask 来自 `MotionCommandResetter.envs_classes_mask`，后者由
  `cfg.commands.motion.envs_classes_ratio` 构建。
- `env.termination_manager.delayed_termination_env_mask`：由
  `DelayedTerminationManager` 写入，由 termination wrappers 读取。
- `env.termination_manager.delayed_termination_active_mask`：由
  `DelayedTerminationManager.compute()` 写入，由 reward wrappers 读取。
- `env._fallen_upward_assist_timeout_stats`：由 fallen upward assist event 拥有。

任何新的共享 env 属性都必须在这里和所属模块合约中记录。

## Command Runtime 预期

Command runtime 被拆分为较小模块：

- `MotionCommand` 编排 loading、selection、reset、cache refresh、metrics 和
  visualization。
- `MotionCommandTimeline` 只拥有 per-env motion cursor state。
- `MotionCommandResetter` 是唯一写 robot root/joint state 到 simulation 的 command
  模块。
- `MotionReferenceCache` 拥有对齐后的 reference body pose cache。
- `MotionSelectionPolicy` 选择 motion frame 并返回 `MotionSelection`。
- `AdaptiveMotionSampler` 拥有 adaptive failed-bin statistics 和 sampling metrics。

Observation、reward、termination 和 event 模块可以读取 `MotionCommand` property，
但不得修改 timeline、sampler、resetter、motion source 或 reference cache 状态。

## 兼容性说明

- Quaternion tensor 使用 IsaacLab `wxyz` 顺序。
- Env-indexed tensor 应位于 simulation device，除非 Isaac/PhysX API 明确要求 CPU
  tensor。
- Observation、reward 和 termination term 的 per-env 输出首维必须是 `env.num_envs`。
- 通过 `__all__` 导出的公开函数是面向配置的 API。重命名或改签名可能破坏
  Hydra/IsaacLab 配置和 checkpoint。
- `cfg.commands.motion.adaptive_uniform_ratio` 仍为兼容性存在，但当前 adaptive
  sampling 使用 `motion_ratio[0]` 作为 uniform probability 权重。
