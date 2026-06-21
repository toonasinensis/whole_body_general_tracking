# Motion Command Sub-Module Contract

本文档定义 `motion_command_runtime.py` / `commands.py` 拆分到 `sub_modules/` 后，各模块的职责边界、允许的副作用、输入输出契约，以及后续待实现工作。

核心原则：`MotionCommand` 只保留 IsaacLab `CommandTerm` 生命周期编排；reference motion 数据、timeline 状态、采样策略、reset 写 sim、cache 对齐、debug visualization、observation feature 组装分别由独立模块负责。

## 总体边界

`MotionCommand` 是 orchestrator：

- 负责从 env/cfg/robot 建立依赖：robot body index、motion anchor index、future step offsets、metrics。
- 负责调用模块：load/resample motion source -> bind selection policy -> select/apply timeline -> reset sim -> refresh cache -> update metrics/debug。
- 不应直接实现 motion 文件加载、采样概率、timeline 索引换算、reset noise、reference alignment、debug marker 创建、observation feature 组装等细节。

`sub_modules` 是 runtime service 层：

- 每个模块只拥有自己领域内的状态。
- 模块之间通过小接口通信，优先使用 `MotionDataSource`、`MotionSelection`、tensor 输入输出，避免传入整个 `MotionCommand`。
- 只有 resetter 可以写 simulator；只有 timeline 可以写 per-env motion cursor；只有 motion source/loader 可以写 motion buffer；只有 visualizer 可以持有 `VisualizationMarkers`。

## 共享接口：`interface.py`

### `MotionDataSource`

职责：

- 描述一个已加载 reference motion 数据源的最小索引接口。
- 给 timeline / sampler / exporter 提供 motion 边界信息。

必须提供：

- `time_step_total: int`
- `time_step_start_idx: torch.Tensor`
- `time_step_end_idx: torch.Tensor`
- `motion_ids_from_timestamps(timestamps: torch.Tensor) -> torch.Tensor`

可选但常用：

- `motion_num`
- `file_names`
- `fps`
- reference tensors: `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`
- anchor views: `anchor_pos_w`, `anchor_quat_w`, `anchor_lin_vel_w`, `anchor_ang_vel_w`

边界：

- `MotionDataSource` 不负责采样策略。
- `MotionDataSource` 不持有 env/robot。
- `MotionDataSource` 不写 timeline，不写 simulator。

待实现：

- 将 `MotionLoader` 从 `commands.py` 移入独立模块，建议文件名 `motion_source.py` 或 `motion_loader.py`。
- 把 `MotionDataSource` 协议扩展成两个层次：`MotionIndexSource` 只暴露索引；`ReferenceMotionSource` 暴露 reference tensor 和 anchor view。
- 明确 tensor shape / device / dtype，例如 global frame 维度必须在第 0 维，timestamp 必须为 long。

### `MotionSelection`

职责：

- 作为采样策略到 timeline 的唯一写入载体。
- 描述每个 env 即将绑定到哪个 motion、该 motion 的 local frame、motion 的 frame end。

字段：

- `motion_ids`: shape `(num_selected_envs,)`, dtype long
- `local_time_steps`: shape `(num_selected_envs,)`, dtype long
- `frame_end`: shape `(num_selected_envs,)`, dtype long，表示当前 motion 的 local exclusive end。

边界：

- `MotionSelection` 不应包含 body/joint reference tensor。
- `MotionSelection` 不应直接应用到 simulator；必须先交给 timeline。

待实现：

- 在 dataclass 上补充 shape 校验 helper，例如 `validate(num_envs, device)`。
- 统一命名：`frame_end` 或 `local_frame_end` 二选一，建议改为 `local_frame_end`。

## Timeline：`motion_timeline.py`

### `MotionCommandTimeline`

职责：

- 拥有所有 env 的当前 reference cursor：`motion_ids`、`local_time_steps`、`frame_end_per_env`。
- 负责 local/global timestamp 换算。
- 负责 future frame timestamp 计算和 clip-to-last-frame。
- 负责判断哪些 env 的 reference 即将过期。
- 在 fixed eval 模式下记录 `eval_cycle_count`。

允许持有状态：

- `motion_ids`
- `local_time_steps`
- `frame_end_per_env`
- `future_step_offsets`
- `eval_cycle_count`

允许的副作用：

- `apply_selection()` 写 timeline 自己的 cursor。
- `step()` 增加 `local_time_steps`。
- `invalidate()` 清空 timeline 自己的 cursor。

禁止：

- 不加载 motion 文件。
- 不读取具体 body/joint tensor，除非通过 `MotionDataSource` 的 index 边界。
- 不计算 sampling probability。
- 不写 simulator。
- 不做 reset noise。
- 不更新 metrics。

调用契约：

- motion 数据 reload 后必须先 `invalidate()`，再由 selection policy `bind_motion_source()`，然后按需要 `apply_selection()`。
- `selection_from_global_timestamps()` 输入 global timestamp，输出 local timeline selection。
- `selection_from_motion_ids()` 默认 local time 为 0，适合 fixed eval 或从 motion 开头采样。
- `_update_command()` 中应先 `timeline.step()`，再用 `expired_env_ids(max_future_step)` 找出需要 resample 的 env。

待实现：

- 将 `future_step_offsets` 的创建和 `max_future_step` 一起封装成 `FutureStepSpec`，避免 cfg 内 `future_step_num` 和 `max_future_step` 不一致。
- `expired_env_ids()` 对 `frame_end_per_env == 0` 的未初始化状态给出更明确行为。
- 增加单元测试覆盖 motion 边界、timestamp bucket 边界、future clamp、empty env ids。
- 统一函数命名：`global_time_steps()` / `future_time_steps()` 都表示 derived index，可考虑迁到 `ReferenceMotionAccessor`，timeline 只保留 cursor 状态。

## Motion Source / Loader：待新增模块

建议文件：`motion_source.py` 或 `motion_loader.py`

当前位置：`commands.py::MotionLoader`

职责：

- 发现 motion npz 文件。
- 按 cfg 策略选择文件：all / max_motion_num random sample / eval sequential chunk / distributed shard / dataset txt。
- 加载 npz 中 reference tensor 并拼接到目标 device。
- 建立 per-motion frame list、start/end global index、file names、fps。
- 暴露 `MotionDataSource` 所需接口。
- 提供 body selection 和 anchor view。

允许持有状态：

- motion raw tensors
- selected body tensors
- frame metadata
- eval sample counter
- file paths / names / fps

禁止：

- 不依赖 `CommandTerm`。
- 不依赖 robot sim state。
- 不写 timeline。
- 不写 simulator。
- 不负责 adaptive sampling。

输入契约：

- cfg 中读取 motion 文件相关字段：`motion_file`、`max_motion_num`、`dataset_txt`、`eval_mode`、`distributed`、`local_rank`、`total_rank`。
- body selection 使用 `body_indexes` 和 `motion_anchor_body_index`，但不应自己查询 robot body names。

输出契约：

- 所有 tensor 的第 0 维为 concatenated global frame。
- `time_step_end_idx` 为 exclusive end。
- `motion_ids_from_timestamps()` 必须 clamp 到合法范围，boundary timestamp 应映射到下一个 motion。

待实现：

- 从 `commands.py` 移出，并把 `_find_npz_files()`、`load_and_cat_npz_with_filenames()`、`resample_motionloader()` 重命名为更清晰的 public API。
- 修复/清理乱码注释和 print 文案。
- 将 file sampling policy 从 loader 中进一步拆出，避免 loader 同时负责 IO 和策略。
- 明确失败文件处理策略：跳过、报错、记录 warning 三选一。
- 增加加载完成后的 shape 校验：joint/body tensors frame 数一致、fps 一致、body index 合法。

## Reference Accessor：待新增模块

建议文件：`motion_accessor.py`

当前位置：`commands.py` 中大量 `@property`：`joint_pos`、`body_pos_w`、`anchor_pos_w`、future variants 等。

职责：

- 根据 `MotionDataSource` + `MotionCommandTimeline` 读取当前帧和 future frames。
- 处理 env origin offset。
- 提供 current/future reference tensor 给 resetter/cache/metrics/debug/features。

输入：

- `motion_source`
- `timeline`
- `env_origins`
- `num_envs`

输出：

- current reference: joint/body/anchor pos, quat, lin vel, ang vel
- future reference: future anchor pos/quat、joint pos/vel 等
- derived index: `global_time_steps`、`future_time_steps`、`motion_num_steps`

边界：

- 只读 motion source 和 timeline。
- 不改变 timeline。
- 不做 reference alignment 到 robot 当前 pose；那是 `MotionReferenceCache` 的职责。
- 不组装 policy observation；那是 `ReferenceFeatureView` 的职责。

待实现：

- 将 `commands.py` 中 reference-motion query properties 迁入该模块。
- 定义 current/future 返回 tensor 的 shape，并保证 future tensor 不越界。
- 处理 env origin offset 的所有权：建议由 accessor 负责 world position 加 origin，motion source 保持 dataset local/world 原始值。

## Reference Cache：`motion_cache.py`

### `MotionReferenceCache`

职责：

- 维护与当前 robot anchor XY-yaw 对齐后的 reference body pose cache。
- 为 reward / metrics / debug 提供 aligned reference body pose。
- 保持 reference motion 的 z height profile，同时把 XY/yaw 对齐到 robot 当前 anchor。

允许持有状态：

- `body_pos_relative_w`
- `body_quat_relative_w`

允许的副作用：

- `refresh()` 写 cache tensor。

禁止：

- 不读取 motion source。
- 不读取 timeline。
- 不读取 robot 对象；robot state 必须由 caller 传入 tensor。
- 不写 simulator。
- 不做 reward 计算，只提供 aligned reference。

输入契约：

- `anchor_pos_w`, `anchor_quat_w`: reference anchor 当前帧。
- `body_pos_w`, `body_quat_w`: reference body 当前帧。
- `robot_anchor_pos_w`, `robot_anchor_quat_w`: robot 当前 anchor。
- quaternion 约定为 wxyz。

输出契约：

- `body_pos_relative_w` shape `(num_envs, body_count, 3)`
- `body_quat_relative_w` shape `(num_envs, body_count, 4)`

待实现：

- 重命名 `relative`，当前语义更像 `aligned_body_pos_w` / `aligned_body_quat_w`，不是 body local relative。
- 增加 shape/device 校验。
- 明确是否只对齐 yaw，roll/pitch 是否永远来自 reference。
- 补测试：anchor yaw difference、z 保持 reference、batch body_count。

## Resetter：`motion_reset.py`

### `MotionCommandResetter`

职责：

- 在 env reset/resample 时，将选中的 reference pose 写回 simulator。
- 添加 root pose/velocity/joint noise。
- clip 到 robot limits。

允许持有状态：

- cfg 中 reset randomization 相关字段。
- robot handle。
- device。

允许的副作用：

- 唯一允许写 simulator 的 sub-module。
- 调用 `robot.write_joint_state_to_sim()`。
- 调用 `robot.write_root_state_to_sim()`。

禁止：

- 不选择 motion。
- 不修改 timeline。
- 不重新加载 motion。
- 不刷新 reference cache。
- 不更新 metrics。

输入契约：

- `env_ids` 是需要 reset 的 env。
- body tensors 应来自当前 timeline selection 对应的 reference frame。
- 当前实现假设 `body_pos_w[:, 0]` 是 root/anchor。这个假设必须在 contract 中显式保留，或改为传入 `root_body_index`。

输出契约：

- 无返回值。
- simulator 中对应 env 的 root state 和 joint state 被更新。

待实现：

- 去掉 `body_names[0] MUST be robot's anchor` 的隐式假设，改为构造时传入 `root_body_index` / `motion_anchor_body_index`。
- 将 cfg 缩小为 `ResetNoiseCfg`，避免 resetter 依赖完整 `MotionCommandCfg`。
- `max_ang_vel_root = 20.0` 移入 cfg。
- joint noise 目前对所有 env 的 `joint_pos` clone 都加 noise，再只写 env_ids；建议只对 `env_ids` 生成/应用 noise。
- 增加 reset tensor shape 校验和 device 校验。

## Sampling：`motion_sampling.py`

### `AdaptiveMotionSampler`

职责：

- 维护 adaptive sampling bin 状态。
- 根据 terminated/failure 历史和 cfg 计算 sampling probability。
- 从 global motion timeline 中采样 timestamps。
- 输出 `MotionSelection`。
- 更新 adaptive sampling metrics。
- 可选导出 high-failure bins 到 json。

允许持有状态：

- `bin_count`
- `bin_failed_count`
- `_current_bin_failed`
- `kernel`
- `success_motion`
- `_last_export_step`

允许的副作用：

- 更新自身 bin failure state。
- 更新传入的 adaptive sampling metrics。
- 写 adaptive bins json 文件。

禁止：

- 不直接 `timeline.apply_selection()`。
- 不写 simulator。
- 不加载 motion 文件。
- 不刷新 reference cache。
- 不创建 debug markers。

输入契约：

- `reset_for_motion_source()` 必须在 motion reload 后调用。
- `sample_selection()` 要求 `env_ids` 非空。
- `terminated` 是 env 维度 bool tensor。
- `metrics` 必须已包含 adaptive metrics keys，或后续改为由 sampler 注册。

输出契约：

- `sample_selection()` 返回 `MotionSelection`，由 caller 或 selection policy 应用到 timeline。
- eval mode 下当前行为会把 `local_time_steps` 清零；后续应明确这是 adaptive eval 语义还是历史兼容。

待实现：

- 移除未使用 import：`quat_apply`、`quat_from_euler_xyz`、`quat_inv`、`quat_mul`、`yaw_quat`。
- 把 json export 拆到 `AdaptiveBinExporter`，sampler 只产出数据。
- 明确 `motion_ratio` 三项含义并校验和是否为 1。
- 处理 `bin_failed_count.mean() == 0` 时 probability 退化为 uniform 的逻辑，避免全 0 分布只靠 epsilon 隐式工作。
- metrics key 由 sampler 提供 `required_metric_names()` 或 `init_metrics(num_envs, device)`，不要散落在 `commands.py`。

## Selection Policy：`motion_selection.py`

### `MotionSelectionPolicy`

职责：

- 统一不同 motion selection 策略的接口。
- 屏蔽 fixed eval / adaptive / unsupported 的差异。
- 输出 `MotionSelection`，由 caller 应用到 timeline。

方法契约：

- `bind_motion_source(motion_source, timeline, decimation, sim_dt)`：motion reload 后重置策略状态。
- `select(env_ids, motion_source, timeline, terminated, metrics, allow_failure_accounting)`：为 env_ids 生成 selection。
- `step_post_update(motion_source, command_step_count)`：每个 command step 结束后的策略状态更新。
- `counts_eval_cycles`：策略是否代表 deterministic per-clip eval，因此 timeline 过期时需要计数。

边界：

- selection policy 可以调用 sampler。
- selection policy 可以在 `bind_motion_source()` 中初始化 timeline，例如 fixed eval。
- selection policy 不应写 simulator。
- selection policy 不应刷新 reference cache。
- selection policy 不应持有完整 `MotionCommand`。

### `FixedEvalMotionSelectionPolicy`

职责：

- eval 模式下让 `env_id == motion_id`。
- 每个 env 固定对应一个 motion，从 local frame 0 开始。

契约：

- 要求 `num_envs == motion_num`。
- `bind_motion_source()` 会初始化 timeline，并清零 `eval_cycle_count`。
- `select()` 对 env_ids 返回固定 motion id，local frame 默认 0。

待实现：

- 支持 `num_envs != motion_num` 的分批 eval 或循环映射，当前只能一一对应。
- 明确 eval 完成条件：所有 `eval_cycle_count` >= 1，还是由外部 manager 判断。

### `AdaptiveMotionSelectionPolicy`

职责：

- 作为 `AdaptiveMotionSampler` 的 adapter。
- 把 common policy interface 转发到 sampler。

待实现：

- 暴露 sampler legacy state 的方式应收敛，避免 `MotionCommand` 复制 `bin_count/kernel/...`。
- adaptive eval 的语义和 fixed eval 分开命名。

### `UnsupportedMotionSelectionPolicy`

职责：

- 保留历史行为：不支持的 cfg 在 select 时显式报错。

待实现：

- 增加 `UniformMotionSelectionPolicy`，覆盖 `adaptive_sample=False` 且非 fixed eval 的普通训练场景。

## Debug Visualizer：`motion_viser.py`

### `MotionCommandDebugVisualizer`

职责：

- 拥有所有 debug visualization marker。
- 根据当前 robot state 和 reference state 渲染 current/goal/future marker。
- 可选渲染 anchor velocity arrow。

允许持有状态：

- `VisualizationMarkers`
- visualizer config
- device

允许的副作用：

- 创建 marker prim。
- 设置 marker visibility。
- 调用 marker `visualize()`。

禁止：

- 不加载 motion。
- 不修改 timeline。
- 不写 simulator。
- 不更新 metrics。
- 不计算 reward。

当前临时依赖：

- 目前 visualizer 持有整个 `command` 并读取大量 properties，这是可工作的过渡方案，但边界过宽。

目标输入契约：

- 改为接收一个只读 `DebugStateProvider` 或每帧 `render(state)`。
- state 应包含：
  - current robot anchor/body pose
  - current reference anchor/body pose
  - future reference anchor pose
  - optional anchor velocity

待实现：

- 文件名建议从 `motion_viser.py` 改为 `motion_visualizer.py`。
- 使用 Protocol 替代完整 `command` 依赖。
- `current_anchor_lin_vel_visualizer` 和 `goal_anchor_lin_vel_visualizer` 类型应允许 `None`。
- 对 `cfg.debug_anchor_speed=False` 路径增加保护，避免访问未创建 visualizer。

## Feature View：待新增模块

建议文件：`motion_features.py`

当前位置：`commands.py::command` 以及 anchor body-frame velocity/gravity/6d rotation properties。

职责：

- 组装 policy observation-facing command tensor。
- 提供 reward/metric 常用 derived features。
- 把 reference accessor 的 raw tensor 转成 body frame、6D rotation、projected gravity 等。

输入：

- reference accessor 或 explicit reference tensors。
- robot/environment constants，例如 `GRAVITY_VEC_W`。

输出：

- `command: torch.Tensor`
- `anchor_lin_vel_b`
- `anchor_ang_vel_b`
- `anchor_project_gravity`
- `anchor_6d_rotation`

边界：

- 不读 motion 文件。
- 不修改 timeline。
- 不写 simulator。
- 不更新 selection policy。
- 不创建 debug markers。

待实现：

- 将 observation 组合从 `MotionCommand.command` property 移入。
- 明确 command tensor layout，并把 layout 写成配置或常量，避免后续改 observation 时破坏 checkpoint 兼容。
- 增加 shape 测试，尤其是 future feature 展平顺序。

## Metrics：待新增或保留在 orchestrator 的边界

当前位置：`commands.py::_update_metrics`

建议：

- 短期可留在 `MotionCommand`，因为 IsaacLab metrics dict 由 `CommandTerm` 管。
- 中期拆为 `MotionCommandMetrics`，只负责根据 robot state、reference accessor、reference cache 更新 metrics dict。

职责：

- 计算 anchor/body/joint error。
- 写 `self.metrics` 中已有 key。

禁止：

- 不选择 motion。
- 不写 simulator。
- 不刷新 cache。

待实现：

- adaptive metrics 初始化应迁到 selection/sampler 模块。
- tracking error metrics 初始化可由 metrics 模块提供 `init_metrics(metrics, num_envs, device)`。

## 推荐调用流程

### 初始化

1. `MotionCommand` 解析 cfg/env/robot，建立 body index 和 anchor index。
2. 创建 `MotionSource`、`Timeline`、`ReferenceAccessor`、`ReferenceCache`、`Resetter`、`SelectionPolicy`、`FeatureView`、`DebugVisualizer`。
3. `motion_source.reload()`
4. `timeline.invalidate()`
5. `selection_policy.bind_motion_source(motion_source, timeline, decimation, sim_dt)`
6. 初始化 metrics。
7. `reference_cache.refresh(...)`

### 每个 command step

1. `command_step_count += 1`
2. `timeline.step()`
3. `env_ids = timeline.expired_env_ids(max_future_step)`
4. 如果 fixed eval，过期 env 增加 `eval_cycle_count`。
5. 如果到达 `resample_interval`，reload motion source，invalidate timeline，bind policy，全 env resample。
6. `selection = selection_policy.select(env_ids, ...)`
7. `timeline.apply_selection(env_ids, selection)`
8. `resetter.apply(env_ids, current_reference_tensors)`
9. `reference_cache.refresh(current_reference_tensors, robot_anchor_tensors)`
10. `selection_policy.step_post_update(...)`

### Debug render

1. IsaacLab debug callback 调用 visualizer。
2. visualizer 只读取当前 state snapshot/provider。
3. 不改变 motion/timeline/sim/metrics。

## 当前拆分的主要问题

- `motion_command_runtime.py` 和 `sub_modules/` 存在重复实现，后续需要选择单一来源，避免两个版本漂移。
- `commands.py` 仍然包含 `MotionLoader`、reference accessor、feature view、metrics 初始化，尚未完成真正边界拆分。
- 多个模块依赖完整 cfg，建议逐步收窄成局部 cfg dataclass。
- visualizer 依赖完整 `MotionCommand`，短期方便，长期会把所有 property 都变成隐式 contract。
- resetter 隐式假设 `body_pos_w[:, 0]` 是 root/anchor，应改为显式 index。
- adaptive sampler 既负责采样，又负责 metrics 写入和 json export，职责偏大。

## 优先实现清单

1. 建立 `sub_modules/__init__.py` 的 public exports，并让 `motion_command_runtime.py` 逐步只 re-export 或删除。
2. 将 `MotionLoader` 迁到 `motion_source.py`，让它明确实现 `MotionDataSource`。
3. 新增 `ReferenceMotionAccessor`，迁出 `commands.py` 中 current/future reference properties。
4. 新增 `MotionCommandFeatures`，迁出 `command` observation 和 body-frame derived features。
5. 收窄 `MotionCommandResetter` cfg 依赖，显式传入 anchor/root body index。
6. 拆出 adaptive bin json exporter，并让 sampler 提供 metrics 初始化/required keys。
7. 将 visualizer 从读取完整 command 改为读取只读 provider/state。
8. 为 timeline、motion source timestamp mapping、reference cache alignment、resetter noise/clip 添加最小单元测试。

