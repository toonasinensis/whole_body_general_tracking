# `feat_motion_lib.facade` Agent Contract

`facade/` 是下游训练、评估和工具脚本应该依赖的稳定 API 层。它负责组织 discovery -> loading -> transform -> store 的流程，并把结果暴露为清晰的查询接口。

## 目录边界

允许：

- 编排多个底层模块。
- 持有 store，并通过 property / method 暴露 flat tensor 查询。
- 暴露稳定查询 property / method。
- 处理 legacy cfg 到 structured config 的适配。

禁止：

- 不把底层 IO 细节泄漏给下游。
- 不直接实现 tensor alignment/resample/index 的细节。
- 不依赖 IsaacLab env/robot。
- 不写 simulator。
- 不实现 command sampling policy。

## 包级公开 API

`facade/__init__.py` 只暴露下游训练、评估和工具脚本应直接依赖的门面类：

- `UnifiedMotionLib`
- `SmplMotionLib`

`config_adapter.py`、`loading.py` 和 `orchestration.py` 中的函数与 outcome dataclass 是 facade 内部编排层，可以被测试直测，但不作为 `feat_motion_lib.facade` 的包级稳定入口。

## `config_adapter.py`

职责：

- 将训练 cfg / SimpleNamespace / legacy object 转成 structured config。

内部入口：

- `cfg_to_unified_load_config(cfg, sample_counter, device, ...) -> UnifiedLoadConfig`
- `cfg_to_smpl_load_config(cfg, device, ...) -> SmplLoadConfig`

输入契约：

- cfg 必须至少提供 motion path 相关字段。
- legacy missing field 应使用明确 default。
- `sample_counter` 由 facade 持有和传入。

输出契约：

- 返回 `config.py` 中 dataclass。
- path 字段转换为 `Path`。
- bool/int/float 字段完成基础 cast。

禁止：

- 不访问文件系统。
- 不加载 motion。
- 不修改 cfg 原对象。

待实现：

- 明确 required cfg fields，缺失时报清晰错误。
- 将 default 值与 `MotionCommandCfg` 或训练 cfg 默认值保持同步。

## `loading.py`

职责：

- 根据已经发现的文件列表加载 robot clips / SMPL clips / paired clips。
- 汇总 loaded/skipped/warnings 到 result dataclass。

内部入口：

- `load_robot_clips(...) -> RobotLoadResult`
- `load_smpl_clips(...) -> list[SmplMotionClip]`
- `load_paired_clips(...) -> PairedLoadResult | None`

输入契约：

- 文件列表由 discovery 提供。
- target fps 和 body/joint name 参数由 orchestration/facade 提供。
- paired loading 必须知道 SMPL dir 和 max frame diff。

输出契约：

- robot clips 已完成 name alignment 和 fps 重采样。
- paired clips 已裁齐。
- report 真实反映 fallback/skipped/warnings。

禁止：

- 不做文件发现。
- 不构建 store。
- 不持有长期状态。

待实现：

- 明确单个文件加载失败时是 fail-fast 还是 skip。
- 将 warning 文案标准化，便于上层展示。

## `orchestration.py`

职责：

- 组织完整 unified load 流程。
- 决定走 paired path 还是 robot-only fallback。
- 构建 store 并返回 load outcome。

内部入口：

- `execute_smpl_load(load_config) -> SmplLoadOutcome`
- `execute_unified_load(...) -> UnifiedLoadOutcome`

输入契约：

- `UnifiedLoadConfig` 已经结构化。
- `body_indexes` / `joint_names` / `motion_body_names` / `all_body_names` 由 facade 构造时提供。
- `smpl_dir` 存在且为目录时才尝试 paired load。

输出契约：

- `UnifiedLoadOutcome.robot_store` 是已构建完成的 `RobotMotionStore`。
- paired 成功时 `smpl_lib` 非 None。
- robot-only fallback 时 `smpl_lib` 为 None 且 report 标明 fallback。
- `base_dir` 传回 facade，用于生成相对 file names。
- `next_sample_counter` 传回 facade 保存。

禁止：

- 不暴露半构建 store 给外部。
- 不做 query API。
- 不写 env/robot。

待实现：

- paired requested but failed 的 fallback 策略应更显式。
- 将 `smpl_requested = bool(smpl_dir and smpl_dir.is_dir())` 的 silent behavior 变成 warning 或错误。

## `unified_motion_lib.py`

职责：

- 面向 robot-only / paired 的统一 facade。
- 持有当前加载后的 robot flat tensors 和可选 SMPL lib。
- 暴露训练侧需要的 reference motion 查询接口。

公开 API：

- `load_from_cfg(cfg) -> LoadReport`
- `load(load_config: UnifiedLoadConfig) -> LoadReport`
- robot tensor properties: `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`
- anchor properties: `anchor_pos_w`, `anchor_quat_w`, `anchor_lin_vel_w`, `anchor_ang_vel_w`
- metadata: `fps`, `time_step_total`, `file_names`, `motion_num`, `frame_list`, `time_step_start_idx`, `time_step_end_idx`
- SMPL query methods: `get_smpl_joints`, `get_smpl_transl`, `get_smpl_pose`, `get_smpl_global_position`
- `motion_ids_from_timestamps(timestamps) -> torch.Tensor`

输入契约：

- `body_indexes` 和 `motion_anchor_body_index` 是针对输出 body order 的索引约定；调用方必须保证它们与 `motion_body_names` 一致。
- `joint_names` / `motion_body_names` / `all_body_names` 只用于 loading alignment。
- query 前必须先 load。

输出契约：

- 所有 robot tensors 在 `self.device` 上。
- `anchor_*` 是从 selected body tensor 中取 `motion_anchor_body_index`。
- global timestamp 对 flat tensor axis 0 索引。
- SMPL methods 在无 SMPL 数据时抛 `MotionNotLoadedError`。

允许副作用：

- `load()` 会 reset 当前 state。
- `load_from_cfg()` 会推进 `_sample_counter`。

禁止：

- 不读取 cfg 中训练 reward/sampling 字段。
- 不写 timeline。
- 不写 simulator。
- 不进行 command reset。

待实现：

- 增加 `_require_metadata`，避免 metadata None 时在下游报隐晦错误。
- load 后校验 quaternion norm。
- 明确 `body_indexes` 与 `motion_anchor_body_index` 的坐标：是 selected body order 还是 full robot body order，当前应坚持 selected body order。

## `smpl_motion_lib.py`

职责：

- 面向 SMPL-only 的 facade。
- 持有 `SmplMotionStore`。
- 暴露 SMPL query API。

公开 API：

- `load_from_cfg(cfg)`
- `load(load_config)`
- `load_motions(clips, target_fps=None)`
- `get_smpl_joints`
- `get_smpl_transl`
- `get_smpl_pose`
- `get_smpl_global_position`

输入契约：

- cfg 或 load_config 必须能解析出 pkl 文件列表。
- query 输入 motion id / local frame。

输出契约：

- query 前未 load 时抛 `MotionNotLoadedError`。
- 输出 tensor 在 facade device 上。

禁止：

- 不加载 robot npz。
- 不做 paired fallback。

待实现：

- 对 `load_motions(clips, target_fps=None)` 的 target fps 行为补充 contract 和测试。

## facade 层测试要求

- load 前 query 抛 `MotionNotLoadedError`。
- robot-only load metadata 正确。
- paired load 成功时 SMPL query 可用。
- paired requested but unavailable 的 fallback/report 行为明确。
- `motion_ids_from_timestamps` boundary 正确。
- `load_from_cfg` sample counter 在 eval chunk 中推进。
