# `feat_motion_lib` Agent Contract

本文档定义 `feat_motion_lib/` 顶层模块的职责边界。更细的目录级 contract 位于：

- `io/agent_contract.md`
- `transform/agent_contract.md`
- `store/agent_contract.md`
- `facade/agent_contract.md`
- `tests/agent_contract.md`

核心原则：`feat_motion_lib` 是 motion data infrastructure，不是训练 runtime。它负责把磁盘上的 robot / SMPL motion 转换成稳定、可索引、可查询的张量状态；它不依赖 IsaacLab env，不写 simulator，不实现 command sampling policy。

## 总体数据流

```text
user cfg / structured config
    -> facade config adapter
    -> discovery
    -> file loaders
    -> transforms
    -> stores
    -> facade query API
```

## 顶层模块边界

### `config.py`

职责：

- 定义结构化 load 配置。
- 表达文件发现策略、加载选项、paired SMPL 选项。
- 作为 facade / orchestration / tests 之间的稳定配置载体。

公开数据结构：

- `DiscoveryOptions`
- `LoadOptions`
- `SmplLoadConfig`
- `UnifiedLoadConfig`

输入契约：

- `motion_dir` / `motion_files` 应在调用 load 前由上层给出。
- `target_fps` 表示输出 clip 的目标 fps。
- `device` 表示 store 最终张量所在设备。
- `up_axis` 只描述 SMPL 坐标转换需求，不应影响 robot `.npz` 原始语义。

禁止：

- 不访问文件系统。
- 不导入 torch。
- 不做 cfg object 兼容解析；这属于 `facade/config_adapter.py`。
- 不包含训练环境字段，除非它们是 motion loading 的真实输入。

待实现：

- 为 rank 字段增加合法性说明：`total_rank > 0`，`0 <= local_rank < total_rank`。
- 明确 `max_motion_num=-1` 表示加载全部，非负整数表示最多加载数量。

### `types.py`

职责：

- 定义跨层共享的数据结构。
- 区分 clip-level 数据、flat-store 数据、load report 数据。
- 替代临时 dict 作为模块之间的主要数据载体。

核心类型：

- `MotionData`: 单个 SMPL clip。
- `RobotMotionData`: 单个 robot clip。
- `PairedMotionData`: 已配对并裁齐的 robot + SMPL clip。
- `MotionIndex`: flat tensor 的 per-motion 起止索引。
- `UnifiedMotionState`: store 输出给 unified facade 的 robot flat state。
- `LoadReport`: 一次 load 的审计信息。
- `RobotLoadResult` / `PairedLoadResult`: loading 层结果。

shape 契约：

- `MotionData.pose_aa`: `(T, J * 3)` 或 `(T, J, 3)`。
- `MotionData.smpl_joints`: `(T, J, 3)`。
- `MotionData.transl`: `(T, 3)`。
- `RobotMotionData.joint_pos`: `(T, num_joints)`。
- `RobotMotionData.joint_vel`: `(T, num_joints)`。
- `RobotMotionData.body_pos_w`: `(T, num_bodies, 3)`。
- `RobotMotionData.body_quat_w`: `(T, num_bodies, 4)`, wxyz, unit quaternion。
- `RobotMotionData.body_lin_vel_w`: `(T, num_bodies, 3)`.
- `RobotMotionData.body_ang_vel_w`: `(T, num_bodies, 3)`.
- `MotionIndex.start_idx` and `MotionIndex.end_idx`: `(motion_num,)`, long, exclusive end.
- `UnifiedMotionState` tensors use concatenated global frame dimension at axis 0.

禁止：

- dataclass 不应加载文件或变换数据。
- 不应把 env、robot、command manager 对象放入这些类型。
- 不应在类型定义中隐藏 device transfer。

待实现：

- 增加轻量 `validate_*` helper，集中检查 shape / dtype / fps / quaternion norm。
- 明确 `duration` 的定义：当前按 `(num_frames - 1) / fps`。

### `errors.py`

职责：

- 定义 motionlib 自己的错误层次。
- 让调用方可以区分文件错误、数据错误、对齐错误、未加载查询错误。

错误语义：

- `MotionFileError`: 文件无法读取、格式无法解析。
- `MotionValidationError`: 必需 key 缺失、shape 不一致、fps 为空等数据校验失败。
- `MotionAlignmentError`: joint/body 名称或维度无法映射。
- `MotionNotLoadedError`: facade/store 在 load 前被查询。

禁止：

- 不捕获或吞掉错误。
- 不打印日志。
- 不依赖具体 IO 实现。

### `api.py`

职责：

- legacy compatibility / script helper 层。
- 当前不属于 `feat_motion_lib/__init__.py` 的包级公开入口。
- 长期应删除或迁移到明确的 CLI / tools 模块。

允许：

- 调用 `io` / `facade` 层公开 API。
- 做极薄的格式转换，例如 tensor -> numpy -> joblib dump。

禁止：

- 不实现新的加载流程。
- 不绕开 facade 直接拼接 store。
- 不加入训练环境逻辑。

待实现：

- 删除未使用的 legacy wrapper，或迁移到明确的脚本工具模块。
- `resample_and_save_motion_file()` 的输出格式应与 `io/smpl_pkl.py` 的输入契约保持一致。

### `__init__.py`

职责：

- 控制包级公开符号。
- 只暴露下游调用方应直接依赖的稳定配置、核心 clip 类型和 facade 类。

公开 API：

- `DiscoveryOptions`
- `LoadOptions`
- `SmplLoadConfig`
- `UnifiedLoadConfig`
- `LoadReport`
- `MotionData`
- `PairedMotionData`
- `PairedPath`
- `RobotMotionData`
- `SmplMotionLib`
- `UnifiedMotionLib`

禁止：

- 不在 import 时执行文件发现、load 或 device 初始化。
- 不转发 `io` / `transform` / `store` 的底层 helper。
- 不通过 `api.py` 暴露 legacy wrapper。

## 跨模块规则

- `io` 层只读文件并产生 clip dataclass。
- `transform` 层只做纯数据变换，不发现文件，不拼接 store。
- `store` 层只拼接和索引 clip，不读取磁盘。
- `facade` 层编排流程并暴露稳定查询 API。
- `tests` 层可以使用临时文件和 fake clip，但不能依赖真实训练数据。

## 关键不变量

- 所有 quaternion 对外统一为 wxyz。
- robot `body_quat_w` 必须保持单位长度，尤其是重采样之后。
- flat global frame index 的 axis 0 是唯一全局时间轴。
- `time_step_end_idx` 是 exclusive end。
- file ordering 必须可解释并在 `LoadReport` 或 `file_names` 中可追踪。
- paired robot / SMPL clip 必须在 clip 粒度和 frame 粒度一致。
