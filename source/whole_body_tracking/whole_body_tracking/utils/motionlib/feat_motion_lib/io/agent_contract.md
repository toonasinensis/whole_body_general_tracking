# `feat_motion_lib.io` Agent Contract

`io/` 负责从磁盘读取 motion 文件，并把原始文件格式转换成 clip-level dataclass。

它可以调用 `transform` 中的对齐和重采样函数，但不负责 store 拼接、facade 查询或训练 runtime 行为。

## 目录边界

允许：

- 访问文件系统。
- 解析 `.npz` / `.pkl` / dataset txt。
- 做文件格式相关校验。
- 调用纯 transform：名称对齐、fps 重采样。
- 返回 `MotionData` / `RobotMotionData` 或文件列表。

禁止：

- 不拼接多个 clip 成 flat tensor；这是 `store` 的职责。
- 不持有全局 load 状态。
- 不依赖 IsaacLab、env、robot。
- 不写 simulator。
- 不实现 adaptive sampling 或 command runtime 逻辑。

## 包级公开 API

`io/__init__.py` 只暴露外部调用方应依赖的加载入口：

- `discover_npz_files`
- `discover_pkl_files`
- `load_smpl_motion_file`
- `load_pkl`
- `load_robot_motion_file`

子模块中的 raw loader、parser、字段读取 helper 可以被测试直测，也可以供同包维护使用，但不作为 `feat_motion_lib.io` 的包级稳定入口。

## `discovery.py`

职责：

- 发现 robot `.npz` 文件和 SMPL `.pkl` 文件。
- 支持目录扫描、`dataset_txt`、eval 顺序切片、distributed rank 切片。
- 返回明确、有序、可复现的文件列表。

包级公开 API：

- `discover_npz_files(options: DiscoveryOptions) -> tuple[list[str], int]`
- `discover_pkl_files(input_path: str | Path) -> list[str]`

模块内部入口：

- `find_npz_files(...) -> tuple[list[str], int]`

输入契约：

- `dir_path` / `motion_dir` 必须指向目录，除非调用的是 `discover_pkl_files()` 且输入可以是单个 `.pkl`。
- `dataset_txt` 中相对路径相对于 `dir_path` 解析。
- `motion_num == -1` 表示不限制数量。
- `sample_counter` 只用于 eval sequential chunk。

输出契约：

- 返回路径为字符串。
- 返回列表顺序必须 deterministic，除非明确执行训练随机采样。
- `next_counter` 只在 eval sequential chunk 成功消耗一个 chunk 后递增。

允许副作用：

- 可以 `warnings.warn()` 跳过 dataset txt 中的无效条目。
- 可以打印简短加载范围信息，但长期建议改为 report/warnings。

禁止：

- 不打开 `.npz` / `.pkl` 读取 motion 内容。
- 不构造 clip dataclass。
- 不改变全局 random seed，除非明确保留 legacy 行为并在 contract 中说明。

待实现：

- 修正 eval slice 的 exclusive end 语义，避免少取最后一个文件。
- 分离 random sampling 与 distributed sharding 的优先级规则。
- 将 print 改成 `LoadReport.warnings` 或可注入 logger。
- 明确 dataset txt 是否允许注释、空行、重复路径。

## `robot_npz.py`

职责：

- 读取 robot motion `.npz`。
- 校验必需 key 和基本 shape。
- 解析 fps。
- 按 joint/body name 或 fallback index 对齐张量。
- 可按 `target_fps` 重采样单个 robot clip。

包级公开 API：

- `load_robot_motion_file(...) -> RobotMotionData`

模块内部入口：

- `read_npz_fps(raw, path) -> float`
- `load_robot_npz_raw(path) -> tuple[np.lib.npyio.NpzFile, float]`
- `parse_robot_npz(...) -> RobotMotionData`

输入契约：

- `.npz` 必须包含：
  - `fps`
  - `joint_pos`
  - `joint_vel`
  - `body_pos_w`
  - `body_quat_w`
  - `body_lin_vel_w`
  - `body_ang_vel_w`
- `body_quat_w` 约定为 wxyz。
- 如果提供 `joint_names` / `motion_body_names`，输出必须按这些顺序排列。
- 如果 `.npz` 中没有名称元数据，则必须能通过维度或 `all_body_names/body_indexes` 安全推断。

输出契约：

- 返回 `RobotMotionData`。
- tensor dtype 为 `torch.float32`。
- frame axis 为第 0 维。
- `num_frames > 0`。
- 若重采样，`fps == target_fps`，`source_fps` 保留原 fps。
- `body_quat_w` 输出必须是单位四元数或尽可能接近单位长度。

允许副作用：

- 只读文件。

禁止：

- 不拼接多个 robot clips。
- 不做 SMPL 配对。
- 不做 device transfer；device transfer 属于 store。

待实现：

- 在 parse 后检查所有 robot tensor 的 frame 数一致。
- 检查 `body_quat_w.shape[-1] == 4`。
- 对未重采样的 `body_quat_w` 可选择 normalize 或至少校验并 warning。
- 将 `np.load` 资源生命周期处理得更明确，避免 raw handle 长期泄漏。

## `smpl_pkl.py`

职责：

- 读取 SMPL `.pkl`。
- 解析 `pose_aa`、`smpl_joints`、`transl`、`fps`。
- 可按 `target_fps` 重采样单个 SMPL clip。

包级公开 API：

- `load_pkl(path) -> dict`
- `load_smpl_motion_file(path, target_fps=None) -> MotionData`

输入契约：

- `.pkl` 中必须能提供 SMPL pose、joints、translation。
- `fps` 缺失时当前 fallback 需明确记录，长期建议变成显式错误或 warning。
- pose 约定为 axis-angle。

输出契约：

- 返回 `MotionData`。
- `pose_aa`、`smpl_joints`、`transl` frame 数一致。
- 若重采样，`fps == target_fps`。

禁止：

- 不做 robot body/joint 对齐。
- 不构造 `SmplMotionStore`。
- 不做 y-up/z-up 坐标转换；这属于 store/facade 中 SMPL store 的职责。

待实现：

- 明确 pkl 支持的 key alias。
- 增加 shape validation 和错误类型区分。
- 对 fallback fps 行为写测试。

## IO 层测试要求

- dataset txt 相对路径解析。
- missing key 抛 `MotionValidationError`。
- joint/body name reorder。
- no metadata fallback。
- target fps 重采样后 frame 数和 fps 正确。
- quaternion 重采样保持单位长度。
