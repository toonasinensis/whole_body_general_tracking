# `feat_motion_lib.transform` Agent Contract

`transform/` 负责纯数据变换。输入是已读取的 tensor / clip dataclass / path list，输出是变换后的 tensor / clip dataclass / paired list。它不访问训练环境，不持有全局状态，不构建 flat store。

## 目录边界

允许：

- 重排 tensor 维度中的 joint/body 轴。
- 重采样时间轴。
- 坐标系转换。
- robot/SMPL clip 配对和 frame 裁齐。

禁止：

- 不扫描目录；文件发现属于 `io/discovery.py`。
- 不读取 `.npz` / `.pkl`；文件解析属于 `io`。
- 不拼接多个 clips 成 flat tensor；属于 `store`。
- 不暴露 facade 查询 API。
- 不依赖 IsaacLab。

## `alignment.py`

职责：

- 从 `.npz` raw metadata 中读取 joint/body 名称。
- 将 robot tensor 按目标 joint/body 顺序重排。
- 在缺少名称元数据时，根据维度、`all_body_names`、`body_indexes` 做安全 fallback。

公开 API：

- `read_name_list(raw, keys) -> list[str] | None`
- `name_indexes(source_names, target_names, path, kind) -> list[int]`
- `align_joint_tensors(tensors, raw, path, joint_names) -> tuple[dict, list[str] | None]`
- `align_body_tensors(tensors, raw, path, motion_body_names, all_body_names, body_indexes) -> tuple[dict, list[str] | None]`

输入契约：

- `tensors` 必须包含需要对齐的 key。
- joint tensors 的 joint 维度在 axis 1。
- body tensors 的 body 维度在 axis 1。
- `motion_body_names` 是目标输出 body 顺序。
- `all_body_names` 是完整 robot body 顺序。
- `body_indexes` 是相对于完整 robot body 顺序的目标索引。

输出契约：

- 输出 tensor dict 中相关 tensor 已按目标顺序重排或切片。
- 返回的 names 若非 None，必须与输出 tensor axis 1 一一对应。

禁止：

- 不改变 frame axis。
- 不重采样。
- 不 normalize quaternion。
- 不吞掉 missing name；必须抛 `MotionAlignmentError`。

待实现：

- 将 body fallback 分支写成更显式的 strategy，减少维度碰巧相等导致的误判。
- 对 `body_indexes=[]` 和 `motion_body_names=None` 的语义给出明确行为。
- 增加所有 body tensors axis 1 一致性的测试。

## `resample.py`

职责：

- 对 SMPL motion 和 robot motion 沿时间轴重采样。
- SMPL pose 使用 slerp 语义。
- robot body quaternion 使用 slerp 并保持单位长度。
- 其它连续量使用线性插值。

公开 API：

- `resample_motion(raw, target_fps) -> MotionData`
- `resample_robot_motion(clip, target_fps) -> RobotMotionData`
- `_resample_quaternion_sequence(quat, source_fps, target_fps) -> torch.Tensor`

输入契约：

- 所有输入 tensor 的时间轴在 axis 0。
- `source_fps > 0`，`target_fps > 0`。
- robot `body_quat_w` shape 最后一维为 4，wxyz。

输出契约：

- 输出 fps 为 `target_fps`。
- 输出 `source_fps` 保留原值。
- 输出 frame 数由原 duration 和 target fps 决定。
- quaternion 输出必须 normalize。

禁止：

- 不读取文件。
- 不改 joint/body 名称。
- 不做 body selection。
- 不做 device transfer，保持输入 tensor 所在 device。

待实现：

- 将 `_compute_frame_indices` 从 smpl_math_utils 暴露出来，避免 resample.py 复制时间索引逻辑。
- 对 1-frame clip 增加测试。
- 对 downsample / upsample / same fps 三种路径增加测试。

## `pairing.py`

职责：

- 根据文件 stem 将 robot `.npz` 与 SMPL `.pkl` 配对。
- 加载 paired clips。
- 校验 paired frame 数差异不超过阈值。
- 将 paired robot / SMPL clip 裁齐到相同 frame 数。

输入契约：

- robot file list 来自 discovery。
- `smpl_dir` 是 SMPL pkl 目录。
- paired stem 规则必须 deterministic。
- `max_frame_diff` 表示允许的 robot/SMPL frame count 差异。

输出契约：

- 返回 `PairedMotionData` list 和 `LoadReport`。
- 每个 pair 中 robot 和 SMPL `num_frames` 一致。
- report 记录 requested/loaded/skipped/warnings。

禁止：

- 不构建 store。
- 不做 global index。
- 不选择训练 sampling subset。

待实现：

- 明确 duplicate stem 的处理策略。
- skipped pair 的原因进入 report。
- 对 frame diff 边界条件加测试。

## `coordinate.py`

职责：

- 提供坐标系转换函数。
- 当前主要服务 SMPL y-up / z-up 转换。

输入契约：

- 输入 tensor shape 和坐标语义必须由函数文档明确。
- 不应隐式改变 quaternion convention。

禁止：

- 不读取文件。
- 不做 fps 重采样。
- 不构建 store。

待实现：

- 补充每个转换函数的数学定义和测试。
- 明确哪些数据在 load 时转换，哪些在 query 时转换。
