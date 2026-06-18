# `feat_motion_lib.store` Agent Contract

`store/` 负责把 clip-level dataclass 拼接成 flat tensors，并建立 motion index。store 是已加载数据的持有者，但不是文件加载器，也不是训练 command runtime。

## 目录边界

允许：

- 持有多个 clip 拼接后的 tensor。
- 将 tensor 移动到目标 device。
- 建立 per-motion start/end index。
- 通过 motion id + local frame 或 global timestamp 查询 flat tensor。
- 构建 `UnifiedMotionState`。

禁止：

- 不访问磁盘。
- 不做文件发现。
- 不做 joint/body name alignment。
- 不做 fps 重采样。
- 不依赖 IsaacLab env/robot。
- 不写 simulator。

## `index.py`

职责：

- 提供 flat motion index 的基础操作。

公开 API：

- `build_length_starts(frame_counts) -> torch.Tensor`
- `build_motion_index(frame_counts, device="cpu") -> MotionIndex`
- `flatten_indices(length_starts, motion_ids, motion_steps) -> torch.Tensor`
- `motion_ids_from_timestamps(timestamps, end_idx, total_frames, motion_num) -> torch.Tensor`

输入契约：

- frame counts 必须为正整数。
- `end_idx` 是 exclusive end。
- timestamps 是 global frame index，可被 clamp 到 `[0, total_frames - 1]`。

输出契约：

- index tensors dtype long。
- `start_idx[0] == 0`。
- `end_idx[-1] == total_frames`。
- boundary timestamp 等于某 motion end 时，应映射到下一个 motion，除非已到最后一帧。

禁止：

- 不依赖 clip dataclass。
- 不读取具体 motion tensor。

待实现：

- 对 empty frame counts 和 zero-length clip 抛清晰错误。
- 为 bucketize boundary 增加测试。

## `robot_store.py`

职责：

- 持有 robot clips 拼接后的 flat robot tensors。
- 建立 robot motion index。
- 暴露 file names、fps、motion_num、frame_list、time_step metadata。
- 构建 `UnifiedMotionState` 供 facade 使用。

输入契约：

- clips 非空。
- 每个 clip 的 robot tensor frame 数与 `num_frames` 一致。
- 每个 clip 的 fps 应一致，或必须先在 loading/transform 层重采样到一致 fps。
- body/joint order 已经在 `io/robot_npz.py` 和 `transform/alignment.py` 中处理完成。
- `body_quat_w` 应为 wxyz unit quaternion。

输出契约：

- flat tensors axis 0 为 concatenated global frame。
- tensors 在 `device` 上。
- `file_names` 与 clip 顺序一致。
- `time_step_start_idx` / `time_step_end_idx` 与 flat tensors 对齐。

允许副作用：

- device transfer。

禁止：

- 不重采样。
- 不做 body selection。
- 不读取文件。

待实现：

- 校验 fps 一致。
- 校验 frame shape 一致。
- 校验 quaternion norm，至少提供 debug assertion 或 warning。
- 大数据集时支持 lazy/batched device transfer，降低峰值显存。

## `smpl_store.py`

职责：

- 持有 SMPL clips 拼接后的 flat tensors。
- 提供按 `motion_id + local frame` 查询 SMPL pose/joints/transl/global state。
- 处理 SMPL 坐标系相关输出。

输入契约：

- clips 非空。
- pose/joints/transl frame 数一致。
- clips 已按 target fps 重采样。
- `up_axis` 明确指定输入/输出坐标语义。

输出契约：

- flat tensors 在目标 device 上。
- query 输入 `motion_ids` 和 `motion_steps` shape 可广播或一致。
- query 输出保持请求 shape 的前缀维度。

禁止：

- 不发现或读取 pkl。
- 不配对 robot clip。
- 不构建 robot state。

待实现：

- 明确 `get_global_positions` / `get_global_rotations` 的坐标系。
- 对 motion id 越界、step 越界给出 clamp 或 error 的统一策略。
- 增加 batch query shape 测试。

## `paired_store.py`

职责：

- 组合 `RobotMotionStore` 和 `SmplMotionStore`。
- 表达 robot/SMPL paired store 的不变量。

输入契约：

- robot store 和 smpl store 的 `motion_num` 一致。
- 每个 motion 的 frame count 一致。
- pair order 一致。

输出契约：

- 对外可访问 robot store 和 smpl store。
- 不复制大 tensor，主要作为结构性绑定。

禁止：

- 不负责配对文件。
- 不裁齐 frame。
- 不读取文件。

待实现：

- 在构造时强校验 frame counts。
- 提供 pair-level metadata，如 stem/file name mapping。
