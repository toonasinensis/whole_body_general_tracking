# `feat_motion_lib` 架构与设计说明

## 1. 背景与目标

`feat_motion_lib` 的目标，是把原先散落在训练逻辑中的 motion loading、文件发现、数据校验、索引构建、SMPL 配对等职责，从单体脚本式实现里拆出来，形成一套可复用、可测试、可逐步演化的运动数据基础设施。

它主要服务于两类场景：

- `robot-only` 场景  
  只使用 retarget 后的 robot motion `.npz`，为 RL / tracking 提供 `joint_pos`、`body_pos_w`、`anchor_pos_w` 等张量。

- `robot + SMPL paired` 场景  
  同时加载 robot `.npz` 和配对的 SMPL `.pkl`，保证两者在 clip 粒度、frame 粒度上可对齐，为训练、评估和可视化提供统一访问接口。

核心设计思想是：

- 将“发现文件”“读文件”“对齐名字”“重采样”“拼接索引”“对外暴露查询接口”分层。
- 对下游暴露稳定的 facade，而不是让训练代码直接依赖底层文件格式。
- 用显式的数据结构和测试覆盖，替代隐式约定和脚本式耦合。

---

## 2. 顶层分层

`feat_motion_lib` 可以分成 6 层：

1. `config` / `types` / `errors`
2. `io`
3. `transform`
4. `store`
5. `facade`
6. `tests`

对应关系如下：

```text
cfg / user input
    -> facade
        -> orchestration / loading
            -> io + transform
                -> typed clips
                    -> store
                        -> flattened state / index
                            -> facade query API
```

---

## 3. 各目录职责

### 3.1 `config.py`

定义结构化配置对象，负责描述“如何发现 motion、如何加载、是否需要 paired SMPL”等信息。

主要类型：

- `DiscoveryOptions`
- `LoadOptions`
- `SmplLoadConfig`
- `UnifiedLoadConfig`

设计动机：

- 将“配置输入”从具体训练 cfg 中抽离出来。
- 让 `UnifiedMotionLib.load()` 和 `load_from_cfg()` 都能工作。

### 3.2 `types.py`

定义全包共享的数据结构，避免不同模块之间传裸 `dict`。

主要类型：

- `SmplMotionClip`
  表示单个 SMPL clip。

- `RobotMotionClip`
  表示单个 robot clip。

- `PairedMotionClip`
  表示一对已经裁齐的 robot + SMPL clip。

- `MotionIndex`
  表示拼接后的索引信息。

- `RobotMotionStore`
  持有已经拼接好的 robot flat tensors；`UnifiedMotionLib` 直接持有该 store 并通过属性转发查询。

- `LoadReport`
  表示一次 load 的结果、警告、fallback 状态。

设计动机：

- 明确 clip 级与 flat tensor 级两种数据表达。
- 降低“中间状态是什么”的心智负担。

### 3.3 `errors.py`

定义 motionlib 自己的错误类型：

- `MotionFileError`
- `MotionValidationError`
- `MotionAlignmentError`
- `MotionNotLoadedError`

设计动机：

- 将“文件坏了”“名字对不上”“未加载即查询”等问题区分开。
- 让上层更容易做定向处理和调试。

### 3.4 `io/`

负责“从磁盘读数据”。

主要文件：

- `discovery.py`
  负责发现 `.npz` / `.pkl` 文件，支持：
  - 目录扫描
  - `dataset_txt`
  - `eval_mode` 顺序切片
  - `distributed` 切分

- `robot_npz.py`
  负责读取 robot motion `.npz`，做 key 校验、fps 解析、基础 shape 校验，并调用对齐逻辑。

- `smpl_pkl.py`
  负责读取 SMPL `.pkl`，解析 `pose_aa / smpl_joints / transl / fps`。

设计动机：

- 将“文件格式相关代码”与“索引和查询代码”分离。
- 让后续支持更多格式时，影响面控制在 `io` 层。

### 3.5 `transform/`

负责对原始 clip 做语义变换，但不负责存储。

主要文件：

- `alignment.py`
  负责按 `joint_names` / `body_names` 重排 robot 张量。

- `resample.py`
  负责按目标 fps 重采样 robot / SMPL motion。

- `pairing.py`
  负责按 stem 匹配 robot 与 SMPL 文件，并在 clip 粒度裁齐 frame 数。

- `coordinate.py`
  负责坐标系转换，例如 `y-up -> z-up`。

设计动机：

- 把“纯变换逻辑”从读写和 facade 中抽出来。
- 让这些逻辑更容易单测，也更容易被复用。

### 3.6 `store/`

负责把多个 clip 连接成统一张量，并建立索引。

主要文件：

- `index.py`
  定义 flat index 的核心操作：
  - `build_motion_index`
  - `flatten_indices`
  - `motion_ids_from_timestamps`

- `robot_store.py`
  将多个 `RobotMotionClip` 拼接成统一的 robot motion flat tensor。

- `smpl_store.py`
  将多个 `SmplMotionClip` 拼接成统一的 SMPL flat tensor，并提供：
  - `get_pose`
  - `get_joints`
  - `get_transl`
  - `get_global_positions`
  - `get_global_rotations`

- `paired_store.py`
  组合 robot store 和 SMPL store，确保 paired 语义成立。

设计动机：

- 把“clip list -> flat tensors + index”的逻辑集中管理。
- 下游统一通过 `motion_id + frame_id` 或 `global timestamp` 访问数据。

### 3.7 `facade/`

这是对外最重要的一层，负责组织加载流程并暴露稳定 API。

主要文件：

- `smpl_motion_lib.py`
  面向 SMPL-only 的 facade。

- `unified_motion_lib.py`
  面向 robot-only / paired 的统一 facade，是当前训练替换的核心入口。

- `loading.py`
  负责“按文件列表加载 clip”，偏局部职责。

- `orchestration.py`
  负责“整体 load 流程编排”，决定走 paired 还是 robot-only。

- `config_adapter.py`
  负责把训练代码里的 cfg 转成 `UnifiedLoadConfig` / `SmplLoadConfig`。

设计动机：

- 让下游依赖稳定接口，而不是依赖内部流程。
- 隐藏底层 `io + transform + store` 的实现细节。

### 3.8 `tests/`

测试覆盖目前主要围绕：

- 名字对齐
- 机器人加载
- paired 加载
- parity 与旧实现一致性
- `commands.py` 的兼容消费方式
- 手动 MuJoCo 可视化验证

设计动机：

- motion 数据处理链条长、隐式约定多，需要回归保护。
- 尤其是“替换旧 MotionLoader”时，要优先保证行为兼容。

---

## 4. 核心数据流

以 `UnifiedMotionLib.load_from_cfg()` 为例，完整数据流如下：

### 4.1 配置适配

训练或外部代码传入一个 duck-typed cfg。

`facade/config_adapter.py` 将其转为：

- `DiscoveryOptions`
- `LoadOptions`
- `UnifiedLoadConfig`

### 4.2 发现文件

`io/discovery.py` 根据配置决定加载哪些 `.npz`：

- 全量扫描
- `dataset_txt`
- `eval_mode` 分块顺序采样
- `distributed` 数据切分

### 4.3 读入 clip

`facade/orchestration.py` 会先判断是否请求 paired SMPL：

- 如果有 `smpl_dir`，尝试 `load_paired_clips()`
- 如果无匹配，或过滤后全失败，则 fallback 到 `load_robot_clips()`

### 4.4 clip 级预处理

在单个 clip 级别，会完成：

- robot npz key / shape 校验
- `joint_names` / `body_names` 对齐
- fps 重采样
- robot / SMPL frame trimming

### 4.5 构建 store

成功加载的 clip list 进入：

- `RobotMotionStore`
- `SmplMotionStore`
- `PairedMotionStore`

然后得到 flat tensors：

- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `smpl_joints`
- `smpl_transl`
- `smpl_poses`

以及索引：

- `frame_list`
- `time_step_start_idx`
- `time_step_end_idx`
- `time_step_total`

### 4.6 对外查询

最终由 `UnifiedMotionLib` / `SmplMotionLib` 提供面向下游的 API，例如：

- `joint_pos`
- `anchor_pos_w`
- `motion_ids_from_timestamps()`
- `get_smpl_joints()`
- `get_smpl_global_position()`

---

## 5. 为什么要做“flat tensor + index”

这是整个包最关键的设计点。

### 5.1 目标

训练时最常见的访问模式不是“按文件遍历”，而是：

- 每个 env 正在播放不同 motion
- 每个 env 正在 motion 的不同 frame
- 需要高频随机索引

这时把数据做成：

- 一组全局拼接的 flat tensor
- 一组 `start_idx/end_idx` 索引

就能把访问统一成：

```text
global_idx = time_step_start_idx[motion_id] + local_step
value = tensor[global_idx]
```

### 5.2 收益

- 查询简单
- 向量化友好
- 与 RL 的 batched env 模式天然匹配
- robot 与 SMPL 可以共享同一套 frame 索引语义

### 5.3 代价

- 加载阶段需要一次性把 clip 拼起来
- 如果数据量很大，GPU 拷贝峰值要特别关注

当前 `RobotMotionStore` 有一个已标记的 TODO：

- 目前是整体 `.to(device)`
- 未来如数据规模继续增大，可以改回 legacy `MotionLoader` 那种 batched CPU->GPU 路径

---

## 6. 与旧 `MotionLoader` 的关系

这个包的一个重要现实目标，是逐步替换训练框架中旧的 `MotionLoader`。

替换策略不是强推训练代码立刻理解所有新抽象，而是：

- 在 facade 层提供与旧接口尽量接近的属性和方法
- 先让 `commands.py` 能平滑迁移
- 再逐步把更多上层能力迁入 `feat_motion_lib`

目前 `UnifiedMotionLib` 已经覆盖了 `MotionCommand` 所需的大部分核心接口：

- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`
- `anchor_pos_w`
- `anchor_quat_w`
- `anchor_lin_vel_w`
- `anchor_ang_vel_w`
- `frame_list`
- `motion_num`
- `time_step_total`
- `time_step_start_idx`
- `time_step_end_idx`
- `motion_ids_from_timestamps()`

因此训练替换可以分阶段完成，而不需要一次性重写所有训练逻辑。

---

## 7. 设计理念总结

### 7.1 facade 优先

下游应依赖 facade，而不是依赖内部文件格式和中间流程。

这让：

- 训练代码更稳定
- 可视化脚本更轻量
- 后续支持新格式更容易

### 7.2 clip 级逻辑与 flat 级逻辑分离

- `io + transform` 负责单 clip 语义正确
- `store` 负责批量组织与索引

这样边界更清晰，也更利于调试。

### 7.3 显式 metadata

通过 `types.py` 把：

- 数据内容
- 索引信息
- 加载报告

都显式建模，减少“靠命名约定传递语义”的情况。

### 7.4 可测试性优先

motion 处理链中很多问题在运行时很隐蔽，比如：

- joint 顺序错位
- body 名字缺失
- fps 不一致
- paired frame mismatch

因此设计上优先把这些逻辑做成小函数 / 小模块，并用测试锁住行为。

### 7.5 兼容与演化并重

`feat_motion_lib` 不是纯粹“重写一套更漂亮的库”，而是在兼容现有训练消费方式的前提下，逐步把 motion 基础设施做规范。

这意味着：

- 有些接口是面向未来的
- 有些接口是为了迁移旧代码保留的
- 需要接受一段时间内 facade 比底层更“宽”的现实

---

## 8. 当前已知边界与后续建议

### 8.1 已知边界

- `UnifiedMotionLib` 当前对外暴露了 `get_smpl_global_position()`，但还没有直接暴露 `get_smpl_global_rotations()`；手动可视化脚本目前通过内部 `_smpl_lib` 访问。
- 大数据集场景下，`RobotMotionStore` 的整体 `.to(device)` 可能带来较高加载峰值显存。
- `pair_motion_paths()` 当前按 stem 一一匹配，要求 robot / SMPL 文件命名保持一致。

### 8.2 建议的后续演进

1. 将 `get_smpl_global_rotations()` 正式补到 `UnifiedMotionLib` facade。
2. 为大规模数据集补 batched CPU->GPU store 构建路径。
3. 为 `manual_visualize_unified_motion_mujoco.py` 沉淀一个更正式的 `extras/visualization` 模块。
4. 若后续支持更多机器人，建议把“joint order preset / body preset / visualization preset”抽成单独 registry。

---

## 9. 推荐阅读顺序

如果你第一次接触这个包，建议按这个顺序阅读：

1. [facade/unified_motion_lib.py](/home/kiki/workspace/dev_whole_body_general_tracking/source/whole_body_tracking/whole_body_tracking/utils/motionlib/feat_motion_lib/facade/unified_motion_lib.py:1)
2. [facade/orchestration.py](/home/kiki/workspace/dev_whole_body_general_tracking/source/whole_body_tracking/whole_body_tracking/utils/motionlib/feat_motion_lib/facade/orchestration.py:1)
3. [io/robot_npz.py](/home/kiki/workspace/dev_whole_body_general_tracking/source/whole_body_tracking/whole_body_tracking/utils/motionlib/feat_motion_lib/io/robot_npz.py:1)
4. [transform/alignment.py](/home/kiki/workspace/dev_whole_body_general_tracking/source/whole_body_tracking/whole_body_tracking/utils/motionlib/feat_motion_lib/transform/alignment.py:1)
5. [store/robot_store.py](/home/kiki/workspace/dev_whole_body_general_tracking/source/whole_body_tracking/whole_body_tracking/utils/motionlib/feat_motion_lib/store/robot_store.py:1)
6. [store/smpl_store.py](/home/kiki/workspace/dev_whole_body_general_tracking/source/whole_body_tracking/whole_body_tracking/utils/motionlib/feat_motion_lib/store/smpl_store.py:1)
7. [tests/test_commands_compat.py](/home/kiki/workspace/dev_whole_body_general_tracking/source/whole_body_tracking/whole_body_tracking/utils/motionlib/feat_motion_lib/tests/test_commands_compat.py:1)

---

## 10. 一句话总结

`feat_motion_lib` 的本质，不是“再写一个 motion loader”，而是把 robot motion / SMPL motion 的加载、校验、对齐、索引和查询，沉淀成一套可复用、可测试、可渐进替换旧训练代码的基础架构。
