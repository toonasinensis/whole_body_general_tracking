# `feat_motion_lib.tests` Agent Contract

`tests/` 负责保护 `feat_motion_lib` 的模块边界和关键不变量。测试应使用临时文件和小型 synthetic tensors，不依赖真实训练数据集。

## 测试边界

允许：

- 使用 `tmp_path` 创建 `.npz` / `.pkl`。
- 使用 synthetic clip dataclass。
- 使用 small tensors 覆盖 shape、index、alignment、resample、store、facade 行为。
- 为 legacy compatibility 构造 fake cfg。

禁止：

- 不依赖 `/home/...` 下的真实 motion 数据。
- 不启动 IsaacLab simulation。
- 不写工作区外文件。
- 不依赖 GPU，除非测试明确标记为 optional。
- 不把大数据文件提交进仓库。

## 必须覆盖的不变量

- `.npz` missing required key 抛 `MotionValidationError`。
- joint/body name alignment 按目标顺序输出。
- body metadata 缺失时 fallback 规则可预测。
- robot quaternion 重采样使用 slerp 并保持单位长度。
- `build_motion_index` start/end/total 正确。
- `motion_ids_from_timestamps` 在 motion boundary 正确。
- robot store flat tensor 与 metadata 对齐。
- SMPL store query shape 正确。
- paired loading 会裁齐 frame。
- unified facade load 前 query 抛 `MotionNotLoadedError`。
- unified facade robot-only load metadata 正确。
- paired load 成功时 SMPL query 可用。

## 测试文件职责

- `test_alignment.py`: 只测 `transform/alignment.py`。
- `test_robot_loader.py`: 只测单文件 robot `.npz` IO 和 alignment integration。
- `test_resample.py`: 只测 resample 纯函数。
- `test_orchestration.py`: 测 load orchestration 的 path 选择和 report。
- `test_paired_store.py` / `test_pairing.py`: 分别测 store invariant 和 pairing transform。
- `test_unified_robot_loading.py`: 测 unified facade robot-only load。
- `test_unified_paired_loading.py`: 测 unified facade paired load。
- `test_unloaded_guards.py`: 测 load 前 guard。
- `test_unified_motion_lib_parity.py`: 只用于 legacy parity，避免新增业务逻辑断言。

## conftest 规则

- 只做 import path bootstrap 和轻量 fixture。
- 不导入训练 env。
- 不加载真实 motion 数据。
- 不修改全局 random seed，除非 fixture 自己恢复。

## 待补测试清单

- `discovery.py` eval chunk 不少取最后一个文件。
- `discovery.py` distributed shard 在不能整除时的行为。
- `resample.py` 1-frame quaternion clip。
- `resample.py` target fps 等于 source fps 时的行为。
- `robot_store.py` fps mismatch 报错或 warning。
- `smpl_store.py` 越界 query 行为。
- `unified_motion_lib.py` anchor index 与 selected body order 的契约。
