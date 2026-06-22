# TerrainPairMulti-G1 Training Notes

## 这是什么

`TerrainPairMulti-G1` 是一个给 G1 机器人用的严格地形-动作配对 tracking 训练 task。

它的目标不是复现完整 HIL 论文流程，而是先把 OmniRetarget 生成的 G1 地形数据和对应动作数据放进现有 whole-body tracking 框架里训练。每个环境里的 STL 地形 cell 只配同名 motion npz，不从全局 motion 池里混采。

当前默认数据是：

- manifest: `data/omniretarget/g1_terrain/pairs.jsonl`
- motion list: `data/omniretarget/g1_terrain/paired_tracking_dataset.txt`
- pair 数量: 145
- task 名: `TerrainPairMulti-G1`
- experiment 名: `g1_paired_terrain`

## 数据约定

严格配对由 `pairs.jsonl` 驱动。每一行是一对地形和动作，核心字段是：

```json
{"name": "climb_00_z_scale_0.8", "tracking_motion": ".../climb_00_z_scale_0.8.npz", "terrain_stl": ".../climb_00_z_scale_0.8.stl"}
```

读取时会做这些校验：

- `tracking_motion` 指向的 `.npz` 必须存在。
- `terrain_stl` 指向的 `.stl` 必须存在。
- motion stem、STL stem、`name` 必须一致。
- manifest 不能为空。
- motion loader 加载出的 `file_names` 顺序必须和 manifest 顺序完全一致。

所以不要随便用 `--motion_file` 或别的 dataset 覆盖这个 task 的默认数据源。这个 task 依赖 manifest 顺序来保证 `pair_id == motion_id == terrain cell index`。

## 代码入口

主要文件：

- `source/whole_body_tracking/whole_body_tracking/terrains/paired_manifest.py`
  - 读取和校验 `pairs.jsonl`
  - 按 manifest 顺序构建 STL terrain grid
  - 提供 pair index 映射工具

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/paired_commands.py`
  - 新 command: `PairedTerrainMotionCommand`
  - 继承原 `MotionCommand`
  - 只改 motion 采样和 env-to-pair 映射

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/paired_terrain_env_cfg.py`
  - 新 env cfg: `G1PairedTerrainEnvCfg`
  - 替换 `scene.terrain`
  - 替换 `commands.motion`

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
  - 新 runner cfg: `G1PairedTerrainRunnerCfg`
  - `experiment_name = "g1_paired_terrain"`

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py`
  - 注册 gym task: `TerrainPairMulti-G1`

## 配对逻辑

默认配置字段：

```python
pairs_jsonl = "data/omniretarget/g1_terrain/pairs.jsonl"
terrain_size = (6.0, 6.0)
terrain_ground_thickness = 0.05
terrain_ground_z = 0.0
strict_pairing = True
env_id_pairing = True
preserve_dataset_order = True
```

terrain grid 按 manifest 顺序生成：

```text
pair_id = row * num_cols + col
```

当前实现默认用 env id 做稳定分配：

```text
pair_id = env_id % pair_count
motion_id = pair_id
terrain cell = pair_id
```

然后 command 会把该 env 的 `terrain_levels`、`terrain_types`、`env_origins` 同步到对应 terrain cell。这样即使 `num_envs > pair_count`，也只是重复使用同一批 pair，不会出现脚下是 A 地形、reference motion 却来自 B 的情况。

每个 paired STL cell 默认会额外合入一块地面底板：

```text
ground size = terrain_size
ground top z = terrain_ground_z
ground thickness = terrain_ground_thickness
```

这样即使原始 STL 只包含障碍块、不包含完整地面，训练里的机器人也不会因为脚下没有 collision 面直接坠落。

如果以后想改回按 IsaacLab 的 terrain level/type 分配，可以把 `env_id_pairing` 设成 `False`，映射就变成：

```text
pair_id = terrain_level * num_cols + terrain_type
motion_id = pair_id
```

## Motion 采样

`PairedTerrainMotionCommand` 仍然沿用原 tracking 的 MotionCommand 大部分逻辑，包括 reset、reference marker、adaptive sampling 统计等。

区别是每个 env reset 或重采样时，只能从自己的 paired motion 中采样合法帧：

```text
env -> pair_id -> motion_id -> local frame/bin
```

adaptive sampling 仍然记录 per-motion/per-bin 失败统计，但采样范围被限制在当前 env 对应的 motion 内，不会跨 motion 随机抽。

启动时会打印前几个 env 的映射预览，例如：

```text
[PairedTerrainMotionCommand] env -> pair -> motion mapping preview:
  env=000 pair=000 motion=000 motion_name=... terrain_name=...
```

看到这里能快速确认动作和地形是否同名。

## 训练命令

推荐 smoke test：

```bash
cd <whole_body_tracking>
python scripts/rsl_rl/train.py \
  --registry_name=test1 \
  --task=TerrainPairMulti-G1 \
  --num_envs=32 \
  --max_iterations=200 \
  --run_name=paired_terrain_smoke
```

正式训练可以增加环境数和迭代数：

```bash
cd <whole_body_tracking>
python scripts/rsl_rl/train.py \
  --registry_name=test1 \
  --task=TerrainPairMulti-G1 \
  --headless \
  --num_envs=4096 \
  --max_iterations=10000 \
  --run_name=paired_terrain_g1
```

注意：这个 task 默认会从 `pairs.jsonl` 推导 motion root 和 `paired_tracking_dataset.txt`。一般不要传 `--motion_file`，否则可能破坏配对顺序。

## 可视化检查

单个 mesh-motion pair 的检查脚本：

```text
scripts/omniretarget/replay_g1_terrain_pair.py
```

只做配对和数据摘要检查，不启动 IsaacSim：

```bash
cd <whole_body_tracking>
scripts/omniretarget/run_with_isaaclab_python.sh \
  scripts/omniretarget/replay_g1_terrain_pair.py \
  --check_only \
  --pair_index 0
```

这个命令会打印 `stem_match`、motion 帧数、body/joint 数量、STL 顶点/面数和 mesh bounds。`stem_match: True` 表示 manifest 名、motion stem、STL stem 是同一对。

这个 replay/check 脚本复用 `source/whole_body_tracking/whole_body_tracking/terrains/paired_manifest.py` 里的 `TerrainMotionPairManifest.from_jsonl()` 读取和校验 pairs。也就是说，除非手动传 `--motion_file/--terrain_file`，可视化入口和训练 task 走的是同一套 manifest 顺序与 stem 校验逻辑。

在 IsaacLab/IsaacSim 里可视化单个 pair：

```bash
cd <whole_body_tracking>
scripts/omniretarget/run_with_isaaclab_python.sh \
  scripts/omniretarget/replay_g1_terrain_pair.py \
  --pair_index 0 \
  --loop
```

一次可视化一批 pair：

```bash
cd <whole_body_tracking>
scripts/omniretarget/run_with_isaaclab_python.sh \
  scripts/omniretarget/replay_g1_terrain_pair.py \
  --pair_indices 0:16 \
  --loop
```

`--pair_indices` 支持这些写法：

```text
0:16       # pair 0 到 15
0:32:2     # 每隔 2 个取一个
0,3,7,12   # 指定几个
all        # 全部 pair，数量多时启动会比较重
```

batch 模式下每个 env 会摆一个机器人和对应 STL，日志会打印：

```text
env=000 pair=000 name=...
env=001 pair=001 name=...
```

默认相机给整批 env 的 overview。如果想跟拍某一个 env：

```bash
cd <whole_body_tracking>
scripts/omniretarget/run_with_isaaclab_python.sh \
  scripts/omniretarget/replay_g1_terrain_pair.py \
  --pair_indices 0:16 \
  --follow_env 3 \
  --loop
```

headless smoke 可以用：

```bash
cd <whole_body_tracking>
scripts/omniretarget/run_with_isaaclab_python.sh \
  scripts/omniretarget/replay_g1_terrain_pair.py \
  --headless \
  --pair_indices 0:4 \
  --max_frames 2
```

如果要看别的 pair，改 `--pair_index`。如果手动传 `--motion_file` 和 `--terrain_file`，默认仍会检查 stem 一致；只有做临时排错时才加 `--allow_stem_mismatch`。

可视化脚本默认也会给每个 STL 加同样的地面底板。想对比原始 STL，可以加：

```bash
--no_ground
```

少量 env GUI 跑起来后，检查两件事：

- robot/reference marker 的动作名称和脚下 STL 名称一致。
- 日志里的 `motion_name` 和 `terrain_name` stem 一致。

示例：

```bash
cd <whole_body_tracking>
python scripts/rsl_rl/train.py \
  --registry_name=test1 \
  --task=TerrainPairMulti-G1 \
  --num_envs=8 \
  --max_iterations=2 \
  --run_name=paired_terrain_visual
```

如果当前机器主要使用 IsaacSim 5.1，并且直接 `python` 启动遇到 Kit/Python 路径问题，可以优先用仓库里的 IsaacLab/IsaacSim 启动 wrapper 做同样的入口命令。

## 已做的安全隔离

这个训练是插件式接入：

- 原 `TR-G1` 不改。
- 原 `FM-G1` 不改。
- 原 `AMP-G1` 不改。
- 原 `RLBC-G1` 不改。
- 原 `MotionCommand` 不改核心行为。
- 只有 `TerrainPairMulti-G1` 使用 paired terrain 和 paired command。

训练脚本只做了一个兼容修正：`--motion_file` 非空时才覆盖 env cfg 默认值。这样新 task 可以默认从 manifest 取 motion，旧 task 仍然可以用 CLI 覆盖。

## 测试

已添加的单元测试：

```bash
cd <whole_body_tracking>
pytest -q tests/test_paired_terrain_manifest.py
```

motion loader 顺序相关测试：

```bash
cd <whole_body_tracking>
PYTHONPATH=source/whole_body_tracking/whole_body_tracking/utils/motionlib/smpl_motion_lib:source/whole_body_tracking/whole_body_tracking/utils/motionlib/smpl_math_utils \
pytest -q source/whole_body_tracking/whole_body_tracking/utils/motionlib/smpl_motion_lib/tests/test_unified_motion_lib_npz_names.py
```

建议每次改 manifest 或 paired command 后至少跑：

```bash
python -m py_compile \
  source/whole_body_tracking/whole_body_tracking/terrains/paired_manifest.py \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/paired_commands.py \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/paired_terrain_env_cfg.py
```

## 当前边界

这一步只是地形-动作严格配对 tracking 训练。

暂时没有加入：

- HIL 的 task-mode 切换。
- 点云编码。
- 判别器。
- teacher/student 蒸馏。
- terrain curriculum 轮换策略。

后续如果要往 HIL 论文完整复现走，可以在这个 task 稳定后再加：地形感知 observation、task command、判别器奖励、reset/curriculum，以及 sim2sim/export 路径。
