# Event Modules 合约

`evt_modules` 拥有 IsaacLab event functions，用于 startup 和 interval effects。
公开 API 以 `evt_modules/__init__.py` 中的 `__all__` 为准。

## 公开导出

- `randomize_joint_default_pos`
- `randomize_rigid_body_com`
- `assist_fallen_robots_with_upward_force`

## 所有权

Event 函数是本 MDP 中唯一有意在 command reset 之外修改 env、asset defaults、
PhysX 属性或外力 buffer 的 term。

允许的副作用：

- `randomize_joint_default_pos()` 可以修改 `asset.data.default_joint_pos`、
  `asset.data.default_joint_pos_nominal` 和 joint position action offset。
- `randomize_rigid_body_com()` 可以通过 `asset.root_physx_view.set_coms()` 修改
  PhysX COM。
- `assist_fallen_robots_with_upward_force()` 可以通过 wrench composer 或
  `asset.set_external_force_and_torque()` 写外力/力矩。
- `assist_fallen_robots_with_upward_force()` 可以向 `env.extras["log"]` 写 log
  metrics，并可以拥有 `env._fallen_upward_assist_timeout_stats`。

禁止的副作用：

- Event 函数不得修改 `MotionCommand` timeline、selection policy、sampler 或
  reference cache。
- 不要在这里执行机器人 reset；command-driven reset 属于 `MotionCommandResetter`。
- 不要替换 `env.termination_manager`；delayed termination 安装属于 `tmt_modules`。

## `assist_fallen_robots_with_upward_force`

该 interval event 通过 `env.command_manager.get_term(command_name)` 读取
`MotionCommand`。

Command 依赖：

- `cfg.anchor_body_name`
- `anchor_pos_w`

Asset 依赖：

- `asset.body_names`
- `asset.data.body_pos_w`
- `asset.data.body_quat_w`
- `asset.data.body_lin_vel_w`
- `asset.data.GRAVITY_VEC_W`
- 可选 `asset.instantaneous_wrench_composer`
- 可选 `asset.permanent_wrench_composer`
- fallback：`asset.set_external_force_and_torque()`

共享 env 属性：

- 当 `use_motion_pose_range_mask=True` 时读取 `env.envs_classes_mask["lying"]`。该
  mask 由 `MotionCommand`/`MotionCommandResetter` 拥有，应选择与 delayed
  termination 相同的 recovery env class。
- 拥有 `env._fallen_upward_assist_timeout_stats`。

行为预期：

- Assist force 只考虑 active mask 选中的 env。
- Fallen 状态根据 command reference height/orientation、当前 robot anchor
  pose/velocity 和 projected gravity 计算。
- Timeout 统计用于在 assisted resets 间衰减 assist force scale。
- 该函数应保持为 recovery aid；不得改变 motion selection、reset policy、
  reference cache 或 termination masks。

性能约束：

- 该函数可能每个 control step 运行。tensor 分配和外力写入应限制在最小必要 env
  子集。
- 生产训练中 debug printing 必须通过默认 `debug_steps=0` 关闭。

## 扩展规则

- 新 event 函数添加到 `evt_functions.py`。
- 公开 event 函数通过 `evt_modules/__init__.py` 导出。
- 任何新的 env 属性、simulator 写入或跨模块依赖，都要在本文档和
  `mdp/agent_contract.md` 中记录。
- 如果 env class masks 改由其他 owner 管理，需要同时更新本文档和
  `tmt_modules/agent_contract.md`。
