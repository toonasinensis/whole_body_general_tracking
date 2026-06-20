# `feat_motion_lib.store` Agent Contract

`store/` owns already-loaded motion clips as concatenated flat tensors and builds per-motion indices for query. It is a data holder/indexing layer, not a file loader, file matcher, resampler, simulator adapter, or training runtime.

## Boundary

Allowed:

- Hold multiple loaded clips as flat tensors concatenated on frame axis 0.
- Move owned tensors to a requested torch device during construction or explicit device transfer.
- Build `MotionIndex` start/end metadata from per-clip frame counts.
- Convert `(motion_id, local_frame)` into flat tensor indices.
- Convert global frame timestamps into motion ids.
- Build `UnifiedMotionState` for the facade layer.
- Validate that paired robot and SMPL stores share the same motion index.

Forbidden:

- Do not discover, open, or parse motion files.
- Do not pair robot files with SMPL files.
- Do not trim clips to matching lengths.
- Do not resample FPS.
- Do not perform joint/body name alignment or body selection.
- Do not depend on IsaacLab env/robot objects.
- Do not write simulator state.

## External Dependencies

Runtime libraries:

- `torch`: tensor concatenation, indexing, dtype/device conversion, random sampling, bucketization, rotation matrix composition.
- `numpy`: stores SMPL per-motion frame count and FPS metadata arrays.
- `os.path`: computes robot relative file names.

Internal package dependencies:

- `..types.MotionIndex`
- `..types.MotionData`
- `..types.RobotMotionData`
- `..types.UnifiedMotionState`
- `..errors.MotionValidationError`
- `..transform.coordinate.yup_to_zup_root_pose`
- `..transform.coordinate.yup_to_zup_points`

External project dependency:

- `whole_body_tracking.utils.motionlib.smpl_math_utils.smpl_math_utils.angle_axis_to_rotation_matrix`
- Fallback import: `smpl_math_utils.angle_axis_to_rotation_matrix`

Input assumptions provided by upstream layers:

- Clips have already been loaded into dataclasses.
- Robot tensors have already been aligned to the target joint/body order.
- Clips have already been resampled if a shared target FPS is required.
- Paired robot/SMPL clips have already been matched and trimmed before store construction.
- SMPL coordinate convention is declared through `up_axis`.

## Public Package API

`store/__init__.py` explicitly exports:

- `RobotMotionStore`
- `SmplMotionStore`
- `PairedMotionStore`
- `build_length_starts`
- `build_motion_index`
- `flatten_indices`
- `motion_ids_from_timestamps`

Importing from submodules is allowed for internal maintenance, but callers should prefer this package API when depending on the store layer.

## `index.py`

Services:

- `build_length_starts(frame_counts) -> torch.Tensor`
- `build_motion_index(frame_counts, device="cpu") -> MotionIndex`
- `flatten_indices(start_idx, motion_ids, motion_steps) -> torch.Tensor`
- `motion_ids_from_timestamps(timestamps, end_idx, total_frames, motion_num) -> torch.Tensor`

Dependencies:

- `torch`
- `..types.MotionIndex`

Contract:

- `frame_counts` should contain positive per-motion frame counts.
- `build_motion_index` accepts `list[int]` or `torch.Tensor` and returns long tensors on `device`.
- `MotionIndex.start_idx` is inclusive, `MotionIndex.end_idx` is exclusive.
- `MotionIndex.total_frames == frame_counts.sum()`.
- `flatten_indices` returns `motion_steps + start_idx[motion_ids]` after moving `start_idx` to the `motion_ids` device.
- `motion_ids_from_timestamps` clamps timestamps to `[0, total_frames - 1]`; a timestamp equal to a motion exclusive end maps to the next motion, except final-frame clamping.

Current limitations:

- Empty `frame_counts` and zero-length clips are not rejected with a custom error.
- `flatten_indices` does not clamp or bounds-check `motion_ids` or `motion_steps`.

## `robot_store.py`

Services:

- `RobotMotionStore(clips, device="cpu")`
- Flat robot tensors:
  - `joint_pos`
  - `joint_vel`
  - `body_pos_w`
  - `body_quat_w`
  - `body_lin_vel_w`
  - `body_ang_vel_w`
- Metadata:
  - `clips`
  - `device`
  - `file_names`
  - `fps`
  - `index`
  - `motion_num`
  - `frame_list`
  - `time_step_total`
  - `time_step_start_idx`
  - `time_step_end_idx`
- `relative_file_names(base_dir) -> list[str]`
- `build_state(base_dir) -> UnifiedMotionState`

Dependencies:

- `os.path`
- `torch`
- `..types.RobotMotionData`
- `..types.UnifiedMotionState`
- `.index.build_motion_index`

Contract:

- `clips` must be non-empty; otherwise construction raises `ValueError`.
- Each `RobotMotionData` tensor is concatenated along frame axis 0 and moved to `device`.
- `file_names` follows clip order and uses `clip.path.name` when present, otherwise `motion_{i}`.
- `fps` is taken from `clips[0].fps`.
- `build_state` packages the flat tensors and index metadata for facade consumption.
- `relative_file_names` uses `os.path.relpath` for clips with a path.

Caller responsibilities:

- Ensure every robot tensor frame dimension matches `clip.num_frames`.
- Ensure all clips have compatible FPS when a shared FPS is required.
- Ensure joint/body ordering and body selection have already been handled.
- Ensure `body_quat_w` uses wxyz unit quaternions.

Current limitations:

- FPS consistency, tensor frame shape consistency, and quaternion norms are not validated.
- Device transfer is eager whole-tensor transfer, which can increase peak memory for large datasets.

## `smpl_store.py`

Services:

- `SmplMotionStore(clips, up_axis="yup", device="cpu")`
- Flat SMPL tensors:
  - `poses_flat`
  - `joints_flat`
  - `transl_flat`
- Metadata:
  - `up_axis`
  - `device`
  - `frame_counts`
  - `motion_fps`
  - `motion_source_fps`
  - `index`
- `to_device(device) -> None`
- `get_pose(motion_ids, motion_steps) -> torch.Tensor`
- `get_joints(motion_ids, motion_steps) -> torch.Tensor`
- `get_transl(motion_ids, motion_steps) -> torch.Tensor`
- `get_global_positions(motion_ids, motion_steps) -> torch.Tensor`
- `get_global_rotations(motion_ids, motion_steps) -> torch.Tensor`
- `get_num_motions() -> int`
- `get_num_frames(motion_id) -> int`
- `get_motion_fps(motion_id) -> float`
- `get_motion_duration(motion_id) -> float`
- `sample_random(batch_size) -> tuple[torch.Tensor, torch.Tensor]`

Dependencies:

- `numpy`
- `torch`
- `angle_axis_to_rotation_matrix`
- `..transform.coordinate.yup_to_zup_root_pose`
- `..transform.coordinate.yup_to_zup_points`
- `..types.MotionData`
- `.index.build_motion_index`
- `.index.flatten_indices`

Contract:

- `up_axis` must be `"yup"` or `"zup"`; otherwise construction raises `ValueError`.
- `clips` must be non-empty; otherwise construction raises `ValueError`.
- Pose, joint, and translation tensors are concatenated along frame axis 0 and moved to `device`.
- `frame_counts`, `motion_fps`, and `motion_source_fps` preserve per-clip metadata as numpy arrays.
- `to_device` moves flat tensors and rebuilds `index` on the target device.
- `get_pose` returns root pose converted to z-up when `up_axis == "yup"`.
- `get_transl` returns translation converted to z-up when `up_axis == "yup"`.
- `get_joints` returns stored joints directly.
- `get_global_positions` returns `get_joints(...) + get_transl(...).unsqueeze(1)`, so translation follows the `up_axis` conversion behavior above.
- `get_global_rotations` converts queried angle-axis pose to local rotation matrices, then composes them through fixed SMPL parents.
- `get_num_frames` raises `IndexError` for out-of-range integer motion ids.
- `sample_random` currently returns CPU tensors.

Caller responsibilities:

- Ensure pose/joints/transl frame counts match each clip's `num_frames`.
- Ensure clips have already been resampled when a shared target FPS is required.
- Pass query tensors with compatible shapes and valid motion/local-frame ranges.

Current limitations:

- Query methods do not clamp or explicitly bounds-check tensor `motion_ids` and `motion_steps`.
- `sample_random` does not return tensors on `self.device`.
- Coordinate semantics of stored `joints_flat` are inherited from upstream data; only pose root and translation are explicitly converted for yup input.

## `paired_store.py`

Services:

- `PairedMotionStore(robot_store, smpl_store)`
- Public attributes:
  - `robot_store`
  - `smpl_store`
- `validate_shared_index() -> None`

Dependencies:

- `..errors.MotionValidationError`
- `.robot_store.RobotMotionStore`
- `.smpl_store.SmplMotionStore`

Contract:

- Construction stores both input stores and immediately calls `validate_shared_index`.
- Validation requires:
  - equal `total_frames`
  - equal motion count
  - equal `frame_counts`
  - equal `start_idx`
  - equal `end_idx`
- Validation failures raise `MotionValidationError`.
- The class does not copy large tensors; it binds two already-built stores structurally.

Caller responsibilities:

- Ensure pair order already matches between robot and SMPL stores.
- Ensure pairing and trimming were done upstream.
