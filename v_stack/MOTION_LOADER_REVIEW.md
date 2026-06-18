# MotionLoader Usage Review

This note records the current review findings for `commands.py` and its use of
`MotionLoader` / `UnifiedMotionLib`. The main theme is that `MotionLoader.load()`
replaces the whole motion buffer and index state, while `MotionCommand` keeps
per-environment cached indices that may still refer to the previous buffer.

## Reviewed Files

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `source/whole_body_tracking/whole_body_tracking/utils/motionlib/feat_motion_lib/facade/unified_motion_lib.py`
- `source/whole_body_tracking/whole_body_tracking/utils/motionlib/feat_motion_lib/facade/config_adapter.py`
- `source/whole_body_tracking/whole_body_tracking/utils/motionlib/feat_motion_lib/facade/orchestration.py`
- `source/whole_body_tracking/whole_body_tracking/utils/motionlib/feat_motion_lib/store/index.py`

## Findings

### 1. Resampling reloads MotionLoader but keeps stale per-env state

Severity: High

Location:

- `commands.py::resample_motion_files`
- `commands.py::_update_command`
- `commands.py::_adaptive_sampling`

`resample_motion_files()` calls:

```python
self.motion.resample_motionloader(device=self.device)
```

`MotionLoader.resample_motionloader()` calls `UnifiedMotionLib.load_from_cfg()`.
That eventually calls `UnifiedMotionLib.load()`, which resets and replaces:

- motion tensors
- `time_step_start_idx`
- `time_step_end_idx`
- `motion_num`
- `time_step_total`
- `file_names`

However, `MotionCommand` keeps these per-environment values outside the loader:

- `self.motion_ids`
- `self.local_time_steps`
- `self.frame_end_per_env`

After a reload, those values still describe the previous concatenated motion
buffer. The next call path can interpret old `motion_ids` and old local time
steps against the new `MotionLoader` indices.

Failure modes:

- If the new motion set has fewer motions, indexing
  `self.motion.time_step_start_idx[self.motion_ids]` can go out of bounds.
- If the new motion set has the same count but different order/content, failure
  statistics and references can silently point at the wrong clip.
- `_adaptive_sampling()` records the failed bin using `self.global_time_steps`
  after the loader may already have been replaced, so failure attribution can be
  corrupted.

Suggested direction:

- Treat a loader reload as an atomic state transition.
- Before replacing the loader, record any failure statistics that depend on the
  old buffer.
- After replacing the loader, reset or immediately resample all envs so that
  `motion_ids`, `local_time_steps`, and `frame_end_per_env` are valid for the
  new buffer.
- Add a small helper such as `_reset_motion_state_after_reload(env_ids)` or make
  `resample_motion_files()` responsible for rebuilding all dependent state.

### 2. Fixed eval mapping is not rebuilt after MotionLoader reload

Severity: High

Location:

- `commands.py::_setup_fixed_eval_motion_assignment`
- `commands.py::_resample_command`
- `commands.py::_update_command`

In fixed evaluation mode, env ids are pinned to motion ids:

```python
self.fixed_eval_motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
```

This mapping is created once. After `resample_motion_files()` reloads the
loader, `_resample_command()` only rebuilds it when:

```python
if self.fixed_eval_motion_ids is None:
    self._setup_fixed_eval_motion_assignment()
```

So a non-`None` stale mapping can survive a loader reload.

Failure modes:

- If the new loader contains a different number of motions, fixed mapping can
  become invalid or fail with an out-of-bounds index.
- If the new loader changes motion order, each env can evaluate a different
  motion than intended.
- `eval_cycle_count` may continue across a different motion assignment.

Suggested direction:

- In fixed eval mode, call `_setup_fixed_eval_motion_assignment()` after every
  loader reload.
- Consider disabling periodic `resample_interval` reloads for fixed eval unless
  reloading is explicitly required.
- If reloading is allowed, validate that `num_envs == motion_num` after each
  reload and reset `eval_cycle_count` intentionally.

### 3. Adaptive failure bins are overwritten within a step

Severity: Medium

Location:

- `commands.py::_adaptive_sampling`

`_resample_command()` documents that it may be called multiple times in one
step. `_adaptive_sampling()` records failures with:

```python
self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)
```

This overwrites any failure bins already collected earlier in the same command
step. Later `_update_command()` folds only the final value into
`bin_failed_count`.

Failure modes:

- Multiple reset paths in the same step can drop earlier failures.
- Adaptive sampling may underestimate failure density.
- Hard bins can be missed if a later call overwrites them with fewer failures.

Suggested direction:

- Accumulate instead of overwrite:

```python
self._current_bin_failed += torch.bincount(fail_bins, minlength=self.bin_count)
```

- Keep the existing `_current_bin_failed.zero_()` at the end of
  `_update_command()` so accumulation is scoped to one command update.

### 4. Future-frame index shape does not match its documented contract

Severity: Medium

Location:

- `commands.py::future_time_steps`
- `commands.py::future_motion_ids`
- `commands.py::anchor_pos_w_future`
- `commands.py::joint_pos_future`
- `commands.py::joint_vel_future`
- `commands.py::anchor_quat_w_future`

The `future_time_steps` docstring says it returns a flattened tensor with shape:

```text
(num_envs * num_future_frames,)
```

The implementation actually returns:

```text
(num_envs, num_future_frames)
```

Current tensor indexing works for several call sites because PyTorch preserves
the two-dimensional leading shape. However, `future_motion_ids` returns a
flattened tensor, so the two properties do not share the same shape convention.

Failure modes:

- Future code can pair flattened `future_motion_ids` with 2D
  `future_time_steps` and get shape mismatches.
- SMPL getters or observation builders can silently flatten/view tensors in the
  wrong order.
- The documented API is misleading for future maintenance.

Suggested direction:

- Pick one convention and make all future-frame helpers follow it.
- If 2D is desired, update the docstring and make `future_motion_ids` return 2D
  or add separate explicit names such as `future_time_steps_flat`.
- If flattened is desired, return:

```python
return (start[:, None] + future_local).long().reshape(-1)
```

and ensure all current consumers reshape intentionally.

### 5. Initial adaptive sampling probabilities are not normalized

Severity: Low to Medium

Location:

- `commands.py::_compute_sampling_probabilities`

When `bin_failed_count` is all zeros, the middle-hard and most-hard probability
branches remain all zeros. The final weights become:

```python
self.cfg.motion_ratio[0] * uniform
```

With the current default `motion_ratio = [0.2, 0.79, 0.01]`, the sum of weights
is `0.2`, not `1.0`.

`torch.multinomial()` accepts non-normalized non-negative weights, so sampling
still works as long as the sum is positive. But the metrics treat these weights
as probabilities.

Failure modes:

- Entropy and top probability metrics are incorrect during the zero-failure
  phase.
- If `motion_ratio[0]` is set to zero, all sampling weights can become zero and
  `torch.multinomial()` will fail.

Suggested direction:

- Normalize final sampling weights before returning.
- Add a fallback to uniform when the sum is zero or non-finite.

Example:

```python
sampling_probabilities = torch.clamp(sampling_probabilities, min=0.0)
total = sampling_probabilities.sum()
if not torch.isfinite(total) or total <= 0:
    sampling_probabilities = torch.full_like(sampling_probabilities, 1.0 / self.bin_count)
else:
    sampling_probabilities = sampling_probabilities / total
```

## Suggested Fix Order

1. Fix loader reload state synchronization.
2. Rebuild fixed eval mapping after every reload.
3. Change adaptive failure bin update from overwrite to accumulation.
4. Normalize adaptive sampling probabilities.
5. Make future-frame helper shapes explicit and consistent.

## Regression Checks To Add

- Reload with a different `motion_num` and verify no stale `motion_ids` are used.
- Fixed eval reload with `num_envs == motion_num` and verify env `i` maps to
  motion `i` after reload.
- Multiple `_resample_command()` calls in one step should accumulate failures
  from all calls.
- `future_time_steps` and `future_motion_ids` should have matching documented
  shape conventions.
- Zero-failure adaptive sampling should return a normalized non-zero
  distribution.
