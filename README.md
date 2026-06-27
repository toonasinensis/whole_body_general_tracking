# Whole Body Tracking

This repository is the local G1 whole-body tracking workspace for Isaac Lab / RSL-RL training, AMP training, motion
NPZ preprocessing, ONNX export, and MuJoCo sim2sim evaluation.

The current code path is built around:

- Isaac Lab manager-based environments registered as `TR-G1`, `FM-G1`, `AMP-G1`, and `Distill-Flat-G1-v0`.
- G1 motion NPZ files under a motion directory, optionally filtered by a `dataset_txt`.
- Name-aware motion loading through optional `joint_names` and `body_names` arrays stored inside each NPZ.
- Adaptive motion sampling that records the real terminate bin, then respawns from an earlier bin in the same motion.
- Sim2sim rollout with per-motion mean metrics and visualization/comparison scripts.

## Environment

Typical local setup:

```bash
cd /home/thl/wt_wbc/wbc_parkour/whole_body_tracking
source /home/thl/miniconda3/etc/profile.d/conda.sh
conda activate my_env
source /home/thl/isaacsim5.1/setup_conda.sh
python -m pip install -e source/whole_body_tracking

git clone https://github.com/toonasinensis/rsl_rl.git
cd rsl_rl
pip install -e .
python -m pip install -U vector-quantize-pytorch
export PYTHONPATH=$PWD/rsl_rl:$PWD/source:$PYTHONPATH
```

For non-Isaac utilities such as sim2sim plots, `my_env` is usually enough. For scripts that launch Isaac Lab, source the
Isaac Sim setup first.

## Data Layout

Common local inputs:

```text
data/lafan/                 named LAFAN NPZ files
data/up/                    get-up NPZ files
dataset_txt/*.txt           file lists used to filter a motion directory
/home/thl/Documents/g1-mimic-npz
```

A dataset txt contains one motion path per line. Paths can be absolute, or relative to the motion root used by the
loader/script.

## Motion NPZ Metadata

New NPZ files should contain:

- `joint_names`: order of `joint_pos` / `joint_vel`.
- `body_names`: order of `body_pos_w` / `body_quat_w` / body velocity arrays.

The loaders are backward-compatible with older files when the dimensions match known G1 layouts, but named NPZ files
are safer. To attach metadata without overwriting the original:

```bash
python scripts/attach_npz_names.py data/lafan --force
```

This writes sidecars like `walk2_subject1.named.npz`. For a dataset txt:

```bash
python scripts/attach_npz_names.py dataset_txt/lafan.txt \
  --motion_root /home/thl/Documents/g1-mimic-npz \
  --force
```

Useful options:

```bash
--body_order auto            # g1_motion or g1_full_isaac selected from body dim
--body_order g1_full_mjcf    # for MuJoCo-order full-body files
--joint_names_file names.txt
--body_names_file names.txt
--in_place                   # overwrite input NPZ, only when you really mean it
```

`scripts/json_to_npz.py` should write `joint_names` and `body_names` for newly generated data.

## Training

Main entry:

```bash
python scripts/rsl_rl/train.py \
  --task=FM-G1 \
  --registry_name=test1 \
  --headless \
  --num_envs=4096 \
  --motion_file=/home/thl/Documents/g1-mimic-npz \
  --dataset_txt=dataset_txt/lafan.txt
```

Distributed launch example:

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
  scripts/rsl_rl/train.py \
  --task=AMP-G1 \
  --registry_name=test1 \
  --headless \
  --distributed \
  --num_envs=8000 \
  --motion_file=/home/thl/Documents/g1-mimic-npz \
  --dataset_txt=dataset_txt/walk2_subject1.txt
```

There is also a local convenience script:

```bash
./scripts/rsl_rl/train_g1.sh
```

Check and edit the hard-coded paths in that script before using it for a new run.

## Play And ONNX Export

Play a checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task=FM-G1 \
  --num_envs=50 \
  --resume_path=logs/rsl_rl/g1_flat/model_60000.pt \
  --motion_file=/home/thl/Documents/g1-mimic-npz \
  --dataset_txt=dataset_txt/lafan.txt
```

Export ONNX only:

```bash
python scripts/rsl_rl/play.py \
  --task=FM-G1 \
  --num_envs=1 \
  --resume_path=logs/rsl_rl/g1_flat/model_60000.pt \
  --motion_file=/home/thl/Documents/g1-mimic-npz \
  --dataset_txt=dataset_txt/lafan.txt \
  --export_onnx \
  --export_only
```

Convenience script:

```bash
./scripts/rsl_rl/play_g1.sh
```

## Adaptive Motion Sampling

The command term uses adaptive sampling to focus on hard parts of motions. The important config fields are in
`MotionCommandCfg`:

```python
motion_sampling_start_frame = 5
adaptive_sample_rewind_min_bins = 1
adaptive_sample_rewind_bins = 2
```

The terminate bin is `m`.

- `motion_sampling_start_frame` prevents sampling from bad first frames.
- `adaptive_sample_rewind_bins` is the maximum rewind distance.
- `adaptive_sample_rewind_min_bins` is the minimum distance away from the terminate bin.

So the respawn window is:

```text
[m - adaptive_sample_rewind_bins, m - adaptive_sample_rewind_min_bins]
```

Examples:

```text
min=1, max=2 -> sample m-1..m-2
min=2, max=5 -> sample m-2..m-5
```

Failure counts are still recorded at the true terminate global bin for debugging/export. Runtime respawn sampling uses a
motion-local spawn-bin table, so a failure in one motion cannot rewind into another motion. Smoothing is also applied
inside each motion only.

Adaptive bin exports are written under:

```text
<log_save_path>/<log_run_name>/adaptive_bins_rank_XX_step_XXXXXXXXX.json
```

## Recovery Curriculum

G1 flat/terrain configs currently set:

```python
pose_range_env_ratio = 0.30
pose_range_init_mode = "lying"
pose_range_lying_height_range = (0.25, 0.45)
```

The pose-range env subset is also used by delayed termination:

```python
install_delayed_termination(..., delay_reset_env_ratio=..., max_delay_steps=250)
```

This means the same env subset can spawn lying down and get a recovery window before early termination resets it.

The optional upward assist event is configured in the G1 env config. Set `debug_steps=0` to silence its debug prints.

## AMP Motion Loader Debug

AMP uses name-aware body mapping. The anchor must be included in `G1_AMP_BODY_NAMES`.

Visualize AMPLoader output:

```bash
python ../rsl_rl/rsl_rl/algorithms/plugins/amp/visualize_motion_loader.py \
  data/lafan/walk2_subject1.named.npz \
  --port 8088
```

Open:

```text
http://127.0.0.1:8088
```

Use this when AMP training looks wrong; it shows the exact body order after the AMP loader, not just the raw NPZ.

## Sim2sim

The MuJoCo runner is robot-config driven. Built-in configs live under
`scripts/deploy/sim2sim_g1/configs/`; use `--robot_config g1` or
`--robot_config roban`, or pass a custom JSON config path.

Run Roban with the existing checkpoint and motion directory:

```bash
EXPORT_ONNX=1 RENDER=0 ./scripts/deploy/sim2sim_roban_mujoco.sh --steps 100
```

After the ONNX has been exported once, omit `EXPORT_ONNX=1`. The wrapper uses
the `mimic` conda environment by default.

Run MuJoCo sim2sim with an existing ONNX:

```bash
EXPORT_ONNX=0 \
ONNX_PATH=logs/rsl_rl/g1_flat/exported/policy.onnx \
MOTION_FILE=/home/thl/Documents/g1-mimic-npz \
DATASET_TXT=dataset_txt/lafan.txt \
METRICS_TAG=my_policy \
./scripts/deploy/sim2sim_g1_mujoco.sh
```

If `DATASET_TXT` is empty, the sim2sim script uses every `.npz` under `MOTION_FILE`:

```bash
DATASET_TXT= MOTION_FILE=data/up ./scripts/deploy/sim2sim_g1_mujoco.sh
```

Metrics CSV naming:

```text
<dataset_txt>.sim2sim_metrics[_tag]_<timestamp>.csv
<motion_file>.sim2sim_metrics[_tag]_<timestamp>.csv   # when DATASET_TXT is empty
```

Set `METRICS_CSV=/path/out.csv` to force a path, or `--metrics_csv ""` to disable metrics.

The metrics are per-motion means:

```text
error_anchor_pos
error_anchor_rot
error_anchor_lin_vel
error_anchor_ang_vel
error_body_pos
error_body_rot
error_body_lin_vel
error_body_ang_vel
error_joint_pos
error_joint_vel
```

## Sim2sim Plots

Visualize one CSV:

```bash
python scripts/deploy/visualize_sim2sim_metrics.py \
  --csv dataset_txt/lafan.txt.sim2sim_metrics_my_policy_20260529_120000.csv
```

Compare multiple policies:

```bash
python scripts/deploy/compare_sim2sim_metrics.py \
  policy_a.csv policy_b.csv \
  --labels policy_a,policy_b
```

Or compare the latest files in a directory:

```bash
python scripts/deploy/compare_sim2sim_metrics.py \
  --metrics_dir dataset_txt \
  --latest 5
```

Both scripts write PNGs, CSV summaries, and an `index.html` report.

## Useful Debug Commands

Validate motion loader / sampling helper:

```bash
pytest -q tests/test_motion_sampling.py
```

Compile changed tracking files:

```bash
python -m py_compile \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_sampling.py
```

Run sim2sim dry-run validation:

```bash
python scripts/deploy/sim2sim_g1_mujoco.py \
  --onnx_path logs/rsl_rl/g1_flat/exported/policy.onnx \
  --motion_file /home/thl/Documents/g1-mimic-npz \
  --dataset_txt dataset_txt/mini_test.txt \
  --dry_run
```

## Code Map

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py
  Motion command, adaptive sampling, reset initialization, tracking metrics.

source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_sampling.py
  Small tested helpers for motion-local sampling windows.

source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py
  Tracking and auxiliary rewards.

source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py
  Early termination and delayed termination wrapper.

source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/
  G1 task registration and env config.

scripts/attach_npz_names.py
  Add `joint_names` / `body_names` metadata to old NPZ files.

scripts/rsl_rl/
  Isaac Lab train/play/export entry points.

scripts/deploy/
  ONNX, MuJoCo sim2sim, metrics visualization, policy comparison.

../rsl_rl/rsl_rl/algorithms/plugins/amp/
  AMP plugin, name-aware AMPLoader, AMPLoader visualizer.
```
