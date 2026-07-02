#!/usr/bin/env bash

set -e

ARGS=(
  --task=AMP-G1
  --num_envs=40
  --resume_path=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/model_46500.pt
  --motion_file=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/g1_amp/WalkandRun
  --dataset_txt=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/g1_amp/WalkandRun/WalkandRun.txt
#   --headless
  # Export/load ONNX.
  --export_onnx
  --use_onnx_policy
  --export_only
  # --onnx_dir=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/exported
  # --onnx_filename=policy.onnx
  # --onnx_path=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/exported/policy.onnx
  # --debug_compare_torch_onnx

  # Play-time TensorBoard logging for command metrics.
  # Logs command/error_anchor_pos, command/error_body_pos, command/error_joint_pos, etc.
#   --logger=tensorboard
#   --tb_log_dir=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/tensorboard/play1
#   --tb_log_interval=50
  # Logs episode_length/env_average_mean/min/max/std and env_average_hist.
  # Uncomment this to also log one average episode length scalar per env id.
  # --tb_log_episode_length_per_env

  # Video recording. Output goes to <checkpoint_dir>/videos/play.
  # --video
  # --video_length=200

  # Exit automatically after N env steps.
  # --max_steps=1000

  # Policy routing/debug options.
  # --encoder_mode=robot
  # --encoder_mode=split
  # --fsq_sample_mode=encode
  # --fsq_sample_mode=random
  # --debug_zero_obs
  # --debug_obs_path=/tmp/isaac_zero_obs.pt
)

python scripts/rsl_rl/play.py "${ARGS[@]}" "$@"
