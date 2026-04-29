python scripts/deploy/export_and_play_onnx.py \
  --task Tracking-Flat-RobanS22-v0 \
  --resume_path logs/rsl_rl/roban_flat/0422_all_kept/model_79000.pt \
  --motion_file data/roban_motions \
  --motion_file_txt data/roban_motions_list/motions_squat_kept.txt \
  --num_envs 1
