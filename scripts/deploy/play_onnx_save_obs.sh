# python scripts/deploy/play_onnx_save_obs.py \
#   --task Tracking-Flat-RobanS22-v0 \
#   --motion_file data/roban_motions \
#   --motion_file_txt data/roban_motions_list/debug_motion.txt \
#   --onnx_path logs/rsl_rl/roban_flat/0422_quick_test/exported/policy.onnx \
#   --output_csv logs/debug_motion.csv \
#   --max_steps 2000

python scripts/deploy/play_onnx_save_obs.py \
  --task Tracking-Flat-RobanS22-v0 \
  --num_envs 1 \
  --motion_file data/roban_motions \
  --motion_file_txt data/roban_motions_list/debug_motion.txt \
  --onnx_path logs/rsl_rl/roban_flat/0422_quick_test/exported/policy.onnx \
  --output_csv logs/debug_motion.csv \
  --history_index 9 \
  --max_steps 3208 \
  --output_dir logs/debug_motion.csv_terms \
  --output_obs_dump_csv logs/debug_motion.csv_terms/observations_dump_isaaclab.csv \
  --zero_unfilled_history
