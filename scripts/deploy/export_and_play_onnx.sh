python scripts/deploy/export_and_play_onnx.py \
  --task Tracking-Flat-RobanS22-v0 \
  --resume_path logs/rsl_rl/roban_flat/2026-04-22_19-57-52_quick_test/model_2000.pt \
  --motion_file data/roban_motions \
  --motion_file_txt data/roban_motions_list/quick_test.txt \
  --num_envs 1
