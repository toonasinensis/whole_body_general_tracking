python scripts/deploy/export_and_play_onnx.py \
  --task Tracking-Flat-RobanS22-v0 \
  --resume_path logs/rsl_rl/roban_flat/2026-05-08_23-45-45_motions_crawl_stand_up/model_37000.pt \
  --motion_file data/roban_motions \
  --motion_file_txt data/roban_motions_list/motions_crawl_stand_up.txt \
  --num_envs 1
