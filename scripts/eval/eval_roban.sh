python scripts/eval/eval.py \
  --task=Tracking-Flat-RobanS22-v0 \
  --motion_file=data/roban_motions \
  --motion_file_txt=data/roban_motions_list/motions_main_kept.txt \
  --num_envs=11116 \
  --resume_path=logs/rsl_rl/roban_flat/0422_all_kept/model_79000.pt \
  --output_path=eval_results/motions_main_kept.json \
  --headless \
  --compute_success_rate
