python scripts/eval/eval.py \
  --task=Tracking-Flat-RobanS22-v0 \
  --motion_file=data/roban_motions \
  --motion_file_txt=data/roban_motions_list/motions_crawl_kept.txt \
  --resume_path=logs/rsl_rl/roban_flat/0420_crawl_kept/model_12400.pt \
  --output_path=eval_results/motions_crawl_kept.json \
  --headless \
  --compute_success_rate
