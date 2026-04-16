#!/usr/bin/env bash
set -euo pipefail

# 1) 先进入项目根目录
cd /home/thl/wt_wbc/wbc_parkour/whole_body_tracking

# 2) 激活 Isaac Sim 环境（按你现在的方式）
# source setup_conda_env.sh

# 3) 配置
INPUT_DIR="/home/thl/wt_wbc/dataset/data/amass_single_motion"      # 例如: /home/thl/.../AMASS_single_motion
BATCH_SIZE=2000

# 4) 统计 npz 文件数（dataset_2_npz.py 当前按 npz 读取）
TOTAL_FILES=$(find "$INPUT_DIR" -type f -name "*.npz" | wc -l)
if [ "$TOTAL_FILES" -eq 0 ]; then
  echo "没有找到 npz 文件: $INPUT_DIR"
  exit 1
fi

TOTAL_BATCHES=$(( (TOTAL_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "total_files=$TOTAL_FILES, batch_size=$BATCH_SIZE, total_batches=$TOTAL_BATCHES"

# 5) 逐批运行
for ((i=0; i<TOTAL_BATCHES; i++)); do
  echo "===== batch $i / $((TOTAL_BATCHES-1)) ====="
  python scripts/dataset_2_npz.py \
    --input_file "$INPUT_DIR" \
    --max_files_per_run "$BATCH_SIZE" \
    --batch_index "$i" \
    --headless
done

echo "全部批次处理完成"
