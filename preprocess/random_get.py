import os
import random
import shutil
from collections import defaultdict

txt_path = "/home/thl/Documents/g1-mimic-npz/hard.txt"  # 你的txt
src_root = "/home/thl/Documents/g1-mimic-npz"  # 数据根目录
dst_dir = "/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/g1/EVAL"
transfer_all = False  # True: 转移全部；False: 每个类别随机抽一个
enable_batch = False  # True: 按 batch 分文件夹；False: 全部放在 dst_dir
batch_size = 2000  # 每个 batch 的文件数量
os.makedirs(dst_dir, exist_ok=True)

groups = defaultdict(list)
all_files = []
missing_files = []

# 1️⃣ 从 txt 读取路径并分组
with open(txt_path) as f:
    for line in f:
        rel_path = line.strip()
        if not rel_path:
            continue

        full_path = os.path.join(src_root, rel_path)

        if os.path.exists(full_path):
            all_files.append(full_path)
            prefix = os.path.basename(rel_path).split("_")[0]
            groups[prefix].append(full_path)
        else:
            missing_files.append(rel_path)

selected = []

# 2️⃣ 根据开关执行转移
if transfer_all:
    if enable_batch:
        # 全量 + 分 batch：按索引划分，每个 batch 目录下生成同名 txt
        batch_map = defaultdict(list)  # batch_label -> [rel_path, ...]
        for i, path in enumerate(all_files):
            start = (i // batch_size) * batch_size
            end = start + batch_size
            label = f"{start}-{end}"
            batch_dir = os.path.join(dst_dir, label)
            os.makedirs(batch_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(batch_dir, os.path.basename(path)))
            batch_map[label].append(os.path.basename(path))

        # 为每个 batch 写 txt
        for label, names in sorted(batch_map.items()):
            txt_out = os.path.join(dst_dir, f"{label}.txt")
            with open(txt_out, "w") as f:
                f.write("\n".join(names) + "\n")
            print(f"  Batch {label}: {len(names)} files -> {txt_out}")
    else:
        # 全量 + 不分 batch：全放 dst_dir，写一个汇总 txt
        for path in all_files:
            shutil.copy2(path, os.path.join(dst_dir, os.path.basename(path)))
        txt_out = os.path.join(dst_dir, "all_files.txt")
        with open(txt_out, "w") as f:
            f.write("\n".join(os.path.basename(p) for p in all_files) + "\n")
        print(f"  All files txt -> {txt_out}")
else:
    # 每个类别抽一个（不走 batch）
    for k, files in groups.items():
        pick = random.choice(files)
        selected.append((k, pick))
        shutil.copy2(pick, os.path.join(dst_dir, os.path.basename(pick)))

    # 写一个汇总 txt
    txt_out = os.path.join(dst_dir, "selected_files.txt")
    with open(txt_out, "w") as f:
        for k, p in selected:
            f.write(os.path.basename(p) + "\n")
    print(f"  Selected files txt -> {txt_out}")

# 3️⃣ 输出统计
print("Total classes:", len(groups))
print("Matched files:", len(all_files))
print("Missing files:", len(missing_files))

if transfer_all:
    print("Transferred(all):", len(all_files))
    if enable_batch:
        print("Batch mode: ON, batch size:", batch_size)
    else:
        print("Batch mode: OFF")
else:
    print("Selected:", len(selected))
    for k, p in selected:
        print(k, "->", os.path.basename(p))
