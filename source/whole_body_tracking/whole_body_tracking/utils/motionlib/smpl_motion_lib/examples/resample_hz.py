"""Batch resample all motion pkl files in a directory from 30Hz to 50Hz.

Pipeline:
  1. Walk src_dir, load raw pkl files one by one into data_list.
  2. Pad & stack into (B, T_max, ...) tensors.
  3. Single-pass batch interpolation via vectorised gather + slerp/lerp.
  4. Save each result back to dst_dir, preserving sub-directory structure.
"""

import os
import torch

import joblib
from smpl_math_utils.interpolation import slerp
from smpl_math_utils.rotation import angle_axis_to_quaternion, quaternion_to_angle_axis
from smpl_motion_lib.loader import load_pkl

# ── config ────────────────────────────────────────────────────────────────────
SRC_DIR = "/home/thl/Downloads/data/TEST"
DST_DIR = "/home/thl/Downloads/data/TEST_50hz"
SRC_FPS = 50.0
TGT_FPS = 100.0
DEVICE = "cuda"  # change to "cuda" if available
# ─────────────────────────────────────────────────────────────────────────────


def _frame_indices(lengths: torch.Tensor, src_fps: float, tgt_fps: float):
    """Return batch gather indices for resampling.

    Returns:
        idx0, idx1: (B, T_new_max)  int64
        blend:      (B, T_new_max)  float32   values in [0, 1]
        out_lengths:(B,)            int64
    """
    B = lengths.shape[0]
    out_lengths = (((lengths.float() - 1) / src_fps) * tgt_fps).floor().long() + 1
    T_new_max = int(out_lengths.max().item())

    t_out = torch.arange(T_new_max, dtype=torch.float32, device=lengths.device) / tgt_fps
    frame_float = t_out * src_fps  # (T_new_max,)

    idx0 = frame_float.floor().long()
    idx0 = idx0.unsqueeze(0).expand(B, -1).clone()  # (B, T_new_max)
    max_idx = (lengths - 1).unsqueeze(1)  # (B, 1)
    idx0 = torch.min(idx0, max_idx)
    idx1 = torch.min(idx0 + 1, max_idx)

    blend = (frame_float.unsqueeze(0).expand(B, -1) - idx0.float()).clamp(0.0, 1.0)
    return idx0, idx1, blend, out_lengths


def batch_resample_linear(data: torch.Tensor, lengths: torch.Tensor, src_fps: float, tgt_fps: float):
    """Linearly resample a padded batch.

    Args:
        data:    (B, T_max, *D)
        lengths: (B,) actual frame counts per sample
    Returns:
        out:        (B, T_new_max, *D)
        out_lengths:(B,)
    """
    trailing = data.shape[2:]
    flat = data.reshape(data.shape[0], data.shape[1], -1)  # (B, T_max, D)
    D = flat.shape[2]
    B = flat.shape[0]

    idx0, idx1, blend, out_lengths = _frame_indices(lengths, src_fps, tgt_fps)
    T_new = idx0.shape[1]

    g0 = idx0.unsqueeze(-1).expand(-1, -1, D)
    g1 = idx1.unsqueeze(-1).expand(-1, -1, D)
    v0 = flat.gather(1, g0)
    v1 = flat.gather(1, g1)

    out = (1.0 - blend.unsqueeze(-1)) * v0 + blend.unsqueeze(-1) * v1
    return out.reshape(B, T_new, *trailing), out_lengths


def batch_resample_pose_slerp(pose_aa: torch.Tensor, lengths: torch.Tensor, src_fps: float, tgt_fps: float):
    """Slerp resample an axis-angle pose batch.

    Args:
        pose_aa: (B, T_max, J*3)  axis-angle
        lengths: (B,) actual frame counts
    Returns:
        out:        (B, T_new_max, J*3)
        out_lengths:(B,)
    """
    B, T_max, _ = pose_aa.shape
    J = pose_aa.shape[2] // 3

    # convert all frames + joints to quaternions in one shot
    quat = angle_axis_to_quaternion(pose_aa.reshape(B * T_max * J, 3))
    quat = quat.reshape(B, T_max, J, 4)

    idx0, idx1, blend, out_lengths = _frame_indices(lengths, src_fps, tgt_fps)
    T_new = idx0.shape[1]

    g0 = idx0.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, J, 4)
    g1 = idx1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, J, 4)
    q0 = quat.gather(1, g0)  # (B, T_new, J, 4)
    q1 = quat.gather(1, g1)

    t_exp = blend.unsqueeze(-1).expand(-1, -1, J).reshape(B * T_new * J)
    q_interp = slerp(q0.reshape(B * T_new * J, 4), q1.reshape(B * T_new * J, 4), t_exp)

    aa = quaternion_to_angle_axis(q_interp).reshape(B, T_new, J * 3)
    return aa, out_lengths


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DST_DIR, exist_ok=True)

    # Phase 1: collect paths + load raw data one by one
    rel_paths, data_list = [], []
    for root, _, files in os.walk(SRC_DIR):
        for f in sorted(files):
            if not f.endswith(".pkl"):
                continue
            abs_path = os.path.join(root, f)
            rel = os.path.relpath(abs_path, SRC_DIR)
            raw = load_pkl(abs_path)
            rel_paths.append(rel)
            data_list.append(raw)
            print(f"Loaded [{len(rel_paths)}] {rel}  ({raw['pose_aa'].shape[0]} frames @ {raw['fps']} fps)")

    B = len(data_list)
    print(f"\n{B} files loaded. Building batch tensors...")

    # Phase 2: pad & stack into (B, T_max, ...) tensors
    lengths = torch.tensor([d["pose_aa"].shape[0] for d in data_list], dtype=torch.long)
    T_max = int(lengths.max().item())

    pose_batch = torch.zeros(B, T_max, 72, device=DEVICE)
    joints_batch = torch.zeros(B, T_max, 24, 3, device=DEVICE)
    transl_batch = torch.zeros(B, T_max, 3, device=DEVICE)

    for i, d in enumerate(data_list):
        T_i = d["pose_aa"].shape[0]
        pose_batch[i, :T_i] = torch.from_numpy(d["pose_aa"])
        joints_batch[i, :T_i] = torch.from_numpy(d["smpl_joints"])
        transl_batch[i, :T_i] = torch.from_numpy(d["transl"])

    lengths = lengths.to(DEVICE)

    # Phase 3: batch interpolation (single pass, no per-file loop)
    print("Batch resampling...")
    pose_out, out_lengths = batch_resample_pose_slerp(pose_batch, lengths, SRC_FPS, TGT_FPS)
    joints_out, _ = batch_resample_linear(joints_batch, lengths, SRC_FPS, TGT_FPS)
    transl_out, _ = batch_resample_linear(transl_batch, lengths, SRC_FPS, TGT_FPS)

    # Phase 4: save each result
    print("Saving...")
    for i, rel in enumerate(rel_paths):
        n = int(out_lengths[i].item())
        out_path = os.path.join(DST_DIR, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        joblib.dump(
            {
                "pose_aa": pose_out[i, :n].cpu().numpy(),
                "smpl_joints": joints_out[i, :n].cpu().numpy(),
                "transl": transl_out[i, :n].cpu().numpy(),
                "fps": TGT_FPS,
                "source_fps": SRC_FPS,
            },
            out_path,
        )
        print(f"  Saved: {out_path}  ({n} frames)")

    print(f"\nDone. {B} files → {DST_DIR}")
