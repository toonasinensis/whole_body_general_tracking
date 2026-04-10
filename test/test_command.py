""" Scripts to test the multi motion command class.
it shall return the correct command and meta information.
"""
from whole_body_tracking.utils.motion_dataset import Motion_Bins_Dataset
from whole_body_tracking.utils.motion_dataloader import Motion_Bins_Dataloader
import torch
import sys

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    dataset_dir = "assets/motion_bins_50"
    body_indexes = [ 0,  3,  4, 12, 20,  5, 13, 21, 10, 18, 26, 11, 19, 27]
    dataset = Motion_Bins_Dataset(dataset_dir=dataset_dir)
    dataloader = Motion_Bins_Dataloader(dataset=dataset, body_indexes=body_indexes)
    
    # Randomly sample the motion ids and time steps to mimic sampling results
    motion_ids = torch.randint(0, dataloader.num_motions, (1024,))
    time_steps = torch.randint(0, 50, (1024,))
    # Check the command properties
    joint_pos = dataloader.motion_buffer.joint_pos[motion_ids, time_steps]
    body_pos_w = dataloader.motion_buffer.body_pos_w[motion_ids, time_steps]
    anchor_pos = dataloader.motion_buffer.body_pos_w[motion_ids, time_steps, 0]
    # Check the commands output
    horizon = 10
    fps_env = dataloader.motion_fps[motion_ids]  # [num_envs]
    interval_steps = torch.clamp((fps_env / float(10)).round().long(), min=1)  # [num_envs]
    base_steps = time_steps  # [num_envs]
    # Offsets for each frame in the horizon, scaled per-env by interval_steps
    frame_ids = torch.arange(horizon, dtype=base_steps.dtype, device=interval_steps.device)      # [horizon] 10 frames
    offsets = interval_steps.unsqueeze(1) * frame_ids.unsqueeze(0) # [num_envs, horizon]
    future_global_time_steps = base_steps.unsqueeze(1).to(offsets.device) + offsets   # [num_envs, horizon]
    # Clamp future timesteps to stay within the current motion for each env
    motion_end_steps = dataloader.motion_lengths[motion_ids] - 1 + dataloader.one_sec_frames # [num_envs]
    future_global_time_steps = torch.clamp(future_global_time_steps, max=motion_end_steps.unsqueeze(1))     # [num_envs, horizon]
    # Gather joint positions / velocities using matching [N, H] motion/time indices
    motion_ids_expanded = motion_ids.unsqueeze(1).expand_as(future_global_time_steps)          # [N, H]
    joint_pos = dataloader.motion_buffer.joint_pos[motion_ids_expanded, future_global_time_steps]  # [N, H, J]
    joint_vel = dataloader.motion_buffer.joint_vel[motion_ids_expanded, future_global_time_steps]  # [N, H, J]

    cmd = torch.cat([joint_pos, joint_vel], dim=-1)  # [N, H, 2*J]
    cmd = cmd.view(1024, -1)
    print(cmd.shape)
