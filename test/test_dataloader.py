""" Scripts to test the motion dataloader class.
it shall return the correct motion bins and meta information.
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
    # Check dimention of dataloader
    print(dataloader.motion_buffer._body_pos_w.shape)
    print(dataloader.motion_buffer._body_quat_w.shape)
    print(dataloader.motion_buffer._body_lin_vel_w.shape)
    print(dataloader.motion_buffer._body_ang_vel_w.shape)
    # Exposed data buffer port
    print("Exposed data buffer port: body_pos_w",     dataloader.motion_buffer.body_pos_w.shape)
    print("Exposed data buffer port: body_quat_w",    dataloader.motion_buffer.body_quat_w.shape)
    print("Exposed data buffer port: body_lin_vel_w", dataloader.motion_buffer.body_lin_vel_w.shape)
    print("Exposed data buffer port: body_ang_vel_w", dataloader.motion_buffer.body_ang_vel_w.shape)
    print("Exposed data buffer port: joint_pos", dataloader.motion_buffer.joint_pos.shape)
    print("Exposed data buffer port: joint_vel", dataloader.motion_buffer.joint_vel.shape)
    # Check meta data
    print("Mean motion length: ", torch.mean(dataloader.motion_lengths.float()))
    print("Mean motion fps: ", torch.mean(dataloader.motion_fps.float()))
    print("Total time steps: ", dataloader.time_step_total)
    print("Total motions: ", dataloader.num_motions)
    print("One sec frames: ", dataloader.one_sec_frames)


    #region Ensure the returned body_* is in the expected order
    body_pos  = dataloader.motion_buffer.body_pos_w[0, 0, :, :] # [num_bodies, 3]
    body_name = [
                 'base_link', 
                 'waist_yaw', 
                 'leg_l2_link', 'leg_l4_link', 'leg_l6_link', 
                 'leg_r2_link', 'leg_r4_link', 'leg_r6_link', 
                 'zarm_l2_link', 'zarm_l4_link', 'zarm_l7_link', 
                 'zarm_r2_link', 'zarm_r4_link', 'zarm_r7_link'
                ]
    # plot the body pos using matploy lab with the body_names annotated to the body_pos points
    import matplotlib.pyplot as plt
    body_pos = body_pos.cpu().numpy()
    # Plot the body positions using matplotlib with annotations
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Create figure with 3D subplot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of all body positions
    scatter = ax.scatter(body_pos[:, 0], body_pos[:, 1], body_pos[:, 2], 
                        c=range(len(body_pos)), cmap='viridis', s=100, alpha=0.8)

    # Annotate each point with its body name
    for i, name in enumerate(body_name):
        ax.text(body_pos[i, 0], body_pos[i, 1], body_pos[i, 2], 
                name, fontsize=8, ha='center', va='bottom')

    # Connect related bodies with lines to show kinematic chains
    # Base to waist
    ax.plot([body_pos[0, 0], body_pos[1, 0]], 
            [body_pos[0, 1], body_pos[1, 1]], 
            [body_pos[0, 2], body_pos[1, 2]], 'r-', alpha=0.5, linewidth=2)

    # Left leg chain
    left_leg_indices = [1, 2, 3, 4]  # waist to leg_l6
    for i in range(len(left_leg_indices)-1):
        idx1, idx2 = left_leg_indices[i], left_leg_indices[i+1]
        ax.plot([body_pos[idx1, 0], body_pos[idx2, 0]], 
                [body_pos[idx1, 1], body_pos[idx2, 1]], 
                [body_pos[idx1, 2], body_pos[idx2, 2]], 'b-', alpha=0.5, linewidth=2)

    # Right leg chain
    right_leg_indices = [1, 5, 6, 7]  # waist to leg_r6
    for i in range(len(right_leg_indices)-1):
        idx1, idx2 = right_leg_indices[i], right_leg_indices[i+1]
        ax.plot([body_pos[idx1, 0], body_pos[idx2, 0]], 
                [body_pos[idx1, 1], body_pos[idx2, 1]], 
                [body_pos[idx1, 2], body_pos[idx2, 2]], 'g-', alpha=0.5, linewidth=2)

    # Left arm chain
    left_arm_indices = [0, 8, 9, 10]  # base to zarm_l7
    for i in range(len(left_arm_indices)-1):
        idx1, idx2 = left_arm_indices[i], left_arm_indices[i+1]
        ax.plot([body_pos[idx1, 0], body_pos[idx2, 0]], 
                [body_pos[idx1, 1], body_pos[idx2, 1]], 
                [body_pos[idx1, 2], body_pos[idx2, 2]], 'c-', alpha=0.5, linewidth=2)

    # Right arm chain
    right_arm_indices = [0, 11, 12, 13]  # base to zarm_r7
    for i in range(len(right_arm_indices)-1):
        idx1, idx2 = right_arm_indices[i], right_arm_indices[i+1]
        ax.plot([body_pos[idx1, 0], body_pos[idx2, 0]], 
                [body_pos[idx1, 1], body_pos[idx2, 1]], 
                [body_pos[idx1, 2], body_pos[idx2, 2]], 'm-', alpha=0.5, linewidth=2)

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Robot Body Positions with Kinematic Chains')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Body Index')

    # Set equal aspect ratio for better visualization
    max_range = np.max(np.abs(body_pos - np.mean(body_pos, axis=0))) * 1.1
    mid_x, mid_y, mid_z = np.mean(body_pos, axis=0)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

    # Print body positions for verification
    print("\nBody positions in expected order:")
    print("-" * 50)
    for i, name in enumerate(body_name):
        print(f"{i:2d}: {name:20s} -> ({body_pos[i, 0]:6.3f}, {body_pos[i, 1]:6.3f}, {body_pos[i, 2]:6.3f})")
    #endregion Ensure the returned body_* is in the expected order


    # ['base_link', 
    #  'leg_l1_link', 'leg_r1_link', 
    #  'waist_yaw', 
    #  'leg_l2_link', 'leg_r2_link', 
    #  'zarm_l1_link', 'zarm_r1_link', 
    #  'leg_l3_link', 'leg_r3_link', 
    #  'zarm_l2_link', 'zarm_r2_link', 
    #  'leg_l4_link', 'leg_r4_link', 
    #  'zarm_l3_link', 'zarm_r3_link', 
    #  'leg_l5_link', 'leg_r5_link', 
    #  'zarm_l4_link', 'zarm_r4_link', 
    #  'leg_l6_link', 'leg_r6_link', 
    #  'zarm_l5_link', 'zarm_r5_link', 
    #  'zarm_l6_link', 'zarm_r6_link', 
    #  'zarm_l7_link', 'zarm_r7_link'
    # ]

    # [MultiMotionCommand] Motion body names: 
    # ['base_link', 
    #  'waist_yaw', 
    #  'leg_l2_link', 'leg_l4_link', 'leg_l6_link', 
    #  'leg_r2_link', 'leg_r4_link', 'leg_r6_link', 
    #  'zarm_l2_link', 'zarm_l4_link', 'zarm_l7_link', 
    #  'zarm_r2_link', 'zarm_r4_link', 'zarm_r7_link'
    # ]

    # [MultiMotionCommand] Body indexes: 
    # [ 0,  3,  4, 12, 20,  5, 13, 21, 10, 18, 26, 11, 19, 27]
