"""
python scripts/fake_data/prepare_fake_data_npy.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mano
from tqdm import tqdm
import os

# Load the MANO model
model_path = 'model/mano'
n_comps = 45
rh_model = mano.load(model_path=model_path,
                     is_rhand=True,
                     num_pca_comps=n_comps,
                     batch_size=1,
                     flat_hand_mean=False)

# Keypoint mapping (same as in prepare_contrastive_data.py)
mmap = [
    16, 15, 14, 13, 0,  # thumb
    1, 4, 10, 7,        # index
    2, 5, 11, 8,        # middle
    3, 6, 12, 9,        # ring
    17, 18, 19, 20      # pinky
]

# Variance parameters (adjusted for fixed orientation)
variance_base = dict(
    betas=0.5,  # Base shape variance
    pose=0.5    # Base pose variance
)
variance_aug = dict(
    betas=0.1,  # Small shape changes
    pose=0.1    # Small pose changes
)

def generate_fixed_orientation_pair(vpow=1):
    """Generate a pair of hand poses with fixed orientation."""
    # Base parameters
    betas_base = torch.rand(1, 10) * variance_base['betas']
    pose_base = (torch.rand(1, n_comps) - 0.5) * 2
    pose_base = (torch.rand(1, 1)**vpow * variance_base['pose']) * pose_base
    
    # Fixed orientation and translation
    global_orient = torch.zeros(1, 3)  # Fixed orientation
    transl = torch.zeros(1, 3)
    
    # Augmented parameters (small perturbations)
    betas_aug = betas_base + torch.randn(1, 10) * variance_aug['betas']
    pose_aug = pose_base + torch.randn(1, n_comps) * variance_aug['pose']
    
    # Generate base pose
    output_base = rh_model(
        betas=betas_base,
        global_orient=global_orient,
        hand_pose=pose_base,
        transl=transl,
        return_verts=True,
        return_tips=True
    )
    
    # Generate augmented pose
    output_aug = rh_model(
        betas=betas_aug,
        global_orient=global_orient,
        hand_pose=pose_aug,
        transl=transl,
        return_verts=True,
        return_tips=True
    )
    
    # Process joints
    joints_base = output_base.joints[0].numpy()[mmap]
    joints_aug = output_aug.joints[0].numpy()[mmap]
    
    # Swap x and y, center at wrist
    joints_base = joints_base[:, [1, 0, 2]] - joints_base[0:1, [1, 0, 2]]
    joints_aug = joints_aug[:, [1, 0, 2]] - joints_aug[0:1, [1, 0, 2]]
    
    # Scale to match the range of real datasets
    joints_base *= 2.5
    joints_aug *= 2.5
    
    return joints_base, joints_aug

def generate_and_save_pairs(num_pairs=100000, output_file='data/kpts/fixed_orientation_pairs.npy'):
    """Generate and save pairs of hand poses."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    pairs = []
    for _ in tqdm(range(num_pairs), desc="Generating hand pairs"):
        base, aug = generate_fixed_orientation_pair()
        pairs.append(np.stack([base, aug]))
    
    pairs = np.stack(pairs)  # Shape: [N, 2, 21, 3]
    np.save(output_file, pairs)
    print(f"Saved {num_pairs} pairs to {output_file}")
    return pairs

def visualize_sample_pairs(pairs, num_samples=5):
    """Visualize sample pairs of hand poses with skeleton connections."""
    fig = plt.figure(figsize=(20, 4*num_samples))
    
    for i in range(num_samples):
        # Plot base pose
        ax1 = fig.add_subplot(num_samples, 2, i*2 + 1, projection='3d')
        plot_3d_hand(ax1, pairs[i, 0], color='b', label='Base')
        ax1.set_title(f'Sample {i+1} - Base')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # Plot augmented pose
        ax2 = fig.add_subplot(num_samples, 2, i*2 + 2, projection='3d')
        plot_3d_hand(ax2, pairs[i, 1], color='r', label='Augmented')
        ax2.set_title(f'Sample {i+1} - Augmented')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Function to plot a 3D hand pose
def plot_3d_hand(ax, joints, color='b', label=None):
    """
    Plot a single 3D hand pose on the given axis with skeleton connections.
    
    Args:
        ax: Matplotlib 3D axis object.
        joints: Numpy array [21, 3] of joint positions (x, y, z coordinates).
        color: Color for the plot (e.g., 'b' for blue).
        label: Label for the legend.
    """
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, marker='o', label=label)
    for connection in [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]:
        start, end = connection
        ax.plot(
            [joints[start, 0], joints[end, 0]],
            [joints[start, 1], joints[end, 1]],
            [joints[start, 2], joints[end, 2]],
            color=color
        )

if __name__ == "__main__":
    # Generate and save pairs
    output_file = 'data/kpts/fake/fixed_orientation_pairs.npy'
    pairs = generate_and_save_pairs(num_pairs=10000000, output_file=output_file)
    
    # Visualize some samples
    sample_pairs = pairs[:5]  # Take first 5 pairs
    # visualize_sample_pairs(sample_pairs)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Shape: {pairs.shape}")
    print(f"Min values (x,y,z): {pairs.min(axis=(0,1,2))}")
    print(f"Max values (x,y,z): {pairs.max(axis=(0,1,2))}")
    print(f"Mean values (x,y,z): {pairs.mean(axis=(0,1,2))}")
    print(f"Std values (x,y,z): {pairs.std(axis=(0,1,2))}")