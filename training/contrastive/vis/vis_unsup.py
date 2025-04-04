import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader

from scripts.fake_data.contrastive_data_dataset import HandPoseContrastiveDataset
from training.contrastive.train_supcon import structured_collate_fn

def plot_hand_skeleton(ax, joints, color='b', label=None):
    """Plot hand skeleton with connections between joints."""
    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, marker='o', label=label)
    
    # Define connections between joints
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    
    # Plot connections
    for start, end in connections:
        ax.plot([joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
                [joints[start, 2], joints[end, 2]], 
                c=color)
        
def visualize_structured_grid(structured_batch, grid_size=32, num_samples=4):
    """
    Visualize samples from the structured grid batch.
    
    Args:
        structured_batch: Tensor of shape [grid_size, grid_size, 21, 3]
        grid_size: Size of the grid (default: 32)
        num_samples: Number of samples to visualize (default: 4)
    """
    fig = plt.figure(figsize=(20, 5*num_samples))
    
    cols = np.random.randint(0, grid_size, 4)  # Select 4 random transformations

    for i in range(num_samples):
        # Select random row and column indices
        row = np.random.randint(0, grid_size)
        
        for j, col in enumerate(cols):
            ax = fig.add_subplot(num_samples, 4, i*4 + j + 1, projection='3d')
            pose = structured_batch[row, col].cpu().numpy()
            plot_hand_skeleton(ax, pose)
            
            ax.set_title(f'Row {row}, Col {col}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=30, azim=45)  # Set viewing angle
    
    plt.tight_layout()
    return fig

def visualize_row_variations(structured_batch, row_idx=None, num_cols=8):
    """
    Visualize variations of the same base pose across a row.
    
    Args:
        structured_batch: Tensor of shape [grid_size, grid_size, 21, 3]
        row_idx: Row index to visualize (if None, randomly selected)
        num_cols: Number of columns to visualize (default: 8)
    """
    grid_size = structured_batch.shape[0]
    if row_idx is None:
        row_idx = np.random.randint(0, grid_size)
        
    fig = plt.figure(figsize=(20, 5))
    
    for j in range(num_cols):
        ax = fig.add_subplot(1, num_cols, j + 1, projection='3d')
        pose = structured_batch[row_idx, j].cpu().numpy()
        plot_hand_skeleton(ax, pose)
        
        ax.set_title(f'Transform {j}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=45)
    
    plt.suptitle(f'Variations of Base Pose (Row {row_idx})')
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # Configuration
    grid_size = 32
    n_aug_pregenerated = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = HandPoseContrastiveDataset(
        npy_file=f'data/kpts/fake/augmented_gesture_groups_{n_aug_pregenerated}.npy'
    )
    
    # Create dataloader with structured_collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=grid_size,
        shuffle=True,
        collate_fn=structured_collate_fn,
        drop_last=True
    )
    
    # Get one batch
    structured_batch = next(iter(dataloader))
    print(structured_batch.shape)
    
    # Visualize random samples from the grid
    fig1 = visualize_structured_grid(structured_batch)
    plt.show()
    # plt.savefig('grid_samples.png')
    # plt.close()
    
    # Visualize variations of the same base pose
    fig2 = visualize_row_variations(structured_batch)
    plt.show()
    # plt.savefig('row_variations.png')
    # plt.close()
    
    # Print statistics about the structured batch
    print("\nStructured Batch Statistics:")
    print(f"Shape: {structured_batch.shape}")
    print(f"Min value: {structured_batch.min().item():.4f}")
    print(f"Max value: {structured_batch.max().item():.4f}")
    print(f"Mean value: {structured_batch.mean().item():.4f}")
    print(f"Std value: {structured_batch.std().item():.4f}")