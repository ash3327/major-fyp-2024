import sys
sys.path.append('.')

import torch
import numpy as np
from matplotlib import pyplot as plt
from scripts.fake_data.contrastive_data_dataset import HandPoseContrastiveDataset
from scripts.hand_only_supervised.hand_supervised_dataset import LabelledHandDataset
from training.contrastive.augments import augment as augment_hand

"""
Acknowledgements: This visualization script is generated with the help of Grok.
"""
# Parameters
num_samples = 1000  # Number of samples to generate or load
vis_batch_size = 3  # Number of samples to visualize from each dataset

# Mediapipe hand gesture connections for drawing the hand skeleton
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Function to plot a 3D hand pose
def plot_3d_hand(ax, joints, color='b', label=None):
    """
    Plot a single 3D hand pose on the given axis.
    
    Args:
        ax: Matplotlib 3D axis object.
        joints: Numpy array [21, 3] of joint positions (x, y, z coordinates).
        color: Color for the plot (e.g., 'b' for blue).
        label: Label for the legend.
    """
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, marker='o', label=label)
    for connection in connections:
        start, end = connection
        ax.plot(
            [joints[start, 0], joints[end, 0]],
            [joints[start, 1], joints[end, 1]],
            [joints[start, 2], joints[end, 2]],
            color=color
        )

# Function to compute statistics
def compute_statistics(dataset, dataset_name):
    """
    Compute min, max, mean, and std of joint positions in the dataset.
    
    Args:
        dataset: Dataset object (either HandPoseContrastiveDataset or LabelledHandDataset).
        dataset_name: Name of the dataset for printing.
    """
    if isinstance(dataset, HandPoseContrastiveDataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
        all_joints = []
        for joints_base, joints_aug in dataloader:
            all_joints.append(joints_base.numpy())
            all_joints.append(joints_aug.numpy())
        all_joints = np.concatenate(all_joints, axis=0)
    elif isinstance(dataset, LabelledHandDataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
        all_joints = []
        for _, joints in dataloader:
            all_joints.append(joints.numpy())
        all_joints = np.concatenate(all_joints, axis=0)
    else:
        raise ValueError("Unknown dataset type")

    # Flatten to [N, 3] where N is total number of joints across samples
    all_joints = all_joints.reshape(-1, 3)
    
    stats = {
        'min': np.min(all_joints, axis=0),
        'max': np.max(all_joints, axis=0),
        'mean': np.mean(all_joints, axis=0),
        'std': np.std(all_joints, axis=0)
    }
    
    print(f"Statistics for {dataset_name}:")
    print(f"  Min (x, y, z): {stats['min']}")
    print(f"  Max (x, y, z): {stats['max']}")
    print(f"  Mean (x, y, z): {stats['mean']}")
    print(f"  Std (x, y, z): {stats['std']}")
    print()

# Function to visualize both datasets side by side in one figure
def visualize_both_datasets_side_by_side(dataset_contrastive, dataset_labelled, batch_size=3):
    """
    Visualize samples from both datasets side by side within the same figure.
    
    Args:
        dataset_contrastive: HandPoseContrastiveDataset instance.
        dataset_labelled: LabelledHandDataset instance.
        batch_size: Number of samples to visualize from each dataset.
    """
    # Load batches from both datasets
    dataloader_contrastive = torch.utils.data.DataLoader(dataset_contrastive, batch_size=batch_size, shuffle=True)
    dataloader_labelled = torch.utils.data.DataLoader(dataset_labelled, batch_size=batch_size, shuffle=True)
    
    joints_base, joints_aug = next(iter(dataloader_contrastive))
    labels, joints_labelled = next(iter(dataloader_labelled))
    
    joints_base = joints_base.numpy()
    joints_aug = joints_aug.numpy()
    joints_labelled = joints_labelled.numpy()
    labels = labels.numpy()
    
    # Create a single figure with 3 columns: Anchor, Augmented Positive, Labelled
    fig = plt.figure(figsize=(9, 3 * batch_size))
    
    for i in range(batch_size):
        # Anchor (Contrastive Dataset)
        ax1 = fig.add_subplot(batch_size, 3, 3 * i + 1, projection='3d')
        plot_3d_hand(ax1, joints_base[i], color='b', label='Anchor')
        ax1.set_title(f'Contrastive Anchor {i+1}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # Augmented Positive (Contrastive Dataset)
        ax2 = fig.add_subplot(batch_size, 3, 3 * i + 2, projection='3d')
        plot_3d_hand(ax2, joints_aug[i], color='r', label='Augmented Positive')
        ax2.set_title(f'Contrastive Augmented {i+1}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        # Labelled Sample (Labelled Dataset)
        ax3 = fig.add_subplot(batch_size, 3, 3 * i + 3, projection='3d')
        plot_3d_hand(ax3, joints_labelled[i], color='g', label=f'Label: {labels[i]}')
        ax3.set_title(f'Labelled Sample {i+1}')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
    
    plt.suptitle('Comparison of HandPoseContrastiveDataset and LabelledHandDataset')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # Initialize datasets
    augment = augment_hand
    dataset_contrastive = HandPoseContrastiveDataset(num_samples=num_samples, augment=augment)
    # dataset_labelled = LabelledHandDataset(dataset_name='lexset', split='train')
    # dataset_labelled = LabelledHandDataset(dataset_name='senz3d')
    dataset_labelled = LabelledHandDataset(dataset_name='handshape', split='test', augment=augment)
    
    # Compute and print statistics
    compute_statistics(dataset_contrastive, "HandPoseContrastiveDataset")
    compute_statistics(dataset_labelled, "LabelledHandDataset")
    
    # Visualize samples side by side in one figure
    print("Visualizing both datasets side by side:")
    visualize_both_datasets_side_by_side(dataset_contrastive, dataset_labelled, batch_size=vis_batch_size)