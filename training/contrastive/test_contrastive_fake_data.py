"""
This file should be executed at root of the project.
"""

import sys
sys.path.append('.')

import torch
from matplotlib import pyplot as plt
from scripts.fake_data.contrastive_data_dataset import HandPoseContrastiveDataset

batch_size = 256
num_samples = 10000
num_epochs = 100

# Mediapipe hand gesture connections for drawing the hand skeleton
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def plot_3d_hand(ax, joints, color='b', label=None):
    """
    Plot a single 3D hand pose on the given axis.

    Args:
        ax: Matplotlib 3D axis object.
        joints: Numpy array [21, 3] of joint positions (x, y, z coordinates for 21 hand joints).
        color: Color for the hand plot (e.g., 'b' for blue, 'r' for red).
        label: Label for the legend (e.g., 'Anchor' or 'Augmented Positive').
    """
    # Plot the 21 joints as scatter points
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, marker='o', label=label)
    
    # Draw lines between joints to form the hand skeleton
    for connection in connections:
        start, end = connection
        ax.plot(
            [joints[start, 0], joints[end, 0]],
            [joints[start, 1], joints[end, 1]],
            [joints[start, 2], joints[end, 2]],
            color=color
        )

def visualize_batch(dataset, batch_size=5):
    """
    Visualize a batch of anchor and augmented positive hand poses side by side.

    Args:
        dataset: Instance of HandPoseContrastiveDataset.
        batch_size: Number of hand pose pairs to visualize (default is 5).
    """
    # Create a DataLoader to fetch a batch of data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    joints_base, joints_aug = next(iter(dataloader))  # Get one batch of anchor and augmented poses
    
    # Convert PyTorch tensors to NumPy arrays for plotting
    joints_base = joints_base.numpy()  # Shape: [batch_size, 21, 3]
    joints_aug = joints_aug.numpy()    # Shape: [batch_size, 21, 3]
    
    # Set up the figure with subplots (each row has an anchor and its augmented positive)
    fig = plt.figure(figsize=(6, 3 * batch_size))
    
    # Loop through the batch and create side-by-side 3D plots
    for i in range(batch_size):
        # Anchor plot (left subplot)
        ax1 = fig.add_subplot(batch_size, 2, 2 * i + 1, projection='3d')
        plot_3d_hand(ax1, joints_base[i], color='b', label='Anchor')
        ax1.set_title(f'Anchor {i+1}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # Augmented positive plot (right subplot)
        ax2 = fig.add_subplot(batch_size, 2, 2 * i + 2, projection='3d')
        plot_3d_hand(ax2, joints_aug[i], color='r', label='Augmented Positive')
        ax2.set_title(f'Augmented Positive {i+1}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
    
    # Adjust layout to prevent overlap and display the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters from your query
    num_samples = 1000
    vis_batch_size = 3  # Number of pairs to visualize (adjust as needed)
    
    # Initialize the dataset (assuming rh_model, variance_diff, variance_sim are defined in your module)
    dataset = HandPoseContrastiveDataset(num_samples=num_samples)
    
    # Visualize the batch
    visualize_batch(dataset, batch_size=vis_batch_size)