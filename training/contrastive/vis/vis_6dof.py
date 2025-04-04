import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('.')

from scripts.fake_data.contrastive_data_dataset import HandPoseContrastiveDataset
from scripts.hand_only_supervised.hand_supervised_dataset import LabelledHandDataset

def plot_hand_with_normal(ax, joints, color='b', title=None):
    """Plot hand skeleton and its normal vector."""
    # Define hand connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    face_nodes = [
        [0, 5, 9, 13, 17, 2], # palm
    ]
    dir_nodes = [
        [0, 1, 2, 3, 4, -1], # thumb
        [5, 6, 7, 8, -1], # index
        [9, 10, 11, 12, -1], # middle
        [13, 14, 15, 16, -1], # ring
        [17, 18, 19, 20, -1] # pinky
    ]

    finger_nodes = {4, 8, 12, 16, 20}
    non_finger_nodes = [i for i in range(21) if i not in finger_nodes]

    out_features = list()

    # # Normalize the joints to fit into [-0.2, 0.2]
    # joints_min, joints_max = joints.min(axis=0), joints.max(axis=0)
    # range_non_zero = np.where(joints_max - joints_min != 0, joints_max - joints_min, 1)
    # joints = (joints - joints_min) / range_non_zero  # Scale to [0, 1]
    # joints = joints * 0.4 - 0.2  # Scale to [-0.2, 0.2]

    # # Normalize x and y for non-finger nodes
    # joints_min, joints_max = joints[non_finger_nodes, :2].min(axis=0), joints[non_finger_nodes, :2].max(axis=0)
    # range_non_zero = np.where(joints_max - joints_min != 0, joints_max - joints_min, 1)
    # joints[:, :2] = (joints[:, :2] - joints_min) / range_non_zero  # Scale x, y to [0, 1]
    # joints[:, :2] = joints[:, :2] * 0.4 - 0.2  # Scale x, y to [-0.2, 0.2]

    def arrow(node, vec, scale=0.2, c='r'):
        ax.quiver(joints[node, 0], joints[node, 1], joints[node, 2],
            vec[0], vec[1], vec[2],
            color=c, length=scale, normalize=True,
            arrow_length_ratio=0.2)
        
    # Plot joints and connections
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, marker='o')
    for start, end in connections:
        ax.plot([joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
                [joints[start, 2], joints[end, 2]], 
                c=color)

    # Compute normal vector using PCA
    def compute_normal(nodes):
        cond = joints[nodes[-1]]-joints[nodes[0]]
        nodes = nodes[:-1]
        centered = joints[nodes] - joints[nodes].mean(axis=0)
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]  # Third principal component is normal to main plane
        if np.dot(cond, normal) < 0:
            normal = -normal
        return normal
    
    def compute_dir(nodes):
        normal = compute_normal(nodes)
        return np.cross(joints[nodes[1]] - joints[nodes[0]], normal)
    
    # Plot normal vector at wrist joint (scaled for visibility)
    for nodes in face_nodes:
        normal = compute_normal(nodes)
        out_features.append(normal) # hand pose
        arrow(nodes[0], normal)
    for nodes in dir_nodes:
        ndir = compute_dir(nodes)
        out_features.append(ndir)
        arrow(nodes[0], ndir, c='g')
    
    # Set equal aspect ratio and bounds
    ax.set_box_aspect([1,1,1])
    bound = 0.5
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])
    ax.set_zlim([-bound, bound])
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def main1():
    # Load a single sample from HandPoseContrastiveDataset
    dataset = HandPoseContrastiveDataset(
        num_samples=1,
        npy_file='data/kpts/fake/augmented_gesture_groups_32.npy'
    )
    dataset = LabelledHandDataset(dataset_name='lexset', split='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    if isinstance(dataset, LabelledHandDataset):
        _, joints_base = next(iter(dataloader))
    else:
        joints_base, *_ = next(iter(dataloader))
    
    # Convert to numpy and squeeze batch dimension
    pose = joints_base[0].numpy()
    
    # Create figure and plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_hand_with_normal(ax, pose, title='Hand Pose with Normal Vector')
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()

def main2():
    # Load multiple samples from both datasets
    contrastive_dataset = HandPoseContrastiveDataset(
        num_samples=5,
        npy_file='data/kpts/fake/augmented_gesture_groups_32.npy'
    )
    labelled_dataset = LabelledHandDataset(dataset_name='lexset', split='test')
    
    contrastive_dataloader = DataLoader(contrastive_dataset, batch_size=1, shuffle=True)
    labelled_dataloader = DataLoader(labelled_dataset, batch_size=1, shuffle=True)
    
    # Create figure for plotting
    fig = plt.figure(figsize=(12, 6))
    
    # Plot samples from HandPoseContrastiveDataset
    for i, (joints_base, *_) in enumerate(contrastive_dataloader):
        if i >= 3:  # Display up to 3 samples
            break
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        pose = joints_base[0].numpy()
        plot_hand_with_normal(ax, pose, title=f'Contrastive Sample {i + 1}')
        ax.view_init(elev=30, azim=45)
    
    # Plot samples from LabelledHandDataset
    for i, data in enumerate(labelled_dataloader):
        if i >= 3:  # Display up to 3 samples
            break
        if isinstance(labelled_dataset, LabelledHandDataset):
            _, joints_base = data
        else:
            joints_base, *_ = data
        ax = fig.add_subplot(2, 3, i + 4, projection='3d')
        pose = joints_base[0].numpy()
        plot_hand_with_normal(ax, pose, title=f'Labelled Sample {i + 1}')
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main2()