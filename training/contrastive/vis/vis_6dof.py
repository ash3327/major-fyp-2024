import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation
import sys
sys.path.append('.')

from scripts.fake_data.contrastive_data_dataset import HandPoseContrastiveDataset
from scripts.hand_only_supervised.hand_supervised_dataset import LabelledHandDataset
from training.contrastive.augments import normalize
from training.contrastive.preprocess import extract_orientations

DEBUG_ORIENTATIONS = False  # Global debug flag

# Define hand connections and important nodes
fingers = np.array(
    [
        [0, 1,  2,  3,  4 ],    # Thumb
        [0, 5,  6,  7,  8 ],    # Index
        [0, 9,  10, 11, 12],    # Middle
        [0, 13, 14, 15, 16],    # Ring
        [0, 17, 18, 19, 20]     # Pinky
    ]
)
connections = [
    (finger[i],finger[i+1]) for finger in fingers for i in range(len(finger)-1)
]
palm_nodes = {0,1,5,9,13,17}

# Define faces for orientation
wrist_face = [0, 5, 9, 13, 17]  # Palm face
finger_faces = dict()
for finger in fingers:
    for j in range(len(finger)-1):
        finger_faces[(finger[j],finger[j+1])] = finger[max(j-2, 0):min(j+3, len(finger))]
        
def plot_rectangle(ax, center, normal, direction, width=0.2, height=0.4, color='b', alpha=0.5):
    """Plot a rectangle with given center, normal and direction."""
    # Calculate the corners of the rectangle
    right = direction / np.linalg.norm(direction)
    up = np.cross(right, normal)
    up = up / np.linalg.norm(up)
    
    # Calculate the four corners
    corners = np.array([
        center - (width/2)*up - (height/2)*right,
        center - (width/2)*up + (height/2)*right,
        center + (width/2)*up + (height/2)*right,
        center + (width/2)*up - (height/2)*right
    ])
    
    # Plot the rectangle
    rect = np.vstack([corners, corners[0]])  # Close the rectangle
    ax.plot3D(rect[:, 0], rect[:, 1], rect[:, 2], color=color, alpha=alpha)
    # ax.fill(rect[:, 0], rect[:, 1], rect[:, 2], color=color, alpha=alpha*0.3)

def get_relative_orientations(absolute_orientations):
    """Convert absolute orientations to relative orientations."""
    relative_orientations = np.zeros_like(absolute_orientations)
    relative_orientations[0] = absolute_orientations[0]  # Keep wrist orientation
    
    # For each joint after wrist
    for i in range(1, len(absolute_orientations)):
        # Find parent orientation
        parent_idx = 0  # Default to wrist
        for finger in fingers:
            if i in finger[1:]:  # If joint is in a finger (not the first joint)
                parent_idx = finger[finger.tolist().index(i) - 1]
                break
        
        # Convert to Rotation objects and compute relative rotation
        r_parent = Rotation.from_quat(absolute_orientations[parent_idx])
        r_current = Rotation.from_quat(absolute_orientations[i])
        r_relative = r_parent.inv() * r_current
        relative_orientations[i] = r_relative.as_quat()
    
    return relative_orientations

def verify_quaternion(ax, joint_pos, quat, scale=0.3):
    """Visualize local coordinate system from quaternion."""
    rot = Rotation.from_quat(quat)
    axes = np.eye(3)  # Identity matrix for x,y,z axes
    rotated_axes = rot.apply(axes)
    
    colors = ['r', 'g', 'b']  # x=red, y=green, z=blue
    for i, (axis, color) in enumerate(zip(rotated_axes.T, colors)):
        ax.quiver(joint_pos[0], joint_pos[1], joint_pos[2],
                 axis[0], axis[1], axis[2],
                 color=color, length=scale, normalize=True)

def plot_hand_with_normal(ax, joints, color='b', title=None, show_arrows=False):
    """Plot hand skeleton and its normal vector and return orientations."""

    def compute_face_normal(nodes):
        """Compute normal vector for a face."""
        points = joints[nodes]
        v1 = points[1] - points[0]
        v2 = points[-1] - points[0]
        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)

    def compute_edge_direction(start, end):
        """Compute direction vector for an edge."""
        return joints[end] - joints[start]
    
    # joints[:, 0] *= np.random.choice([-1, 1])

    joints = normalize(joints).numpy()

    # Plot base skeleton
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, marker='o')
    for start, end in connections:
        ax.plot([joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
                [joints[start, 2], joints[end, 2]], c=color)

    # Initialize orientations array
    orientations = np.zeros((21, 4))  # [21, 4] for quaternions

    # Compute wrist orientation
    wrist_normal = compute_face_normal(wrist_face)
    wrist_direction = compute_edge_direction(0, 9)  # Using middle finger direction
    if np.dot(wrist_normal, compute_edge_direction(0, 2)) < 0: # ensures normal points in the direction of palm
        wrist_normal = -wrist_normal

    # Create rotation matrix for wrist
    def get_rotation(forward, normal): # forward must be correct
        # Adjust normal to be in the plane formed by normal and direction
        forward = forward / np.linalg.norm(forward)
        normal = normal - np.dot(normal, forward) * forward
        # rotation matrix
        x_axis = forward
        z_axis = normal / np.linalg.norm(normal)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        return np.row_stack([x_axis, y_axis, z_axis])
    
    R_wrist = get_rotation(wrist_direction, wrist_normal)
    orientations[0] = Rotation.from_matrix(R_wrist).as_quat()
    print(R_wrist)

    # Plot wrist orientation
    plot_rectangle(ax, joints[0], wrist_normal, wrist_direction, width=0.2, height=0.8, color='r')
    if show_arrows:
        ax.quiver(joints[0, 0], joints[0, 1], joints[0, 2],
                wrist_normal[0], wrist_normal[1], wrist_normal[2],
                color='r', length=0.5, normalize=True)

    # Plot finger orientations
    for (prev_node, base_node), face_nodes in finger_faces.items():
        normal = compute_face_normal(face_nodes)
        direction = compute_edge_direction(prev_node, base_node)
        normal = np.cross(normal, direction)
        if prev_node in palm_nodes:
            if np.dot(normal, wrist_normal) < 0: # normal points towards wrist
                normal = -normal
        else:
            if np.dot(normal, compute_edge_direction(base_node, 0)) < 0: # normal points towards wrist
                normal = -normal

        # Create rotation matrix
        R = get_rotation(direction, normal)
        orientations[base_node] = Rotation.from_matrix(R).as_quat()

        plot_rectangle(ax, joints[base_node], normal, direction, color='g')
        if show_arrows:
            ax.quiver(joints[base_node, 0], joints[base_node, 1], joints[base_node, 2],
                    normal[0], normal[1], normal[2],
                    color='g', length=0.5, normalize=True)

    # # Debug visualization of quaternions
    # if DEBUG_ORIENTATIONS:
    #     # Verify wrist orientation
    #     da_orientations = dict()
    #     da_orientations[0] = orientations[0]
    #     verify_quaternion(ax, joints[0], orientations[0])
        
    #     # Verify finger joint orientations
    #     for (prev_node, base_node), _ in finger_faces.items():
    #         da_orientations[base_node] = (Rotation.from_quat(orientations[base_node]) * 
    #                                       Rotation.from_quat(da_orientations[prev_node])).as_quat()
    #         verify_quaternion(ax, joints[base_node], orientations[base_node])

    # Set visualization parameters
    ax.set_box_aspect([1,1,1])
    if title:
        ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Convert absolute orientations to relative orientations
    relative_orientations = get_relative_orientations(orientations)
    relative_orientations = extract_orientations(np.array([joints,joints]), batched=True)[0]

    # Debug visualization of quaternions
    if DEBUG_ORIENTATIONS:
        # Verify wrist orientation
        da_orientations = dict()
        da_orientations[0] = relative_orientations[0]
        verify_quaternion(ax, joints[0], da_orientations[0])
        
        # Verify finger joint orientations
        for (prev_node, base_node), _ in finger_faces.items():
            da_orientations[base_node] = (Rotation.from_quat(da_orientations[prev_node]) * 
                                          Rotation.from_quat(relative_orientations[base_node]) 
                                          ).as_quat()
            verify_quaternion(ax, joints[base_node], da_orientations[base_node])
    
    # print(relative_orientations)
    return relative_orientations

def main1():
    global DEBUG_ORIENTATIONS
    DEBUG_ORIENTATIONS = True  # Enable debug visualization
    # Load a single sample from HandPoseContrastiveDataset
    dataset = HandPoseContrastiveDataset(
        num_samples=1,
        npy_file='data/kpts/fake/augmented_gesture_groups_32.npy'
    )
    # dataset = LabelledHandDataset(dataset_name='lexset', split='test')
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
    global DEBUG_ORIENTATIONS
    DEBUG_ORIENTATIONS = True  # Enable debug visualization
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
    main1()