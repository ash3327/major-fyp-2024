import torch
import torch.nn.functional as F
import math
import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation

# Define hand topology
fingers = np.array(
    [
        [0, 1,  2,  3,  4 ],    # Thumb
        [0, 5,  6,  7,  8 ],    # Index
        [0, 9,  10, 11, 12],    # Middle
        [0, 13, 14, 15, 16],    # Ring
        [0, 17, 18, 19, 20]     # Pinky
    ]
)
wrist_face = [0, 5, 9, 13, 17]  # Palm face
palm_nodes = {0,1,5,9,13,17}

def compute_face_normal_batch(joints, nodes):
    """Compute normal vector for a batch of faces."""
    points = joints[:, nodes]
    v1 = points[:, 1] - points[:, 0]
    v2 = points[:, -1] - points[:, 0]
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)

def compute_edge_direction_batch(joints, start_nodes, end_nodes):
    """Compute direction vector for a batch of edges."""
    return joints[:, end_nodes] - joints[:, start_nodes]

def get_rotation_matrix_batch(forward, normal):
    """Get rotation matrix from forward and normal vectors for a batch."""
    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
    normal = normal - np.sum(normal * forward, axis=-1, keepdims=True) * forward
    x_axis = forward
    z_axis = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis, axis=-1, keepdims=True)
    return np.stack([x_axis, y_axis, z_axis], axis=1)

def extract_orientations(joints, batched=False):
    """
    Extract relative orientations from joints.
    Input: joints [B, 21, 3] or [21, 3] as numpy array or torch tensor
    Output: relative quaternions [B, 21, 4] or [21, 4] as torch tensor
    """
    if isinstance(joints, torch.Tensor):
        joints_np = joints.cpu().numpy()
    else:
        joints_np = joints

    if joints_np.ndim == 2 and joints_np.shape == (21, 3):
        joints_np = np.expand_dims(joints_np, axis=0)
        batched = True
        single_input = True
    elif joints_np.ndim == 3 and joints_np.shape[1:] == (21, 3):
        batched = True
        single_input = False
    else:
        raise ValueError(f"Joints should be of shape [B, 21, 3] or [21, 3] instead of {joints_np.shape}")

    batch_size = joints_np.shape[0]
    orientations_batch = np.zeros((batch_size, 21, 4))

    # Compute wrist orientation for the batch
    wrist_normal_batch = compute_face_normal_batch(joints_np, wrist_face)
    wrist_direction_batch = compute_edge_direction_batch(joints_np, 0, 9)
    wrist_direction_batch = wrist_direction_batch / np.linalg.norm(wrist_direction_batch, axis=-1, keepdims=True)

    wrist_normal_check_direction_batch = compute_edge_direction_batch(joints_np, 0, 2)
    dot_product_wrist = np.sum(wrist_normal_batch * wrist_normal_check_direction_batch, axis=-1)
    wrist_normal_batch[dot_product_wrist < 0] *= -1

    R_wrist_batch = get_rotation_matrix_batch(wrist_direction_batch, wrist_normal_batch)
    try:
        orientations_batch[:, 0] = Rotation.from_matrix(R_wrist_batch).as_quat()
    except np.linalg.LinAlgError as e:
        print(joints_np)
        raise ValueError(f"Invalid rotation matrix for wrist orientation: {e}")

    # Compute finger orientations for the batch
    for finger in fingers:
        for j in range(len(finger) - 1):
            prev_node, base_node = finger[j], finger[j + 1]
            face_nodes = finger[max(j - 2, 0):min(j + 3, len(finger))]

            normal_batch = compute_face_normal_batch(joints_np, face_nodes)
            direction_batch = compute_edge_direction_batch(joints_np, prev_node, base_node)
            direction_batch = direction_batch / np.linalg.norm(direction_batch, axis=-1, keepdims=True)
            normal_batch = np.cross(normal_batch, direction_batch)
            normal_batch = normal_batch / np.linalg.norm(normal_batch, axis=-1, keepdims=True)

            wrist_normal_at_batch = wrist_normal_batch  # Use the pre-computed wrist normal

            if prev_node in palm_nodes:
                wrist_normal_dot = np.sum(normal_batch * wrist_normal_at_batch, axis=-1)
                normal_batch[wrist_normal_dot < 0] *= -1
            else:
                direction_to_wrist_batch = compute_edge_direction_batch(joints_np, base_node, 0)
                direction_to_wrist_batch = direction_to_wrist_batch / np.linalg.norm(direction_to_wrist_batch, axis=-1, keepdims=True)
                normal_dot_to_wrist = np.sum(normal_batch * direction_to_wrist_batch, axis=-1)
                normal_batch[normal_dot_to_wrist < 0] *= -1

            R_batch = get_rotation_matrix_batch(direction_batch, normal_batch)
            try:
                orientations_batch[:, base_node] = Rotation.from_matrix(R_batch).as_quat()
            except np.linalg.LinAlgError as e:
                print(joints_np)
                raise ValueError(f"Invalid rotation matrix for node {base_node}: {e}")

    # Convert to relative orientations (batched)
    relative_orientations_batch = np.zeros_like(orientations_batch)
    relative_orientations_batch[:, 0] = orientations_batch[:, 0]

    for finger in fingers:
        for j in range(1, len(finger)):
            prev_idx, curr_idx = finger[j - 1], finger[j]
            parent_rotations = Rotation.from_quat(orientations_batch[:, prev_idx])
            current_rotations = Rotation.from_quat(orientations_batch[:, curr_idx])
            relative_rotations = parent_rotations.inv() * current_rotations
            relative_orientations_batch[:, curr_idx] = relative_rotations.as_quat()

    if hasattr(single_input, 'value') and single_input:
        return torch.from_numpy(relative_orientations_batch).float().squeeze(0)
    else:
        return torch.from_numpy(relative_orientations_batch).float()

def get_6dof(joints, batched=True):
    """
    Get 6D representation of hand joints.
    Input: joints [B, 21, 3] or [21, 3] as numpy array or torch tensor
    Output: 6D representation [B, 21, 6] or [21, 6] as torch tensor
    """
    if isinstance(joints, torch.Tensor):
        joints_np = joints.cpu().numpy()
    else:
        joints_np = joints

    orientations = extract_orientations(joints, batched=batched)

    if orientations.ndim == 3:
        return torch.cat([torch.from_numpy(joints_np).float(), orientations[:, :, :3]], dim=-1)
    else:
        return torch.cat([torch.from_numpy(joints_np).float(), orientations[:, :3]], dim=-1)

def get_3dof(joints, batched=True):
    feats = extract_orientations(joints, batched=batched)
    if feats.ndim == 3:
        return feats[:, :, :3]
    return feats[:,:3]