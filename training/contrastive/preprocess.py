import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

sys.path.append('.')

# hand topology
from training.contrastive.topology import fingers, wrist_face, palm_nodes

def normalize(v):
    vlen = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(vlen != 0, v / vlen, 0)

def compute_face_normal_batch(joints, nodes):
    """Compute normal vector for a batch of faces."""
    points = joints[:, nodes]
    v1 = points[:, 1] - points[:, 0]
    v2 = points[:, -1] - points[:, 0]
    normal = np.cross(v1, v2)
    return normalize(normal)

def compute_edge_direction_batch(joints, start_nodes, end_nodes):
    """Compute direction vector for a batch of edges."""
    return joints[:, end_nodes] - joints[:, start_nodes]

def get_rotation_matrix_batch(forward, normal):
    """Get rotation matrix from forward and normal vectors for a batch."""
    forward = normalize(forward)
    normal = normal - np.sum(normal * forward, axis=-1, keepdims=True) * forward
    x_axis = forward
    z_axis = normalize(normal)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = normalize(y_axis)
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
    orientations_batch = np.zeros((batch_size, 21, 4)) # [B,21,4]

    # Compute wrist orientation for the batch
    wrist_normal_batch = compute_face_normal_batch(joints_np, wrist_face)
    wrist_direction_batch = compute_edge_direction_batch(joints_np, 0, 9)
    wrist_direction_batch = normalize(wrist_direction_batch)

    wrist_normal_check_direction_batch = compute_edge_direction_batch(joints_np, 0, 2)
    dot_product_wrist = np.sum(wrist_normal_batch * wrist_normal_check_direction_batch, axis=-1)
    wrist_normal_batch[dot_product_wrist < 0] *= -1

    R_wrist_batch = get_rotation_matrix_batch(wrist_direction_batch, wrist_normal_batch)
    try:
        null_entries = np.linalg.det(R_wrist_batch) == 0 # [B]
        R_wrist_batch[null_entries] = np.eye(3)[np.newaxis]
        orientations_batch[~null_entries, 0] = Rotation.from_matrix(R_wrist_batch[~null_entries]).as_quat() # [B, 4]
        orientations_batch[null_entries,0,:] = np.zeros((1,4)) # [B, 4]
        if np.any(null_entries):
            print("!!!!WRIST ", wrist_direction_batch[null_entries], wrist_normal_batch[null_entries])
    except np.linalg.LinAlgError as e:
        print(joints_np, R_wrist_batch, np.linalg.det(R_wrist_batch))
        raise ValueError(f"Invalid rotation matrix for wrist orientation: {e}")

    # Compute finger orientations for the batch
    for finger in fingers:
        for j in range(len(finger) - 1):
            prev_node, base_node = finger[j], finger[j + 1]
            face_nodes = finger[max(j - 2, 0):min(j + 3, len(finger))]

            normal_batch = compute_face_normal_batch(joints_np, face_nodes)
            direction_batch = compute_edge_direction_batch(joints_np, prev_node, base_node)
            direction_batch = normalize(direction_batch)
            normal_batch = np.cross(normal_batch, direction_batch)
            normal_batch = normalize(normal_batch)

            wrist_normal_at_batch = wrist_normal_batch  # Use the pre-computed wrist normal

            if prev_node in palm_nodes:
                wrist_normal_dot = np.sum(normal_batch * wrist_normal_at_batch, axis=-1)
                normal_batch[wrist_normal_dot < 0] *= -1
            else:
                direction_to_wrist_batch = compute_edge_direction_batch(joints_np, base_node, 0)
                direction_to_wrist_batch = normalize(direction_to_wrist_batch)
                normal_dot_to_wrist = np.sum(normal_batch * direction_to_wrist_batch, axis=-1)
                normal_batch[normal_dot_to_wrist < 0] *= -1

            R_batch = get_rotation_matrix_batch(direction_batch, normal_batch) # [B, 3, 3]
            try:
                null_entries = np.linalg.det(R_wrist_batch) == 0
                R_batch[null_entries] = np.eye(3)[np.newaxis]
                orientations_batch[~null_entries, base_node] = Rotation.from_matrix(R_batch[~null_entries]).as_quat() # [B, 4]
                orientations_batch[null_entries,base_node,:] = np.zeros((1,4)) # [B, 4]
                if np.any(null_entries):
                    print("!!!!CNET ", direction_batch[null_entries,base_node], normal_batch[null_entries,base_node], base_node)
            except np.linalg.LinAlgError as e:
                print(joints_np, R_wrist_batch, np.linalg.det(R_wrist_batch))
                raise ValueError(f"Invalid rotation matrix for node {base_node}: {e}")

    # Convert to relative orientations (batched)
    relative_orientations_batch = np.zeros_like(orientations_batch) # [B,21,4]
    relative_orientations_batch[:, 0] = orientations_batch[:, 0]

    le_norms = np.linalg.norm(orientations_batch, axis=-1)
    le_null_entries = le_norms == 0 # [B, 21]

    batch_size = orientations_batch.shape[0]
    for finger in fingers:
        for j in range(1, len(finger)):
            prev_idx, curr_idx = finger[j - 1], finger[j]
            for i in range(batch_size):
                try:
                    null_entry = np.logical_or(le_null_entries[i, prev_idx], le_null_entries[i, curr_idx]) # [1]
                    if not null_entry:
                        parent_rotation = Rotation.from_quat(orientations_batch[i, prev_idx])
                        current_rotation = Rotation.from_quat(orientations_batch[i, curr_idx])
                        relative_rotation = parent_rotation.inv() * current_rotation
                        relative_orientations_batch[i, curr_idx] = relative_rotation.as_quat()
                    elif np.any(null_entry):
                        print(f'WARNING: ENCOUNTERED ZERO NORM QUATERNION AT (entry {i}) {null_entry}: {orientations_batch[i]}: {joints_np[i]}')
                except ValueError as e:
                    print(f'ERROR: ENCOUNTERED ZERO NORM QUATERNION (entry {i}). Skipping relative orientation for node {curr_idx}.')
                    relative_orientations_batch[i, curr_idx] = np.zeros(4)
                    continue

    if np.any(np.isnan(orientations_batch)):
        nan_indices = np.where(np.isnan(orientations_batch))
        print(f"NaN values found at indices:")
        for i in range(len(nan_indices[0])):
            # nan_indices is a tuple of arrays, one for each dimension
            # For a 2D array, nan_indices[0] contains row indices and nan_indices[1] contains column indices
            index_tuple = tuple(nan_indices[j][i] for j in range(orientations_batch.ndim))
            print(f"  {index_tuple}, {orientations_batch[index_tuple[0]]}, {joints[index_tuple[0]]}")
        raise ValueError(f"NAN ERROR: {orientations_batch}")

    if hasattr(single_input, 'value') and single_input:
        return torch.from_numpy(relative_orientations_batch).float().squeeze(0)
    else:
        return torch.from_numpy(relative_orientations_batch).float()

def get_6dof(joints:torch.Tensor, batched=True):
    """
    Get 6D representation of hand joints.
    Input: joints [B, 21, 3] or [21, 3] as numpy array or torch tensor
    Output: 6D representation [B, 21, 6] or [21, 6] as torch tensor
    """
    orientations = extract_orientations(joints, batched=batched).to(joints.device)
    if orientations.ndim == 3:
        return torch.cat([joints, orientations[:, :, :3]], dim=-1)
    else:
        return torch.cat([joints, orientations[:, :3]], dim=-1)

def get_3dof(joints:torch.Tensor, batched=True):
    feats = extract_orientations(joints, batched=batched)
    if feats.ndim == 3:
        return feats[:, :, :3]
    return feats[:,:3]