import torch
import torch.nn.functional as F
import math
import numpy as np
import sys
import os

# Linkages

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]
num_joints = 21
num_edges = len(connections)

children_map = {i: [] for i in range(num_joints)}
parent_map = {}
edge_idx_to_joints = {}
joints_to_edge_idx = {}
for i, (u, v) in enumerate(connections):
    children_map[u].append(v)
    parent_map[v] = u
    edge_idx_to_joints[i] = (u, v)
    joints_to_edge_idx[(u, v)] = i

# ==================
# Feature Extraction
# ==================

def extract_geometric_features(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts geometric node and edge features from hand joint coordinates.

    Args:
        joints (torch.Tensor): Tensor of joint coordinates.
                               Shape: (batch_size, 21, 3) or (21, 3).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - node_features (torch.Tensor): Features per joint.
                Shape: (batch_size, 21, 4) or (21, 4)
                Features: [rel_x, rel_y, rel_z, angle]
            - edge_features (torch.Tensor): Features per bone connection.
                Shape: (batch_size, 20, 4) or (20, 4)
                Features: [dir_x, dir_y, dir_z, length]
    """
    was_batched = joints.dim() == 3
    if not was_batched:
        joints = joints.unsqueeze(0) # Temporarily add batch dimension

    batch_size = joints.shape[0]
    device = joints.device
    dtype = joints.dtype

    node_features = torch.zeros(batch_size, num_joints, 4, device=device, dtype=dtype)
    edge_features = torch.zeros(batch_size, num_edges, 4, device=device, dtype=dtype)

    # Calculate Edge Features
    for edge_i, (parent_idx, child_idx) in edge_idx_to_joints.items():
        parent_coords = joints[:, parent_idx, :]
        child_coords = joints[:, child_idx, :]
        bone_vector = child_coords - parent_coords
        bone_length = torch.norm(bone_vector, p=2, dim=-1, keepdim=True) + 1e-8
        bone_direction = bone_vector / bone_length
        edge_features[:, edge_i, 0:3] = bone_direction
        edge_features[:, edge_i, 3:4] = bone_length

    # Calculate Node Features
    wrist_coords = joints[:, 0:1, :]
    node_features[:, :, 0:3] = joints - wrist_coords

    for joint_idx in range(num_joints):
        if joint_idx in parent_map and children_map[joint_idx]:
            parent_idx = parent_map[joint_idx]
            child_idx = children_map[joint_idx][0]
            try:
                edge_in_idx = joints_to_edge_idx[(parent_idx, joint_idx)]
                edge_out_idx = joints_to_edge_idx[(joint_idx, child_idx)]
                vec_joint_to_parent = -edge_features[:, edge_in_idx, 0:3]
                vec_joint_to_child = edge_features[:, edge_out_idx, 0:3]
                dot_product = torch.sum(vec_joint_to_parent * vec_joint_to_child, dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                angle = torch.acos(dot_product)
                node_features[:, joint_idx, 3] = angle
            except KeyError:
                 pass

    if not was_batched:
        node_features = node_features.squeeze(0)
        edge_features = edge_features.squeeze(0)

    return node_features, edge_features