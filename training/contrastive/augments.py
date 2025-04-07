import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

def random_rotate(joints, max_angle=np.pi/3):
    # quarternion
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
    q_vec = np.random.uniform(-1, 1, 3)
    q_vec /= np.linalg.norm(q_vec)  # Normalize to unit vector

    # quarternion: cos(a/2), sin(a/2)*v
    # even rotation opportunities
    angle = np.random.uniform(-max_angle, max_angle) # radians
    w = np.cos(angle / 2)
    q_vec = np.sin(angle / 2) * q_vec
    q_vec = np.concatenate([q_vec,[w]])
    
    rot = R.from_quat(q_vec)  # Create rotation object from quaternion
    rotated_joints = rot.apply(joints)  # Apply rotation to joints

    return rotated_joints

def augment(joints, scale_range=(0, 5), max_angle=np.pi/3):
    # Apply random scaling (0–5x) per axis
    joints = random_rotate(joints, max_angle=max_angle)

    binary_choice = np.random.choice([0, 1], size=3)  # Binary choice for each axis
    scale = np.where(binary_choice == 0, 
                     np.random.uniform(low=scale_range[0], high=1),  # Resample in (low, 1)
                     np.random.uniform(low=1, high=scale_range[1]))  # Resample in (1, high)
    joints *= scale  # Independent scaling for x, y, z

    return joints

def augment_pair(joints_base, joints_aug, scale_range=(0, 5), max_angle=np.pi*2):
    """Apply identical augmentation to both base and augmented joints."""
    # Generate random rotation parameters
    q_vec = np.random.uniform(-1, 1, 3)
    q_vec /= np.linalg.norm(q_vec)
    angle = np.random.uniform(-max_angle, max_angle)
    w = np.cos(angle / 2)
    q_vec = np.sin(angle / 2) * q_vec
    q_vec = np.concatenate([q_vec,[w]])
    
    # Create rotation object
    rot = R.from_quat(q_vec)
    
    # Apply same rotation to both joints
    rotated_base = rot.apply(joints_base)
    rotated_aug = rot.apply(joints_aug)
    
    # Generate same scaling for both joints
    binary_choice = np.random.choice([0, 1], size=3)
    scale = np.where(binary_choice == 0, 
                     np.random.uniform(low=scale_range[0], high=1),
                     np.random.uniform(low=1, high=scale_range[1]))
    
    # Apply same scaling to both joints
    rotated_base *= scale
    rotated_aug *= scale
    
    return rotated_base, rotated_aug

def generate_random_rotation_object(batch_size=None, max_angle=np.pi):
    """Generates a random scipy Rotation object with a maximum rotation angle."""    
    _bs = batch_size if batch_size is not None else 1
    axes = np.random.randn(_bs, 3)  # Random axis
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)  # Normalize to unit vector
    
    angle = np.random.uniform(-max_angle, max_angle, size=(_bs,))  # Random angle within the range # [B]
    quats = np.concatenate([
        np.sin(angle / 2)[:, np.newaxis] * axes, # [B, 1] x [B, 3]
        np.cos(angle / 2)[:, np.newaxis] # [B, 1]
    ], axis=1)  # Quaternion representation [B, 4]
    
    if batch_size is None:
        quats = quats[0]
    return R.from_quat(quats)

def generate_random_scaling_vector(batch_size=None, scale_range=(0, 2), n_dims=3):
    """Generates a random scaling vector."""
    _bs = batch_size if batch_size is not None else 1
    bin = np.random.choice([0, 1], size=(_bs, n_dims))
    scaling_vector = np.where(
        bin == 0,
        np.random.uniform(low=scale_range[0], high=1, size=(_bs, n_dims)),
        np.random.uniform(low=1, high=scale_range[1], size=(_bs, n_dims))
    ) # [B, 3]
    if batch_size is None:
        return scaling_vector[0]
    return scaling_vector

def apply_transform(joints, rotation_obj, scaling_vector):
    """
    Applies rotation and scaling to a single set of joints.
    Assumes joints are centered at the origin.
    Input: joints (np.array or torch.Tensor [21, 3])
    Output: transformed joints (torch.Tensor [21, 3])
    """
    # Ensure joints is a numpy array for scipy rotation
    if isinstance(joints, torch.Tensor):
        joints_np = joints.cpu().numpy()
    else:
        joints_np = joints

    # Apply rotation
    rotated_joints = rotation_obj.apply(joints_np)

    # Apply scaling
    scaled_joints = rotated_joints * scaling_vector # Broadcasting scaling vector

    # Return as torch tensor
    return torch.from_numpy(scaled_joints).float()

def normalize(joints):
    # shape: [B,21,3]
    if not isinstance(joints, torch.Tensor):
        joints = torch.from_numpy(joints).float()
    dim_is_2 = joints.ndim == 2
    B = 1 if dim_is_2 else joints.shape[0]
    joints = F.normalize(joints.view(B, -1), dim=1).view(B, 21, 3)
    # jmin, jmax = torch.min(joints, dim=-2).values[...,torch.newaxis,:], torch.max(joints, dim=-2).values[...,torch.newaxis,:]
    # joints = (joints-jmin)/(jmax-jmin)*2-1
    # joints = joints - joints[...,0,torch.newaxis,:]
    if dim_is_2:
        joints = joints.view(21,3)
    return joints

def vectorized_apply_transform(poses_batch, rotation, scaling):
    """Vectorized transformation for a batch of poses."""
    # poses_batch: [B, 21, 3]
    # Apply scaling
    scaled = poses_batch * scaling.view(1, 1, 3)
    scaled = scaled.view(-1, 3)
    # Apply rotation
    rotated = torch.from_numpy(rotation.apply(scaled.numpy()))
    return rotated.view(-1, 21, 3)