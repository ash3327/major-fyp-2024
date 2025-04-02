import numpy as np
import torch
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

def generate_random_rotation_object():
    """Generates a random scipy Rotation object."""
    quat = np.random.randn(4) # Interesting way for generating random quaternion learnt from gemini
    # Note for self: SO(3) sampling, fulfills property that w+ai+bj+ck satisfies w^2+a^2+b^2+c^2 = 1
    # Original formulation: cos(t/2)+sin(t/2)(ai+bj+ck) with a^2+b^2+c^2=1 (conventional sampling)
    quat /= np.linalg.norm(quat)
    return R.from_quat(quat)

def generate_random_scaling_vector(scale_range=(0.7, 1.3), n_dims=3):
    """Generates a random scaling vector."""
    # Simple uniform scaling for now, can be made axis-independent
    # scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    # return np.full(n_dims, scale_factor)
    # Or independent scaling per axis:
    return np.random.uniform(scale_range[0], scale_range[1], size=n_dims)

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