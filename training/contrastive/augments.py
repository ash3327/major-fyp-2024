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

def generate_random_rotation_object(max_angle=np.pi):
    """Generates a random scipy Rotation object with a maximum rotation angle."""    
    axis = np.random.randn(3)  # Random axis
    axis /= np.linalg.norm(axis)  # Normalize to unit vector
    
    angle = np.random.uniform(-max_angle, max_angle)  # Random angle within the range
    quat = np.concatenate([np.sin(angle / 2) * axis, [np.cos(angle / 2)]])  # Quaternion representation
    
    return R.from_quat(quat)

def generate_random_scaling_vector(scale_range=(0, 2), n_dims=3):
    """Generates a random scaling vector."""
    bin = np.random.choice([0, 1], size=n_dims)
    scaling_vector = np.where(
        bin == 0,
        np.random.uniform(low=scale_range[0], high=1, size=n_dims),
        np.random.uniform(low=1, high=scale_range[1], size=n_dims)
    )
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
    # shape: [21,3]
    jmin, jmax = torch.min(joints, dim=0).values, torch.max(joints, dim=0).values
    joints = (joints-jmin)/(jmax-jmin)*2-1
    return joints