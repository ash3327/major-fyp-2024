import numpy as np
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

def augment(joints, scale_range=(0, 5)):
    # Apply random scaling (0–5x) per axis
    joints = random_rotate(joints, max_angle=np.pi/3)

    binary_choice = np.random.choice([0, 1], size=3)  # Binary choice for each axis
    scale = np.where(binary_choice == 0, 
                     np.random.uniform(low=scale_range[0], high=1),  # Resample in (low, 1)
                     np.random.uniform(low=1, high=scale_range[1]))  # Resample in (1, high)
    joints *= scale  # Independent scaling for x, y, z

    return joints