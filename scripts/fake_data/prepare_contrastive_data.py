import torch
import numpy as np
import mano
import os

# Load the MANO model
model_path = 'model/mano'
n_comps = 45
rh_model = mano.load(model_path=model_path,
                     is_rhand=True,
                     num_pca_comps=n_comps,
                     batch_size=1,  # Adjusted dynamically in functions
                     flat_hand_mean=False)
mmap = [
    16, 15, 14, 13, 0, # thumb
    1, 4, 10, 7, # index
    2, 5, 11, 8, # middle
    3, 6, 12, 9, # ring
    17, 18, 19, 20 # pinky
]

# Variance parameters
variance_sim = dict(betas=0.1, pose=0.1, global_orient=1)    # For similar poses (positive pairs)
variance_diff = dict(betas=0.5, pose=0.5, global_orient=5)  # For different poses (base diversity)

def generate_hand_poses(rh_model, batch_size, variance, vpow=1):
    """
    Generate a batch of hand poses.
    
    Args:
        rh_model: MANO hand model.
        batch_size: Number of poses to generate.
        variance: Dict with betas, pose, global_orient variances.
        vpow: Power parameter for pose variance (default=1).
    
    Returns:
        joints: Numpy array [batch_size, 21, 3] of 3D joint positions.
    """
    betas = torch.rand(batch_size, 10) * variance['betas']
    pose = (torch.rand(batch_size, n_comps) - 0.5) * 2
    pose = (torch.rand(batch_size, 1)**vpow * variance['pose']) * pose
    global_orient = torch.rand(batch_size, 3) * variance['global_orient']
    transl = torch.zeros(batch_size, 3)
    
    output = rh_model(betas=betas,
                      global_orient=global_orient,
                      hand_pose=pose,
                      transl=transl,
                      return_verts=True,
                      return_tips=True)
    
    joints = output.joints.numpy()  # [batch_size, 21, 3]
    joints = joints[:, mmap, :]
    joints = joints[:, :, [1, 0, 2]]  # Swap x and y
    joints -= joints[:, [0], :]  # Center at wrist
    return joints

def generate_positive_pair(variance_base=variance_diff, variance_aug=variance_sim, vpow=1, scale_range=(0, 5)):
    """
    Generate a positive pair: base pose and augmented (similar) pose.
    
    Args:
        variance_base: Variance for base pose (diverse).
        variance_aug: Variance for augmentation (small changes).
        vpow: Power parameter for pose variance.
        scale_range: Tuple (min, max) for x, y, z scaling (0–5x).
    
    Returns:
        joints_base, joints_aug: Numpy arrays [21, 3] for base and augmented poses.
    """
    # Base parameters (diverse pose)
    betas_base = torch.rand(1, 10) * variance_base['betas']
    pose_base = (torch.rand(1, n_comps) - 0.5) * 2
    pose_base = (torch.rand(1, 1)**vpow * variance_base['pose']) * pose_base
    global_orient_base = torch.rand(1, 3) * variance_base['global_orient']
    transl_base = torch.zeros(1, 3)
    
    # Augmented parameters (small perturbations)
    betas_aug = betas_base + torch.randn(1, 10) * variance_aug['betas']
    pose_aug = pose_base + torch.randn(1, n_comps) * variance_aug['pose']
    global_orient_aug = global_orient_base + torch.randn(1, 3) * variance_aug['global_orient']
    
    # Generate base pose
    output_base = rh_model(betas=betas_base,
                           global_orient=global_orient_base,
                           hand_pose=pose_base,
                           transl=transl_base,
                           return_verts=True,
                           return_tips=True)
    joints_base = output_base.joints[0].numpy()
    joints_base = joints_base[mmap, :]
    joints_base = joints_base[:, [1, 0, 2]]
    joints_base -= joints_base[0]
    
    # Generate augmented pose
    output_aug = rh_model(betas=betas_aug,
                          global_orient=global_orient_aug,
                          hand_pose=pose_aug,
                          transl=transl_base,
                          return_verts=True,
                          return_tips=True)
    joints_aug = output_aug.joints[0].numpy()
    joints_aug = joints_aug[mmap, :]
    joints_aug = joints_aug[:, [1, 0, 2]]
    joints_aug -= joints_aug[0]
    
    # Apply random scaling (0–5x) per axis
    scale = np.random.uniform(scale_range[0], scale_range[1], size=3)
    joints_aug *= scale  # Independent scaling for x, y, z
    
    return joints_base, joints_aug