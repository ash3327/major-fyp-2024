# FILE NAME: prepare_many_augmentations_per_gesture.py
"""
Generates groups of slightly augmented hand poses for contrastive learning.
For each base gesture, it creates 32 slightly different versions, all in a
canonical orientation after processing (mapping, axis swap, center, scale).

Output format: Numpy array [N_Gestures, 32, 21, 3]

Usage: python scripts/fake_data/prepare_many_augmentations_per_gesture.py
"""

import torch
import numpy as np
import mano
from tqdm import tqdm
import os

# Load the MANO model
model_path = 'model/mano'
n_comps = 45

# Generate variants sequentially for one base pose
rh_model = mano.load(model_path=model_path,
                     is_rhand=True,
                     num_pca_comps=n_comps,
                     batch_size=1,
                     flat_hand_mean=False)

# Keypoint mapping (same as in prepare_contrastive_data.py)
mmap = [
    0, 13, 14, 15, 16,  # thumb
    1, 2, 3, 17,        # index
    4, 5, 6, 18,        # middle
    10, 11, 12, 19,        # ring
    7, 8, 9, 20      # pinky
]

# --- Variance Parameters ---
# Define finger-specific variance parameters
pose_fb_var = 2  # Variance for flexion/bend fingers
pose_non_fb_var = 0.3  # Variance for other movements

# Define which components control flexion/bending
fb_fingers = {i for i in range(9*4) if i % 9 in {2,5,8}}
fb_fingers.add(36)  # Add thumb component

# Variance parameters with finger-specific controls
variance_base = dict(
    betas=2.0,  # Base shape variance
    pose=np.array([pose_fb_var if i in fb_fingers else pose_non_fb_var 
                   for i in range(45)], dtype=np.float32),
    global_orient=0
)

# Bias parameters to control default pose
bias = dict(
    pose=np.array([1.5 if i == 36 else 0.5 if i in fb_fingers else 0 
                   for i in range(45)], dtype=np.float32)
)

# Smaller variations for augmentations
variance_aug = dict(
    betas=0.1,
    pose=variance_base['pose'] * 0.1  # Scale down base variance for augmentations
)

vpow = 2

# --- Generation Parameters ---
N_GESTURES = 1000000  # Total number of unique base gestures to generate (adjust)
N_AUGMENTATIONS_PER_GESTURE = 32 # Number of slight variations per gesture
OUTPUT_DIR = 'data/kpts/fake/'
# Use a distinct filename
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, f'augmented_gesture_groups_{N_AUGMENTATIONS_PER_GESTURE}.npy')
SCALE_FACTOR = 2.5 # Scaling factor from your script
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# Helper Function
# =============================
def generate_one_gesture_group(n_augmentations=N_AUGMENTATIONS_PER_GESTURE, vpow=1):
    """Generate a base pose and its augmentations with finger-specific controls."""
    gesture_variants = np.zeros((n_augmentations, 21, 3), dtype=np.float32)

    # Generate base parameters
    betas_base = (torch.rand(1, 10)*2-1) * variance_base['betas']
    
    # Generate pose with finger-specific controls
    pose_base = torch.randn(1, n_comps) * torch.from_numpy(variance_base['pose']) * 0
    pose_base[:, list(fb_fingers)] = (torch.rand(1, len(fb_fingers))**vpow * 
                                     torch.from_numpy(variance_base['pose'][list(fb_fingers)]))
    pose_base -= torch.ones(1, n_comps) * torch.from_numpy(bias['pose'])

    # Fixed canonical orientation
    global_orient = torch.zeros(1, 3)
    transl = torch.zeros(1, 3)
    
    # Generate base pose
    output_base = rh_model(
        betas=betas_base,
        global_orient=global_orient,
        hand_pose=pose_base,
        transl=transl,
        return_verts=True,
        return_tips=True
    )
    
    # Process base pose
    joints = output_base.joints.detach().cpu().numpy().squeeze()[mmap]
    joints = joints[:, [1, 0, 2]] - joints[0:1, [1, 0, 2]]
    joints[:,1:3] = (joints[:,0:2]+joints[:,2:4]).copy()/2
    joints *= SCALE_FACTOR
    gesture_variants[0] = joints

    # Generate augmentations
    for i in range(1, n_augmentations):
        # Small perturbations from base parameters
        betas_aug = betas_base + torch.randn(1, 10) * variance_aug['betas']
        pose_aug = pose_base + torch.randn(1, n_comps) * torch.from_numpy(variance_aug['pose'])

        output_aug = rh_model(
            betas=betas_aug,
            global_orient=global_orient,
            hand_pose=pose_aug,
            transl=transl,
            return_verts=True,
            return_tips=True
        )
        
        joints_aug = output_aug.joints.detach().cpu().numpy().squeeze()[mmap]
        joints_aug = joints_aug[:, [1, 0, 2]] - joints_aug[0:1, [1, 0, 2]]
        joints_aug[1:3] = (joints_aug[0:2]+joints_aug[2:4]).copy()/2
        joints_aug *= SCALE_FACTOR
        gesture_variants[i] = joints_aug
    
    return gesture_variants


# Generation Script
all_gesture_groups = np.zeros((N_GESTURES, N_AUGMENTATIONS_PER_GESTURE, 21, 3), dtype=np.float32)

print(f"Generating {N_GESTURES} gesture groups with {N_AUGMENTATIONS_PER_GESTURE} variants each...")

for i in tqdm(range(N_GESTURES), desc="Generating Gesture Groups"):
    all_gesture_groups[i] = generate_one_gesture_group(n_augmentations=N_AUGMENTATIONS_PER_GESTURE, vpow=vpow)

print(f"Saving data to {OUTPUT_FILENAME}...")
np.save(OUTPUT_FILENAME, all_gesture_groups)
print("Gesture group data generation complete.")
print(f"Data shape: {all_gesture_groups.shape}")

# Verification
print("\nDataset Statistics (averaged over all variants):")
# Flatten variants for stats calculation
flat_data = all_gesture_groups.reshape(-1, 21, 3)
print(f"Total Samples: {flat_data.shape[0]}")
print(f"Min values (x,y,z): {flat_data.min(axis=(0,1))}")
print(f"Max values (x,y,z): {flat_data.max(axis=(0,1))}")
print(f"Mean values (x,y,z): {flat_data.mean(axis=(0,1))}")
print(f"Std values (x,y,z): {flat_data.std(axis=(0,1))}")