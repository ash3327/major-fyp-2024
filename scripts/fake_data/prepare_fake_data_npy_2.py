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

# Variance parameters
variance_base = dict(
    betas=0.5,  # Base shape variance
    pose=0.5    # Base pose variance
)
# variance_base = dict(
#     betas=0.3,  # Base shape variance
#     pose=1    # Base pose variance
# )
variance_aug = dict(
    betas=0.1,  # Small shape changes for augmentations
    pose=0.05    # Small pose changes for augmentations
)
vpow = 1 # Keep if used before

# --- Generation Parameters ---
N_GESTURES = 500#00  # Total number of unique base gestures to generate (adjust)
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
    """
    Generates a base pose and N-1 slightly augmented versions.
    All poses are processed (map, swap, center, scale) to a canonical orientation.
    Returns: np.array [n_augmentations, 21, 3]
    """
    gesture_variants = np.zeros((n_augmentations, 21, 3), dtype=np.float32)

    # 1. Generate Base Parameters
    betas_base = torch.rand(1, 10) * variance_base['betas']
    # pose_base = (torch.rand(1, n_comps) - 0.5) * 2
    # pose_base = (torch.rand(1, 1)**vpow * variance_base['pose']) * pose_base
    pose_base = torch.rand(1, n_comps) * variance_base['pose']

    # Fixed canonical orientation and translation for ALL variants
    global_orient = torch.zeros(1, 3)
    transl = torch.zeros(1, 3)
    
    # 2. Generate and Process Base Pose (first variant)
    output_base = rh_model(
        betas=betas_base, global_orient=global_orient, hand_pose=pose_base, transl=transl,
        return_verts=True, return_tips=True
    )
    joints = output_base.joints.detach().cpu().numpy().squeeze()[mmap]
    joints = joints[:, [1, 0, 2]] - joints[0:1, [1, 0, 2]] # Swap, Center
    joints *= SCALE_FACTOR # Scale
    gesture_variants[0] = joints

    # 3. Generate and Process Augmented Poses (remaining variants)
    for i in range(1, n_augmentations):
        # Augmented parameters (small perturbations from base)
        betas_aug = betas_base + torch.randn(1, 10) * variance_aug['betas']
        pose_aug = pose_base + torch.randn(1, n_comps) * variance_aug['pose']

        # Generate augmented pose with SAME canonical orientation
        output_aug = rh_model(
            betas=betas_aug, global_orient=global_orient, hand_pose=pose_aug, transl=transl,
            return_verts=True, return_tips=True
        )
        joints_aug = output_aug.joints.detach().cpu().numpy().squeeze()[mmap]
        joints_aug = joints_aug[:, [1, 0, 2]] - joints_aug[0:1, [1, 0, 2]] # Swap, Center
        joints_aug *= SCALE_FACTOR # Scale
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