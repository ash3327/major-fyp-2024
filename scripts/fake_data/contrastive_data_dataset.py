import os
import numpy as np
import torch
from torch.utils.data import Dataset

class HandPoseContrastiveDataset(Dataset):
    """
    Dataset for loading pre-generated groups of hand pose augmentations.
    Each group contains N slight variations of the same base gesture,
    all in a canonical orientation.

    Adapts indexing logic from HandPoseContrastiveDataset, allowing for
    an epoch size (`num_samples`) potentially different from the total
    number of gestures available.
    """
    def __init__(self, num_samples=10000, npy_file='data/kpts/fake/augmented_gesture_groups_32.npy', **kwargs):
        """
        Args:
            npy_file (str): Path to the .npy file.
                            Expected shape: [N_Gestures, N_Augmentations, 21, 3]
            num_samples (int, optional): The effective number of samples per epoch.
                                         If None, defaults to the total number of
                                         unique gestures in the npy file.
            fixed_indexing (bool): If True, use simple idx % total_gestures.
                                   If False (default), use the random offset logic.
        """
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"Could not find {npy_file}")

        print(f"Loading augmented gesture groups from {npy_file}...")
        self.gesture_groups = np.load(npy_file) # Shape: [N_Gestures, N_Augs, 21, 3]
        self.total_gestures = self.gesture_groups.shape[0]
        self.num_augs_per_gesture = self.gesture_groups.shape[1]

        if num_samples is None:
            self.num_samples = self.total_gestures
            print(f"`num_samples` not provided, setting epoch size to total gestures: {self.total_gestures}")
        else:
            self.num_samples = min(num_samples, self.total_gestures)
            print(f"Dataset epoch size set to `num_samples`: {self.num_samples}")

        print(f"\nAugmented Gesture Dataset Statistics:")
        print(f"Total unique gestures in file: {self.total_gestures}")
        print(f"Number of augmentations per gesture: {self.num_augs_per_gesture}")
        print(f"Loaded data shape: {self.gesture_groups.shape}")
        # Note: Augmentations (rotation, scaling) are applied dynamically in collate_fn
        # Stats below are for the canonical poses stored in the file
        flat_data = self.gesture_groups.reshape(-1, 21, 3)
        print(f"Min values (x,y,z): {flat_data.min(axis=(0,1))}")
        print(f"Max values (x,y,z): {flat_data.max(axis=(0,1))}")
        print(f"Mean values (x,y,z): {flat_data.mean(axis=(0,1))}")
        print(f"Std values (x,y,z): {flat_data.std(axis=(0,1))}")

    def __len__(self):
        """Returns the defined number of samples per epoch."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns the group of augmentations for a selected gesture index.
        The index selection depends on `self.fixed`.
        """
        # Random offset indexing logic adapted from HandPoseContrastiveDataset
        assert self.num_samples > 0, "num_samples cannot be zero for non-fixed indexing."

        # Calculate how many times the epoch size fits into the total gestures
        res = self.total_gestures // self.num_samples
        idx = (idx + self.num_samples*np.random.random_integers(0,res)) % self.total_gestures

        # Retrieve the corresponding gesture group
        gesture_group = self.gesture_groups[idx] # Shape: [N_Augs, 21, 3]
        gesture_group = gesture_group[:,:,(2,1,0)].copy()

        # Return as numpy array. Tensor conversion and augmentation happen in collate_fn.
        return gesture_group
