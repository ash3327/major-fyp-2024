import os
import numpy as np
import torch
from torch.utils.data import Dataset

class HandPoseContrastiveDataset(Dataset):
    """Dataset for loading pre-generated hand pose pairs with fixed orientation."""
    def __init__(self, num_samples=10000, npy_file='data/kpts/fake/fixed_orientation_pairs.npy', augment=None, base_augment=None, **kwargs):
        """
        Initialize the dataset.
        
        Args:
            npy_file (str): Path to the .npy file containing pre-generated pairs.
            augment (callable, optional): Augmentation function to apply to the poses.
        """
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"Could not find {npy_file}")
            
        self.num_samples = num_samples
        self.pairs = np.load(npy_file)  # Shape: [N, 2, 21, 3]
        self.augment = augment
        self.fixed = False
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Number of pairs: {len(self.pairs)}")
        print(f"Min values (x,y,z): {self.pairs.min(axis=(0,1,2))}")
        print(f"Max values (x,y,z): {self.pairs.max(axis=(0,1,2))}")
        print(f"Mean values (x,y,z): {self.pairs.mean(axis=(0,1,2))}")
        print(f"Std values (x,y,z): {self.pairs.std(axis=(0,1,2))}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.fixed:
            idx = idx % self.num_samples
        else:
            res = len(self.pairs)//self.num_samples
            idx = ((idx % self.num_samples) + self.num_samples*np.random.random_integers(0,res)) % len(self.pairs)
        
        joints_base, joints_aug = self.pairs[idx]  # Each is shape [21, 3]
        
        # Center at wrist (in case it's not already done)
        joints_base = joints_base - joints_base[0]
        joints_aug = joints_aug - joints_aug[0]
        
        # Apply augmentation if provided
        if self.base_augment:
            joints_base, joints_aug = self.base_augment(joints_base,joints_aug)
        if self.augment:
            joints_base = self.augment(joints_base)
            joints_aug = self.augment(joints_aug)
        
        return torch.from_numpy(joints_base).float(), torch.from_numpy(joints_aug).float()

if __name__ == '__main__':
    # Test the dataset
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from prepare_fake_data_npy import plot_3d_hand
    
    # Initialize dataset
    dataset = HandPoseContrastiveDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Test dataloader
    for joints_base, joints_aug in dataloader:
        print("Batch shapes:")
        print(f"Base joints: {joints_base.shape}")
        print(f"Augmented joints: {joints_aug.shape}")
        break
    
    # Visualize some samples
    fig = plt.figure(figsize=(20, 12))
    for i in range(5):
        joints_base, joints_aug = dataset[i]
        
        # Plot base pose
        ax1 = fig.add_subplot(5, 2, i*2 + 1, projection='3d')
        plot_3d_hand(ax1, joints_base.numpy(), color='b', label='Base')
        ax1.set_title(f'Sample {i+1} - Base')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # Plot augmented pose
        ax2 = fig.add_subplot(5, 2, i*2 + 2, projection='3d')
        plot_3d_hand(ax2, joints_aug.numpy(), color='r', label='Augmented')
        ax2.set_title(f'Sample {i+1} - Augmented')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()