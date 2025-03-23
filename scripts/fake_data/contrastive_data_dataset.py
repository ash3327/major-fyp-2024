from torch.utils.data import Dataset
import torch

from .prepare_contrastive_data import generate_positive_pair

class HandPoseContrastiveDataset(Dataset):
    def __init__(self, num_samples, vpow=1, scale_range=(0, 5)):
        """
        Dataset for on-the-fly generation of positive pairs.
        
        Args:
            num_samples: Total number of pairs.
            vpow: Power parameter for pose variance.
            scale_range: Scaling range for augmentation (0-5x).
        """
        self.num_samples = num_samples
        self.vpow = vpow
        self.scale_range = scale_range
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        joints_base, joints_aug = generate_positive_pair(vpow=self.vpow, scale_range=self.scale_range)
        return torch.from_numpy(joints_base).float(), torch.from_numpy(joints_aug).float()