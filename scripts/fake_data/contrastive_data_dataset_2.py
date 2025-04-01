from torch.utils.data import Dataset
import torch

from .prepare_contrastive_data import generate_positive_pair

class HandPoseContrastiveDataset(Dataset):
    def __init__(self, num_samples, vpow=1, scale_range=(0, 5), augment=None):
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
        self.augment = augment
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        joints_base, joints_aug = generate_positive_pair(vpow=self.vpow, scale_range=self.scale_range)
        """
        Before adjustments:

        Statistics for HandPoseContrastiveDataset:
            Min (x, y, z): [   -0.14282    -0.13709    -0.15487]
            Max (x, y, z): [    0.12503     0.12755      0.1392]
            Mean (x, y, z): [  -0.001927   0.0086811   -0.016303]
            Std (x, y, z): [    0.04153    0.042865    0.044661]

            Statistics for LabelledHandDataset: # lexset
            Min (x, y, z): [   -0.83551          -1     -1.4672]
            Max (x, y, z): [     0.8914     0.96627      1.1815]
            Mean (x, y, z): [ 0.00096391       -0.21    -0.10805]
            Std (x, y, z): [    0.11838     0.15705     0.16907]

            Statistics for LabelledHandDataset: # senz3d
            Min (x, y, z): [   -0.32398    -0.81078     -1.4202]
            Max (x, y, z): [    0.45164     0.33794     0.26407]
            Mean (x, y, z): [   0.034734     -0.2311    -0.17816]
            Std (x, y, z): [   0.081709     0.13831     0.11482]

            Statistics for LabelledHandDataset: # handshape
            Min (x, y, z): [   -0.86802    -0.52049    -0.84314]
            Max (x, y, z): [    0.95333           1      0.2794]
            Mean (x, y, z): [   0.066712    -0.13533    -0.14399]
            Std (x, y, z): [    0.12579     0.11963      0.1062]
        """
        joints_base, joints_aug = joints_base * 2.5, joints_aug * 2.5
        # scale to range [-0.1, 0.1]
        # shape: (21, 3)
        if self.augment:
            joints_base = self.augment(joints_base, scale_range=self.scale_range)
            joints_aug = self.augment(joints_aug, scale_range=self.scale_range)
        return torch.from_numpy(joints_base).float(), torch.from_numpy(joints_aug).float()