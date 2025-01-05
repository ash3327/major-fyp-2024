"""
Lexset Loader

Accepted layout: 
root(train/test)/
    class1/
        image.jpg/png/...

receives config format:
    root: "data/raw" # subfolder under project root
    path: "synthetic-asl-alphabet" # subfolder under root
    train: "Train_Alphabet" # subfolder under path
    test: "Test_Alphabet" # subfolder under path
    layoutType: "lexset"

layout type: lexset
"""

import os
import typing
from typing import List, Optional, Callable, Union, Dict, Any

import torch
import yaml
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LexsetDataset(Dataset):
    """
    Loader for layout type: lexset
    
    Accepted layout: 
    root(train/test)/
        class1/
            image.jpg/png/...

    receives config format:
        root: "data/raw" # subfolder under project root
        path: "synthetic-asl-alphabet" # subfolder under root
        train: "Train_Alphabet" # subfolder under path
        test: "Test_Alphabet" # subfolder under path
        layoutType: "lexset"
    """
    def __init__(
        self, 
        config: Dict[str, Any],
        split: str = 'train',
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    ):
        # Validate config
        if config.get('layoutType') != 'lexset':
            raise ValueError("Configuration must be for a lexset layout type")
        
        # Construct full path
        root_dir = os.path.join(
            config.get('root', ''), 
            config.get('path', ''), 
            config.get(split, '')
        )
        
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) # ImageNet
        
        self.target_transform = target_transform
        self.extensions = extensions
        
        # Collect image paths and labels
        self.images: List[str] = []
        self.labels: List[str] = []
        
        self._scan_directory(root_dir)
    
    def _scan_directory(self, directory: str):
        """
        Recursively scan directory for images and collect paths
        """
        for class_name in os.listdir(directory):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if any(file.lower().endswith(ext) for ext in self.extensions):
                        full_path = os.path.join(class_path, file)
                        self.images.append(full_path)
                        self.labels.append(class_name)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and transform image
        
        Returns:
            Dict with 'image' and 'label' keys
        """
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return {
            'image': image,
            'label': label
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: str, split: str = 'train'):
        """
        Create dataset from YAML configuration file
        
        Args:
            yaml_path (str): Path to YAML configuration file
            split (str): Dataset split to load (train/test)
        
        Returns:
            LexsetDataset instance
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(config, split=split)

def get_lexset_dataloader(
    yaml_path: str, 
    split: str = 'train',
    batch_size: int = 32, 
    shuffle: bool = True, 
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """
    Convenience function to create a DataLoader from config
    
    Args:
        config (Dict[str, Any]): Dataset configuration
        split (str): Dataset split to load
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle dataset
        num_workers (int): Number of subprocesses for data loading
    
    Returns:
        PyTorch DataLoader
    """
    dataset = LexsetDataset.from_yaml(yaml_path, split=split)
    
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
    )