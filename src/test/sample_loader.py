"""
Example of loading data from a dataset.

ONLY WORKS IF EXECUTED FROM ROOT
"""

import torch

import sys
sys.path.append('.')

from src.data import LexsetDataset
from multiprocessing import freeze_support

if __name__ == '__main__':
    split: str = 'train'
    batch_size: int = 32 
    shuffle: bool = True 
    num_workers: int = 4

    yaml_path = 'configs/data_configs/lexset_configs.yaml'

    dataset = LexsetDataset.from_yaml(
        yaml_path, 
        split=split
    )

    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )

    for batch in dataloader:
        images = batch['image']
        labels = batch['label']
        # Process your batch
        print(images.shape)
        print(labels)
        break
    