import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Literal

from .data import ImageDataset

back = lambda x: os.path.dirname(x)
base_path = back(back(os.getcwd()))

class ASLDataLoader:
    def __init__(self, cfgs:dict):
        self.using_kaggle = cfgs['using_kaggle']

        if self.using_kaggle and 'kaggle_override' in cfgs:
            cfgs.update(cfgs['kaggle_override'])

        self.use_dataset = cfgs['use_dataset']

        if self.use_dataset not in cfgs['data']:
            raise Exception(f'The dataset specified ({self.use_dataset}) does not exist.')

        data_cfgs = cfgs['data'][self.use_dataset]
        
        self.input_folder = os.path.normpath(os.path.join(base_path, cfgs['input_folder'], data_cfgs['path']))
        self.train_data_path = os.path.join(self.input_folder, data_cfgs['train'])
        self.test_data_path = os.path.join(self.input_folder, data_cfgs['test'])

        self.dataset = None
        self.dataloader = None

    def check(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")

    def get_dataset(self, train:bool=True, transform=None, mode: Literal["standard", "triplets", "contrastive"] = "standard"):
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        folder_path = self.train_data_path if train else self.test_data_path
        
        match mode:
            case "standard":
                self.dataset = ImageDataset(folder_path, transform=transform)
                return self.dataset
            case "contrastive":
                raise Exception(f"Mode '{mode}' not implemented yet.")
            case "triplets":
                raise Exception(f"Mode '{mode}' not implemented yet.")
            case _:
                raise Exception(f"Mode '{mode}' not found.")
        
    def get_dataloader(self, train:bool=True, batch_size=32, shuffle=True, transform=None, mode: Literal["standard", "triplets", "contrastive"] = "standard"):
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        dataset = self.get_dataset(train, transform)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return self.dataloader
    
    def display_data_info(self, train: bool = True):
        dataset = self.get_dataset(train)
        num_images = dataset.get_image_count()
        num_people = dataset.get_unique_labels()
        print(f"Number of images: {num_images}")
        print(f"Number of unique classes: {num_people}")