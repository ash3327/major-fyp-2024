import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.load_data()

    def load_data(self):
        self.classes = []
        for folder_name in os.listdir(self.data_path):
            subfolder_path = os.path.join(self.data_path, folder_name)
            self.classes.append(folder_name)
            for image_name in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(folder_name)
        self.unique_labels = {label: i for i, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        # target = torch.nn.functional.one_hot(, num_classes=self.get_unique_labels())
        return image, label, self.unique_labels[label]
    
    def get_image_count(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def get_unique_labels(self):
        """Return the number of unique labels (people) in the dataset."""
        return len(set(self.labels))
