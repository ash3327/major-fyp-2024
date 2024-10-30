import os
import cv2
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.load_data()

    def load_data(self):
        self.classes = []
        # Dictionary to store image paths for each class
        self.class_to_images = {}
        
        for folder_name in os.listdir(self.data_path):
            subfolder_path = os.path.join(self.data_path, folder_name)
            self.classes.append(folder_name)
            self.class_to_images[folder_name] = []
            
            for image_name in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(folder_name)
                self.class_to_images[folder_name].append(image_path)

    def __len__(self):
        return len(self.image_paths)  # Keep original dataset size

    def __getitem__(self, index):
        # Get anchor image and its class
        anchor_path = self.image_paths[index]
        anchor_class = self.labels[index]
        
        # Select positive sample (same class, different image)
        possible_positives = [p for p in self.class_to_images[anchor_class] if p != anchor_path]
        if possible_positives:  # If there are other images in the same class
            positive_path = random.choice(possible_positives)
        else:  # If no other images in the same class, use the anchor image
            positive_path = anchor_path
            
        # Select negative sample (different class)
        negative_classes = [c for c in self.classes if c != anchor_class]
        if negative_classes:  # If there are other classes
            negative_class = random.choice(negative_classes)
            negative_path = random.choice(self.class_to_images[negative_class])
        else:  # If no other classes, use the anchor image (this should rarely happen)
            negative_path = anchor_path
        
        # Load and transform images
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        
        return (anchor, positive, negative), (anchor_class, negative_class)
    
    def _load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image
    
    def get_image_count(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def get_unique_labels(self):
        """Return the number of unique labels (people) in the dataset."""
        return len(self.classes)