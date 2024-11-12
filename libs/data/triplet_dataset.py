"""
Without Semi-Hard Mining
Dataset size reduced by factor of 3 to reduce runtime required.
"""

import os
import cv2
import numpy as np
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
        self.class_to_images = {}
        
        for folder_name in os.listdir(self.data_path):
            subfolder_path = os.path.join(self.data_path, folder_name)
            self.classes.append(folder_name)
            self.class_to_images[folder_name] = []
            
            for image_name in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_name)
                self.class_to_images[folder_name].append(image_path)

        # Flatten the image paths and labels for sampling
        for folder_name in self.classes:
            self.image_paths.extend(self.class_to_images[folder_name])
            self.labels.extend([folder_name] * len(self.class_to_images[folder_name]))

    def __len__(self):
        # Return a third of the total images for the dataset
        return len(self.image_paths) // 3  # Adjusted dataset size

    def __getitem__(self, index):
        # Get all images and labels
        all_images = self.image_paths
        all_labels = self.labels

        # Adjust the index to select a random image from the third of the dataset
        actual_index = index * 3  # Sample every third image
        if actual_index >= len(all_images) - 3:
            actual_index = random.randint(0, len(all_images) - 3)  # Fallback to random selection if out of bounds
        offset = random.randint(0, 3)

        # Get anchor image and its class
        anchor_path = all_images[actual_index+offset]
        anchor_class = all_labels[actual_index+offset]
        
        # Select positive sample (same class, different image)
        possible_positives = [p for p in self.class_to_images[anchor_class] if p != anchor_path]
        if possible_positives:
            positive_path = random.choice(possible_positives)
        else:
            positive_path = anchor_path
            
        # Select negative sample (different class)
        negative_classes = [c for c in self.classes if c != anchor_class]
        if negative_classes:
            negative_class = random.choice(negative_classes)
            negative_path = random.choice(self.class_to_images[negative_class])
        else:
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
        return len(self.image_paths)

    def get_unique_labels(self):
        return len(self.classes)