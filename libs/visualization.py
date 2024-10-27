import os
import cv2
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def visualize_sample_images(folder_path):
    num_images = len(os.listdir(folder_path))
    num_rows = min((num_images + 4) // 5, 2)
    num_cols = min(num_images, 5)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows))

    for i, image_name in enumerate(os.listdir(folder_path)):
        if i >= 2 * num_cols:
            break
        image_path = os.path.join(folder_path, image_name)
        sample_image = cv2.imread(image_path)
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        row = i // num_cols
        col = i % num_cols

        if num_rows == 1:
              ax = axes[col]
        else:
            ax = axes[row, col]

        ax.imshow(sample_image)
        ax.axis('off')

    for ax in axes.flat[num_images:]:
          ax.remove()

    plt.suptitle(f'Person ID: {os.path.basename(folder_path)}')
    plt.tight_layout()
    plt.show()

def visualize_dataset_images(dataset):
    # Create a DataLoader for the dataset
    data_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

    visualize_dataloader_images(data_loader)

def visualize_dataloader_images(data_loader):
    # Load a batch of images
    for images, labels, targets in data_loader:
        # Plot the images
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        axs = axs.flatten()
        for i, (img, label) in enumerate(zip(images, labels)):
            if i >= 10: 
                break
            img = img.cpu().numpy().transpose((1, 2, 0))  # Convert to numpy array and transpose
            
            axs[i].imshow(img)
            axs[i].set_title(f'ID: {label}', fontsize=20)
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()
        break  # Only show the first batch