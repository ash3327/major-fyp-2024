"""
Running the code:
    python training/contrastive/train_contrastive_fake_data.py

Monitor with TensorBoard:
    tensorboard --logdir runs/hand_contrastive_learning/v1
"""

# train.py
import sys
sys.path.append('.')  # Ensure imports work from the project root

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scripts.fake_data.contrastive_data_dataset import HandPoseContrastiveDataset
from model import HandEncoder
from losses import info_nce_loss
from datetime import datetime

from torch.optim.lr_scheduler import CosineAnnealingLR
import math

# Train info
version_id = 1
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
train_path_root = f'runs/hand_contrastive_learning/v{version_id}/{current_time}'

# Hyperparameters
batch_size = 256
num_samples = 100 * batch_size
num_epochs = 200
base_learning_rate = 0.01  # Base LR
warmup_epochs = 10  # Warmup period

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataset and dataloader
dataset = HandPoseContrastiveDataset(num_samples=num_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = HandEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# Initialize TensorBoard writer
os.makedirs(train_path_root, exist_ok=True)
writer = SummaryWriter(os.path.join(train_path_root,'logs'))

# Model save paths
model_save_path_root = os.path.join(train_path_root,'checkpoints')
os.makedirs(model_save_path_root, exist_ok=True)
best_model_path = os.path.join(model_save_path_root,'best.pth')
last_model_path = os.path.join(model_save_path_root,'last.pth')

# Create the models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Initialize best loss tracker
best_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (joints_base, joints_aug) in tqdm(enumerate(dataloader), total=len(dataloader)):
        joints_base, joints_aug = joints_base.to(device), joints_aug.to(device)
        
        # Forward pass
        embeddings_base = model(joints_base)
        embeddings_aug = model(joints_aug)
        
        # Compute loss and backward pass
        loss = info_nce_loss(embeddings_base, embeddings_aug)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
    
    if avg_loss < best_loss: # save the best model
        best_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
    
    torch.save(model.state_dict(), last_model_path)
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Clean up
writer.close()