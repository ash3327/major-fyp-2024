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

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import math

# Train info
version_id = 2
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
train_path_root = f'runs/hand_contrastive_learning/v{version_id}/{current_time}'

# Hyperparameters
batch_size = 256
num_samples = 100 * batch_size
num_samples_eval = 10 * batch_size
num_epochs = 2000
base_learning_rate = 0.01  # Base LR
warmup_epochs = 10  # Warmup period

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataset and dataloader
dataset = HandPoseContrastiveDataset(num_samples=num_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataset_val = HandPoseContrastiveDataset(num_samples=num_samples_eval)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = HandEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
# scheduler = CosineAnnealingLR(optimizer, T_max=10)
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer, patience=500)

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
    # Training phase
    model.train()  # Enable training mode (dropout and batch norm active)
    total_train_loss = 0.0
    for batch_idx, (joints_base, joints_aug) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        joints = torch.cat([joints_base, joints_aug], dim=0).to(device)
        # joints_base, joints_aug = joints_base.to(device), joints_aug.to(device)
        
        # Forward pass
        features = model(joints) # [2B, 21, 3] -> [2B, D]
        # embeddings_base = model(joints_base)
        # embeddings_aug = model(joints_aug)
        
        # Compute loss and backward pass
        loss = info_nce_loss(features, device=device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(dataloader)
    
    # Evaluation phase on the same dataset
    model.eval()  # Switch to evaluation mode (dropout off, batch norm fixed)
    total_eval_loss = 0.0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_idx, (joints_base, joints_aug) in tqdm(enumerate(dataloader_val), total=len(dataloader_val), desc=f"Epoch {epoch+1}/{num_epochs} - Evaluation"):
            joints = torch.cat([joints_base, joints_aug], dim=0).to(device)
            # joints_base, joints_aug = joints_base.to(device), joints_aug.to(device)
            
            # Forward pass
            features = model(joints)
            # embeddings_base = model(joints_base)
            # embeddings_aug = model(joints_aug)
            
            # Compute loss and backward pass
            loss = info_nce_loss(features, device=device)
            total_eval_loss += loss.item()
    
    avg_eval_loss = total_eval_loss / len(dataloader_val)
    
    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/eval', avg_eval_loss, epoch)
    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
    
    # Save the best model based on evaluation loss
    if avg_eval_loss < best_loss:  # Use eval loss for best model selection
        best_loss = avg_eval_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with eval loss: {best_loss:.4f}")
    
    # Save the last model
    torch.save(model.state_dict(), last_model_path)
    
    # Step the scheduler (assuming it supports loss-based adjustment, e.g., ReduceLROnPlateau)
    scheduler.step(avg_train_loss)
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Clean up
writer.close()