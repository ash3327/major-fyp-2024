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

# Hyperparameters
batch_size = 256
num_samples = 10000
num_epochs = 100
learning_rate = 0.001
version_id = 1
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
train_path_root = f'runs/hand_contrastive_learning/v{version_id}/{current_time}'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataset and dataloader
dataset = HandPoseContrastiveDataset(num_samples=num_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = HandEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    for batch_idx, (joints_base, joints_aug) in tqdm(enumerate(dataloader), total=num_samples // batch_size + 1):
        joints_base, joints_aug = joints_base.to(device), joints_aug.to(device)
        
        # Forward pass
        embeddings_base = model(joints_base)
        embeddings_aug = model(joints_aug)
        
        # Compute loss
        loss = info_nce_loss(embeddings_base, embeddings_aug)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    
    # Log the average loss to TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    
    # Check if this is the best loss so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        # Save the best model
        torch.save(model.state_dict(), best_model_path)

    # Save the last model after training completes
    torch.save(model.state_dict(), last_model_path)
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Clean up
writer.close()