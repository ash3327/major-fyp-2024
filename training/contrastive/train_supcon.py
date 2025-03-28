"""
Running the code:
    python training/contrastive/train_supcon.py

Monitor with TensorBoard:
    tensorboard --logdir runs/hand_contrastive_learning/v3
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
from scripts.hand_only_supervised.hand_supervised_dataset import LabelledHandDataset
from model import HandEncoder
from losses import info_nce_loss, supcon_loss
from evals import extract_embeddings, evaluate_knn
from datetime import datetime

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import math

# Train info
version_id = 3
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
train_path_root = f'runs/hand_contrastive_learning/v{version_id}/{current_time}'

# Hyperparameters
batch_size = 256
num_samples_unsup = 100 * batch_size
num_samples_sup = 100 * batch_size
num_epochs = 2000
base_learning_rate = 0.01  # Base LR
warmup_epochs = 10  # Warmup period
eval_interval = 10  # Evaluate every 10 epochs
k_neighbors = 5    # Number of neighbors for k-NN

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize datasets and dataloaders
# Unsupervised dataset
dataset_unsup = HandPoseContrastiveDataset(num_samples=num_samples_unsup)
dataloader_unsup = DataLoader(dataset_unsup, batch_size=batch_size, shuffle=True)
# Labelled dataset (e.g., senz3d)
dataset_sup = LabelledHandDataset(dataset_name='lexset', split='train')
dataloader_sup = DataLoader(dataset_sup, batch_size=batch_size, shuffle=True)
# Test dataset
dataset_test = LabelledHandDataset(dataset_name='lexset', split='test')
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Initialize model and optimizer
model = HandEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
# scheduler = CosineAnnealingLR(optimizer, T_max=10)
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer, patience=20)

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
    model.train()
    total_train_loss = 0.0
    sup_iter = iter(dataloader_sup)  # Iterator for labelled data
    unsup_iter = iter(dataloader_unsup)  # Iterator for unlabelled data

    # Alternate between supervised and unsupervised batches
    for i in tqdm(range(max(len(dataloader_sup), len(dataloader_unsup))),
                  desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        # Supervised batch (if available)
        try:
            labels, joints = next(sup_iter)
            joints, labels = joints.to(device), labels.to(device)
            features = model(joints)  # [B, D]
            features = features.unsqueeze(1)  # [B, 1, D]
            loss = supcon_loss(features, labels, device=device)
        except StopIteration:
            loss = 0.0  # Skip if no more labelled data

        # Unsupervised batch (if available)
        try:
            joints_base, joints_aug = next(unsup_iter)
            joints = torch.cat([joints_base, joints_aug], dim=0).to(device)
            features = model(joints)  # [2B, D]
            unsup_loss = info_nce_loss(features, device=device)
            loss = loss + unsup_loss if loss != 0.0 else unsup_loss
        except StopIteration:
            pass  # Continue with supervised loss if no more unlabelled data

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / max(len(dataloader_sup), len(dataloader_unsup))

    # Log metrics
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

    # Save models
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with train loss: {best_loss:.4f}")
    torch.save(model.state_dict(), last_model_path)

    # Update scheduler
    scheduler.step(avg_train_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Evaluation with k-NN
    if (epoch + 1) % eval_interval == 0:
        print(f"Evaluating on test set at epoch {epoch+1} using k-NN (k={k_neighbors})")
        
        # Extract embeddings
        train_embeddings, train_labels = extract_embeddings(model, dataloader_sup, device)
        test_embeddings, test_labels = extract_embeddings(model, dataloader_test, device)

        # Perform k-NN evaluation
        accuracy, f1 = evaluate_knn(train_embeddings, train_labels, test_embeddings, test_labels, k=k_neighbors)

        # Log evaluation metrics
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        writer.add_scalar('F1/test', f1, epoch)
        print(f"Test Accuracy: {accuracy:.4f}, Test F1: {f1:.4f}")

# Clean up
writer.close()