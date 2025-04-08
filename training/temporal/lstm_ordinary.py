"""
tensorboard --logdir runs/ipn_classifiers/v1 --port 6007
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
sys.path.append('.')

from tqdm import tqdm
from datetime import datetime

from scripts.hand_only_supervised.video_dataset import get_dataloader

# == Model ==
# Define LSTM model
class LSTMGestureModel(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=256, output_dim=128, num_layers=3, dropout=0.2):
        super(LSTMGestureModel, self).__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x) # shape: [B,L,D_in] -> [B,L,D_lstm]
        out = self.fc(x) # shape: -> [B,L,D_out]
        return out
        # return self.fc(x[:, -1, :])  # Take last time step output

# === Info ===
# Log
version_id = 1
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_name = f'ipn_classifiers'
run_name = f'v{version_id}/{current_time}'
train_path_root = f'runs/{experiment_name}/{run_name}'
log_dir = os.path.join(train_path_root, 'logs')
checkpoint_dir = os.path.join(train_path_root, 'checkpoints')

eval_interval = 2
patience = 100
extra_text = "Bi-directional LSTM, direct classification"

# Training setup
batch_size = 32
num_epochs = 1000
lr = 0.001
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
model = LSTMGestureModel(177, output_dim=21).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss(ignore_index=-100) # Ignore padding index
scheduler = ReduceLROnPlateau(optimizer, patience=patience)

# Pre-processing
def post_fn(sequences:torch.Tensor, labels):
    B,L,_,_ = sequences.shape
    return sequences.reshape(B,L,-1), labels

# Data
train_loader = get_dataloader(dataset='ipn', split='train', batch_size=batch_size, post_fn=post_fn)
val_loader = get_dataloader(dataset='ipn', split='test', batch_size=batch_size, post_fn=post_fn)

# === Misc ===
# initialize TensorBoard writer
os.makedirs(train_path_root, exist_ok=True)
writer = SummaryWriter(os.path.join(train_path_root,'logs'))

# Log model class name
writer.add_text('Model', f'Model class: {model.__class__.__name__} <{train_loader.dataset.__class__.__name__}> (ipn.ipynb) {extra_text}', 0)

# model save paths
os.makedirs(train_path_root, exist_ok=True)
model_save_path_root = os.path.join(train_path_root,'checkpoints')
os.makedirs(model_save_path_root, exist_ok=True)
best_model_path = os.path.join(model_save_path_root,'best.pth')
last_model_path = os.path.join(model_save_path_root,'last.pth')

# === Util Functions ===
def evaluate_model():
    """Evaluates the model on the validation set using global variables."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0 # Total non-padded frames

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating", leave=False)
        for sequences, labels in progress_bar:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            outputs_flat = outputs.view(-1, model.output_dim)
            labels_flat = labels.view(-1)

            mask = labels_flat != -100
            num_valid_samples = mask.sum().item()
            if num_valid_samples > 0:
                loss = criterion(outputs_flat, labels_flat)
                total_loss += loss.item() * sequences.size(0)

                _, predicted = torch.max(outputs_flat[mask], 1)
                total_correct += (predicted == labels_flat[mask]).sum().item()
                total_samples += num_valid_samples
            else:
                 loss = torch.tensor(0.0)

    avg_loss = total_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
    avg_acc = (total_correct / total_samples) if total_samples > 0 else 0.0
    return avg_loss, avg_acc

# Training
def train():
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_correct_train = 0
        total_samples_train = 0
        for i, (sequences, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            # print(sequences.shape)
            outputs = model(sequences)
            outputs_flat = outputs.view(-1, model.output_dim)
            labels_flat = labels.view(-1)

            mask = labels_flat != -100
            num_valid = mask.sum().item()
            if num_valid == 0:
                continue # Skip batch if no valid labels

            loss = criterion(outputs_flat, labels_flat)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * sequences.size(0)

            # Calculate training accuracy for monitoring
            with torch.no_grad():
                _, predicted = torch.max(outputs_flat[mask], 1)
                total_correct_train += (predicted == labels_flat[mask]).sum().item()
                total_samples_train += num_valid

            # Update progress bar postfix dynamically
            current_lr = optimizer.param_groups[0]['lr']

        # Logs
        avg_train_loss = total_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
        avg_train_acc = (total_correct_train / total_samples_train) if total_samples_train > 0 else 0.0

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Evaluation
        if (epoch + 1) % eval_interval == 0:
            epoch_val_loss, epoch_val_acc = evaluate_model()
            writer.add_scalar('Loss/validation', epoch_val_loss, epoch)
            writer.add_scalar('Accuracy/validation', epoch_val_acc, epoch)

            # LR Scheduler Step (based on validation loss)
            scheduler.step(epoch_val_loss)

            # Save Best Model Checkpoint (based on validation loss)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_path = os.path.join(checkpoint_dir, 'best.pth')
                # Save only model state dict for 'best' for simplicity
                torch.save(model.state_dict(), best_model_path)
                print(f"*** Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.4f} ***")

        # Save Last Model Checkpoint
        last_model_path = os.path.join(checkpoint_dir, 'last.pth')
        torch.save(model.state_dict(), last_model_path)

# === Main ===
train()