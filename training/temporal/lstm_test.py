# %%
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
from sklearn.metrics import accuracy_score, f1_score

import random
sys.path.append('.')

from tqdm import tqdm
from datetime import datetime

from scripts.hand_only_supervised.video_dataset import get_dataloader, visualize_video_with_labels


# %%
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

# %%


# %%
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

# Checkpoint
model_checkpoint_name = None
model_checkpoint_name = 'v1/20250408_221008'

# load model from file
def load_model(model_checkpoint_name):
    if model_checkpoint_name:
        model_checkpoint_path = f'runs/{experiment_name}/{model_checkpoint_name}/checkpoints/best.pth'
        if os.path.exists(model_checkpoint_path):
            model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
            print(f"Model loaded from {model_checkpoint_path}")
            # You might want to load optimizer state as well for resuming training
            # optimizer.load_state_dict(torch.load(model_checkpoint_path.replace('best.pth', 'optimizer.pth')))
        else:
            print(f"Model file not found at {model_checkpoint_path}")

load_model(model_checkpoint_name)

# Pre-processing
def post_fn(sequences:torch.Tensor, labels, *others):
    B,L,_,_ = sequences.shape
    return sequences.reshape(B,L,-1), labels, *others

# Data
train_loader = get_dataloader(dataset='ipn', split='train', batch_size=batch_size, post_fn=post_fn, with_id=True)
val_loader = get_dataloader(dataset='ipn', split='test', batch_size=batch_size, post_fn=post_fn, with_id=True)

# %%

def evaluate_model(dataloader):
    """Evaluates the model on the validation set using global variables."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0 # Total non-padded frames
    true_labels = []
    predicted_labels = []
    misclassified_videos = dict()

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for sequences, labels, idxs in progress_bar:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            L = outputs.shape[1] # [B,L,C]
            outputs_flat = outputs.view(-1, model.output_dim) # [B*L,C]
            labels_flat = labels.view(-1) # [B*L]

            mask = labels_flat != -100
            num_valid_samples = mask.sum().item()
            if num_valid_samples > 0:
                loss = criterion(outputs_flat, labels_flat)
                total_loss += loss.item() * sequences.size(0)

                _, predicted = torch.max(outputs_flat[mask], 1)
                total_correct += (predicted == labels_flat[mask]).sum().item()
                total_samples += num_valid_samples
                true_labels.extend(labels_flat[mask].cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
            else:
                loss = torch.tensor(0.0)
            print(sequences.shape, outputs.shape, labels.shape)

            # Identify misclassified videos
            batch_size = sequences.size(0)
            for i in range(batch_size):
                start_idx = i * L
                end_idx = (i + 1) * L
                video_labels_flat = labels_flat[start_idx:end_idx]
                video_mask = video_labels_flat != -100
                valid_video_labels = video_labels_flat[video_mask]
                if valid_video_labels.numel() > 0:
                    video_outputs_flat = outputs_flat[start_idx:end_idx]
                    _, video_predicted = torch.max(video_outputs_flat[video_mask], 1)
                    if not torch.equal(video_predicted, valid_video_labels):
                        misclassified_videos[idxs[i]] = (sequences[i][video_mask], video_predicted, valid_video_labels)

    avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
    avg_acc = (total_correct / total_samples) if total_samples > 0 else 0.0

    f1 = 0.0
    if true_labels:
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        
    return avg_loss, avg_acc, f1, misclassified_videos

# %%
# Evaluation
epoch_val_loss, epoch_val_acc, epoch_val_f1, misclassified_videos = evaluate_model(val_loader)
# print(misclassified_video)
for k, (seq, pred, lab) in misclassified_videos.items():
    visualize_video_with_labels(seq, pred, lab, val_loader.dataset)
    break

