# %%
"""
tensorboard --logdir runs/ipn_classifiers/v1 --port 6007
"""

import os
import sys
import numpy as np
from itertools import islice
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

from scripts.hand_only_supervised.video_dataset import get_dataloader
from training.temporal.model import LSTMGestureModel, LSTMGestureModel_Windowed, LSTMGestureModel_Hierachical_Windowed
from training.contrastive.losses import softcon_loss
from training.contrastive.losses import info_nce_loss, supcon_loss, info_nce_loss_from_matrix

# %%
# Log
version_id = 1
dump = False #True

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_name = f'ipn_classifiers' if not dump else 'ipn_dump'
run_name = f'v{version_id}/{current_time}'
train_path_root = f'runs/{experiment_name}/{run_name}'
log_dir = os.path.join(train_path_root, 'logs')
checkpoint_dir = os.path.join(train_path_root, 'checkpoints')

eval_interval = 2
patience = 100
extra_text = f"Bi-directional LSTM, contrastive hierachy, fine-tune [{__file__}]"

# Training setup
batch_size = 32
num_epochs = 10000
lr = 0.001
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

feature_dim = 128
window_size = 16
window_stride = window_size//2
num_classes = 14
num_splits = 2

# LSTM component checkpoint
# mid_model = LSTMGestureModel(177, output_dim=21)
# mid_model_checkpoint_id = '20250408_221008'
# if mid_model_checkpoint_id:
#     mid_model_checkpoint_path = f'runs/ipn_classifiers/v1/{mid_model_checkpoint_id}/checkpoints/last.pth'
#     if os.path.exists(mid_model_checkpoint_path):
#         mid_model.load_state_dict(torch.load(mid_model_checkpoint_path, map_location=device))
#         print(f"Model loaded from {mid_model_checkpoint_path}")
#         # You might want to load optimizer state as well for resuming training
#         # optimizer.load_state_dict(torch.load(model_checkpoint_path.replace('best.pth', 'optimizer.pth')))
#     else:
#         print(f"Model file not found at {mid_model_checkpoint_path}")

# Model
model = LSTMGestureModel_Hierachical_Windowed(output_dim=feature_dim, window_size=16, window_stride=window_stride).to(device)
# if mid_model_checkpoint_id:
#     model.lstm = mid_model.lstm
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# criterion = nn.CrossEntropyLoss(ignore_index=-100) # Ignore padding index
criterion = supcon_loss
scheduler = ReduceLROnPlateau(optimizer, patience=patience)

# Checkpoint
model_checkpoint_name = None
# model_checkpoint_name = 'v1/20250408_221008'
# model_checkpoint_name = 'v1/20250414_150209'

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
timescale_min = .2
timescale_max = 2
clip_max_size = 1000

def aug(sequences, labels):
    augmented_sequences = []
    augmented_labels = []

    for seq, label in zip(sequences, labels):
        original_length = len(seq)

        timescale = random.uniform(timescale_min, timescale_max)
        target_length = int(original_length * timescale)

        indices = np.linspace(0, original_length - 1, target_length)
        L = len(indices)
        if L > clip_max_size:
            max_start = (L-clip_max_size) % clip_max_size
            start = np.random.randint(0,max(max_start,1))
        else:
            start = 0

        for k in range(L//clip_max_size+1):    
            indices = indices[start+clip_max_size*k:min(start+clip_max_size*(k+1),len(indices))]

            sampled_indices = np.floor(indices).astype(int)
            sampled_indices = np.clip(sampled_indices, 0, original_length - 1)

            augmented_sequences.append(seq[sampled_indices])
            augmented_labels.append(label[sampled_indices])

    len_aug = len(augmented_sequences)
    len_aug_divisible = len_aug//num_splits*num_splits
    if len_aug % num_splits != 0:
        assert len_aug > num_splits, f"Batch size (output batch size={len_aug} for this batch) should be > the number of splits {num_splits}"
        augmented_sequences, augmented_labels = augmented_sequences[:len_aug_divisible], augmented_labels[:len_aug_divisible]
    return augmented_sequences, augmented_labels

def post_fn(sequences:torch.Tensor, labels, *others):
    B,L,_,_ = sequences.shape
    return sequences.reshape(B,L,-1), labels, *others

# Data
train_loader = get_dataloader(dataset='ipn2', split='train', batch_size=batch_size, clip_max_size=clip_max_size*2, post_fn=post_fn, aug=aug)
val_loader = get_dataloader(dataset='ipn2', split='test', batch_size=batch_size, post_fn=post_fn)

# for sequences, labels in train_loader:
#     print(sequences.shape, labels.shape)

# exit(0)
# %%
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

def pool_labels(outputs, labels, soft=False):
    """
    Sample: 
        outputs: [B,L//W,D]
        labels: [B,L]
    Output:
        features: [B,L//W,D_lstm]
        soft_labels: [B,L//W,C]
        mask: [B,L//W]
    """
    mask = labels != -100 # [B,L]

    labels[~mask] = num_classes
    labels_onehot = F.one_hot(labels, num_classes=num_classes+1).float() # [B,L,C]
    
    labels_onehot = labels_onehot.permute(0,2,1)
    labels_onehot = F.avg_pool1d(labels_onehot, kernel_size=window_size, stride=window_stride)
    labels_onehot = labels_onehot.permute(0,2,1)

    # outputs_flat = outputs.view(-1, model.output_dim)
    # labels_flat = labels_onehot.view(-1,num_classes+1)
    mask = labels_onehot[...,-1] == 0
    labels_onehot = F.normalize(labels_onehot[...,:num_classes], dim=-1)
    return outputs, labels_onehot, mask

def batchify(data, batch_size):
    it = iter(data)
    while batch := list(islice(it, batch_size)):
        yield batch

# %%
def evaluate_model(centroids):
    """Evaluates the model on the validation set using global variables."""
    model.eval() # Set model to evaluation mode
    # total_loss = 0.0
    total_correct = 0
    total_samples = 0 # Total non-padded frames
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating", leave=False)
        for sequences, labels in progress_bar:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)

            outputs_flat, labels_flat, mask = pool_labels(outputs, labels)
            outputs_flat = outputs_flat.view(-1,feature_dim)
            labels_flat = labels_flat.view(-1,num_classes)
            mask = mask.view(-1)

            if not mask.any():
                continue

            feats = F.normalize(outputs_flat[mask], dim=-1) # [N,D]
            labls = labels_flat[mask] # [N,C]
            # print(feats.shape, labls.shape)

            sim = torch.mm(feats, centroids.T) # [N,D] * [D,C]
            preds = sim.argmax(dim=-1) # [N,]

            true_labls = labls.argmax(dim=-1) # [N,]

            total_correct += (preds == true_labls).sum().item()
            total_samples += len(true_labls)
            true_labels.extend(true_labls.detach().cpu())
            predicted_labels.extend(preds.detach().cpu())            

    # avg_loss = total_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
    avg_acc = (total_correct / total_samples) if total_samples > 0 else 0.0

    f1 = 0.0
    if true_labels:
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"Test Acc: {avg_acc}, F1: {f1}")

    return 0, avg_acc, f1

# %%
# Training
def train():
    best_val_loss = float('inf')
    centroids = torch.zeros(num_classes, feature_dim, device=device)
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_correct_train = 0
        total_samples_train = 0
        total_batches = 0

        new_centroids = torch.zeros_like(centroids, device=device)

        for i, (a_sequences, a_labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)):
            num_splits = len(a_sequences)//batch_size
            
            combined = list(zip(a_sequences, a_labels))
            random.shuffle(combined)
            batch = batchify(combined, batch_size=batch_size)
            
            for v in batch:
                sequences, labels = list(zip(*v))
                sequences = torch.from_numpy(np.array(sequences))
                labels = torch.from_numpy(np.array(labels))

                sequences, labels = sequences.to(device), labels.to(device)

                optimizer.zero_grad()
                
                outputs = model(sequences)
                
                outputs_flat, labels_flat, mask = pool_labels(outputs, labels)

                num_valid = mask.sum().item()
                if num_valid == 0:
                    continue # Skip batch if no valid labels

                labels_flat_c = labels_flat.argmax(dim=-1)
                loss = criterion(outputs_flat[mask].unsqueeze(1), labels_flat_c[mask])

                loss.backward()

                max_norm = 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

                total_train_loss += loss.item()
                total_batches += 1

                # Update centroids:
                B, L, D = outputs_flat.shape
                outputs_flat = F.normalize(outputs_flat, dim=-1)
                # print(labels_flat.shape, outputs_flat.shape)
                for k, v, m in zip(labels_flat.view(B*L,-1), outputs_flat.view(B*L,-1), mask.view(B*L,-1)): # [B,L//W,C]
                    # k [C,], v [D,]
                    new_centroids[k != 0] += k[k != 0].unsqueeze(1) * v.unsqueeze(0)

                # Evaluate Accuracy
                outputs_flat = outputs_flat.view(-1,feature_dim)
                labels_flat = labels_flat.view(-1,num_classes)
                mask = mask.view(-1)

                if not mask.any():
                    continue
                
                feats = F.normalize(outputs_flat[mask], dim=-1) # [N,D]
                labls = labels_flat[mask] # [N,D]

                sim = torch.mm(feats, centroids.T) # [N,D] * [D,C]
                preds = sim.argmax(dim=-1) # [N,D]

                true_labls = labls.argmax(dim=-1)

                total_correct_train += (preds == true_labls).sum().item()
                total_samples_train += len(true_labls)
        
        centroids = F.normalize(new_centroids, dim=-1)

        # Logs
        avg_train_loss = total_train_loss / total_batches if total_batches > 0 else 0.0
        avg_train_acc = (total_correct_train / total_samples_train) if total_samples_train > 0 else 0.0

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
        # writer.add_scalar('F1/train', avg_train_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        print(f"Train Loss: {avg_train_loss}, Acc: {avg_train_acc}")

        # LR Scheduler Step
        scheduler.step(avg_train_loss)

        # Evaluation
        if (epoch + 1) % eval_interval == 0:
            epoch_val_loss, epoch_val_acc, epoch_val_f1 = evaluate_model(centroids)
            writer.add_scalar('Loss/validation', epoch_val_loss, epoch)
            writer.add_scalar('Accuracy/validation', epoch_val_acc, epoch)
            writer.add_scalar('F1/validation', epoch_val_f1, epoch)

            # Save Best Model Checkpoint (based on validation loss)
            # if epoch_val_loss < best_val_loss:
            #     best_val_loss = epoch_val_loss
            #     best_model_path = os.path.join(checkpoint_dir, 'best.pth')
            #     # Save only model state dict for 'best' for simplicity
            #     torch.save(model.state_dict(), best_model_path)
            #     # Save centroids
            #     centroids_save_path = os.path.join(checkpoint_dir, 'centroids_best.pth')
            #     torch.save(centroids, centroids_save_path)
            #     print(f"*** Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.4f} ***")

        # Save Last Model Checkpoint
        last_model_path = os.path.join(checkpoint_dir, 'last.pth')
        torch.save(model.state_dict(), last_model_path)
        # Save centroids
        centroids_save_path = os.path.join(checkpoint_dir, 'centroids_last.pth')
        torch.save(centroids, centroids_save_path)

# %%
train() 


