import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append('.')

from training.temporal.model import LSTMGestureModel_Hierachical_Windowed
from scripts.hand_only_supervised.video_dataset import get_dataloader

# Configuration
feature_dim = 128
window_size = 16 
window_stride = window_size//2
num_classes = 14
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pool_labels(outputs, labels):
    """Same pooling function from training"""
    mask = labels != -100
    labels[~mask] = num_classes
    labels_onehot = F.one_hot(labels, num_classes=num_classes+1).float()
    
    labels_onehot = labels_onehot.permute(0,2,1)
    labels_onehot = F.avg_pool1d(labels_onehot, kernel_size=window_size, stride=window_stride)
    labels_onehot = labels_onehot.permute(0,2,1)

    mask = labels_onehot[...,-1] == 0
    labels_onehot = F.normalize(labels_onehot[...,:num_classes], dim=-1)
    return outputs, labels_onehot, mask

def post_fn(sequences, labels, *others):
    B,L,_,_ = sequences.shape
    return sequences.reshape(B,L,-1), labels, *others

def generate_centroids(model_checkpoint_id):
    # Load model
    model = LSTMGestureModel_Hierachical_Windowed(
        output_dim=feature_dim,
        window_size=window_size, 
        window_stride=window_stride
    ).to(device)
    
    checkpoint_path = f'runs/ipn_classifiers/v1/{model_checkpoint_id}/checkpoints/last.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Get validation loader
    train_loader = get_dataloader(dataset='ipn2', split='train', batch_size=batch_size, post_fn=post_fn)

    # Initialize centroids tensor
    centroids = torch.zeros(num_classes, feature_dim, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    # Compute features and accumulate
    with torch.no_grad():
        for sequences, labels in tqdm(train_loader, desc="Computing centroids"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            outputs_flat, labels_flat, mask = pool_labels(outputs, labels)
            
            outputs_flat = outputs_flat.view(-1, feature_dim)
            labels_flat = labels_flat.view(-1, num_classes)
            mask = mask.view(-1)

            if not mask.any():
                continue

            feats = F.normalize(outputs_flat[mask], dim=-1)
            labls = labels_flat[mask]

            # Accumulate features per class
            for k in range(num_classes):
                class_mask = labls[:, k] > 0
                if class_mask.any():
                    centroids[k] += feats[class_mask].sum(dim=0)
                    class_counts[k] += class_mask.sum()

    # Average and normalize
    valid_classes = class_counts > 0
    centroids[valid_classes] /= class_counts[valid_classes].unsqueeze(1)
    centroids = F.normalize(centroids, dim=-1)

    # Save centroids
    save_path = f'runs/ipn_classifiers/v1/{model_checkpoint_id}/checkpoints/centroids_last.pth'
    torch.save(centroids, save_path)
    print(f"Saved centroids to {save_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python generate_centroids.py <checkpoint_id>")
        sys.exit(1)
        
    model_checkpoint_id = sys.argv[1]
    generate_centroids(model_checkpoint_id)
