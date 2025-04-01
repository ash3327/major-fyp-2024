"""
Evaluation Script:
    python training/contrastive/eval_model.py
"""

import os
import sys
sys.path.append('.')
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.hand_only_supervised.hand_supervised_dataset import LabelledHandDataset
from evals import extract_embeddings, evaluate_knn
from model import HandEncoder

# Configuration
model_checkpoint_path = 'runs/hand_contrastive_learning/v4/20250401140023/checkpoints/best.pth'
batch_size = 256
k_neighbors = 5  # Number of neighbors for k-NN

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
dataset_sup = LabelledHandDataset(dataset_name='lexset', split='train')
dataloader_sup = DataLoader(dataset_sup, batch_size=batch_size, shuffle=False)

dataset_test = LabelledHandDataset(dataset_name='lexset', split='test')
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Load model
model = HandEncoder().to(device)
if os.path.exists(model_checkpoint_path):
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    print(f"Model loaded from {model_checkpoint_path}")
else:
    raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint_path}")

# Evaluation
print("Evaluating the model using k-NN...")
model.eval()

# Extract embeddings
train_embeddings, train_labels = extract_embeddings(model, dataloader_sup, device)
test_embeddings, test_labels = extract_embeddings(model, dataloader_test, device)

# Perform k-NN evaluation
accuracy, f1 = evaluate_knn(train_embeddings, train_labels, test_embeddings, test_labels, k=k_neighbors)

# Print results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
