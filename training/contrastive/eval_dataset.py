"""
Evaluation Script:
    python training/contrastive/eval_model.py
"""

import os
import sys
sys.path.append('.')

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.hand_only_supervised.hand_supervised_dataset import LabelledHandDataset
from evals import extract_embeddings, evaluate_knn
from model import HandEncoder

# Configuration
model_checkpoint_path = 'runs/hand_contrastive_learning/v4/20250401140023/checkpoints/best.pth'
model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250402210020/checkpoints/best.pth'
model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250403104741/checkpoints/best.pth'

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

# class DoNothing:
#     def eval(self):
#         pass
#     def forward(self, x):
#         return x
#     def __call__(self, x):
#         return x.view(x.shape[0],-1)
# model = DoNothing()

# Extract embeddings
train_embeddings, train_labels = extract_embeddings(model, dataloader_sup, device)
test_embeddings, test_labels, test_joints = extract_embeddings(model, dataloader_test, device, output_joints=True)

# Perform k-NN evaluation
accuracy, f1, dumps = evaluate_knn(train_embeddings, train_labels, test_embeddings, test_labels, k=k_neighbors, have_class_outputs=True)

# Print results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Class accu: {dumps['per_class_accuracy']}")
print(f"Class f1: {dumps['per_class_f1']}")

# preds = dumps['preds']

# def handle_blank_landmarks(landmarks, labels, predictions):
#     """
#     Check if landmarks are all zeros (blank) and display the label and prediction.
#     Count the number of blank predictions and blank labels (class = 26).

#     Args:
#         landmarks (np.ndarray): Array of landmark values.
#         labels (np.ndarray): Array of true labels.
#         predictions (np.ndarray): Array of predicted labels.
#     """
#     blank_mask = torch.all(landmarks == 0, dim=(1, 2))  # Check if all landmarks are zeros
#     blank_label_count = 0
#     blank_prediction_count = 0

#     for i, is_blank in enumerate(blank_mask):
#         if is_blank:
#             print(f"Blank landmarks detected at index {i}:")
#             print(f"  Label: {labels[i]}")
#             print(f"  Prediction: {predictions[i]}")
#             if labels[i] == 26:  # Count blank labels
#                 blank_label_count += 1
#             if predictions[i] == 26:  # Count blank predictions
#                 blank_prediction_count += 1

#     print(f"Total blank labels (class = 26): {blank_label_count}")
#     print(f"Total blank predictions (class = 26): {blank_prediction_count}")

# handle_blank_landmarks(test_joints, test_labels, preds)
