"""
Evaluation Script:
    python training/contrastive/eval_model.py
"""

# train_supcon_structured.py
import sys
sys.path.append('.')  # Ensure imports work from the project root

import os
import torch
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime

from scripts.fake_data.contrastive_data_dataset import HandPoseContrastiveDataset
from scripts.hand_only_supervised.hand_supervised_dataset import LabelledHandDataset, CombinedLabelledHandDataset
from training.contrastive.augments import augment as augment_hand, augment_pair as augment_handpair
from training.contrastive.augments import generate_random_rotation_object, generate_random_scaling_vector, apply_transform, vectorized_apply_transform, normalize  # Make sure path is correct

from training.contrastive.model import HandEncoder, HandEncoder_6DOF
from training.contrastive.model_gat import HandEncoderGAT3dof, HandEncoderGAT6dof, graph_transform, graph_transform_complex
from training.contrastive.model_gcn import HandEncoderGCN3dof, HandEncoderGCN6dof
from training.contrastive.losses import info_nce_loss, supcon_loss, info_nce_loss_from_matrix
from training.contrastive.evals import extract_embeddings, evaluate_knn

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import math

# Configuration
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

model_checkpoint_path = 'runs/hand_contrastive_learning/v4/20250401140023/checkpoints/best.pth'
model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250402210020/checkpoints/best.pth'
model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250403104741/checkpoints/best.pth'
model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250405195023/checkpoints/best.pth' # supcon, no aug, HandEncoder model.


embedding_dim = 128
# model = HandEncoderGAT6dof(embedding_size=embedding_dim, do_norm_after_input=False, fn=graph_transform_complex).to(device)
# model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250405195023/checkpoints/best.pth'

# model = HandEncoderGAT6dof(embedding_size=embedding_dim, do_norm_after_input=False).to(device)
# model_checkpoint_path = 'runs/hand_contrastive_learning_structured_2/v1/20250405231456/checkpoints/best.pth'

# model = HandEncoderGCN6dof(embedding_size=embedding_dim, do_norm_after_input=False).to(device)
# model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250406000135/checkpoints/best.pth'

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

# from training.contrastive.preprocess import get_6dof

# class Do6DoF:
#     def eval(self):
#         pass
#     def forward(self, x):
#         return x.view(x.shape[0],-1)
#     def __call__(self, x):
#         return torch.stack([get_6dof(sample) for sample in x], dim=0).view(x.shape[0], -1)
# model = Do6DoF()

# Extract embeddings
train_embeddings, train_labels = extract_embeddings(model, dataloader_sup, device)
test_embeddings, test_labels, test_joints = extract_embeddings(model, dataloader_test, device, output_joints=True)

train_embeddings = F.normalize(train_embeddings, dim=-1)
test_embeddings = F.normalize(test_embeddings, dim=-1)

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
