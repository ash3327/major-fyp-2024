# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HandEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(21 * 3, 128),  # Input: 21 joints * 3 coordinates
            nn.ReLU(),
            nn.Linear(128, 64)       # Output: 64-dimensional embedding
        )
    
    def forward(self, x):
        embeddings = self.fc(x.view(x.size(0), -1))  # Compute raw embeddings
        return F.normalize(embeddings, p=2, dim=1)   # L2 normalization to unit norm