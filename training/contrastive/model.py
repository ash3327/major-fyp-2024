# model.py
import torch
import torch.nn as nn

class HandEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(21 * 3, 128),  # Input: 21 joints * 3 coordinates
            nn.ReLU(),
            nn.Linear(128, 64)       # Output: 64-dimensional embedding
        )
    
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))  # Flatten input and pass through layers