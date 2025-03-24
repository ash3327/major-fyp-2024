# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HandEncoderV1(nn.Module):
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

class HandEncoder(nn.Module):
    def __init__(self, input_size=21 * 3, hidden_sizes=[256, 128, 64], embedding_size=128, dropout_rate=0.3, leaky_slope=0.01):
        """
        HandEncoder with dynamically adjustable hidden layers, dropout, and modern enhancements.
        
        Args:
            input_size (int): Size of the flattened input (default: 21 joints * 3 coords = 63).
            hidden_sizes (list): List of integers specifying the sizes of hidden layers.
                                Default: [256, 128, 64] for a simpler, effective MLP.
            embedding_size (int): Size of the final embedding output (default: 128).
            dropout_rate (float): Dropout probability for hidden layers (default: 0.5).
            leaky_slope (float): Slope for LeakyReLU activation (default: 0.01).
        """
        super().__init__()
        
        # Validate inputs
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer size")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        # Build the layer list dynamically
        layers = []
        prev_size = input_size
        
        # Add hidden layers based on hidden_sizes
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))  # Before activation, as per modern practice
            layers.append(nn.LeakyReLU(negative_slope=leaky_slope))  # Modern default
            layers.append(nn.Dropout(p=dropout_rate))  # Dropout after activation
            prev_size = size
        
        # Add the final embedding layer (no activation or dropout)
        layers.append(nn.Linear(prev_size, embedding_size))
        
        # Combine into a Sequential container
        self.fc = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Apply Kaiming (He) normal initialization to linear layers, tuned for LeakyReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use 'fan_in' mode and adjust for LeakyReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass: flatten input, compute embeddings, and normalize."""
        x = x.view(x.size(0), -1)  # Flatten input to [batch_size, input_size]
        embeddings = self.fc(x)    # Pass through the sequential layers
        return embeddings  # L2 normalize embeddings