"""
Without Semi-hard mining
"""

import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def train_one_epoch(model, dataloader, criterion, optimizer, num_epochs, device='cuda', epoch=0):
    model.train()
    running_loss = 0.0
    
    for (anchors, positives, negatives), (label, neg_label) in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        # Move to device
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
        
        optimizer.zero_grad()
        
        # Get embeddings
        _, anchor_embeddings = model(anchors)
        _, positive_embeddings = model(positives)
        _, negative_embeddings = model(negatives)
        
        # Compute loss
        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    return epoch_loss

def test_one_epoch(model, dataloader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for (anchors, positives, negatives), (label, neg_label) in tqdm(dataloader, desc="Testing"):
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            
            _, anchor_embeddings = model(anchors)
            _, positive_embeddings = model(positives)
            _, negative_embeddings = model(negatives)
            
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            running_loss += loss.item()
    
    test_loss = running_loss / len(dataloader)
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss