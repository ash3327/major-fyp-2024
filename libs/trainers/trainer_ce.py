import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_one_epoch(model, dataloader, criterion, optimizer, num_epochs:int, device='cuda', epoch:int=0):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    # Iterate over the dataloader
    for inputs, _, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients
        outputs, _ = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimization step
        
        # Calculate the number of correct predictions
        _, preds = torch.max(outputs, 1)  # Get the predicted class
        running_corrects += torch.sum(preds == labels.data)  # Update correct predictions
        running_loss += loss.item() * inputs.size(0)  # Accumulate loss

    epoch_loss = running_loss / len(dataloader.dataset)  # Average loss for the epoch
    epoch_accuracy = running_corrects.double() / len(dataloader.dataset)  # Accuracy

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    return epoch_loss, epoch_accuracy

def test_one_epoch(model, dataloader, criterion, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_corrects = 0  # To track the number of correct predictions

    # Disable gradient calculation for testing
    with torch.no_grad():
        # Iterate over the dataloader
        for inputs, _, labels in tqdm(dataloader, desc="Testing"):
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            
            # Calculate the number of correct predictions
            _, preds = torch.max(outputs, 1)  # Get the predicted class
            running_corrects += torch.sum(preds == labels.data)  # Update correct predictions
            
            running_loss += loss.item() * inputs.size(0)  # Accumulate loss

    # Calculate average loss and accuracy
    epoch_loss = running_loss / len(dataloader.dataset)  # Average loss for the epoch
    epoch_accuracy = running_corrects.double() / len(dataloader.dataset)  # Accuracy

    print(f'Test Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    return epoch_loss, epoch_accuracy