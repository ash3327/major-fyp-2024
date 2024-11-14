"""
Without Semi-hard mining
"""

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class LatentClassifier:
    def __init__(self, method='knn', n_clusters=27):
        self.method = method
        self.n_clusters = n_clusters
        
    def fit(self, latents, labels=None):
        self.latents = latents.cpu().numpy()
        self.labels = labels.cpu().numpy()
        
        if self.method == 'knn':
            self.clf = KNeighborsClassifier(n_neighbors=5)
            self.clf.fit(latents, labels)
            
        elif self.method == 'kmeans':
            self.clf = KMeans(n_clusters=self.n_clusters)
            self.clusters = self.clf.fit_predict(latents)
            
        elif self.method == 'gmm':
            self.clf = GaussianMixture(n_components=self.n_clusters)
            self.clusters = self.clf.fit_predict(latents)
            
    def predict(self, query_latents):
        if self.method == 'knn':
            return self.clf.predict(query_latents)
            
        elif self.method == 'cosine':
            # Compute similarities
            similarities = cosine_similarity(query_latents, self.latents)
            # Get most similar indices
            most_similar_idx = np.argmax(similarities, axis=1)
            # Return corresponding labels
            return self.labels[most_similar_idx]
            
        elif self.method in ['kmeans', 'gmm']:
            return self.clf.predict(query_latents)
        
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
    """
    This method tests only on TRIPLET losses, and uses TRIPLET DATASET.
    """
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

def fit_classifier(model, dataloader, classifier:LatentClassifier, device='cuda'):
    """
    This method uses the IMAGE DATASET.
    """
    model.eval()  # Set the model to evaluation mode
    embeddings_list = []
    labels_list = []
    
    # Disable gradient calculation for testing
    with torch.no_grad():
        # Iterate over the dataloader
        for inputs, _, labels in tqdm(dataloader, desc="Fitting"):
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)

            _, embeddings = model(inputs)  # Forward pass
            embeddings_list.append(embeddings)
            labels_list.append(labels)

    # Convert embeddings_list and labels_list to a format compatible with cosine_similarity(query_latents, self.latents)
    embeddings_list = torch.cat(embeddings_list, dim=0)
    labels_list = torch.cat(labels_list, dim=0)

    # Fit the classifier
    classifier.fit(embeddings_list, labels_list)

    print('Finished fitting.')

def test_celoss_one_epoch(model, dataloader, classifier:LatentClassifier, device='cuda'):
    """
    This method uses the IMAGE DATASET.
    """
    model.eval()  # Set the model to evaluation mode
    running_corrects = 0  # To track the number of correct predictions

    # Disable gradient calculation for testing
    with torch.no_grad():
        # Iterate over the dataloader
        for inputs, _, labels in tqdm(dataloader, desc="Testing"):
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels

            _, embeddings = model(inputs)  # Forward pass
            preds = classifier.predict(embeddings.cpu().numpy())
            
            # Calculate the number of correct predictions
            running_corrects += torch.sum(preds == labels.data)  # Update correct predictions

    # Calculate average loss and accuracy
    epoch_accuracy = running_corrects.double() / len(dataloader.dataset)  # Accuracy

    print(f'Test Accuracy: {epoch_accuracy:.4f}')
    return epoch_accuracy