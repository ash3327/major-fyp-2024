import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm

def evaluate_knn(train_embeddings, train_labels, test_embeddings, test_labels, k=5, have_class_outputs=False):
    """
    Evaluate the model using k-NN with cosine similarity.
    
    Args:
        train_embeddings (torch.Tensor): Training embeddings, shape [N_train, D].
        train_labels (torch.Tensor): Training labels, shape [N_train].
        test_embeddings (torch.Tensor): Test embeddings, shape [N_test, D].
        test_labels (torch.Tensor): Test labels, shape [N_test].
        k (int): Number of nearest neighbors to consider (default: 5).
    
    Returns:
        tuple: (accuracy, f1, per_class_accuracy, per_class_f1) - 
               Accuracy, weighted F1 score, per-class accuracy, and per-class F1 score as floats.
    """
    # Convert PyTorch tensors to NumPy arrays
    train_embeddings = train_embeddings.cpu().numpy()
    train_labels = train_labels.cpu().numpy()
    test_embeddings = test_embeddings.cpu().numpy()
    test_labels = test_labels.cpu().numpy()
    
    # Initialize k-NN with cosine similarity
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(train_embeddings)
    
    # Find the k nearest neighbors for each test embedding
    distances, indices = nn.kneighbors(test_embeddings)
    
    # Get labels of the nearest neighbors
    neighbor_labels = train_labels[indices]  # Shape: [N_test, k]
    
    # Predict labels by majority vote among k neighbors
    preds = np.array([np.bincount(labels).argmax() for labels in neighbor_labels])
    
    # Compute accuracy and F1 score
    accuracy = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average='weighted')
    
    if not have_class_outputs:
        return accuracy, f1
    
    # Compute per-class accuracy and F1 score
    unique_classes = np.unique(test_labels)
    per_class_accuracy = {}
    per_class_f1 = {}
    for cls in unique_classes:
        cls_indices = test_labels == cls
        cls_preds = preds[cls_indices]
        cls_labels = test_labels[cls_indices]
        per_class_accuracy[cls] = np.mean(cls_preds == cls_labels)
        per_class_f1[cls] = f1_score(cls_labels, cls_preds, average='weighted')
        
    return accuracy, f1, dict(
        per_class_accuracy=per_class_accuracy, 
        per_class_f1=per_class_f1,
        preds=preds
    )

def extract_embeddings(model, dataloader, device, output_joints=False):
    """
    Extract embeddings from a dataloader using the model.
    
    Args:
        model: The trained HandEncoder model.
        dataloader: DataLoader for the dataset.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
    
    Returns:
        tuple: (embeddings, labels) - Tensors of embeddings and corresponding labels.
    """
    model.eval()
    embeddings = []
    labels = []
    joints = []
    
    with torch.no_grad():
        for batch_labels, batch_joints in tqdm(dataloader):
            batch_joints = batch_joints.to(device)
            batch_labels = batch_labels.to(device)
            feats = model(batch_joints)  # [batch_size, embedding_dim]
            embeddings.append(feats)
            labels.append(batch_labels)
            if output_joints:
                joints.append(batch_joints)
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    if output_joints:
        joints = torch.cat(joints, dim=0)
        return embeddings, labels, joints
    return embeddings, labels