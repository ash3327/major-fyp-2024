import os
import sys
sys.path.append('.')
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

from scripts.hand_only_supervised.hand_supervised_dataset import LabelledHandDataset
from model import HandEncoder
from evals import extract_embeddings

def visualize_labelled_pca(embeddings, labels, label_to_idx, title="PCA visualization of labelled embeddings"):
    """Visualize embeddings using PCA with different colors for each label."""
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Get unique labels and create color map
    unique_labels = sorted(label_to_idx.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot points for each label
    for label, color in zip(unique_labels, colors):
        label_idx = label_to_idx[label]
        mask = labels == label_idx
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[color], alpha=0.6, label=label)
    
    plt.title(title)
    plt.xlabel(f'First PC (explained variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'Second PC (explained variance: {pca.explained_variance_ratio_[1]:.3f})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    
    return embeddings_2d, pca

def analyze_clusters(embeddings, labels, label_to_idx):
    """Analyze cluster sizes and distances."""
    # Calculate cluster means
    cluster_means = {}
    cluster_sizes = {}
    
    for label, idx in label_to_idx.items():
        mask = labels == idx
        cluster_embeddings = embeddings[mask]
        cluster_means[label] = np.mean(cluster_embeddings, axis=0)
        
        # Calculate distances from points to cluster center
        distances = cdist([cluster_means[label]], cluster_embeddings)[0]
        cluster_sizes[label] = {
            'count': len(cluster_embeddings),
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    
    # Calculate inter-cluster distances
    labels_list = sorted(label_to_idx.keys())
    means_matrix = np.array([cluster_means[label] for label in labels_list])
    distances = cdist(means_matrix, means_matrix)
    
    return cluster_sizes, distances, labels_list

def plot_distance_heatmap(distances, labels_list, title="Inter-cluster Distances"):
    """Plot heatmap of inter-cluster distances."""
    plt.figure(figsize=(12, 10))
    plt.imshow(distances, cmap='viridis')
    plt.colorbar(label='Distance')
    
    # Add labels
    plt.xticks(range(len(labels_list)), labels_list, rotation=45, ha='right')
    plt.yticks(range(len(labels_list)), labels_list)
    
    # Add distance values as text
    for i in range(len(labels_list)):
        for j in range(len(labels_list)):
            plt.text(j, i, f'{distances[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if distances[i, j] > distances.mean() else 'black')
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Configuration
    model_checkpoint_path = 'runs/hand_contrastive_learning/v4/20250401140023/checkpoints/best.pth'
    batch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = LabelledHandDataset(dataset_name='lexset', split='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    model = HandEncoder().to(device)
    if os.path.exists(model_checkpoint_path):
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
        print(f"Model loaded from {model_checkpoint_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint_path}")
    
    # Extract embeddings
    embeddings, labels = extract_embeddings(model, dataloader, device)
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Visualize PCA
    embeddings_2d, pca = visualize_labelled_pca(embeddings, labels, dataset.label_to_idx)
    
     # Analyze clusters
    cluster_sizes, distances, labels_list = analyze_clusters(embeddings, labels, dataset.label_to_idx)
    
    # Print cluster statistics
    print("\nCluster Statistics:")
    print("==================")
    for label in labels_list:
        stats = cluster_sizes[label]
        print(f"\nLabel: {label}")
        print(f"  Sample count: {stats['count']}")
        print(f"  Average distance to center: {stats['avg_distance']:.4f}")
        print(f"  Standard deviation: {stats['std_distance']:.4f}")
        print(f"  Minimum distance to center: {stats['min_distance']:.4f}")
        print(f"  Maximum distance to center: {stats['max_distance']:.4f}")
    
    # Plot distance heatmap
    plot_distance_heatmap(distances, labels_list)
    
    # Find and print most similar and most different clusters
    min_dist = np.inf
    max_dist = -np.inf
    min_pair = max_pair = None
    
    for i, label1 in enumerate(labels_list):
        for j, label2 in enumerate(labels_list[i+1:], i+1):
            dist = distances[i,j]
            if dist < min_dist:
                min_dist = dist
                min_pair = (label1, label2)
            if dist > max_dist:
                max_dist = dist
                max_pair = (label1, label2)
    
    print("\nMost Similar Clusters:")
    print(f"{min_pair[0]} <-> {min_pair[1]}: {min_dist:.4f}")
    
    print("\nMost Different Clusters:")
    print(f"{max_pair[0]} <-> {max_pair[1]}: {max_dist:.4f}")