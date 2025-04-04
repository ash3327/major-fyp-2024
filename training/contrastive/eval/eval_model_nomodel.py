import os
import sys
sys.path.append('.')
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

import argparse

from scripts.hand_only_supervised.hand_supervised_dataset import LabelledHandDataset
from training.contrastive.evals import extract_embeddings, evaluate_knn
from training.contrastive.model import HandEncoder

def save_plot(fig, output_dir, filename):
    """Helper function to save a plot to a file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    print(f"Saved plot to {filepath}")
    plt.close(fig)

def _visualize_2d_embeddings(embeddings_2d, labels, label_to_idx, title, output_dir, filename):
    """Helper function to visualize 2D embeddings with class labels over cluster centers and save the plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    unique_labels = sorted(label_to_idx.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    cluster_centers = {}

    for label, color in zip(unique_labels, colors):
        label_idx = label_to_idx[label]
        mask = labels == label_idx
        cluster_embeddings = embeddings_2d[mask]
        ax.scatter(cluster_embeddings[:, 0], cluster_embeddings[:, 1],
                   c=[color], alpha=0.3, label=label)
        # Calculate cluster center
        cluster_center = np.mean(cluster_embeddings, axis=0)
        cluster_centers[label] = cluster_center
        # Add text annotation for the cluster center
        ax.text(cluster_center[0], cluster_center[1], label, fontsize=12,
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    ax.grid(True)
    save_plot(fig, output_dir, filename)

def visualize_labelled_pca(embeddings, labels, label_to_idx, output_dir, title="PCA visualization of labelled embeddings"):
    """Visualize embeddings using PCA with different colors for each label and save the plot."""
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    _visualize_2d_embeddings(embeddings_2d, labels, label_to_idx, title, output_dir, "pca_visualization.png")
    return embeddings_2d, pca

def visualize_labelled_tsne(embeddings, labels, label_to_idx, output_dir, title="t-SNE visualization of labelled embeddings"):
    """Visualize embeddings using t-SNE with different colors for each label and save the plot."""
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    _visualize_2d_embeddings(embeddings_2d, labels, label_to_idx, title, output_dir, "tsne_visualization.png")
    return embeddings_2d, tsne

def visualize_labelled_umap(embeddings, labels, label_to_idx, output_dir, title="UMAP visualization of labelled embeddings"):
    """Visualize embeddings using UMAP with different colors for each label and save the plot."""
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    _visualize_2d_embeddings(embeddings_2d, labels, label_to_idx, title, output_dir, "umap_visualization.png")
    return embeddings_2d, reducer

def analyze_clusters(embeddings, labels, label_to_idx, output_dir):
    """Analyze cluster sizes and distances and store results as a text file."""
    cluster_means = {}
    cluster_sizes = {}
    for label, idx in label_to_idx.items():
        mask = labels == idx
        cluster_embeddings = embeddings[mask]
        cluster_means[label] = np.mean(cluster_embeddings, axis=0)
        distances = cdist([cluster_means[label]], cluster_embeddings)[0]
        cluster_sizes[label] = {
            'count': len(cluster_embeddings),
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    labels_list = sorted(label_to_idx.keys())
    means_matrix = np.array([cluster_means[label] for label in labels_list])
    distances = cdist(means_matrix, means_matrix)

    # Save cluster analysis to a text file
    os.makedirs(output_dir, exist_ok=True)
    analysis_filepath = os.path.join(output_dir, "cluster_analysis.txt")
    with open(analysis_filepath, "w") as f:
        f.write("Cluster Analysis:\n")
        for label in labels_list:
            f.write(f"Label: {label}\n")
            for key, value in cluster_sizes[label].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        f.write("Inter-cluster Distances:\n")
        for i, label1 in enumerate(labels_list):
            for j, label2 in enumerate(labels_list):
                f.write(f"Distance between {label1} and {label2}: {distances[i, j]:.4f}\n")
            f.write("\n")
    print(f"Cluster analysis saved to {analysis_filepath}")

    return cluster_sizes, distances, labels_list

def plot_distance_heatmap(distances, labels_list, cluster_sizes, output_dir, use_max=False, title="Inter-cluster Distances"):
    """Plot heatmap of inter-cluster distances with diagonal replaced by cluster statistics and save the plot."""
    fig, ax = plt.subplots(figsize=(12, 10))
    distances_with_diag = distances.copy()
    for i, label in enumerate(labels_list):
        distances_with_diag[i, i] = cluster_sizes[label]['max_distance'] if use_max else cluster_sizes[label]['avg_distance']
    cax = ax.imshow(distances_with_diag, cmap='viridis')
    fig.colorbar(cax, label='Distance')
    ax.set_xticks(range(len(labels_list)))
    ax.set_xticklabels(labels_list, rotation=45, ha='right')
    ax.set_yticks(range(len(labels_list)))
    ax.set_yticklabels(labels_list)
    for i in range(len(labels_list)):
        for j in range(len(labels_list)):
            ax.text(j, i, f'{distances_with_diag[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if distances_with_diag[i, j] > distances_with_diag.mean() else 'black')
    ax.set_title(title)
    plt.tight_layout()
    save_plot(fig, output_dir, f"heatmap_{'max' if use_max else 'avg'}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model and visualize embeddings.")
    parser.add_argument('-d', '--dataset', type=str, default='lexset', help='Name of the dataset (default: lexset)')
    parser.add_argument('-s', '--split', type=str, default=None, help='Dataset split (default: train)')
    # parser.add_argument('-c', '--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    args = parser.parse_args()

    # model_checkpoint_path = args.checkpoint
    dataset_name = args.dataset
    split = args.split

    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250402191732/checkpoints/best.pth'
    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250402210020/checkpoints/best.pth'
    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250403104741/checkpoints/best.pth'

    # ckpt_id = model_checkpoint_path.rsplit("/checkpoints/", 1)[0].rsplit('/', 1)[1]
    batch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LabelledHandDataset(dataset_name=dataset_name, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # model = HandEncoder().to(device)
    # if os.path.exists(model_checkpoint_path):
    #     model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    #     print(f"Model loaded from {model_checkpoint_path}")
    # else:
    #     raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint_path}")
    class DoNothing:
        def eval(self):
            pass
        def forward(self, x):
            return x
        def __call__(self, x):
            return x.view(x.shape[0],-1)

    model = DoNothing()
    embeddings, labels = extract_embeddings(model, dataloader, device)
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()

    output_dir = f"runs/eval/none/{dataset.dataset_name}-{dataset.split}/"
    visualize_labelled_pca(embeddings, labels, dataset.label_to_idx, output_dir)
    visualize_labelled_tsne(embeddings, labels, dataset.label_to_idx, output_dir)
    visualize_labelled_umap(embeddings, labels, dataset.label_to_idx, output_dir)
    cluster_sizes, distances, labels_list = analyze_clusters(embeddings, labels, dataset.label_to_idx, output_dir)
    plot_distance_heatmap(distances, labels_list, cluster_sizes, output_dir, use_max=False)
    plot_distance_heatmap(distances, labels_list, cluster_sizes, output_dir, use_max=True)
