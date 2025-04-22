import os
import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from training.contrastive.model import HandEncoder, HandEncoder_6DOF
from training.contrastive.model_gat import HandEncoderGAT3dof, HandEncoderGAT6dof, graph_transform
from training.contrastive.model_gcn import HandEncoderGCN3dof, HandEncoderGCN6dof

use_cosine = True

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
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
        cluster_embeddings = F.normalize(torch.from_numpy(cluster_embeddings), dim=-1).numpy()
        cluster_means[label] = np.mean(cluster_embeddings, axis=0)
        cluster_means[label] = F.normalize(torch.from_numpy(cluster_means[label]), dim=-1).numpy()
        distances = np.dot(cluster_embeddings, cluster_means[label])
        if len(cluster_embeddings) == 0:
            cluster_sizes[label] = {
                'count': 0,
                'avg_distance': np.nan,
                'std_distance': np.nan,
                'min_distance': np.nan,
                'max_distance': np.nan
            }
        else:
            cluster_sizes[label] = {
                'count': len(cluster_embeddings),
                'avg_distance': np.nanmean(distances),
                'std_distance': np.nanstd(distances),
                'min_distance': np.nanmin(distances),
                'max_distance': np.nanmax(distances)
            }
    labels_list = sorted(label_to_idx.keys())
    means_matrix = np.array([cluster_means[label] for label in labels_list])
    means_matrix = F.normalize(torch.from_numpy(means_matrix), dim=-1).numpy()
    distances = np.dot(means_matrix, means_matrix.T)

    # Calculate interclass and intraclass statistics
    intraclass_distances = []
    for label in labels_list:
        intraclass_distances.append(cluster_sizes[label]['avg_distance'])
    interclass_distances = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            interclass_distances.append(distances[i, j])

    intraclass_mean = np.nanmean(intraclass_distances)
    intraclass_std = np.nanstd(intraclass_distances)
    interclass_mean = np.nanmean(interclass_distances)
    interclass_std = np.nanstd(interclass_distances)

    print(f"Intraclass Mean: {intraclass_mean:.4f}, Intraclass Std: {intraclass_std:.4f}")
    print(f"Interclass Mean: {interclass_mean:.4f}, Interclass Std: {interclass_std:.4f}")

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
        f.write(f"Intraclass Mean: {intraclass_mean:.4f}, Intraclass Std: {intraclass_std:.4f}\n")
        f.write(f"Interclass Mean: {interclass_mean:.4f}, Interclass Std: {interclass_std:.4f}\n")
    print(f"Cluster analysis saved to {analysis_filepath}")

    return cluster_sizes, distances, labels_list

def plot_distance_heatmap(distances, labels_list, cluster_sizes, output_dir, use_max=False, title="Inter-cluster Distances"):
    """Plot heatmap of inter-cluster distances with diagonal replaced by cluster statistics and save the plot."""
    fig, ax = plt.subplots(figsize=(12, 10))
    distances_with_diag = distances.copy()
    for i, label in enumerate(labels_list):
        distances_with_diag[i, i] = cluster_sizes[label]['min_distance' if use_cosine else 'max_distance'] if use_max else cluster_sizes[label]['avg_distance']
    cax = ax.imshow(distances_with_diag, cmap='viridis')
    fig.colorbar(cax, label='Distance')
    ax.set_xticks(range(len(labels_list)))
    ax.set_xticklabels(labels_list, rotation=45, ha='right')
    ax.set_yticks(range(len(labels_list)))
    ax.set_yticklabels(labels_list)
    if len(labels_list) < 20:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint_path = None

    # === model ===
    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250402191732/checkpoints/best.pth'
    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250402210020/checkpoints/best.pth'
    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250403104741/checkpoints/best.pth'
    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250405195023/checkpoints/best.pth' # supcon, no aug, HandEncoder model.
    # model = HandEncoder().to(device)

    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250405224236/checkpoints/best.pth' # supcon, no aug, HandEncoderGCN3dof model.
    # model = HandEncoderGCN3dof().to(device=device)

    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250405224236/checkpoints/best.pth' # supcon, no aug, HandEncoderGCN3dof model.
    # model = HandEncoderGCN3dof().to(device=device)

    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250406000135/checkpoints/best.pth' # supcon, no aug, HandEncoderGCN3dof model.
    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250406164630/checkpoints/best.pth' # sup+unsup, linear curriculum scheduling, HandEncoderGCN3dof model.
    # model = HandEncoderGCN6dof().to(device=device)
    
    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/best/20250408122338/checkpoints/last.pth' # unsup all, linear curriculum scheduling, HandEncoderGCN6dof model.
    # model = HandEncoderGCN6dof(do_norm_before_input=False).to(device=device)
    
    # Load model
    model = HandEncoder().to(device)

    model_checkpoint_path = 'runs/hand_contrastive_learning/v4/20250401140023/checkpoints/best.pth'
    model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250402210020/checkpoints/best.pth'
    model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250403104741/checkpoints/best.pth'
    model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250405195023/checkpoints/best.pth' # supcon, no aug, HandEncoder model.


    embedding_dim = 128
    # model = HandEncoderGAT6dof(embedding_size=embedding_dim, do_norm_after_input=False, fn=graph_transform_complex).to(device)
    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250405195023/checkpoints/best.pth'

    ## Base
    # model = HandEncoderGAT6dof(embedding_size=embedding_dim, do_norm_after_input=False).to(device)
    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured_2/v1/20250405231456/checkpoints/best.pth'

    # model = HandEncoderGCN6dof(embedding_size=embedding_dim, do_norm_after_input=False).to(device)
    # model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250406000135/checkpoints/best.pth'
    # model_checkpoint_path = 'runs/stor/preliminary/20250406000135/checkpoints/best.pth' 

    ## Curriculum
    model = HandEncoderGAT6dof(embedding_size=embedding_dim, do_norm_after_input=True).to(device)
    model_checkpoint_path = 'runs/stor/curriculum/20250410184054/checkpoints/best.pth' # pink
    # model_checkpoint_path = 'runs/stor/curriculum/20250410184054/checkpoints/last.pth' # pink
    # model_checkpoint_path = 'runs/stor/curriculum/20250406185135/checkpoints/best.pth' # grey
    # model_checkpoint_path = 'runs/stor/curriculum/20250406185135/checkpoints/last.pth' # grey

    # model = HandEncoderGCN6dof(embedding_size=embedding_dim, do_norm_after_input=True).to(device)
    # model_checkpoint_path = 'runs/stor/curriculum/20250407201859/checkpoints/best.pth' # blue
    # model_checkpoint_path = 'runs/stor/curriculum/20250407201859/checkpoints/last.pth' # blue

    
    if os.path.exists(model_checkpoint_path):
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
        print(f"Model loaded from {model_checkpoint_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint_path}")

    # == extra info ==
    if model_checkpoint_path is not None:
        ckpt_id = model_checkpoint_path.rsplit("/checkpoints/", 1)[0].rsplit('/', 1)[1]
        if 'last' in model_checkpoint_path:
            ckpt_id += '_last'
    batch_size = 256
    dataset = LabelledHandDataset(dataset_name=dataset_name, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # from training.contrastive.preprocess import get_3dof

    # class Do3DoF:
    #     def eval(self):
    #         pass
    #     def forward(self, x):
    #         return x.view(x.shape[0],-1)
    #     def __call__(self, x):
    #         out = get_3dof(x)
    #         return out.reshape(out.shape[0], -1)
    # model = Do3DoF()
    # ckpt_id = '3dof'

    # from training.contrastive.preprocess import get_6dof

    # class Do6DoF:
    #     def eval(self):
    #         pass
    #     def forward(self, x):
    #         return x.view(x.shape[0],-1)
    #     def __call__(self, x):
    #         out = get_6dof(x)
    #         return out.reshape(out.shape[0], -1)
    # model = Do6DoF()
    # ckpt_id = '6dof/cosinesim'

    # class DoNothing:
    #     def eval(self):
    #         pass
    #     def forward(self, x):
    #         return x
    #     def __call__(self, x):
    #         return x.view(x.shape[0],-1)
    # model = DoNothing()
    # ckpt_id = 'none/cosinesim/withlabel'
    
    embeddings, labels = extract_embeddings(model, dataloader, device)
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()

    output_dir = f"runs/eval_cos/{ckpt_id}/{dataset.dataset_name}-{dataset.split}/"
    visualize_labelled_pca(embeddings, labels, dataset.label_to_idx, output_dir)
    visualize_labelled_tsne(embeddings, labels, dataset.label_to_idx, output_dir)
    visualize_labelled_umap(embeddings, labels, dataset.label_to_idx, output_dir)
    cluster_sizes, distances, labels_list = analyze_clusters(embeddings, labels, dataset.label_to_idx, output_dir)
    plot_distance_heatmap(distances, labels_list, cluster_sizes, output_dir, use_max=False)
    plot_distance_heatmap(distances, labels_list, cluster_sizes, output_dir, use_max=True)
