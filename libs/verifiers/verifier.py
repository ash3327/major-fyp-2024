import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def extract_features(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()  # Set to evaluation mode
    features_list = []
    labels_list = []

    with torch.no_grad(): 
        for inputs, labels, _ in tqdm(dataloader):
            inputs = inputs.to(device)
            _, last_layer_input = model(inputs)  # Get last layer input
            features_list.append(last_layer_input.cpu().numpy())  # Store features
            labels_list.append(labels)  # Store labels

    return np.concatenate(features_list), np.concatenate(labels_list)

def plot_tsne(features, labels_id):       
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Plotting the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_id, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title('t-SNE of Last Layer Input Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

def plot_similarity_matrix(all_embeddings, all_labels):
    # Extract unique class labels
    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)

    # Initialize the cosine similarity matrix
    cosine_sim_matrix = np.zeros((4, num_classes, num_classes))

    # Compute cosine similarities between every pair of classes
    for i, lab_i in enumerate(unique_labels):
        for j, lab_j in enumerate(unique_labels):
            # Extract embeddings for class i and class j
            embeddings_class_i = np.array(all_embeddings[all_labels==lab_i])
            embeddings_class_j = np.array(all_embeddings[all_labels==lab_j])
            
            # Compute pairwise cosine similarities
            similarity = cosine_similarity(embeddings_class_i, embeddings_class_j)
            
            # Take the mean similarity as the representative similarity between class i and class j
            min_similarity = np.min(similarity)
            mean_similarity = np.mean(similarity)
            max_similarity = np.max(similarity)
            cosine_sim_matrix[:, i, j] = [mean_similarity, min_similarity, max_similarity, max_similarity]

    cosine_sim_matrix[3, np.arange(num_classes), np.arange(num_classes)] = cosine_sim_matrix[1, np.arange(num_classes), np.arange(num_classes)]

    # Plotting the similarity matrix as a heatmap
    for mat, label in zip(cosine_sim_matrix, ["Mean", "Min", "Max", "Comparison"]):
        plt.figure(figsize=(10, 8))
        plt.imshow(mat, cmap='viridis', interpolation='nearest')
        plt.colorbar(label=f'{label} Cosine Similarity')
        plt.xticks(range(num_classes), unique_labels, rotation=90)
        plt.yticks(range(num_classes), unique_labels)
        plt.title(f'Cosine Similarity Matrix Between Classes ({label})')
        plt.xlabel('Class B')
        plt.ylabel('Class A')
        plt.tight_layout()
        plt.show()