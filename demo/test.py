import os
import numpy as np
import cv2
import torch
from gesture_recognition import HandGestureRecognizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.lines import Line2D
import pickle
from tqdm import tqdm
from scipy.signal import savgol_filter, find_peaks
from matplotlib.animation import FuncAnimation

# Define cache directory
CACHE_DIR = "temp/"
os.makedirs(CACHE_DIR, exist_ok=True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="kpt_contrastive_1", help="Model to use for gesture recognition, advised list under demo/configs/model_configs/ (file name only, without .yaml suffix)")
args = parser.parse_args()

# Caching function
def cache_embeddings(cache_file, generate_fn, *args):
    """
    Check for cached embeddings, or generate and save them if not present.

    Args:
        cache_file (str): Path to the cache file.
        generate_fn (function): Function to generate the embeddings if not cached.
        *args: Arguments to pass to the `generate_fn`.

    Returns:
        tuple: (latent_embeddings, labels)
    """
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        print("Generating embeddings...")
        result = generate_fn(*args)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Embeddings cached to {cache_file}.")
        return result

# Your embedding generation function
def process_image_sequence(file_paths, config_path=None):
    """
    Process a sequence of image files and return their latent embeddings and labels.
    """
    if config_path is None:
        config_path = f'configs/model_configs/{args.model}.yaml'
    
    print(f"\nUsing model config: {config_path}\n")

    recognizer = HandGestureRecognizer(config_path)
    latent_embeddings, labels = [], []

    for image_path in tqdm(file_paths):
        image = cv2.imread(image_path)
        result = recognizer.process_frame(image, return_landmarks=True)
        if isinstance(result, tuple):
            pred, emb, _ = result
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            latent_embeddings.append(emb)
            labels.append(pred)

    latent_embeddings = np.array(latent_embeddings) if latent_embeddings else None
    return latent_embeddings, labels

def visualize_embeddings_with_interaction(latent_embeddings, images):
    """
    Interactive visualization for embeddings with PCA, t-SNE, and distances within a window.
    """
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, random_state=42)

    pca_result = pca.fit_transform(latent_embeddings)
    tsne_result = tsne.fit_transform(latent_embeddings)

    # Default window size
    W = 5

    def calculate_windowed_distances(window_size):
        """Calculate windowed distances: norm(latent[t-w] - latent[t+w]) for all t."""
        distances = np.zeros(len(latent_embeddings))
        for t in range(len(latent_embeddings)):
            t_minus_w = max(0, t - window_size)
            t_plus_w = min(len(latent_embeddings) - 1, t + window_size)
            distances[t] = np.linalg.norm(latent_embeddings[t_minus_w] - latent_embeddings[t_plus_w])
        smoothed_distances = savgol_filter(distances, window_length=11, polyorder=3)
        local_minima_indices, _ = find_peaks(-smoothed_distances)
        return smoothed_distances, local_minima_indices

    # Compute initial distances
    distances, local_minima_indices = calculate_windowed_distances(W)

    # Set up the plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={"height_ratios": [1, 4]})
    plt.subplots_adjust(bottom=0.25, top=0.9)

    # Distance plot
    axs[0, 0].remove()  # Remove unused subplot
    axs[0, 2].remove()  # Remove unused subplot
    ax_distance = axs[0, 1]
    line_distance, = ax_distance.plot(distances, color="blue", label="Windowed Distance")
    
    # ax_distance.scatter(local_minima_indices, distances[local_minima_indices], color="red", label="Local Minima")

    vertical_line = Line2D([0, 0], [0, max(distances)], color="red", linestyle="--", lw=2)
    ax_distance.add_line(vertical_line)
    ax_distance.set_xlim(0, len(latent_embeddings) - 1)
    ax_distance.set_ylim(0, max(distances) * 1.1)
    ax_distance.set_title("Latent Space Windowed Distance")
    ax_distance.set_xlabel("Frame Index")
    ax_distance.set_ylabel("Distance")

    # PCA plot
    scatter_pca = axs[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], c=np.arange(len(latent_embeddings)), cmap="viridis", s=30)
    current_point_pca = axs[1, 0].scatter([pca_result[0, 0]], [pca_result[0, 1]], c="red", s=50, marker="o")
    axs[1, 0].set_title("PCA Embeddings")
    axs[1, 0].set_xlabel("PCA 1")
    axs[1, 0].set_ylabel("PCA 2")

    # t-SNE plot
    scatter_tsne = axs[1, 2].scatter(tsne_result[:, 0], tsne_result[:, 1], c=np.arange(len(latent_embeddings)), cmap="viridis", s=30)
    current_point_tsne = axs[1, 2].scatter([tsne_result[0, 0]], [tsne_result[0, 1]], c="red", s=50, marker="o")
    axs[1, 2].set_title("t-SNE Embeddings")
    axs[1, 2].set_xlabel("t-SNE 1")
    axs[1, 2].set_ylabel("t-SNE 2")

    # Image display
    ax_image = axs[1, 1]
    ax_image.axis("off")
    img_display = ax_image.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))

    # Add slider for frame
    ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])  # [left, bottom, width, height]
    slider = Slider(ax_slider, "Frame", 0, len(latent_embeddings) - 1, valinit=0, valstep=1)

    def update_plot(val):
        """Update plots and image based on slider value."""
        index = int(slider.val)

        # Update the vertical line position
        vertical_line.set_xdata([index, index])

        # Update PCA and t-SNE plots
        current_point_pca.set_offsets([pca_result[index]])
        current_point_tsne.set_offsets([tsne_result[index]])

        # Update the displayed image
        img_display.set_data(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))

        fig.canvas.draw_idle()

    # Bind slider
    slider.on_changed(update_plot)

    # Initialize distance plot with the first frame's data
    update_plot(None)

    # Auto-increment slider using FuncAnimation
    def auto_increment(frame):
        current_val = slider.val
        next_val = (current_val + 5) % len(latent_embeddings)  # Loop back to 0 at the end
        slider.set_val(next_val)

    ani = FuncAnimation(fig, auto_increment, interval=100)  # Update every 100ms

    plt.show()

# Main script
if __name__ == "__main__":
    # Path to your images
    base_path = "D:\\kht3327\\_Projects\\Major FYP\\proj\\data\\raw\\B2Counting"
    file_paths = [f"SK_color_{i}.png" for i in range(1500)]
    file_paths = [os.path.join(base_path, f) for f in file_paths]

    # Cache file
    cache_file = os.path.join(CACHE_DIR, f"embeddings_cache_{args.model}.pkl")

    # Generate or load cached embeddings
    latent_embeddings, _ = cache_embeddings(cache_file, process_image_sequence, file_paths)
    latent_embeddings = latent_embeddings.squeeze(1)
    images = [cv2.imread(path) for path in file_paths]

    # Visualize embeddings
    visualize_embeddings_with_interaction(latent_embeddings, images)
