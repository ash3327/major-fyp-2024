# Cell 1: Imports
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from tqdm.notebook import tqdm

sys.path.append('.')
from scripts.datasets.prepare_dataset import get_info

# Cell 2: Load Dataset Information
# Get dataset info
data_dir, dataset, subfolders, output_dir, dyn, is_video, infodict, *_ = get_info('IPN_Hand')

# Load metadata
metadata_path = os.path.join(data_dir, 'IPN_Hand', 'annotations', 'metadata.csv')
metadata_df = pd.read_csv(metadata_path)

print("Dataset Information:")
print(f"Data directory: {data_dir}")
print(f"Is video dataset: {is_video}")
print(f"\nMetadata shape: {metadata_df.shape}")
print("\nMetadata columns:")
print(metadata_df.columns.tolist())

# Cell 3: Analyze Annotations
def read_annotation_file(filepath):
    annotations = []
    with open(filepath, 'r') as f:
        for line in f:
            video_name, gesture, gesture_id, start_frame, end_frame, duration = line.strip().split(',')
            annotations.append({
                'video_name': video_name,
                'gesture': gesture,
                'gesture_id': int(gesture_id),
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'duration': int(duration)
            })
    return pd.DataFrame(annotations)

# Load train and test annotations
train_annot = read_annotation_file(os.path.join(data_dir, 'IPN_Hand', 'annotations', 'Annot_TrainList.txt'))
test_annot = read_annotation_file(os.path.join(data_dir, 'IPN_Hand', 'annotations', 'Annot_TestList.txt'))

print("Training set statistics:")
print(f"Number of sequences: {len(train_annot)}")
print("\nGesture distribution:")
print(train_annot['gesture'].value_counts())

print("\nTest set statistics:")
print(f"Number of sequences: {len(test_annot)}")
print("\nGesture distribution:")
print(test_annot['gesture'].value_counts())

# Cell for plot_hand_skeleton function
def plot_hand_skeleton(ax, joints, color='b', title=None):
    """Plot hand skeleton with connections between joints."""
    # Define connections between joints
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    
    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, marker='o')
    
    # Plot connections with different colors for each finger
    finger_colors = ['r', 'g', 'b', 'c', 'm']
    finger_ranges = [(0,4), (5,8), (9,12), (13,16), (17,20)]
    
    for (start_idx, end_idx), color in zip(finger_ranges, finger_colors):
        relevant_connections = [conn for conn in connections 
                              if conn[0] >= start_idx and conn[1] <= end_idx 
                              or conn[0] == 0 and conn[1] >= start_idx and conn[1] <= end_idx]
        
        for start, end in relevant_connections:
            ax.plot([joints[start, 0], joints[end, 0]],
                   [joints[start, 1], joints[end, 1]],
                   [joints[start, 2], joints[end, 2]], 
                   c=color, linewidth=2)
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Set axis limits
    bound = 1
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])
    ax.set_zlim([-bound, bound])

    # Set viewing angle
    ax.view_init(elev=30, azim=45)

# Cell for visualization
# Load sample features
def load_sequence_features(video_name, start_frame, end_frame):
    """Load features for a sequence from the extracted features file."""
    features_path = os.path.join('data/kpts/IPN_Hand/vid', f"{video_name}.npy")
    if not os.path.exists(features_path):
        return None
    
    features = np.load(features_path, allow_pickle=True)
    return features[start_frame:end_frame]

# Get a sample sequence from training set
sample_seq = train_annot.iloc[17]
features = load_sequence_features(sample_seq['video_name'], 
                                sample_seq['start_frame'], 
                                sample_seq['end_frame'])

# if features is not None:
#     print(f"Sequence: {sample_seq['video_name']}")
#     print(f"Gesture: {sample_seq['gesture']} (ID: {sample_seq['gesture_id']})")
#     print(f"Duration: {sample_seq['duration']} frames")
    
#     # Create figure
#     fig = plt.figure(figsize=(15, 5))
    
#     # Plot both hands
#     hands = features[0][3]['hands']  # First frame, hand landmarks
#     for i, hand in enumerate(hands):
#         if np.any(hand):  # If hand is detected
#             ax = fig.add_subplot(1, 2, i+1, projection='3d')
#             plot_hand_skeleton(ax, hand, title=f'Hand {i+1}')
    
#     plt.tight_layout()
#     plt.show()
# print(np.all(features == 0, axis=1))
empty_frames = []

for i, frame in enumerate(features):
    hands = frame[3]['hands']
    if np.all(hands == 0):
        empty_frames.append(i)

print()
print(f"File: {sample_seq['video_name']}")
print(f"Number of empty frames: {len(empty_frames)}")
print(f"Empty frames: {empty_frames}")

def animate_sequence(features, sample_seq):
    """Create an interactive animation of hand poses."""
    # Create figure and subplots
    fig = plt.figure(figsize=(15, 7))  # Made taller to accommodate slider
    ax1 = plt.subplot(121, projection='3d')
    ax2 = plt.subplot(122, projection='3d')
    
    # Add slider
    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])  # [left, bottom, width, height]
    frame_slider = Slider(
        ax=slider_ax,
        label='Frame',
        valmin=0,
        valmax=len(features)-1,
        valinit=0,
        valstep=1
    )
    
    def update(frame):
        """Update function for animation."""
        ax1.clear()
        ax2.clear()
        
        # Get hand data for current frame
        hands = features[frame][3]['hands']
        
        # Plot each hand
        for i, hand in enumerate(hands):
            ax = ax1 if i == 0 else ax2
            if np.any(hand):
                plot_hand_skeleton(ax, hand, title=f'Hand {i+1} - Frame {frame}')
        
        # Maintain same view for both plots
        for ax in [ax1, ax2]:
            ax.view_init(elev=30, azim=45)
            ax.set_box_aspect([1,1,1])
        
        fig.canvas.draw_idle()
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        update,
        frames=len(features),
        interval=50,  # 50ms between frames
        repeat=True
    )
    
    # Connect slider to animation
    def slider_update(val):
        anim.pause()
        update(int(val))
    
    frame_slider.on_changed(slider_update)
    
    # Add play/pause button
    button_ax = plt.axes([0.85, 0.02, 0.1, 0.03])
    play_button = Button(button_ax, 'Play/Pause')
    
    def play_pause(event):
        if anim.running:
            anim.pause()
        else:
            anim.resume()
    
    play_button.on_clicked(play_pause)
    
    # Print sequence info
    print(f"Sequence: {sample_seq['video_name']}")
    print(f"Gesture: {sample_seq['gesture']} (ID: {sample_seq['gesture_id']})")
    print(f"Duration: {sample_seq['duration']} frames")
    
    plt.tight_layout()
    plt.show()
    
    return anim  # Return animation object to prevent garbage collection

# Update the visualization code
if features is not None:
    # Create and show animation
    anim = animate_sequence(features, sample_seq)