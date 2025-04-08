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
data_dir, dataset, subfolders, output_dir, dyn, is_video, infodict, *_ = get_info('lsa64')
split = None

splits = list(subfolders.keys())

inpt = split
if split is None or split not in splits:
    inpt = input(f"Choose from the splits: {splits}\n>>> ")\
        if len(splits) > 1 else splits[0]

if inpt in splits:
    split = inpt
    print(f"Fetching dataset with split {split}...")
else:
    print("Invalid split. Terminating...")
    exit(1)

dataset = 'lsa64_raw_2'
kpt_root = os.path.join(output_dir, dataset, split)

# Cell 3: Analyze Annotations
gesture_dict = {
    1: "Opaque", 2: "Red", 3: "Green", 4: "Yellow", 5: "Bright", 6: "Light-blue", 7: "Colors", 8: "Pink",
    9: "Women", 10: "Enemy", 11: "Son", 12: "Man", 13: "Away", 14: "Drawer", 15: "Born", 16: "Learn",
    17: "Call", 18: "Skimmer", 19: "Bitter", 20: "Sweet milk", 21: "Milk", 22: "Water", 23: "Food", 24: "Argentina",
    25: "Uruguay", 26: "Country", 27: "Last name", 28: "Where", 29: "Mock", 30: "Birthday", 31: "Breakfast", 32: "Photo",
    33: "Hungry", 34: "Map", 35: "Coin", 36: "Music", 37: "Ship", 38: "None", 39: "Name", 40: "Patience",
    41: "Perfume", 42: "Deaf", 43: "Trap", 44: "Rice", 45: "Barbecue", 46: "Candy", 47: "Chewing-gum", 48: "Spaghetti",
    49: "Yogurt", 50: "Accept", 51: "Thanks", 52: "Shut down", 53: "Appear", 54: "To land", 55: "Catch", 56: "Help",
    57: "Dance", 58: "Bathe", 59: "Buy", 60: "Copy", 61: "Run", 62: "Realize", 63: "Give", 64: "Find"
}

def get_annotations():
    annotations = []
    for i in range(1,65):
        for j in range(1,11):
            for k in range(1,6):
                annotations.append({
                    'video_name': f"{i:03d}_{j:03d}_{k:03d}",
                    'gesture': gesture_dict[i],
                    'gesture_id': i
                })
    return pd.DataFrame(annotations)

# Load train and test annotations
annot = get_annotations()

print("Set statistics:")
print(f"Number of sequences: {len(annot)}")
print("\nGesture distribution:")
print(annot['gesture'].value_counts())

# Cell for plot_hand_skeleton function
def plot_pose(ax, pose):
    if pose.ndim == 3:
        ax.scatter(pose[:,0],pose[:,1],pose[:,2])
    else:
        ax.scatter(pose[:,0],pose[:,1],np.zeros_like(pose[:,0]))

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

# Cell for visualization
# Load sample features
def load_sequence_features(video_name):
    """Load features for a sequence from the extracted features file."""
    features_path = os.path.join(kpt_root, f"{video_name}.npy")
    if not os.path.exists(features_path):
        return None
    
    features = np.load(features_path, allow_pickle=True)
    return features

# Get a sample sequence from training set
sample_seq = annot.iloc[0]
features = load_sequence_features(sample_seq['video_name'])

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
        pose = features[frame][3]['pose'].copy()
        zeros = pose == 0
        if pose.ndim == 3:
            # pose = pose[...,(1,0,2)]
            pose[...,0] -= 0.5
        else:
            # pose = pose[...,(1,0)]
            pose[...,0] -= 0.5
        pose[zeros] = 0

        hands = features[frame][3]['hands'].copy()#[...,(1,0,2)]
        zeros = hands == 0
        hands[...,0] -= 0.5
        hands[...,0] *= -1
        hands[zeros] = 0

        # print(hands[1])
        # Plot each hand
        plot_pose(ax1, pose)
        for i, hand in enumerate(hands):
            ax = ax1 if i == 0 else ax2
            if np.any(hand):
                plot_hand_skeleton(ax1, hand, title=f'Hand {i+1} - Frame {frame}')
        
        # Maintain same view for both plots
        for ax in [ax1, ax2]:
            # ax.view_init(elev=30, azim=45)
            ax.view_init(elev=90, azim=90)
            ax.set_title(f"{sample_seq['gesture']} | {sample_seq['gesture_id']} (Frame {frame})")
            # Set axis limits
            bound = 1
            ax.set_xlim([-bound/2, bound/2])
            ax.set_ylim([0, bound])
            ax.set_zlim([-bound/2, bound/2])
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
    print(f"Duration: {len(features)} frames")
    
    plt.tight_layout()
    plt.show()
    
    return anim  # Return animation object to prevent garbage collection

# Update the visualization code
if features is not None:
    # Create and show animation
    anim = animate_sequence(features, sample_seq)