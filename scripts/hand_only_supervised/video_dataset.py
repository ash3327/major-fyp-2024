import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

from tqdm import tqdm

# Define dataset class
class IPNGestureDataset(Dataset):
    def __init__(self, dataset=None, dataset_info=None, split='train', batch_size=32, with_id=False, **kwargs):
        if dataset_info is None:
            if dataset == 'ipn':
                dataset_info = dict(
                    data_path='data/kpts_flat/IPN_Hand/vid',
                    annot=dict(
                        train='data/raw/IPN_Hand/annotations/Annot_TrainList.txt',
                        test='data/raw/IPN_Hand/annotations/Annot_TestList.txt'
                    ),
                    classes=['D0X','B0A','B0B','G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G11'],
                    aggregated=False
                )
            elif dataset == 'ipn2':
                dataset_info = dict(
                    data_path='data/kpts_flat/IPN_Hand2/record_vid.npy',
                    labels_path='data/kpts/IPN_Hand2/record_vid.npy',
                    annot=dict(
                        train='data/raw/IPN_Hand/annotations/Annot_TrainList.txt',
                        test='data/raw/IPN_Hand/annotations/Annot_TestList.txt'
                    ),
                    classes=['D0X','B0A','B0B','G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G11'],
                    aggregated=True
                )
        self.data = []
        self.labels = []
        self.source_file_features = dict()
        self.source_file_labels = dict()
        self.batch_size = batch_size
        self.classes = dataset_info['classes']
        self.aggregated = dataset_info['aggregated']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.with_id = with_id

        assert split in dataset_info['annot'], f"Split {split} is not within the list of splits {dataset_info['annot'].keys()}"

        annot_file = dataset_info['annot'][split]
        err_count = 0
        minstart = 9999
        maxend = 0
        if self.aggregated:
            data_path = dataset_info['data_path']
            file_names_path = dataset_info['labels_path']
            if not os.path.exists(data_path):
                raise Exception(f'Error in reading file {data_path}')
            if not os.path.exists(file_names_path):
                raise Exception(f'Error in reading file {file_names_path}')
            features = np.load(data_path, allow_pickle=True)
            file_names = np.load(file_names_path, allow_pickle=True)[:,0]
            source_file_features = dict()
            source_file_ids = dict()
            # print(self.source_file_features.shape,file_names[:15])
            ### SPECIFIC TO IPN DATASET
            for f, l in tqdm(zip(features, file_names)):
                # self.source_files[str(l)] = (f, np.zeros(len(f)))
                file_name, frame_id = str(l).rsplit('/',1)
                # self.source_file_features[vid_map[frame_id]]
                if file_name not in source_file_features:
                    source_file_features[file_name] = list()
                    source_file_ids[file_name] = list()
                frame_id = int(frame_id.rsplit('_',1)[1].split('.')[0])
                source_file_features[file_name].append(f)
                source_file_ids[file_name].append(frame_id-1) # starts from 1.
                # print(file_name, frame_id, str(l))
                # break
            for file_name, features in source_file_features.items():
                # a[b] = a.copy() reorders dataset by the index referred in b.
                f = source_file_features[file_name] = np.array(source_file_features[file_name])
                f[source_file_ids[file_name]] = f.copy()
        
        with open(annot_file, 'r') as f:
            for line in tqdm(f):
                video_name, gesture, gesture_id, start_frame, end_frame, duration = line.strip().split(',')
                start_frame, end_frame = int(start_frame)-1, int(end_frame)-1
                
                if video_name not in self.source_file_features:
                    if self.aggregated:
                        features = source_file_features[video_name]
                    else:
                        features_path = os.path.join(dataset_info['data_path'],f"{video_name}.npy")
                        if not os.path.exists(features_path):
                            print(f'Error in reading file {features_path}')
                            err_count += 1
                        features = np.load(features_path, allow_pickle=True)
                    self.source_file_features[video_name] = features
                    self.source_file_labels[video_name] = np.zeros(len(features))
                    # features = features[start_frame:end_frame]
                # if end_frame > len(self.source_files[video_name][0]):
                #     print(f"NOT MATCH: {video_name} have {len(self.source_files[video_name][0])} frames but frame interval [{start_frame},{end_frame}) is queried (class {gesture})")
                self.source_file_labels[video_name][start_frame:end_frame] = self.class_to_idx[gesture]
                minstart = min(minstart,start_frame)
                maxend = max(maxend,end_frame)

        self.data, self.labels = list(self.source_file_features.values()), list(self.source_file_labels.values())
        print(minstart,maxend)
        print(f"Loaded {len(self.data)} samples from {dataset_info['data_path']}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.with_id:
            return torch.from_numpy(self.data[idx].astype(np.float32)), \
                torch.from_numpy(self.labels[idx].astype(np.int64)), \
                idx
        return torch.from_numpy(self.data[idx].astype(np.float32)), \
            torch.from_numpy(self.labels[idx].astype(np.int64))

# Nothing
def nothing(*x):
    return x

# Collate function for padding
def collate_fn(batch,aug=None):
    if aug is None:
        aug = nothing
    sequences, labels, *others = zip(*batch)
    sequences, labels = aug(sequences, labels)
    sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return sequences, labels, *others

# Get loader
def get_dataloader(dataset='ipn', split='train', batch_size=32, post_fn=None, aug=None, **kwargs):
    if post_fn is None:
        post_fn = nothing
    if aug is None:
        aug = nothing
    dataset = IPNGestureDataset(dataset=dataset, split=split, batch_size=batch_size, **kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: post_fn(*collate_fn(x,aug=aug)))
    return loader

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
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Set axis limits
    bound = 1
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])
    ax.set_zlim([-bound, bound])

    # Set viewing angle
    ax.view_init(elev=30, azim=45)

def visualize_video_with_labels(features, predicted_labels, true_labels, dataset):
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
        predicted_labels = predicted_labels.detach().cpu().numpy()
        true_labels = true_labels.detach().cpu().numpy()
    features = features.reshape(-1,59,3)

    num_frames = len(features)
    print(num_frames, len(predicted_labels), len(true_labels))
    if len(predicted_labels) != num_frames or len(true_labels) != num_frames:
        print("Error: Length of label lists must match the number of frames.")
        return

    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
    ax = plt.subplot(gs[0], projection='3d')
    ax_labels = plt.subplot(gs[1])
    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
    frame_slider = Slider(slider_ax, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
    text_ax = fig.add_axes([0.1, 0.95, 0.8, 0.03]) # For displaying labels
    text_ax.axis('off')
    label_text = text_ax.text(0.01, 0.5, '', transform=text_ax.transAxes, fontsize=12)

    def update(frame_idx):
        nonlocal ax
        ax.clear()

        pose = features[frame_idx,:17].copy()
        zeros = pose == 0
        if pose.ndim == 3:
            pose[...,0] -= 0.5
        else:
            pose[...,0] -= 0.5
        pose[zeros] = 0

        hands = features[frame_idx,17:].copy()
        zeros = hands == 0
        hands[...,0] -= 0.5
        hands[...,0] *= -1
        hands[zeros] = 0
        # print(pose.shape, hands.shape)

        plot_pose(ax, pose)
        for i, hand in enumerate([hands[:21],hands[21:]]):
            if np.any(hand):
                plot_hand_skeleton(ax, hand, title=f'Hand {i+1}')

        ax.view_init(elev=90, azim=90)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        bound = 1
        ax.set_xlim([-bound/2, bound/2])
        ax.set_ylim([0, bound])
        ax.set_zlim([-bound/2, bound/2])    

        predicted_label_idx = predicted_labels[frame_idx]
        true_label_idx = true_labels[frame_idx]
        predicted_label_name = dataset.classes[predicted_label_idx] if 0 <= predicted_label_idx < len(dataset.classes) else "Unknown"
        true_label_name = dataset.classes[true_label_idx] if 0 <= true_label_idx < len(dataset.classes) else "Unknown"

        label_text.set_text(f"Frame: {frame_idx} | Predicted: {predicted_label_name} | True: {true_label_name}")
        ax.set_title(f"Predicted Label: {predicted_label_name} vs True: {true_label_name}\n(Frame {frame_idx})")

        fig.canvas.draw_idle()

    frame_slider.on_changed(update)

    label_data = np.array([predicted_labels, true_labels])
    ax_labels.imshow(label_data, aspect='auto', cmap='viridis')
    ax_labels.set_yticks([0, 1])
    ax_labels.set_yticklabels(['Predicted ID', 'True ID'])
    ax_labels.set_xticks(np.arange(num_frames))
    ax_labels.set_xticklabels(np.arange(num_frames))
    ax_labels.set_xlabel("Frame Number")

    plt.tight_layout(rect=[0, 0.05, 1, 0.92]) # Adjust layout to make space for slider and text
    plt.show()