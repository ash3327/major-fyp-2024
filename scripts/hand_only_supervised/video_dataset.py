import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

from tqdm import tqdm

# Define dataset class
class IPNGestureDataset(Dataset):
    def __init__(self, dataset=None, dataset_info=None, split='train', batch_size=32):
        if dataset_info is None:
            if dataset == 'ipn':
                dataset_info = dict(
                    data_path='data/kpts_flat/IPN_Hand/vid',
                    annot=dict(
                        train='data/raw/IPN_Hand/annotations/Annot_TrainList.txt',
                        test='data/raw/IPN_Hand/annotations/Annot_TestList.txt'
                    ),
                    classes=['D0X','B0A','B0B','G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G11']
                )
        self.data = []
        self.labels = []
        self.source_files = dict()
        self.batch_size = batch_size
        self.classes = dataset_info['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        assert split in dataset_info['annot'], f"Split {split} is not within the list of splits {dataset_info['annot'].keys()}"

        annot_file = dataset_info['annot'][split]
        err_count = 0
        minstart = 9999
        maxend = 0
        with open(annot_file, 'r') as f:
            for line in tqdm(f):
                video_name, gesture, gesture_id, start_frame, end_frame, duration = line.strip().split(',')
                start_frame, end_frame = int(start_frame), int(end_frame)
                if video_name not in self.source_files:
                    features_path = os.path.join(dataset_info['data_path'],f"{video_name}.npy")
                    if not os.path.exists(features_path):
                        print(f'Error in reading file {features_path}')
                        err_count += 1
                    features = np.load(features_path, allow_pickle=True)
                    self.source_files[video_name] = (features, np.zeros(len(features)))
                    # features = features[start_frame:end_frame]
                self.source_files[video_name][1][start_frame:end_frame] = self.class_to_idx[gesture]
        self.data, self.labels = list(zip(*list(self.source_files.values())))
        print(minstart,maxend)
        print(f"Loaded {len(self.data)} samples from {dataset_info['data_path']}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].astype(np.float32)), \
            torch.from_numpy(self.labels[idx].astype(np.int64))
    
# Collate function for padding
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return sequences, labels

def nothing(*x):
    return x

# Get loader
def get_dataloader(dataset='ipn', split='train', batch_size=32, post_fn=None):
    if post_fn is None:
        post_fn = nothing
    dataset = IPNGestureDataset(dataset=dataset, split=split, batch_size=batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: post_fn(*collate_fn(x)))
    return loader