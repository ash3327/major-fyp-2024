import os
import sys
sys.path.append('.')

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from scripts.datasets.prepare_dataset import get_info  # Assuming this provides dataset metadata

class LabelledHandDataset(Dataset):
    """
    NOTE: Currently only handles static hand, one-hand datasets including lexset, senz3d and handshape.
    """
    def __init__(self, dataset_name, split=None, augment=None, max_num_hands=2, ignore_flat=True, normalize_to_wrist=True):
        """
        Dataset class to extract labels and hand landmarks from .npy files.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'asl_alphabet', 'lsa64_raw').
            split (str): Dataset split (e.g., 'train', 'test').
        """
        self.dataset_name = dataset_name
        self.split = split
        self.data = []  # List of (label_idx, hand_landmarks) tuples
        self.label_to_idx = {}  # Mapping from string labels to integer indices

        data_dir, dataset, subfolders, output_dir, dyn, is_video, infodict, *_ = get_info(self.dataset_name)
        self.data_dir = data_dir
        self.dataset_name = dataset
        self.subfolders = subfolders

        self.max_num_hands = max_num_hands
        if dataset == 'synthetic-asl-alphabet':
            self.max_num_hands = 1

        self.ignore_flat = ignore_flat
        self.normalize_to_wrist = normalize_to_wrist

        splits = list(subfolders.keys())
        if self.split is None and len(splits) == 1:
            self.split = splits[0]
        elif self.split not in splits:
            raise Exception(f'Split {self.split} is not in the list of splits for this dataset: {splits}.')

        self.is_video = is_video
        self.annotations = None
        self._load_annotation_list()
        self._get_labels_from_annotated_doc(infodict)

        self.augment = augment

        self.load_data(infodict)

    def _load_annotation_list(self):
        labels = None

        match self.dataset_name:
            case 'synthetic-asl-alphabet':
                labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['Blank']
            case 'senz3d_dataset':
                labels = [f'G{i}' for i in range(1, 11)]
            case 'ph2014-handshape':
                labels = ["1", "2", "3", "3_hook", "4", "5", "6", "7", "8", "a",
                          "b", "b_nothumb", "b_thumb", "cbaby", "obaby", "by", "c",
                          "d", "e", "f", "f_open", "fly", "fly_nothumb", "g", "h", "h_hook",
                          "h_thumb", "i", "jesus", "k", "l_hook", "middle", "m", "n", "o",
                          "index", "index_flex", "index_hook", "pincet", "ital", "ital_thumb",
                          "ital_nothumb", "ital_open", "r", "s", "write", "spoon", "t", "v",
                          "v_flex", "v_hook", "v_thumb", "w", "y", "ae", "ae_thumb", "pincet_double",
                          "obaby_double", "m2", "jesus_thumb"]

        if labels:
            self.label_to_idx = {label: v for v, label in enumerate(labels)}
            

    def _get_label_from_filename(self, filename, infodict=None):
        """
        Extract the label from the filename based on dataset conventions.

        Args:
            filename (str): The filename from features[0].

        Returns:
            str: The extracted label.
        """
        # This is a placeholder; adjust based on actual filename conventions
        # Example: For 'asl_alphabet', filename might be 'A/001.png' -> label 'A'
        # For video datasets, filename might be the video name indicating the gesture
        
        if self.dataset_name == 'senz3d_dataset':
            label = filename.rsplit('/',1)[0].split('/')[1]
        elif self.dataset_name == 'ph2014-handshape':
            fold_names = dict(
                test="images/",
                train="/work/cv2/koller/features/danish_nz_ph2014/hand.20151016/data/joint/danish_nz_ph2014"
            )
            assert self.split in fold_names, f"Split {self.split} not in fold_names"
            fold_name = fold_names[self.split]
            label = fold_name+''.join(char if char.isascii() else '*' for char in filename)
            if label in self.annotations:
                label = self.annotations[label]
        else:
            label = filename.rsplit('/',1)[0]
        
        # Map label to integer index
        if label not in self.label_to_idx:
            self.label_to_idx[label] = len(self.label_to_idx)
        return self.label_to_idx[label]
    
    def _get_labels_from_annotated_doc(self, infodict=None):
        """
        Extract labels from an annotated document.
        """
        if infodict is not None and 'annotations' in infodict and self.split in infodict['annotations']:
            annotation_path = os.path.join(self.data_dir,self.dataset_name,infodict['annotations'][self.split])
            if not os.path.exists(annotation_path):
                raise FileNotFoundError(f"Could not find {annotation_path}")
            self.annotations = dict()
            with open(annotation_path, 'r') as f:
                for line in f:
                    # Assuming the format is: "path/to/image.png class_label"
                    path, class_label = line.strip().split(' ')
                    self.annotations[path] = class_label
        pass

    def load_data(self, infodict=None):
        """
        Load .npy files and extract labels and hand landmarks.
        """
        if not self.is_video:
            # Non-video dataset: single .npy file
            npy_file = os.path.join("data/kpts", self.dataset_name, f"record_{self.split}.npy")
            if not os.path.exists(npy_file):
                raise FileNotFoundError(f"Could not find {npy_file}")
            data = np.load(npy_file, allow_pickle=True)
            self._process_data(data, infodict)
        else:
            # Video dataset: multiple .npy files in split directory
            root_path = os.path.join("data/kpts", self.dataset_name, self.split)
            if not os.path.exists(root_path):
                raise FileNotFoundError(f"Could not find directory {root_path}")
            for root, _, files in os.walk(root_path):
                for file in tqdm(files, desc="Fetching data"):
                    if file.lower().endswith('.npy'):
                        npy_file = os.path.join(root, file)
                        # Use the video filename as the label source
                        label_idx = self._get_label_from_filename(file, infodict)
                        data = np.load(npy_file, allow_pickle=True)
                        self._process_video_data(data, label_idx)

    def _process_data(self, data, infodict=None):
        """
        Process data for non-video datasets.

        Args:
            data: Loaded .npy data (list of feature tuples).
        """
        for item in tqdm(data, desc="Fetching data"):
            filename = item[0]  # features[0] is the label source
            num_hands = item[2]  # Number of hands in the sample
            hands = item[3]['hands']  # Array of shape (2, 21, 3)
            label_idx = self._get_label_from_filename(filename, infodict)
            self._extract_hands(label_idx, hands, num_hands)

    def _process_video_data(self, data, video_label_idx):
        """
        Process data for video datasets, assigning the video label to all frames.

        Args:
            data: Loaded .npy data (list of per-frame feature tuples).
            video_label_idx (int): Integer label index for the entire video.
        """
        for item in data:
            num_hands = item[2]
            hands = item[3]['hands']
            self._extract_hands(video_label_idx, hands, num_hands)

    def _extract_hands(self, label_idx, hands, num_hands):
        """
        Extract hand landmarks based on the number of hands.

        Args:
            label_idx (int): Integer label index.
            hands (np.ndarray): Array of shape (2, 21, 3) containing hand landmarks.
            num_hands (int): Number of hands (0, 1, or 2).
        """
        if self.ignore_flat and np.all(hands[:,2] == 0):
            return # skip flat handmarks
        if num_hands == 0:
            self.data.append((label_idx, hands[0])) # Do not skip samples with no hands
        elif num_hands == 1:
            # Add the non-zero hand
            if np.any(hands[0]):
                self.data.append((label_idx, hands[0]))
            elif np.any(hands[1]):
                self.data.append((label_idx, hands[1]))
        elif num_hands == 2:
            # Add both hands as separate samples with the same label
            self.data.append((label_idx, hands[0]))
            if self.max_num_hands > 1:
                self.data.append((label_idx, hands[1]))

    def __len__(self):
        """
        Return the total number of samples.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (label_idx, hand_landmarks) where label_idx is an int and
                   hand_landmarks is a torch.Tensor of shape (21, 3).
        """
        label_idx, hand_landmarks = self.data[idx]
        if self.normalize_to_wrist:
            hand_landmarks -= hand_landmarks[0]
        if self.augment and np.any(hand_landmarks):
            hand_landmarks = self.augment(hand_landmarks)
        hand_landmarks = torch.from_numpy(hand_landmarks).float()
        return label_idx, hand_landmarks

    def get_label_map(self):
        """
        Return the label-to-index mapping.

        Returns:
            dict: Mapping from string labels to integer indices.
        """
        return {idx: label for label, idx in self.label_to_idx.items()}
    
if __name__ == '__main__':
    # Test the dataset class
    # Load the dataset
    # dataset = LabelledHandDataset(dataset_name='lexset', split='train')
    # dataset = LabelledHandDataset(dataset_name='senz3d')
    dataset = LabelledHandDataset(dataset_name='handshape', split='test')
    # dataset = LabelledHandDataset(dataset_name='lsa64')

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate over the data
    for labels, landmarks in dataloader:
        print(labels.shape)  # torch.Size([batch_size])
        print(landmarks.shape)  # torch.Size([batch_size, 21, 3])
        break

    # Get the label mapping
    label_map = dataset.get_label_map()
    print(label_map)  # e.g., {0: 'A', 1: 'B', ...}