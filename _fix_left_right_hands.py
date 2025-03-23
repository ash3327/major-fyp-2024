"""
python _fix_left_right_hands.py -d <dataset_name> -s <split_name>
"""

import os
import numpy as np
from tqdm import tqdm

import argparse

from prepare_dataset import get_info

def fix_npy_features(data_dir, dataset, subfolders, output_dir):
    """
    Updates the features[0] entries in existing .npy files for video datasets by setting them
    to the correct format: f"{video_path}${idx}", where idx is the frame index. This fixes
    missing or incorrect features[0] entries without rerunning the full feature extraction.

    Parameters:
    - data_dir (str): Path to the directory containing the dataset (e.g., "path/to/data").
    - dataset (str): Name of the dataset (e.g., "dataset_name").
    - subfolders (dict): Mapping from split names to subfolder names (e.g., {"train": "train_subfolder", "test": "test_subfolder"}).
    - output_dir (str): Path to the directory where .npy files are stored (e.g., "path/to/output").
    """
    for root, _, files in os.walk(os.path.join(output_dir, dataset)):
        for file in tqdm(files):
            if file.lower().endswith((".npy")):
                npy_path = os.path.join(root,file)
                if os.path.exists(npy_path):
                    try:
                        # Load the .npy file with allow_pickle=True since it contains object arrays
                        data = np.load(npy_path, allow_pickle=True)

                        # Update entries
                        for features in data:
                            body = features[3]['pose']
                            hand1, hand2 = np.copy(features[3]['hands'][0]), np.copy(features[3]['hands'][1])
                            b1, b2 = (body[9] != 0).all(), (body[10] != 0).all()
                            flag = True
                            if features[2] == 2 and b1 and b2:
                                if np.linalg.norm(body[9]-hand1[0,:2]) > np.linalg.norm(body[9]-hand2[0,:2]):
                                    features[3]['hands'][0],features[3]['hands'][1] = hand2, hand1
                            elif features[2] == 1:
                                hand1, hand2 = (hand1, hand2) if (hand2 == 0).all() else (hand2, hand1)
                                if b1 and b2:
                                    if np.linalg.norm(body[9]-hand1[0,:2]) < np.linalg.norm(body[10]-hand1[0,:2]):
                                        b2 = False
                                    else:
                                        b1 = False
                                if b1: # have left hand
                                    features[3]['hands'][0],features[3]['hands'][1] = hand1, hand2
                                elif b2:
                                    features[3]['hands'][0],features[3]['hands'][1] = hand2, hand1
                            else:
                                flag = False
                            # if flag:
                            #     print(f'Switched {features[0]}')
                        # Save the updated array back to the .npy file
                        np.save(npy_path, data)
                        print(f"Successfully updated: {npy_path}")

                    except Exception as e:
                        print(f"Error processing {npy_path}: {e}")
                        raise e
                else:
                    print(f"Warning: .npy file not found at {npy_path}")
                    # pass

def parse_dataset(dataset, split=None):
    data_dir, dataset, subfolders, output_dir, dyn, is_video, *args = get_info(dataset)
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

    # print(data_dir, dataset, subfolders, output_dir)
    fix_npy_features(data_dir, dataset, subfolders, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Loader")
    parser.add_argument('-d','--dataset', type=str, default='lexset', help='Name of the dataset')
    parser.add_argument('-s', '--split', type=str, default=None, help='Name of the split')
    args = parser.parse_args()
    parse_dataset(args.dataset, split=args.split)