"""
python scripts/datasets/_fix_video_names.py -d <dataset_name> -s <split_name>
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
    for split, subfolder in subfolders.items():
        # Construct the path to the subfolder containing video files
        main_dir = os.path.join(data_dir, dataset, subfolder)
        print(f"Processing subfolder: {main_dir}")

        # Collect all video file paths in the subfolder
        video_paths = []
        for root, _, files in os.walk(main_dir):
            for file in files:
                # Check for common video file extensions
                if file.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                    video_path = os.path.join(root, file)
                    video_paths.append(video_path)

        # Process each video and its corresponding .npy file
        for video_path in tqdm(video_paths, desc=f"Fixing {split} videos"):
            # Extract the video name without extension to match the .npy filename
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            npy_path = os.path.join(output_dir, dataset, split, f"{video_name}.npy")

            # Check if the .npy file exists
            if os.path.exists(npy_path):
                try:
                    # Load the .npy file with allow_pickle=True since it contains object arrays
                    data = np.load(npy_path, allow_pickle=True)

                    # Update each entry's features[0] to f"{video_path}${idx}"
                    for i, entry in enumerate(data):
                        # Assuming entry is a tuple or array with at least 4 elements
                        # Create a new tuple with updated features[0]
                        updated_entry = (f"{video_path}${i}", entry[1], entry[2], entry[3])
                        data[i] = updated_entry

                    # Save the updated array back to the .npy file
                    np.save(npy_path, data)
                    print(f"Successfully updated: {npy_path}")

                except Exception as e:
                    print(f"Error processing {npy_path}: {e}")
            else:
                # print(f"Warning: .npy file not found at {npy_path}")
                pass

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