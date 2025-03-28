"""
python scripts/datasets/fix_holistic.py -d <dataset_name> -s <split> --output_dir <output_directory>
  
This script iterates through .npy files in the dataset’s output directory,
checks for entries where features[2] != 2, re-runs holistic MediaPipe detection,
updates the hand landmarks, and saves the .npy file back.
"""

import os
import sys
sys.path.append('.')

import cv2
import numpy as np
import argparse
from tqdm import tqdm

from .prepare_dataset import get_info

from .feature_extractor_holistic import extract_features_from_image as holistic_extract_features

def load_image(identifier, data_dir, dataset, subfolder, is_video=False):
    """
    Loads an image based on the identifier.
    If is_video is True, identifier is expected to be "<video_path>$<frame_index>".
    For images, identifier is a relative image path.
    """
    if is_video:
        # For video entries, split identifier to get video path and frame index.
        try:
            video_path, idx_str = identifier.split('$')
            frame_idx = int(idx_str)
        except Exception as e:
            print(f"Error parsing video identifier {identifier}: {e}")
            return None
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Could not load frame {frame_idx} from video {video_path}")
            return None
        # Convert frame from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return image
    else:
        # For image entries, construct the full path from data_dir, dataset, and subfolder.
        full_path = os.path.join(data_dir, dataset, subfolder, identifier)
        if not os.path.exists(full_path):
            print(f"Image file {full_path} does not exist.")
            return None
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

def process_npy_file(npy_path, data_dir, dataset, subfolder, is_video=False):
    """
    Loads the .npy file, updates entries where features[2] != 2 using holistic detection,
    and overwrites the .npy file with updated entries.
    
    Also, for video entries, ensures features[0] is in the format "<video_path>$<frame_index>".
    """
    print(f"Processing file: {npy_path}")
    try:
        data = np.load(npy_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load {npy_path}: {e}")
        return
    
    updated = False
    num_updated = 0
    num_checked = 0
    for idx, features in enumerate(pbar:=tqdm(data, desc="Processing", dynamic_ncols=True)):
        pbar.set_description(f"Processing {features[0]} ({num_updated}/{num_checked})")
        # features is a tuple: (identifier, num_bodies, num_hands, landmarks)
        identifier = features[0]
        num_hands = features[2]
        if num_hands == 2:
            continue  # Skip if already 2 hands detected
        # print(identifier)
        # continue
        # Load the corresponding image/frame using our helper
        image = load_image(identifier, data_dir, dataset, subfolder, is_video=is_video)
        if image is None:
            continue
        num_checked += 1
        # Run holistic detection on the image/frame
        # The holistic_extract_features function returns a tuple: (image_path, num_poses, num_hands, landmarks)
        new_features = holistic_extract_features(image, image_path=identifier, dyn=True)
        
        # Overwrite hand landmarks with the holistic detection results
        # We assume new_features[3] has the updated 'hands' field.
        if new_features[2] > features[2]:
            # print(f'UPDATED {features[0]}')
            features[3]['hands'] = new_features[3]['hands']
            features[2] = 2
            num_updated += 1
            updated = True
    
    if updated:
        np.save(npy_path, data)
        print(f"Updated {npy_path}")
    else:
        print(f"No updates needed for {npy_path}")

def fix_dataset(data_dir, dataset, subfolders, output_dir, is_video=False):
    """
    Walks through the output directory (data/kpts/<dataset>) and processes each .npy file,
    applying holistic detection to update entries with incomplete hand landmarks.
    """
    dataset_dir = os.path.join(output_dir, dataset)
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(".npy"):
                npy_path = os.path.join(root, file)
                # Derive subfolder name from npy_path relative to dataset_dir
                rel_path = os.path.relpath(npy_path, dataset_dir)
                subfolder = os.path.dirname(rel_path) if is_video else file.rsplit('.npy',1)[0].split('record_',1)[1]  # may be empty for top-level npy file    
                subfolder = subfolders[subfolder]
                # print(subfolder, npy_path)
                process_npy_file(npy_path, data_dir, dataset, subfolder, is_video=is_video)

def main():
    parser = argparse.ArgumentParser(description="Fix Holistic Features in Dataset npy files")
    parser.add_argument("-d", "--dataset", type=str, default="lexset", help="Name of the dataset")
    parser.add_argument("-s", "--split", type=str, default=None, help="Name of the split (if needed)")
    parser.add_argument("--output_dir", type=str, default="data/kpts", help="Directory where npy files are stored")
    args = parser.parse_args()
    
    data_dir, dataset, subfolders, output_dir, dyn, is_video, *rest = get_info(args.dataset)
    
    fix_dataset(data_dir, dataset, subfolders, args.output_dir, is_video=is_video)

if __name__ == "__main__":
    main()
