"""
python scripts/datasets/_fix_order.py -d <dataset_name> -s <split> --output_dir <output_directory>
  
Output Shape: [N,59,3]
where, first 17 is the pose, later 21 and 21 are the two hands.
p[:17] pose
p[17:38] hand1
p[38:59] hand2
"""

import os
import sys
sys.path.append('.')

import cv2
import numpy as np
import argparse
from tqdm import tqdm

from scripts.datasets.prepare_dataset import get_info

from scripts.datasets.feature_extractor_holistic import extract_features_from_image as holistic_extract_features

def process_npy_file(npy_path):
    """
    Loads the .npy file, updates entries where features[2] != 2 using holistic detection,
    and overwrites the .npy file with updated entries.
    
    Also, for video entries, ensures features[0] is in the format "<video_path>$<frame_index>".
    """
    print(f"Processing file: {npy_path} -> {npy_path.replace('kpts','kpts_flat')}")
    try:
        data = np.load(npy_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load {npy_path}: {e}")
        return
    
    out = list()
    updated = False
    num_updated = 0
    num_checked = 0
    for idx, features in enumerate(pbar:=tqdm(data, desc="Processing", dynamic_ncols=True)):
        pbar.set_description(f"Processing {features[0]} ({num_updated}/{num_checked})")
        # features is a tuple: (identifier, num_bodies, num_hands, landmarks)
        pose_features = features[3]['pose']
        if pose_features.shape[1] == 2:
            pose_features = np.pad(pose_features, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        out.append(np.concatenate([pose_features,features[3]['hands'][0],features[3]['hands'][1]],axis=0))
        # print(out[-1].shape)
    # if updated:
    output_path = npy_path.replace('kpts','kpts_flat')
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    np.save(output_path, out)
    print(f"Updated {output_path}")
    # else:
    #     print(f"No updates needed for {npy_path}")

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
                if 'record_' not in file and not is_video:
                    continue
                process_npy_file(npy_path)

def main():
    parser = argparse.ArgumentParser(description="Fix Holistic Features in Dataset npy files")
    parser.add_argument("-d", "--dataset", type=str, default="lexset", help="Name of the dataset")
    # parser.add_argument("-s", "--split", type=str, default=None, help="Name of the split (if needed)")
    parser.add_argument("--output_dir", type=str, default="data/kpts", help="Directory where npy files are stored")
    args = parser.parse_args()
    
    data_dir, dataset, subfolders, output_dir, dyn, is_video, *rest = get_info(args.dataset)
    
    fix_dataset(data_dir, dataset, subfolders, args.output_dir, is_video=is_video)

if __name__ == "__main__":
    main()
