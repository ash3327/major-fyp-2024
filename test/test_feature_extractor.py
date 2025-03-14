"""
python test/test_feature_extractor.py -d lexset -s train
python test/test_feature_extractor.py -d lexset -s test

# First Round Test Problems
python test/test_feature_extractor.py -f data/raw/hands_dataset/Subject1/Subject1/3371_color.png
python test/test_feature_extractor.py -f data/raw/hands_dataset/Subject2/Subject2/2497_color.png

# Second Round Test Problems
python test/test_feature_extractor.py -f data/raw/synthetic-asl-alphabet/Test_Alphabet/A/38ba5f14-0117-4d8f-a263-a506540208e9.rgb_0000.png
python test/test_feature_extractor.py -f data/raw/hands_dataset/Subject1/Subject1/3379_color.png
python test/test_feature_extractor.py -f data/raw/hands_dataset/Subject2/Subject2/2445_color.png
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

sys.path.append('.')

from prepare_dataset import get_info
from feature_extractor import (
    get_image,
    extract_image,
    extract_features_from_clipped_region
)

def parse_image(image_path):
    # Read the image (RGB) using our helper
    image_rgb = get_image(image_path)
    # Also process image with CLAHE adjustments if needed
    image_rgb_clahe = extract_image(image_path)
    
    # Extract features from the clipped region using YOLO detection
    features, bbox, num_person = extract_features_from_clipped_region(image_rgb, dyn=False)
    landmarks = features[3]
    
    # Skip items that match expected counts (if applicable)
    # if features[2] >= 1:
    # if (features[1], features[2]) in expected_cnts.get(dataset, [(None, None)]):
        # return
    print(features[2])
        # pass
    # if features[2] != 0:
    #     continue
    # if num_person == 1 and features[1] == 1:
        # continue
    # print(features[1])

    plt.figure(figsize=(6, 6))
    plt.title(f"Image: {image_path}")
    plt.imshow(image_rgb_clahe)

    # If bounding box was found, plot it
    if bbox is not None:
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    # Plot landmarks on the original image space
    for key, value in landmarks.items():
        if key == 'pose':
            plt.scatter(value[:, 0] * image_rgb.shape[1], value[:, 1] * image_rgb.shape[0],
                        s=10, marker='.', c='b')
        elif key == 'hands':
            for hand in value:
                plt.scatter(hand[:, 0] * image_rgb.shape[1], hand[:, 1] * image_rgb.shape[0],
                            s=10, marker='.', c='r')
    
    plt.tight_layout()
    plt.show()

def parse_dataset(dataset, split):
    data_dir, dataset, subfolders, output_dir, dyn = get_info(dataset)
    splits = list(subfolders.keys())

    # Choose split if not provided or invalid
    if split is None or split not in splits:
        inpt = input(f"Choose from the splits: {splits}\n>>> ") if len(splits) > 1 else splits[0]
        if inpt in splits:
            split = inpt
            print(f"Fetching dataset with split {split}...")
        else:
            print("Invalid split. Terminating...")
            exit(1)
    else:
        print(f"Fetching dataset with split {split}...")

    npy_file = os.path.join("data/kpts", dataset, f"record_{split}.npy")
    data = np.load(npy_file, allow_pickle=True)

    # Expected counts for filtering (if applicable)
    expected_cnts = {
        'synthetic-asl-alphabet': [(0, 1),(1, 1)],
        'senz3d_dataset': [(1, 1)],
        'hands_dataset': [(1, 2)]
    }

    for i, item in enumerate(data):
        image_path = os.path.join(data_dir, dataset, subfolders[split], item[0])
        parse_image(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Loader")
    parser.add_argument('-d', '--dataset', type=str, default='lexset', help='Name of the dataset')
    parser.add_argument('-s', '--split', type=str, default=None, help='Name of the split')
    parser.add_argument('-f', '--file', type=str, help='Path to a single image file')
    args = parser.parse_args()

    if args.file:
        parse_image(args.file)
    else:
        parse_dataset(args.dataset, split=args.split)
