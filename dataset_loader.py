'''
python dataset_loader.py -d lexset -s train
'''

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from prepare_dataset import get_info

#-----------
# Visualization
#-----------
def visualize_image(fname, fpath, features):
    image = Image.open(fpath)
    lmks = (features[3]['pose'], features[3]['hands'][0], features[3]['hands'][1])
    plt.imshow(image)
    plt.title(fname)
    for landmarks, c in zip(lmks, ['b','g','r','k']):
        plt.scatter(landmarks[:, 0]*image.size[0], landmarks[:, 1]*image.size[1], s=10, marker='.', c=c)
    plt.show()

def visualize_video_frame(fname, frame_id, features):
    cap = cv2.VideoCapture(fname)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {fname}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Unable to read frame {frame_id} from video {fname}")
        cap.release()
        return

    lmks = (features[3]['pose'], features[3]['hands'][0], features[3]['hands'][1])
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Frame {frame_id} of {fname}: {features[2]} hands")
    for landmarks, c in zip(lmks, ['b', 'g', 'r', 'k']):
        plt.scatter(landmarks[:, 0] * frame.shape[1], landmarks[:, 1] * frame.shape[0], s=10, marker='.', c=c)
    plt.show()

    cap.release()

#-----------
# Prepare Dataset
#-----------

def parse_dataset(dataset, split):
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

    # Assuming data is a list of dictionaries with 'image' and 'landmarks' keys
    count = 0
    cnt0, cnt1, cnt2, cnt3 = 0, 0, 0, 0
    total = 0
    expected_cnts = {
        'asl_alphabet': [(0, 1)],
        'synthetic-asl-alphabet': [(0, 1),(1, 1)],
        'senz3d_dataset': [(1, 1)],
        'hands_dataset': [(1, 2)],
        'ph2014-handshape': [(1, 1)],
        'lsa64_preprocessed': [(1, 2)]
    }
    actual_cnts = dict()
    expected_cnts = expected_cnts[dataset]
    shapes = dict()

    def evaluate(data, rpath=None):
        nonlocal count, cnt0, cnt1, cnt2, cnt3, total
        for i, item in tqdm(enumerate(data)):
            total += 1
            shapes[(item[3]['pose'].shape, item[3]['hands'].shape)] = shapes.get((item[3]['pose'].shape, item[3]['hands'].shape),0)+1
            if dataset == 'synthetic-asl-alphabet' and item[0].startswith('Blank'):
                continue
            if not (item[1], item[2]) in expected_cnts:
            # if item[1] != 1 or item[2] != 2:
                # print(item[0], item[1], item[2])
                count += 1
                a,b,c = True, True, True
                for exp in expected_cnts:
                    a = a and item[1] != exp[0]
                    b = b and item[2] != exp[1]
                    c = c and item[1] < exp[0] or item[2] < exp[1]
                cnt0 += a and not b
                cnt1 += b and not a
                cnt2 += a and b
                cnt3 += c
                actual_cnts[(item[1], item[2])] = actual_cnts.get((item[1], item[2]), 0) + 1

                if is_video:
                    print(rpath, i)
                else:
                    fname = os.path.join(subfolders[split], item[0])
                    fpath = os.path.join(data_dir, dataset, fname)
                    print(fname, item[2])

            if item[2] == 0:
            # if item[0] == "final_phoenix_noPause_noCompound_lefthandtag_noClean/30July_2010_Friday_tagesschau_default-7/1/.png_fn000135-0.png":
                if is_video:
                    fname, frame_idx = item[0].rsplit('$',1)
                    visualize_video_frame(fname, int(frame_idx), item)
                else:
                    fname = os.path.join(subfolders[split], item[0])
                    fpath = os.path.join(data_dir, dataset, fname)
                    visualize_image(fname, fpath, item)

    if is_video:
        root_path = os.path.join("data/kpts", dataset, split)
        print(root_path)
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.lower().endswith(('.npy')):
                    npy_file = os.path.join(root, file)
                    print(npy_file)
                    data = np.load(npy_file, allow_pickle=True)
                    evaluate(data, rpath=npy_file)
        # return
    else:
        npy_file = os.path.join("data/kpts", dataset, f"record_{split}.npy")
        data = np.load(npy_file, allow_pickle=True)
        evaluate(data)

    cnts = np.array((count, cnt0, cnt1, cnt2, cnt3))
    print('Problematic Items:',cnts,'out of',total)
    print('Problematic Ratio:',cnts/total)
    print('Actual Counts:',actual_cnts)
    print('Shapes',shapes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Loader")
    parser.add_argument('-d','--dataset', type=str, default='lexset', help='Name of the dataset')
    parser.add_argument('-s', '--split', type=str, default=None, help='Name of the split')
    args = parser.parse_args()
    parse_dataset(args.dataset, split=args.split)
