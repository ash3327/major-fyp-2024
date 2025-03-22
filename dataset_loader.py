'''
python dataset_loader.py -d lexset -s train
--show3d shows the 3d plot of the hands.
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
from mpl_toolkits.mplot3d import Axes3D

#-----------
# Visualization
#-----------
def visualize_image(fname, fpath, features, show3d=False):
    image = Image.open(fpath)
    only_hands = False
    lmks = [features[3]['pose'], features[3]['hands'][0], features[3]['hands'][1]]
    if only_hands:
        # offset back to (0,0,0) for wrist,
        lmks = np.array([lmks[1]-lmks[1][0], lmks[2]-lmks[2][0]])
        # orientation issues: hand is pointing upwards (-y)

    if show3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        step = max(1, image.size[0] // 120)  # Reduce resolution to 1/50 of the image size
        xx, yy = np.meshgrid(np.linspace(0, 1, image.size[0] // step), np.linspace(0, 1, image.size[1] // step))
        zz = np.zeros_like(xx)
        reduced_image = np.array(image.resize((xx.shape[1], yy.shape[0]))) / 255  # Downsample the image

        for landmarks, c in zip(lmks, ['b', 'g', 'r', 'k']):
            ax.scatter(landmarks[:, 0], 
               landmarks[:, 1], 
               landmarks[:, 2] if landmarks.shape[1] > 2 else 0, 
               s=10, marker='.', c=c)
        if only_hands: # In the cube [-.5,.5]^3
            ax.set_xlim(-.5, .5)
            ax.set_ylim(-.5, .5)
            ax.set_zlim(-.5, .5)
        else: # In the cuboid [0,1]^2 x [-1,1]
            ax.scatter([lmks[0][9][0]],[lmks[0][9][1]],[0], s=20, c='k')
            ax.scatter([lmks[0][10][0]],[lmks[0][10][1]],[0], s=20, c='k')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(-.5, .5)
        
        # Paint and label the axes
        ax.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='X-axis')  # X-axis in red
        ax.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Y-axis')  # Y-axis in green
        ax.quiver(0, 0, 0, 0, 0, 0.5, color='b', label='Z-axis')  # Z-axis in blue
        ax.text(0.5, 0, 0, 'X', color='r')
        ax.text(0, 0.5, 0, 'Y', color='g')
        ax.text(0, 0, 0.5, 'Z', color='b')

        ax.set_title(fname)
        ax.legend()

        # Ensure the scale of the three axes is the same and adjust based on landmarks
        # all_landmarks = np.vstack([lmks[0], lmks[1], lmks[2]])
        # max_range = np.array([all_landmarks[:, 0].max() - all_landmarks[:, 0].min(),
        #               all_landmarks[:, 1].max() - all_landmarks[:, 1].min(),
        #               all_landmarks[:, 2].max() - all_landmarks[:, 2].min()]).max() / 2.0
        # mid_x = (all_landmarks[:, 0].max() + all_landmarks[:, 0].min()) * 0.5
        # mid_y = (all_landmarks[:, 1].max() + all_landmarks[:, 1].min()) * 0.5
        # mid_z = (all_landmarks[:, 2].max() + all_landmarks[:, 2].min()) * 0.5
        # ax.set_xlim(mid_x - max_range, mid_x + max_range)
        # ax.set_ylim(mid_y - max_range, mid_y + max_range)
        # ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()
    else:
        plt.imshow(image)
        plt.title(fname)
        for landmarks, c in zip(lmks, ['b', 'g', 'r', 'k']):
            plt.scatter(landmarks[:, 0] * image.size[0], 
                        landmarks[:, 1] * image.size[1], 
                        s=10, marker='.', c=c)
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

def visualize_video(fname, features):
    cap = cv2.VideoCapture(fname)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {fname}")
        return

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id < len(features):
            frame_lmks = features[frame_id][3]
            lmks = (frame_lmks['pose'], frame_lmks['hands'][0], frame_lmks['hands'][1])
            for landmarks, c in zip(lmks, ['b', 'g', 'r', 'k']):
                for landmark in landmarks:
                    x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (255, 0, 0) if c == 'b' else (0, 255, 0) if c == 'g' else (0, 0, 255), -1)

            # Add dynamic title to the frame
            num_hands = features[frame_id][2]  # Number of hands
            title_text = f"Frame: {frame_id} | Hands: {num_hands}"
            cv2.putText(frame, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
        cv2.imshow(f"Video: {fname}", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to quit
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

#-----------
# Prepare Dataset
#-----------

def parse_dataset(dataset, split, show3d=False):
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
        'lsa64_raw': [(1, 2)],
        'IPN_Hand': [(1, 2)]
    }
    actual_cnts = dict()
    expected_cnts = expected_cnts[dataset]
    shapes = dict()

    def evaluate(data, rpath=None):
        nonlocal count, cnt0, cnt1, cnt2, cnt3, total
        flag = False
        features = list()
        cum_total = 0
        cum_count = 0
        # mindepth, maxdepth = 0,0
        for i, item in tqdm(enumerate(data)):
            total += 1
            cum_total += 1
            shapes[(item[3]['pose'].shape, item[3]['hands'].shape)] = shapes.get((item[3]['pose'].shape, item[3]['hands'].shape),0)+1
            if is_video:
                features.append(item)
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
                # cum_count += c
                actual_cnts[(item[1], item[2])] = actual_cnts.get((item[1], item[2]), 0) + 1

                if is_video:
                    # print(rpath, i)
                    pass
                else:
                    fname = os.path.join(subfolders[split], item[0])
                    fpath = os.path.join(data_dir, dataset, fname)
                    print(fname, item[2])

            if item[2] == 0:
            # if item[1] != 0:
            # if item[0] == "final_phoenix_noPause_noCompound_lefthandtag_noClean/30July_2010_Friday_tagesschau_default-7/1/.png_fn000135-0.png":
                if is_video:
                    cum_count += 1
                    fname, frame_idx = item[0].rsplit('$',1)
                    # if not flag:
                    #     visualize_video_frame(fname, int(frame_idx), item)
                    # flag = True
                else:
                    fname = os.path.join(subfolders[split], item[0])
                    fpath = os.path.join(data_dir, dataset, fname)
                    visualize_image(fname, fpath, item, show3d=show3d)
                    # mindepth = min(mindepth, np.min(item[3]['hands'][:,:,2]))
                    # maxdepth = max(maxdepth, np.max(item[3]['hands'][:,:,2]))
        # if flag and is_video:
        if cum_count/cum_total > .3 and is_video:
            visualize_video(fname, features)

        # print(f"MIN={mindepth}, MAX={maxdepth}")

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
    parser.add_argument('--show3d', action='store_true', help="whether 3d plot is shown instead of 2d")
    args = parser.parse_args()
    parse_dataset(args.dataset, split=args.split, show3d=args.show3d)
