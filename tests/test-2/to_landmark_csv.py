"""
python tests/test-2/to_landmark_csv.py

edit paths directly in this document

VERSION 1
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from math import radians, sin, cos
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Paths
data_dir = "data/raw"
dataset = "synthetic-asl-alphabet"
subfolder = "Test_Alphabet"
output_dir = "tests/test-2"
output_file = "kpts/basic_hand_landmarks"
have_class = True
ordered = False
filter_fmt = None

# Trials (Hard-coded)
# dataset, subfolder, output_file = "synthetic-asl-alphabet", "Test_Alphabet", "kpts/basic_hand_landmarks_"
# dataset, subfolder, output_file = "synthetic-asl-alphabet", "Train_Alphabet", "kpts/basic_hand_landmarks_train"
# dataset, subfolder, output_file, have_class, filter_fmt, ordered = "B2Counting", ".", "kpts/b2_counting_landmarks", False, "SK_color_", True

# Enable GPU acceleration for MediaPipe
mp.solutions.hands.HAND_CONNECTIONS
mp_hands = mp.solutions.hands

# Configure MediaPipe to use GPU
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_image(image_path, hands, have_class=False):
    # Read image in BGR format for CPU processing
    image = cv2.imread(image_path)
    # Flip channels from BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Make image contiguous for better CPU memory access
    image_rgb = np.ascontiguousarray(image_rgb)

    # Process with CPU
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        
        # Merge channels
        lab_clahe = cv2.merge((l_clahe, a, b))
        
        # Flip channels from BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        # Process again
        results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks)
            landmarks[1:] -= landmarks[0] # relative position to wrist

            return landmarks[1:].flatten(), image_path # Exclude landmark 0

    return None, None

def check_filename(filename):
    return (filename.endswith(".jpg") or filename.endswith(".png")) and (filter_fmt is None or filter_fmt in filename)

def process_folder(data_dir, dataset, subfolder, output_dir, output_file, have_class):
    # Initialize MediaPipe Hands with GPU optimization
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        model_complexity=1  # Use 1 for better performance with GPU
    ) as hands:
        
        # Define the main directory containing subfolders
        main_dir = os.path.join(data_dir, dataset, subfolder) #"../kaggle/input/synthetic-asl-alphabet/Train_Alphabet"
        print(f"Processing {main_dir}...")

        landmarks_data = []
        labels_data = []
        image_paths = []
        failed_paths = []

        if have_class:
            total_images = sum([len([f for f in os.listdir(os.path.join(main_dir, label)) 
                                if f.endswith(('.jpg', '.png'))])
                            for label in os.listdir(main_dir)
                            if os.path.isdir(os.path.join(main_dir, label))])

            pbar = tqdm(total=total_images, desc="Processing images")
        
            for label in os.listdir(main_dir):
                subfolder_path = os.path.join(main_dir, label)
                if os.path.isdir(subfolder_path):
                    for filename in os.listdir(subfolder_path):
                        if check_filename(filename):
                            image_path = os.path.join(subfolder_path, filename)
                            landmarks, _ = process_image(image_path, hands, have_class)
                            if landmarks is not None:
                                landmarks_data.append(landmarks)
                                labels_data.append(label)
                                image_paths.append(image_path)
                            elif label != "Blank":
                                failed_paths.append(image_path)
                                pbar.set_description(f"Processing images, {len(failed_paths)} failed")

                            pbar.update(1)
        else:
            if ordered:
                load_image_paths = list(sorted([os.path.join(main_dir, filename) for filename in os.listdir(main_dir) if check_filename(filename)],
                                     key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))))
            else:
                load_image_paths = [os.path.join(main_dir, filename) for filename in os.listdir(main_dir) if check_filename(filename)]

            pbar = tqdm(total=len(load_image_paths), desc="Processing images")
            
            for image_path in load_image_paths:
                landmarks, _ = process_image(image_path, hands, have_class)
                if landmarks is not None:
                    landmarks_data.append(landmarks)
                    labels_data.append("No Label")
                    image_paths.append(image_path)
                else:
                    failed_paths.append(image_path)
                    pbar.set_description(f"Processing images, {len(failed_paths)} failed")

                pbar.update(1)

        pbar.close()

    print("Converting to DataFrame...")
    # Convert landmarks and labels to a DataFrame for easy handling
    data = pd.DataFrame(landmarks_data)
    data['label'] = labels_data
    data['image_path'] = image_paths

    print(f"Saving {len(data)} samples to CSV...")

    # Save the data to CSV for use in the Siamese model training
    output_folder = os.path.join(output_dir, output_file)
    os.makedirs(output_folder, exist_ok=True)

    data.to_csv(os.path.join(output_folder, 'landmarks.csv'), index=False)
    with open(os.path.join(output_folder, 'instructions.txt'), "w") as f:
        f.write("Keypoints: \n0~59: (x,y,z) relative to wrist, in the order x0 y0 z0 x1 y1 z1 ...\n")
        f.write(f"Done with {len(data)} samples and {len(failed_paths)} failed images.")

    formats = pd.DataFrame({"start_id": [0], "end_id": [60], "n_features_per_channel": [3], "name": ["hand 1"]})
    formats.to_csv(os.path.join(output_folder, 'landmark_formats.csv'), index=False)

    with open(os.path.join(output_folder, 'failed_paths.txt'), "w") as f:
        for path in failed_paths:
            f.write(f"{path}\n")

    print(f"Done with {len(data)} samples and {len(failed_paths)} failed images.")
    print(f"Saved to {output_folder}")

if __name__ == "__main__":
    process_folder(data_dir, dataset, subfolder, output_dir, output_file, have_class)