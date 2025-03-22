"""
With parallelization, faster, but may not work on your computer.
Use noparallel one if this script fails.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

import concurrent.futures

mp_holistic = mp.solutions.holistic

def get_holistic_landmarks(image_rgb, dyn=False):
    """
    Extract holistic landmarks (pose, face, hands) using MediaPipe Holistic.
    """
    with mp_holistic.Holistic(
        static_image_mode=not dyn,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.5
    ) as holistic:
        results = holistic.process(image_rgb)
    return results

def extract_features_from_image(image_rgb, image_path=None, dyn=False):
    """
    Extract pose and hand landmarks using MediaPipe Holistic.
    """
    results = get_holistic_landmarks(image_rgb, dyn=dyn)
    landmarks = {
        'pose': [],
        'hands': [[], []]  # Left hand, Right hand
    }

    # Extract pose landmarks
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks['pose'].append([landmark.x, landmark.y, landmark.z])
    else:
        landmarks['pose'] = [[0, 0, 0]] * 33  # Default to 33 pose landmarks

    # Extract left hand landmarks
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            landmarks['hands'][0].append([landmark.x, landmark.y, landmark.z])
    else:
        landmarks['hands'][0] = [[0, 0, 0]] * 21  # Default to 21 hand landmarks

    # Extract right hand landmarks
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            landmarks['hands'][1].append([landmark.x, landmark.y, landmark.z])
    else:
        landmarks['hands'][1] = [[0, 0, 0]] * 21  # Default to 21 hand landmarks

    # Convert to numpy arrays
    landmarks['pose'] = np.array(landmarks['pose'])
    landmarks['hands'] = np.array(landmarks['hands'])

    num_poses = int(np.any(landmarks['pose'] != 0))
    num_hands = sum(np.any(hand != 0) for hand in landmarks['hands'])
    return image_path, num_poses, num_hands, landmarks

def extract_features_from_video(video_path, output_dir, dataset, split, skip=False):
    """
    Extract features from video frames using MediaPipe Holistic.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, dataset, split, f"{video_name}.npy")

    if skip and os.path.exists(output_path):
        print(f"Skipping {video_path}, features already extracted.")
        return

    cap = cv2.VideoCapture(video_path)
    data = []

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = extract_features_from_image(image_rgb, image_path=f"{video_path}${idx}", dyn=True)
        data.append(landmarks)
        idx += 1

    cap.release()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(data, dtype=object))
    print(f"Saved video features to {output_path}")

def extract_features_from_subfolder(data_dir, dataset, subfolder, output_dir, split, dyn=False, skip=False):
    main_dir = os.path.join(data_dir, dataset, subfolder)
    print('Extracting', main_dir, output_dir)
    
    data = []
    image_paths = []
    video_paths = []
    for root, _, files in os.walk(main_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_paths.append((root, os.path.join(root, file)))
            elif file.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                video_paths.append(os.path.join(root, file))
    num_images = len(image_paths)
    num_videos = len(video_paths)
    print(num_images, num_videos)

    def extract_features(image_path):
        image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        _, num_bodies, num_hands, landmarks = extract_features_from_image(image_rgb, dyn=dyn, image_path=image_path)
        dpath = image_path.split(main_dir)[1].replace("\\", "/")
        dpath = dpath[1:] if dpath and dpath[0] == "/" else dpath
        return (dpath, num_bodies, num_hands, landmarks)
    
    if num_images != 0:
        pbar = tqdm(total=num_images, desc="Processing images")
        num_threads = min(os.cpu_count(), num_images, 16)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(extract_features, image_path) for label, image_path in image_paths]
            
            for future in concurrent.futures.as_completed(futures):
                data.append(future.result())
                pbar.update(1)
                
        pbar.close()
        print("Saving data to npy file...")
        data = np.array(data, dtype=object)
        output_dir = os.path.join(output_dir, dataset)
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"record_{split}.npy"), data)

    if num_videos != 0:
        print("Processing videos...")
        for video_path in tqdm(video_paths, desc="Processing videos"):
            extract_features_from_video(video_path, output_dir, dataset, split, skip=skip)
    
    print("Feature extraction complete!")

def extract_features(data_dir, dataset, subfolders, output_dir, dyn=False, *args, skip=False, **kwargs):
    for split, subfolder in subfolders.items():
        extract_features_from_subfolder(data_dir, dataset, subfolder, output_dir, split, dyn=dyn, skip=skip)
