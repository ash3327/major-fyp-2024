import os
import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
import concurrent.futures
import threading

# Mediapipe and YOLO Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pose_model = YOLO("model/yolo11n-pose.pt")
pose_model.fuse()
pose_model.to(device)
hand_model = YOLO("model/best-3.pt")
hand_model.fuse()
hand_model.to(device)

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def extract_features_from_video(video_path, output_dir, dataset):
    frames = extract_frames_from_video(video_path)
    data = []
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    def process_frame(idx, frame):
        features, _, _ = extract_features_from_clipped_region(frame)
        data.append(features)
    
    threads = []
    for idx, frame in enumerate(frames):
        thread = threading.Thread(target=process_frame, args=(idx, frame))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    output_path = os.path.join(output_dir, dataset, f"{video_name}.npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(data, dtype=object))
    print(f"Saved video features to {output_path}")

def extract_features_from_subfolder(data_dir, dataset, subfolder, output_dir, split):
    main_dir = os.path.join(data_dir, dataset, subfolder)
    print(f'Extracting {main_dir} to {output_dir}')
    
    image_paths = []
    video_paths = []
    for root, _, files in os.walk(main_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
            elif file.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                video_paths.append(os.path.join(root, file))
    
    image_features = []
    
    def process_image(image_path):
        features, _, _ = extract_features_from_clipped_region(get_image(image_path))
        image_features.append(features)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, image_paths), total=len(image_paths), desc="Processing images"))
    
    image_output_path = os.path.join(output_dir, dataset, subfolder, f"{split}_images.npy")
    os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
    np.save(image_output_path, np.array(image_features, dtype=object))
    print(f"Saved image features to {image_output_path}")
    
    for video_path in tqdm(video_paths, desc="Processing videos"):
        extract_features_from_video(video_path, output_dir, dataset)
    
    print("Feature extraction complete!")

def extract_features(data_dir, dataset, subfolders, output_dir):
    for split, subfolder in subfolders.items():
        extract_features_from_subfolder(data_dir, dataset, subfolder, output_dir, split)

def get_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def extract_features_from_clipped_region(image_rgb):
    yolo_results = pose_model(image_rgb, verbose=False)
    results = hand_model(image_rgb, verbose=False)
    
    num_hands = 0
    landmarks = {'pose': np.zeros((17,2)), 'hands': np.zeros((2,21,3))}
    
    for result in results:
        if result.boxes.xyxy.shape[0] > 0:
            num_hands += len(result.boxes.xyxy)
            for i in range(min(2, len(result.boxes.xyxy))):
                landmarks['hands'][i] = result.keypoints.xy[i].cpu().numpy()
    
    return np.array((num_hands, landmarks))

