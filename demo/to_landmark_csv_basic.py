import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from math import radians, sin, cos
from tqdm import tqdm

# Enable GPU acceleration for MediaPipe
mp.solutions.hands.HAND_CONNECTIONS
mp_hands = mp.solutions.hands

# Configure MediaPipe to use GPU
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to rotate landmarks around an axis
def rotate_landmarks(landmarks, axis, angle):
    angle_rad = radians(angle)
    rotation_matrix = None

    if axis == 'xy':  # Rotation in the xy-plane
        rotation_matrix = np.array([
            [cos(angle_rad), -sin(angle_rad), 0],
            [sin(angle_rad), cos(angle_rad), 0],
            [0, 0, 1]
        ])
    elif axis == 'xz':  # Rotation in the xz-plane
        rotation_matrix = np.array([
            [cos(angle_rad), 0, -sin(angle_rad)],
            [0, 1, 0],
            [sin(angle_rad), 0, cos(angle_rad)]
        ])
    elif axis == 'yz':  # Rotation in the yz-plane
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos(angle_rad), -sin(angle_rad)],
            [0, sin(angle_rad), cos(angle_rad)]
        ])

    # Apply rotation
    rotated_landmarks = np.dot(landmarks, rotation_matrix.T)
    return rotated_landmarks

# Initialize MediaPipe Hands with GPU optimization
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    model_complexity=1  # Use 1 for better performance with GPU
) as hands:
    # Define the main directory containing subfolders
    main_dir = "../kaggle/input/synthetic-asl-alphabet/Train_Alphabet"
    landmarks_data = []
    labels_data = []

    # Get total number of images for progress bar
    total_images = sum([len([f for f in os.listdir(os.path.join(main_dir, label)) 
                           if f.endswith(('.jpg', '.png'))])
                       for label in os.listdir(main_dir)
                       if os.path.isdir(os.path.join(main_dir, label))])
    
    pbar = tqdm(total=total_images, desc="Processing images")
    
    for label in os.listdir(main_dir):
        subfolder_path = os.path.join(main_dir, label)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(subfolder_path, filename)
                    # Read image in BGR format for GPU processing
                    image = cv2.imread(image_path)
                    # Flip channels from BGR to RGB for MediaPipe
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Make image contiguous for better GPU memory access
                    image_rgb = np.ascontiguousarray(image_rgb)
                    
                    # Process with GPU acceleration
                    results = hands.process(image_rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Extract hand landmark coordinates
                            landmarks = []
                            for landmark in hand_landmarks.landmark:
                                landmarks.append([landmark.x, landmark.y, landmark.z])
                            landmarks = np.array(landmarks)

                            landmarks_data.append(landmarks[1:].flatten())  # Exclude landmark 0
                            labels_data.append(label)
                    
                    pbar.update(1)
    
    pbar.close()

print("Converting to DataFrame...")
# Convert landmarks and labels to a DataFrame for easy handling
data = pd.DataFrame(landmarks_data)
data['label'] = labels_data

print(f"Saving {len(data)} samples to CSV...")
# Save the data to CSV for use in the Siamese model training
data.to_csv("basic_hand_landmarks.csv", index=False)
print("Done!")
