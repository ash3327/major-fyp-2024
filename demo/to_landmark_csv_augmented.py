import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from math import radians, sin, cos

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

# Function to augment data
def augment_landmarks(landmarks, num_rotations=10, num_scalings=5):
    augmented_data = []
    # Subtract the landmark 0 coordinate (origin adjustment)
    origin = landmarks[0]
    relative_landmarks = landmarks - origin

    # Generate rotations
    for axis in ['xy', 'xz', 'yz']:
        for _ in range(num_rotations):
            angle = np.random.uniform(-5, 5)  # Random angle in degrees
            rotated_landmarks = rotate_landmarks(relative_landmarks, axis, angle)
            augmented_data.append(rotated_landmarks)

    # Generate scalings
    for _ in range(num_scalings):
        scale_factor = np.random.uniform(0.8, 1.2)  # Random scale factor
        scaled_landmarks = relative_landmarks * scale_factor
        augmented_data.append(scaled_landmarks)

    return augmented_data

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Define the main directory containing subfolders
main_dir = "Train_Alphabet"
landmarks_data = []
labels_data = []

# Process each subfolder and extract hand landmarks
with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
    for label in os.listdir(main_dir):
        subfolder_path = os.path.join(main_dir, label)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(subfolder_path, filename)
                    image = cv2.imread(image_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Extract hand landmark coordinates
                            landmarks = []
                            for landmark in hand_landmarks.landmark:
                                landmarks.append([landmark.x, landmark.y, landmark.z])
                            landmarks = np.array(landmarks)

                            # Augment the data
                            augmented_landmarks = augment_landmarks(landmarks)

                            # Append the original and augmented data
                            for augmented in augmented_landmarks:
                                landmarks_data.append(augmented[1:].flatten())  # Exclude landmark 0
                                labels_data.append(label)

# Convert landmarks and labels to a DataFrame for easy handling
data = pd.DataFrame(landmarks_data)
data['label'] = labels_data

# Save the data to CSV for use in the Siamese model training
data.to_csv("augmented_hand_landmarks.csv", index=False)
