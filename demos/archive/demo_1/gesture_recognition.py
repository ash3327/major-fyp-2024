import json
import cv2
import mediapipe as mp
import numpy as np
import torch
import time

import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(62, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 128)
        )

    def forward_once(self, x):
        x = x.permute(0, 2, 1)  # reshape to (B, 62, 41)
        return self.encoder(x)


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNet().to(device)
model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
model.eval()

# Load class means
with open("class_means.json", "r") as f:
    class_means = json.load(f)

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Time tracking for frame differences
last_frame_time = None

def classify_hand_landmarks(embedding, class_means, threshold=0.5):
    """
    Classify hand gesture based on adjusted landmarks.
    """
    distances = {cls: np.linalg.norm(embedding - np.array(mean)) for cls, mean in class_means.items()}
    closest_class, closest_distance = min(distances.items(), key=lambda x: x[1])
    if closest_distance < threshold:
        return closest_class
    return "No Gesture"

def is_embedding_far_enough(embedding, class_means, threshold):
    """
    Check if the embedding is at least 'threshold' distance from all class means.
    """
    temp = (np.linalg.norm(embedding - np.array(mean)) for mean in class_means.values())
    for te in temp:
        print(f"{te:.2f}", "", end="")
    return all(np.linalg.norm(embedding - np.array(mean)) >= threshold for mean in class_means.values())


# Initialize last_origin as a zero vector (3,)
last_origin = np.zeros(3)
rolling_window = []

def process_frame(frame, threshold=0.5, return_embedding=False):
    """
    Process a video frame, detect hand landmarks, and classify the hand gesture.
    Optionally returns the embedding for real-time visualization.
    """
    global last_frame_time
    global rolling_window

    current_time = time.time()
    if last_frame_time is not None:
        time_diff = current_time - last_frame_time
        # print(f"Time difference between frames: {time_diff:.4f} seconds")
    last_frame_time = current_time

    global last_origin

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            origin = landmarks[0]  # (3,)

            # Compute change in origin (difference from last frame's origin)
            delta_origin = origin - last_origin

            # Adjust landmarks by subtracting the origin and append the change in origin
            adjusted_landmarks = landmarks[1:] - origin  # Shape: (20, 3)
            input_data = np.concatenate((delta_origin[:2], adjusted_landmarks.flatten()))  # Shape: (63,)

            # Update last_origin to the current origin for the next frame
            last_origin = origin

            # Convert to PyTorch tensor
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

            # Pass through model
            rolling_window.append(input_data)
            if len(rolling_window) >= 41:
                rolling_window = rolling_window[-41:]
                rolling_window_array = np.array(rolling_window, dtype=np.float32)
                with torch.no_grad():
                    embedding = model.forward_once(torch.tensor(rolling_window_array).unsqueeze(0).to(device)).squeeze().cpu().numpy()  # Shape: (128,)
            else:
                return None, None



            # Normalize embedding for visualization (outdated)
            embedding_norm = embedding

            if is_embedding_far_enough(embedding_norm, class_means, 2 * threshold):
                print("Yes: Embedding is far enough from all class means.")
            else:
                print("No: Embedding is not far enough from all class means.")

            # Classify the gesture
            classification_result = classify_hand_landmarks(embedding_norm, class_means, threshold=0.5)

            if return_embedding:
                return classification_result, embedding_norm

            return classification_result

    # Reset last_origin to zero if no hand is detected
    last_origin = np.zeros(3)

    return "No Gesture Detected", None if return_embedding else "No Gesture Detected"
