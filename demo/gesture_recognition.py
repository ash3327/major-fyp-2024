import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load the embedding model and class means
embedding_model = load_model("embedding_model.h5")
with open("class_means.json", "r") as f:
    class_means = json.load(f)

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def classify_hand_landmarks(landmarks, embedding_model, class_means, threshold=0.5):
    """
    Classify hand gesture based on adjusted landmarks.
    """
    embedding = embedding_model.predict(np.expand_dims(landmarks, axis=0))[0]
    distances = {cls: np.linalg.norm(embedding - np.array(mean)) for cls, mean in class_means.items()}
    closest_class, closest_distance = min(distances.items(), key=lambda x: x[1])
    if closest_distance < threshold:
        return closest_class
    return "No Class"

def process_frame(frame, threshold=0.5):
    """
    Process a video frame, detect hand landmarks, adjust coordinates, 
    and classify the hand gesture.
    
    Args:
        frame: A single video frame (BGR format).
        threshold: Distance threshold for classification.
    
    Returns:
        classification_result: The predicted gesture or "No Gesture Detected".
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Subtract landmark 0 (set as origin)
            origin = landmarks[0]
            adjusted_landmarks = landmarks[1:] - origin  # Use landmarks 1 to 20 only

            # Flatten the adjusted landmarks
            flattened_landmarks = adjusted_landmarks.flatten()

            # Classify the gesture
            return classify_hand_landmarks(flattened_landmarks, embedding_model, class_means)

    # If no hand is detected
    return "No Gesture Detected"
