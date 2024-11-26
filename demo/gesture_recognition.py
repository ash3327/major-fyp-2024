import json
import numpy as np
import yaml

import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

class HandGestureRecognizer:
    def __init__(self, config_path="configs/old_model.yaml"):
        """
        Initialize the HandGestureRecognizer with the embedding model and class means.
        """
        self.config = yaml.safe_load(open(config_path))
        self.embedding_model = load_model(self.config["model"])
        with open(self.config["class_means"], "r") as f:
            self.class_means = json.load(f)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    def classify_hand_landmarks(self, landmarks, threshold=0.5, return_landmarks=True):
        """
        Classify hand gesture based on adjusted landmarks.
        """
        # Handle both single input and batch input
        if landmarks.ndim == 1:
            landmarks = np.expand_dims(landmarks, axis=0)
        
        embeddings = self.embedding_model.predict(landmarks)
        
        distances = np.array([[np.linalg.norm(embedding - np.array(mean)) for label, mean in self.class_means.items()] for embedding in embeddings])
        closest_indices = np.argmin(distances, axis=1)
        closest_distances = np.min(distances, axis=1)
        
        labels = list(self.class_means.keys())
        closest_labels = [labels[i] if d < threshold else "No Class" for i, d in zip(closest_indices, closest_distances)]
        
        return (closest_labels[0] if len(closest_labels) == 1 else closest_labels), embeddings

    def process_frame(self, frame, threshold=0.5, return_landmarks=False):
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
        results = self.hands.process(frame_rgb)

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
                classification_result, pred_embedding = self.classify_hand_landmarks(flattened_landmarks, return_landmarks=True)
                if return_landmarks:
                    return classification_result, pred_embedding
                return classification_result
        
        # If no hand is detected
        return "No Gesture Detected"
