import sys
import json
import numpy as np
import yaml
sys.path.append(".")

import cv2
import torch
import mediapipe as mp
from tensorflow.keras.models import load_model

from libs.models.efficientnet_mlp_classifier import SimpleClassifier
from libs.model_loader import load_model as torch_load_model

class HandGestureRecognizer:
    NO_GESTURE_DETECTED = "No Gesture Detected"

    def __init__(self, config_path="configs/old_model.yaml"):
        """
        Initialize the HandGestureRecognizer with the embedding model and class means.
        """
        self.config = yaml.safe_load(open(config_path))
        self.framework = self.config["framework"]
        self.channels = self.config["channels"]

        self.embedding_model = self.load_model(self.config["model"])
        try:
            self.class_means = json.load(open(self.config["class_means"]))
        except FileNotFoundError:
            print("No class means found. Using empty class means.")
            self.class_means = {}

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    def load_model(self, model_path, device='cuda'):
        if self.framework == "tensorflow":
            return load_model(model_path)
        elif self.framework == "torch":
            loaded_model = SimpleClassifier(num_classes=27)
            loaded_model = torch_load_model(loaded_model, model_path)
            loaded_model.to(device)
            return loaded_model
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def extract_embeddings(self, input):
        if self.framework == "tensorflow":
            return self.embedding_model.predict(input, verbose=0)
        elif self.framework == "torch":
            self.embedding_model.eval()
            with torch.no_grad():
                input_tensor = torch.tensor(input, dtype=torch.float32)/256
                input_tensor = input_tensor.permute(2, 0, 1)
                input_tensor = input_tensor.unsqueeze(0)
                input_tensor = input_tensor.to('cuda')
                _, features = self.embedding_model(input_tensor)
                return features.cpu().numpy()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def classify_hand_landmarks(self, landmarks, threshold=0.5, return_landmarks=True, supply_class_means=None):
        """
        Classify hand gesture based on adjusted landmarks.
        """
        class_means = supply_class_means if supply_class_means else self.class_means
            
        embeddings = self.extract_embeddings(landmarks)
        
        if len(class_means) == 0:
            return self.NO_GESTURE_DETECTED, embeddings

        distances = np.array([[np.linalg.norm(embedding - np.array(mean)) for label, mean in class_means.items()] for embedding in embeddings])
        closest_indices = np.argmin(distances, axis=1)
        closest_distances = np.min(distances, axis=1)
        
        labels = list(class_means.keys())
        closest_labels = [labels[i] if d < threshold else "No Class" for i, d in zip(closest_indices, closest_distances)]
        
        return (closest_labels[0] if len(closest_labels) == 1 else closest_labels), embeddings

    def process_frame(self, frame, threshold=0.5, return_landmarks=False, mode='bgr', supply_class_means=None):
        """
        Process a video frame, detect hand landmarks, adjust coordinates, 
        and classify the hand gesture.
        
        Args:
            frame: A single video frame (BGR format).
            threshold: Distance threshold for classification.
        
        Returns:
            classification_result: The predicted gesture or "No Gesture Detected".
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if mode == 'bgr' else frame
        
        if self.channels == "keypoints":
            results = self.hands.process(frame_rgb)

            model_inputs = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                    # Subtract landmark 0 (set as origin)
                    origin = landmarks[0]
                    adjusted_landmarks = landmarks[1:] - origin  # Use landmarks 1 to 20 only

                    # Flatten the adjusted landmarks
                    flattened_landmarks = adjusted_landmarks.flatten()

                    # Handle both single input and batch input
                    if flattened_landmarks.ndim == 1:
                        flattened_landmarks = np.expand_dims(flattened_landmarks, axis=0)
                    
                    model_inputs = flattened_landmarks
                    break

        elif self.channels == "images":
            model_inputs = frame_rgb

        if model_inputs is not None:
            # Classify the gesture
            classification_result, pred_embedding = self.classify_hand_landmarks(model_inputs, return_landmarks=True, supply_class_means=supply_class_means)
            if return_landmarks:
                return classification_result, pred_embedding
            return classification_result
        
        # If no hand is detected
        if return_landmarks:
            return self.NO_GESTURE_DETECTED, None
        return self.NO_GESTURE_DETECTED
        
        
