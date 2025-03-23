"""
Train the yolo hand keypoints model (as in model/hand/best.pt)

Utilizes the default yolo hand keypoints detection dataset, 
with poor detection accuracy since only a small dataset is used.

Instructions: 
- python training/yolo_hand/yolo_keypoints.py
- wait.
- go to runs/train/hand<run_id>/weights/best.pt and replace the current model
"""

from ultralytics import YOLO
from multiprocessing import freeze_support

# Load a model
model = YOLO("model/yolo11n-pose.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    freeze_support()
    # Train the model
    results = model.train(
        data="hand-keypoints.yaml",  # Path to your custom dataset YAML file
        epochs=100,                  # Number of epochs to train
        imgsz=640,                   # Image size for training
        project="runs/train",        # Directory to save training results
        name="hand"          # Name of the subdirectory to save the model
    )