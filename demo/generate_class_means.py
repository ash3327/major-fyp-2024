"""
Only execute at project root.

Execution format: 
python demo/generate_class_means.py demo/configs/img_crossentropy.yaml

NOTE: The changes are overriding.
"""
import argparse
import cv2
import json
import numpy as np
import os
import sys
import yaml
from tqdm import tqdm

sys.path.append(".")
from gesture_recognition import HandGestureRecognizer

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_image_data(data_path):
    """Load and preprocess images from directory structure."""
    X = []
    y = []
    classes = sorted(os.listdir(data_path))
    
    for class_idx, class_name in enumerate(tqdm(classes, desc="Loading images")):
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                # Read and preprocess image
                img = cv2.imread(img_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.resize(img, target_size)
                # img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
                
                X.append(img)
                y.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    return np.array(X), np.array(y), classes

def compute_class_means(model, X, y, classes):
    """Compute mean embeddings for each class."""
    class_means = {}
    total_samples = len(y)
    with tqdm(total=total_samples, desc="Computing class means") as pbar:
        for idx, cls in enumerate(classes):
            indices = np.where(y == idx)[0]
            if len(indices) == 0:
                continue
            embeddings_list = []
            for i in indices:
                _, embedding = model.process_frame(X[i], supply_class_means=[], return_landmarks=True)
                if embedding is not None:
                    embeddings_list.append(embedding)
                pbar.update(1)
            if len(embeddings_list) > 0:
                embeddings = np.concatenate(embeddings_list, axis=0)
                class_means[cls] = np.mean(embeddings, axis=0).tolist()
    return class_means

def main():
    parser = argparse.ArgumentParser(description='Generate class means for a model')
    parser.add_argument('config_path', help='Path to the model config YAML file')
    parser.add_argument('--data', default='kaggle/input/synthetic-asl-alphabet/Test_Alphabet',
                      help='Path to the training data directory')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)
    
    # Load image data
    print("Loading data...")
    X, y, classes = load_image_data(args.data)
    print(f"Loaded {len(X)} images from {len(classes)} classes")

    # Initialize model
    print("Initializing model...")
    model = HandGestureRecognizer(config_path=args.config_path)
    
    # Compute class means
    class_means = compute_class_means(model, X, y, classes)
    
    # Save class means to JSON
    output_path = config['class_means']
    with open(output_path, 'w') as f:
        json.dump(class_means, f)
    print(f"Class means saved to {output_path}")

if __name__ == '__main__':
    main()