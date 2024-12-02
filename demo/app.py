"""
Now do python demo/app.py instead of just python app.py.
"""

import os
import cv2
import json
import numpy as np
import threading
from flask import Flask, render_template, Response, jsonify, request
from gesture_recognition import HandGestureRecognizer
from pca_visualizer import PCAVisualizer
import base64
import argparse

# Create argument parser
parser = argparse.ArgumentParser(description='Hand Gesture Recognition Demo')
parser.add_argument('-m', '--model', type=str, choices=['kpt_contrastive', 'img_ce', 'img_triplet'],
                    default='kpt_contrastive', help='Model to use for gesture recognition')
args = parser.parse_args()

# Map model choice to config file
MODEL_CONFIGS = {
    'kpt_contrastive': 'demo/configs/kpt_ce_augmented_deep.yaml',
    'img_ce': 'demo/configs/img_crossentropy.yaml',
    'img_triplet': 'demo/configs/img_triplet.yaml'
}

config_path = MODEL_CONFIGS[args.model]
print(f"Using model config: {config_path}")

# Initialize Flask app with proper template folder
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Global variables
current_prediction = "No Gesture Detected"
current_embedding = None
current_predictions = []  # Store all predictions
current_frame = None
custom_gestures = {}  # Store custom gesture embeddings
gesture_counter = 1
include_alphabets = True  # Toggle for alphabet classification
is_webcam_mode = True  # New variable to track mode
lock = threading.Lock()

# Initialize models
try:
    print(f"Initializing model with config: {config_path}")
    model = HandGestureRecognizer(config_path)
    pca_viz = PCAVisualizer(model.config["class_means"])
    print(f"Successfully loaded model: {args.model}")
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    raise

# Ensure the custom gestures directory exists
CUSTOM_GESTURES_DIR = os.path.join('demo/static', 'custom_gestures')
os.makedirs(CUSTOM_GESTURES_DIR, exist_ok=True)

# Video capture function
def gen_frames():
    global current_prediction, current_embedding, current_predictions, current_frame
    
    try:
        print("Opening video capture...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open video capture")
            return
            
        print("Video capture opened successfully")
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame")
                break
                
            # Flip frame horizontally for a mirror-like effect
            frame = cv2.flip(frame, 1)
            
            # Store current frame for gesture capture
            with lock:
                current_frame = frame.copy()
            
            # Process the frame for gesture recognition
            if custom_gestures:
                # Create a combined dictionary of model's class means and custom gestures
                all_class_means = model.class_means.copy()
                all_class_means.update(custom_gestures)
                result = model.process_frame(frame, return_landmarks=True, supply_class_means=all_class_means)
            else:
                result = model.process_frame(frame, return_landmarks=True)
            
            # Update the prediction result and embedding
            with lock:
                if isinstance(result, tuple):
                    pred, emb, all_preds = result
                    # print(f"Prediction: {pred}")
                    # Check if we should skip alphabets
                    if not include_alphabets and pred.isalpha():
                        current_prediction = "No Class"
                    else:
                        current_prediction = pred
                    current_embedding = emb
                    current_predictions = all_preds
                else:
                    current_prediction = result
                    current_embedding = None
                    current_predictions = []
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Could not encode frame")
                break
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
    except Exception as e:
        print(f"Error in video capture: {str(e)}")
        
    finally:
        print("Releasing video capture...")
        cap.release()

# API endpoint for current prediction
@app.route('/prediction')
def get_prediction():
    global current_prediction, current_predictions
    with lock:  # Ensure thread-safe access
        return jsonify({
            'prediction': current_prediction,
            'all_predictions': current_predictions
        })

# API endpoint for PCA data
@app.route('/pca_data')
def get_pca_data():
    global current_embedding
    with lock:
        if current_embedding is not None:
            # If current_embedding is a batch with one sample, get the first sample
            if current_embedding.ndim > 1 and current_embedding.shape[0] == 1:
                embedding_data = current_embedding[0]
            else:
                embedding_data = current_embedding
            viz_data = pca_viz.get_visualization_data(embedding_data)
        else:
            viz_data = pca_viz.get_visualization_data()
    return jsonify(viz_data)

@app.route('/store_gesture', methods=['POST'])
def store_gesture():
    global gesture_counter, current_frame, current_embedding
    
    with lock:
        if current_frame is None or current_embedding is None:
            return jsonify({'error': 'No frame or embedding available'}), 400
        
        # Generate gesture name
        gesture_name = f"Gesture {gesture_counter}"
        gesture_counter += 1
        
        # Save the frame as an image
        img_filename = f"{gesture_name.lower().replace(' ', '_')}.jpg"
        img_path = os.path.join(CUSTOM_GESTURES_DIR, img_filename)
        cv2.imwrite(img_path, current_frame)
        
        # Store the embedding
        custom_gestures[gesture_name] = current_embedding
        
        # Update PCA visualizer with new gesture
        pca_viz.add_class(gesture_name, current_embedding)
        
        return jsonify({
            'gesture_name': gesture_name,
            'image_path': f'static/custom_gestures/{img_filename}'
        })

@app.route('/toggle_alphabets', methods=['POST'])
def toggle_alphabets():
    global include_alphabets
    include_alphabets = not include_alphabets
    return jsonify({'include_alphabets': include_alphabets})

@app.route('/get_stored_gestures')
def get_stored_gestures():
    gestures = []
    for gesture_name in custom_gestures.keys():
        img_filename = f"{gesture_name.lower().replace(' ', '_')}.jpg"
        gestures.append({
            'name': gesture_name,
            'image_path': f'static/custom_gestures/{img_filename}'
        })
    return jsonify(gestures)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    global current_frame, current_prediction, current_embedding, is_webcam_mode
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
            
        # Read and process the uploaded image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400
            
        # Update current frame
        with lock:
            current_frame = frame
            is_webcam_mode = False
            
            # Process the frame for gesture recognition
            if custom_gestures:
                all_class_means = model.class_means.copy()
                all_class_means.update(custom_gestures)
                result = model.process_frame(frame, return_landmarks=True, supply_class_means=all_class_means)
            else:
                result = model.process_frame(frame, return_landmarks=True)
                
            # Update prediction and embedding
            if isinstance(result, tuple):
                pred, emb, all_preds = result
                if not include_alphabets and pred.isalpha():
                    current_prediction = "No Class"
                else:
                    current_prediction = pred
                current_embedding = emb
                current_predictions = all_preds
            else:
                current_prediction = result
                current_embedding = None
                current_predictions = []
                
        # Convert the processed frame to base64 for display
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'prediction': current_prediction
        })
        
    except Exception as e:
        print(f"Error processing uploaded image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    global is_webcam_mode
    is_webcam_mode = not is_webcam_mode
    return jsonify({'is_webcam_mode': is_webcam_mode})

@app.route('/current_frame')
def get_current_frame():
    global current_frame
    
    if current_frame is None:
        return jsonify({'error': 'No frame available'}), 404
        
    # Convert the current frame to base64
    _, buffer = cv2.imencode('.jpg', current_frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_base64}'
    })

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
