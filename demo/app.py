from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import numpy as np
import os
from gesture_recognition import HandGestureRecognizer
from pca_visualizer import PCAVisualizer
import json
import base64

# Initialize Flask app
app = Flask(__name__)

# Global variables
current_prediction = "No Gesture Detected"
current_embedding = None
current_frame = None
custom_gestures = {}  # Store custom gesture embeddings
gesture_counter = 1
include_alphabets = True  # Toggle for alphabet classification
lock = threading.Lock()

# Initialize models
model = HandGestureRecognizer('configs/ce_augmented_deep.yaml')
pca_viz = PCAVisualizer(model.config["class_means"])

# Ensure the custom gestures directory exists
CUSTOM_GESTURES_DIR = os.path.join('static', 'custom_gestures')
os.makedirs(CUSTOM_GESTURES_DIR, exist_ok=True)

# Video capture function
def gen_frames():
    global current_prediction, current_embedding, current_frame
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)

        # Store current frame
        with lock:
            current_frame = frame.copy()

        # Process the frame for gesture recognition
        result = model.process_frame(frame, return_landmarks=True)
        
        # Update the prediction result and embedding
        with lock:
            if isinstance(result, tuple):
                pred, emb = result
                # Check custom gestures first
                if custom_gestures:
                    min_dist = float('inf')
                    closest_gesture = None
                    for gesture_name, gesture_emb in custom_gestures.items():
                        dist = np.linalg.norm(emb - gesture_emb)
                        if dist < min_dist:
                            min_dist = dist
                            closest_gesture = gesture_name
                    # If a custom gesture is closer than the threshold, use it
                    if min_dist < 0.5:  # You may need to adjust this threshold
                        current_prediction = closest_gesture
                    else:
                        # Check if we should skip alphabets
                        if not include_alphabets and pred.isalpha():
                            current_prediction = "No Gesture Detected"
                        else:
                            current_prediction = pred
                else:
                    # Check if we should skip alphabets
                    if not include_alphabets and pred.isalpha():
                        current_prediction = "No Gesture Detected"
                    else:
                        current_prediction = pred
                current_embedding = emb
            else:
                current_prediction = result
                current_embedding = None

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# API endpoint for current prediction
@app.route('/prediction')
def get_prediction():
    global current_prediction
    with lock:  # Ensure thread-safe access
        return jsonify(prediction=current_prediction)

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
            'image_path': f'custom_gestures/{img_filename}'
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
            'image_path': f'custom_gestures/{img_filename}'
        })
    return jsonify(gestures)

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
