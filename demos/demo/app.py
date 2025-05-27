import os
import cv2
import json
import numpy as np
import threading
import base64
import argparse
from flask import Flask, render_template, Response, jsonify, request
from gesture_recognition import HandGestureRecognizer
from pca_visualizer import PCAVisualizer

# Create argument parser
parser = argparse.ArgumentParser(description='Hand Gesture Recognition Demo')
parser.add_argument('-m', '--model', type=str, default='kpt_contrastive_1',
                    help='Model to use for gesture recognition, advised list under demo/configs/model_configs/ (file name only, without .yaml suffix)')
args = parser.parse_args()

# Map model choice to config file
config_path = f'configs/model_configs/{args.model}.yaml'
if not os.path.exists(config_path):
    print(f"Error: Config file {config_path} does not exist.")
    exit(1)
print(f"Using model config: {config_path}\n")

# Initialize Flask app
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Global variables
current_prediction = "No Gesture Detected"
current_embedding = None
current_predictions, current_frame = [], None
custom_gestures, gesture_counter = {}, 1
include_alphabets, is_webcam_mode = True, True
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
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video capture")
            return
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            with lock:
                current_frame = frame.copy()
            process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        print(f"Error in video capture: {str(e)}")
    finally:
        cap.release()

# Frame processing function
def process_frame(frame):
    global current_prediction, current_embedding, current_predictions
    supply_class_means = {**model.class_means, **custom_gestures} if include_alphabets else custom_gestures
    result = model.process_frame(frame, return_landmarks=True, supply_class_means=supply_class_means)
    with lock:
        if isinstance(result, tuple):
            pred, emb, all_preds = result
            current_prediction = "No Class" if not include_alphabets and pred.isalpha() else pred
            current_embedding, current_predictions = emb, all_preds
        else:
            current_prediction, current_embedding, current_predictions = result, None, []

# Generic response for image processing
def process_uploaded_image(file):
    global current_frame, is_webcam_mode
    image_bytes = file.read()
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid image format'}), 400
    with lock:
        current_frame, is_webcam_mode = frame, False
        process_frame(frame)
    _, buffer = cv2.imencode('.jpg', frame)
    return jsonify({'success': True, 'image': f'data:image/jpeg;base64,{base64.b64encode(buffer).decode()}','prediction': current_prediction, 'all_predictions': current_predictions})

@app.route('/prediction')
def get_prediction():
    with lock:
        return jsonify({'prediction': current_prediction, 'all_predictions': current_predictions})

@app.route('/pca_data')
def get_pca_data():
    with lock:
        embedding_data = current_embedding[0] if current_embedding is not None and current_embedding.ndim > 1 and current_embedding.shape[0] == 1 else current_embedding
        return jsonify(pca_viz.get_visualization_data(embedding_data))

@app.route('/store_gesture', methods=['POST'])
def store_gesture():
    global gesture_counter
    with lock:
        if current_frame is None or current_embedding is None:
            return jsonify({'error': 'No frame or embedding available'}), 400
        gesture_name = f"Gesture {gesture_counter}"
        gesture_counter += 1
        img_filename = f"{gesture_name.lower().replace(' ', '_')}.jpg"
        img_path = os.path.join(CUSTOM_GESTURES_DIR, img_filename)
        cv2.imwrite(img_path, current_frame)
        custom_gestures[gesture_name] = current_embedding
        pca_viz.add_class(gesture_name, current_embedding)
        return jsonify({'gesture_name': gesture_name, 'image_path': f'static/custom_gestures/{img_filename}'})

@app.route('/toggle_alphabets', methods=['POST'])
def toggle_alphabets():
    global include_alphabets
    include_alphabets = not include_alphabets
    return jsonify({'include_alphabets': include_alphabets})

@app.route('/get_stored_gestures')
def get_stored_gestures():
    return jsonify([{'name': name, 'image_path': f'static/custom_gestures/{name.lower().replace(" ", "_")}.jpg'} for name in custom_gestures.keys()])

@app.route('/upload_image', methods=['POST'])
def upload_image():
    return process_uploaded_image(request.files['image']) if 'image' in request.files else jsonify({'error': 'No image uploaded'}), 400

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    global is_webcam_mode
    is_webcam_mode = not is_webcam_mode
    return jsonify({'is_webcam_mode': is_webcam_mode})

@app.route('/current_frame')
def get_current_frame():
    with lock:
        if current_frame is None:
            return jsonify({'error': 'No frame available'}), 404
        _, buffer = cv2.imencode('.jpg', current_frame)
        return jsonify({'image': f'data:image/jpeg;base64,{base64.b64encode(buffer).decode()}'})

@app.route('/model_info')
def get_model_info():
    return jsonify({'model_name': args.model, 'config_path': config_path})

@app.route('/')
def index():
    return render_template('index.html', model_name=args.model)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)
