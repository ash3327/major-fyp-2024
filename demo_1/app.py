from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import cv2
import threading
import time
import numpy as np
import pickle
import torch
import umap
from collections import deque
from gesture_recognition import process_frame
from training.temporal.model import LSTMGestureModel_Hierachical_Windowed
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
current_prediction = "No Gesture Detected"
latest_frame = None
frame_lock = threading.Lock()
prediction_lock = threading.Lock()
stored_umap_data_sent = False

# Frame buffer for temporal processing
frame_buffer = deque(maxlen=32)  # Store 32 frames for temporal processing
embeddings_buffer = deque(maxlen=32)  # Store embeddings for 32 frames

# Model configuration
feature_dim = 128 
window_size = 16
window_stride = window_size // 2
num_classes = 14

# Load LSTM model and centroids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMGestureModel_Hierachical_Windowed(
    output_dim=feature_dim, 
    window_size=window_size,
    window_stride=window_stride
).to(device)

model_checkpoint_path = 'runs/ipn_classifiers/v1/YOUR_CHECKPOINT_ID/checkpoints/last.pth'
centroids_path = 'runs/ipn_classifiers/v1/YOUR_CHECKPOINT_ID/checkpoints/centroids_last.pth'

model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
centroids = torch.load(centroids_path, map_location=device)
model.eval()

# Capture video from webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

# Gesture recognition loop
def recognize_gesture():
    global current_prediction, latest_frame, frame_buffer, embeddings_buffer
    frame_rate = 60
    frame_time = 1 / frame_rate

    while True:
        start_time = time.time()

        with frame_lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        # Process frame and get embedding
        _, embedding = process_frame(frame_copy, return_embedding=True)
        
        # Add to buffers
        frame_buffer.append(frame_copy)
        embeddings_buffer.append(embedding)

        # When we have enough frames, run LSTM prediction
        if len(embeddings_buffer) == window_size:
            # Convert buffer to tensor
            embeddings_tensor = torch.from_numpy(
                np.stack(list(embeddings_buffer))
            ).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                # Get hierarchical LSTM features
                outputs = model(embeddings_tensor)
                # Normalize features
                features = F.normalize(outputs.squeeze(0), dim=-1)
                # Get cosine similarity with centroids
                sim = torch.mm(features, centroids.T)
                # Get prediction from highest similarity
                prediction = sim.argmax(dim=-1).item()
                
                # Map prediction index to gesture label
                # TODO: Replace with your gesture mapping from IPN dataset
                gesture_labels = [str(i) for i in range(num_classes)]
                predicted_gesture = gesture_labels[prediction]

                # Update prediction safely
                with prediction_lock:
                    current_prediction = predicted_gesture

        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)

# Start gesture recognition thread
threading.Thread(target=recognize_gesture, daemon=True).start()

# Video streaming generator
def gen_frames():
    global latest_frame
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        with frame_lock:
            latest_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/prediction')
def get_prediction():
    """ API endpoint for gesture prediction. """
    with prediction_lock:
        return jsonify(prediction=current_prediction)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """ Video stream route. """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=False, port=5001)
