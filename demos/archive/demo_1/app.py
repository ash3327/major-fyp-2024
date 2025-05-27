from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import cv2
import threading
import time
import numpy as np
import pickle
import torch
import umap
from gesture_recognition import process_frame

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
current_prediction = "No Gesture Detected"
latest_frame = None
frame_lock = threading.Lock()
prediction_lock = threading.Lock()
stored_umap_data_sent = False

# Capture video from webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

# Gesture recognition loop
def recognize_gesture():
    global current_prediction, latest_frame
    frame_rate = 60
    frame_time = 1 / frame_rate

    while True:
        start_time = time.time()

        with frame_lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        with prediction_lock:
            # Run gesture recognition
            classification_result, embedding = process_frame(frame_copy, return_embedding=True)

        # Update prediction safely
        with prediction_lock:
            current_prediction = classification_result


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
