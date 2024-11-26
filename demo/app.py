from flask import Flask, render_template, Response, jsonify
import cv2
import threading
from gesture_recognition import HandGestureRecognizer
from pca_visualizer import PCAVisualizer
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Global variables for prediction and embedding
current_prediction = "No Gesture Detected"
current_embedding = None
lock = threading.Lock()  # Lock for thread-safe updates

# Initialize models
model = HandGestureRecognizer('configs/ce_augmented_deep.yaml')
pca_viz = PCAVisualizer(model.config["class_means"])

# Video capture function
def gen_frames():
    global current_prediction, current_embedding
    cap = cv2.VideoCapture(0)  # Access webcam
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)

        # Process the frame for gesture recognition
        result = model.process_frame(frame, return_landmarks=True)
        
        # Update the prediction result and embedding
        with lock:
            if isinstance(result, tuple):
                current_prediction, current_embedding = result
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
