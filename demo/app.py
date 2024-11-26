from flask import Flask, render_template, Response, jsonify
import cv2
import threading
from gesture_recognition import HandGestureRecognizer

# Initialize Flask app
app = Flask(__name__)

# Global variables for prediction
current_prediction = "No Gesture Detected"
lock = threading.Lock()  # Lock for thread-safe prediction updates

# model = HandGestureRecognizer('configs/old_model.yaml')
model = HandGestureRecognizer('configs/ce_augmented_deep.yaml')

# Video capture function
def gen_frames():
    global current_prediction
    cap = cv2.VideoCapture(0)  # Access webcam
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)

        # Process the frame for gesture recognition
        classification_result = model.process_frame(frame)

        # Update the prediction result
        with lock:
            current_prediction = classification_result

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
