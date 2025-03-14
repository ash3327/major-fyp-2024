from ultralytics import YOLO

# Load the model and run inference on the webcam
YOLO('model/hand/best.pt').predict(source=0, show=True, save=True, project='runs/pose/output', name='tests')