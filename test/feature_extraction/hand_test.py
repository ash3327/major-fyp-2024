import cv2
import sys
sys.path.append('.')
from feature_extractor import extract_features_from_image  # Ensure this function exists in feature_extractor.py

def extract_keypoints(frame):
    """
    Wrapper function to extract keypoints from a frame using the feature_extractor module.
    """
    _, _, _, landmarks = extract_features_from_image(frame)
    keypoints = []
    for hand in landmarks['hands']:
        for point in hand:
            keypoints.append((point[0] * frame.shape[1], point[1] * frame.shape[0]))  # Scale back to image dimensions
    for point in landmarks['pose']:
        keypoints.append((point[0] * frame.shape[1], point[1] * frame.shape[0]))
    return keypoints

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert frame to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to extract keypoints
        keypoints = extract_keypoints(frame_rgb)

        # Visualize keypoints on the frame
        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Display the resulting frame
        cv2.imshow('Keypoint Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()