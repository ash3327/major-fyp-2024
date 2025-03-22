import cv2
import sys
sys.path.append('.')

import numpy as np
from matplotlib import pyplot as plt

from feature_extractor import extract_features_from_clipped_region  # Ensure this function exists in feature_extractor.py

def extract_keypoints(frame):
    """
    Wrapper function to extract keypoints from a frame using the feature_extractor module.
    """
    features, _, _ = extract_features_from_clipped_region(frame)

    landmarks = features[3]

    return landmarks['pose'], landmarks['hands'][0], landmarks['hands'][1], landmarks['pose'][9:10,:]

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
        lmks = extract_keypoints(frame_rgb)

        # print(lmks)
        # Visualize keypoints on the frame
        for landmarks, c in zip(lmks, [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]):
            for p in landmarks:
                if len(p) == 3:
                    x, y, z = p
                else:
                    x, y = p
                    z = 0
                # Draw the circle
                cx, cy = int(x * frame.shape[1]), int(y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 3, c, -1)
                
                # Draw the vertical line to indicate the 3rd dimension (z)
                line_length = int(frame.shape[0] / 10)
                cz = int(z * line_length)
                cv2.line(frame, (cx, cy), (cx, cy - cz), c, 1)

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