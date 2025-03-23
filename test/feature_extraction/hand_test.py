import cv2
import numpy as np
import sys
sys.path.append('.')

from feature_extractor import extract_features_from_clipped_region, get_image

def extract_keypoints(frame):
    """
    Wrapper function to extract landmarks from a frame using extract_features_from_clipped_region.
    Returns:
      - pose: np.array of shape [N,2] (normalized) for pose (if available)
      - hand0: np.array of shape [M,3] for first hand (we use x,y)
      - hand1: np.array of shape [M,3] for second hand (we use x,y)
    """
    features, _, _ = extract_features_from_clipped_region(frame)
    landmarks = features[3]
    return landmarks.get('pose', np.zeros((0,2))), landmarks['hands'][0], landmarks['hands'][1]

def plot_hand(hand_landmarks, m):
    """
    Generates an m x m image plotting the hand landmarks in 2D.
    
    The hand landmarks are shifted such that the first landmark becomes (0,0),
    then normalized to [-1,1]. These normalized coordinates are mapped to pixel values
    in [0, m-1] (with y-axis inverted for display).
    
    Args:
      hand_landmarks (np.array): shape [M, 3] or [M,2]. Only x,y are used.
      m (int): The size of the square plot (m x m).
      
    Returns:
      img (np.array): An image (m x m x 3) with the landmarks plotted.
    """
    # Ensure we have at least one landmark
    if hand_landmarks.shape[0] == 0:
        return 255 * np.ones((m, m, 3), dtype=np.uint8)
    
    # Shift landmarks: subtract the first point from all points
    shifted = hand_landmarks[:, :2] - hand_landmarks[0, :2]
    
    # Normalize to [-1, 1] by dividing by the maximum absolute value among x and y (avoid division by zero)
    max_val = np.max(np.abs(shifted)) if np.max(np.abs(shifted)) != 0 else 1.0
    normalized = shifted# / max_val  # now in [-1,1]
    
    # Create a blank white image
    img = 255 * np.ones((m, m, 3), dtype=np.uint8)
    
    # Map normalized coordinates [-1, 1] to pixel coordinates [0, m-1]\n    # For x: x_pixel = ((x + 1)/2) * (m-1)\n    # For y: y_pixel = ((1 - (y + 1)/2)) * (m-1) to flip the y-axis
    pixels = np.zeros_like(normalized)
    pixels[:, 0] = ((normalized[:, 0] + 1) / 2) * (m - 1)
    pixels[:, 1] = ((1 - (normalized[:, 1] + 1) / 2)) * (m - 1)
    
    # Draw each landmark as a circle
    for (x, y) in pixels.astype(np.int32):
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    
    return img

def main():
    # Set base size m (each small plot is m x m; left frame is (2*m)x(2*m))
    m = 200
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Convert captured frame to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract landmarks from the frame
        pose, hand0, hand1 = lmks = extract_keypoints(frame_rgb)

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
        
        # Prepare left frame: resize original frame to a square of size (2*m) x (2*m)
        left_frame = cv2.resize(frame, (2*m*frame_rgb.shape[1]//frame_rgb.shape[0], 2*m))
        
        # Prepare right frame: create two subplots, one for each hand, each of size m x m
        top_plot = plot_hand(hand0, m)
        bottom_plot = plot_hand(hand1, m)
        right_frame = np.vstack((top_plot, bottom_plot))  # vertical stack gives (2*m) x m
        
        # Pad or resize the right frame to make it square (2*m x 2*m) if needed
        if right_frame.shape[1] < 2*m:
            pad_total = (2*m) - right_frame.shape[1]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            right_frame = cv2.copyMakeBorder(right_frame, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255,255,255])
        else:
            right_frame = cv2.resize(right_frame, (2*m, 2*m))
        
        # Combine left and right frames side-by-side
        combined = np.hstack((left_frame, right_frame))
        cv2.imshow('Original (left) | Hand Landmark Plot (right)', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
