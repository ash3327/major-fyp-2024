"""
This module transforms images to gestures.
"""
import mediapipe as mp
import cv2
import numpy as np

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
def extract_hand_keypoints(image):
    try:
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
            image = (image * 255).astype(np.uint8)  # Scale and convert to uint8
            
            ##########
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_clahe = clahe.apply(l)
            
            # Merge channels
            lab_clahe = cv2.merge((l_clahe, a, b))
            
            # Convert back to RGB
            image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            ##########

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            
            # Create a blank image with same dimensions
            output_image = np.zeros(image.shape, dtype=np.uint8)
            
            keypoints_coords = None
            if results.multi_hand_landmarks:
                keypoints_coords = []
                kpt_coords = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for i, point in enumerate(hand_landmarks.landmark):
                        x, y = int(point.x * image.shape[1]), int(point.y * image.shape[0])
                        keypoints_coords.append((x, y))
                        kpt_coords.append((point.x, point.y, point.z))
                keypoints_coords = np.array(keypoints_coords)
                kpt_coords = np.array(kpt_coords)

                draw_keypoints_on_image(output_image, keypoints_coords)
            else:
                kpt_coords = np.zeros((21, 3))
            
            return image, output_image, kpt_coords
    except Exception as e:
        print(f'Exception {e} occurred')

def draw_keypoints_on_image(image, keypoints_coords):
    if keypoints_coords is not None:
        for pair in [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                     (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                     (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                     (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                     (0, 17), (17, 18), (18, 19), (19, 20)]:  # Pinky
            start = keypoints_coords[pair[0]]
            end = keypoints_coords[pair[1]]
            cv2.line(image, start, end, (255, 255, 255), 2)
        
        for i, (x, y) in enumerate(keypoints_coords):
            if i == 0:  # Palm
                cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
            elif i == 1 or i == 2 or i == 3 or i == 4:  # Thumb
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # Red
            elif i == 5 or i == 6 or i == 7 or i == 8:  # Index finger
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green
            elif i == 9 or i == 10 or i == 11 or i == 12:  # Middle finger
                cv2.circle(image, (x, y), 5, (0, 255, 255), -1)  # Cyan
            elif i == 13 or i == 14 or i == 15 or i == 16:  # Ring finger
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Blue
            elif i == 17 or i == 18 or i == 19 or i == 20:  # Pinky
                cv2.circle(image, (x, y), 5, (255, 0, 255), -1)  # Magenta
    return image

def extract_keypoints(image):
    try:
        with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False  # Image is no longer writeable
            results = holistic.process(image)  # Make prediction
            results = _extract_keypoints(results)
            return results
    except Exception as e: 
        print(f'Exception {e} occurred')

def _extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] if res.visibility > 0.5 else [0, 0, 0] for res in
                     results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros([33, 3])
    # face = np.array([[res.x, res.y, res.z] for res in
    #                  [results.face_landmarks.landmark[i] for i in
    #                   FACEMESH_LIPS]]) if results.face_landmarks else np.zeros(
    #     [len(FACEMESH_LIPS), 3])
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros(
        [21, 3])
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros(
        [21, 3])
    return np.concatenate([pose, lh, rh])