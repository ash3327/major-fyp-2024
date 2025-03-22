import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def process_input(input_file=None):
    # Determine input source
    if input_file is None:
        cap = cv2.VideoCapture(0)  # Webcam
        is_video = True
    else:
        if input_file.lower().endswith(('.avi', '.mp4')):
            cap = cv2.VideoCapture(input_file)  # Video file
            is_video = True
        else:
            cap = cv2.VideoCapture()  # Image file
            is_video = False

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        if is_video:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of file or empty frame.")
                    break

                # Process the frame
                process_frame(frame, holistic, is_video=True)
        else:
            # Read and process the image
            frame = cv2.imread(input_file)
            if frame is None:
                print("Error reading the image file.")
                return
            process_frame(frame, holistic, is_video=False)

    # Release the input source and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame, holistic, is_video=False):
    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the frame with Mediapipe Holistic
    results = holistic.process(image)

    # Convert the image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw landmarks on the frame
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

    # Display the frame
    cv2.imshow('Mediapipe Holistic', image)

    # Break the loop on pressing 'q'
    if cv2.waitKey(10 if is_video else 0) & 0xFF == ord('q'):
        return

# Example usage:
# process_input()  # Use webcam

# process_input("data/raw/synthetic-asl-alphabet/Test_Alphabet/A/4ab5c2a9-7a18-4453-aed3-f0af875f69e2.rgb_0000.png")
# process_input("data/raw/senz3d_dataset/acquisitions/S1/G11/30-color.png")
process_input("data/raw/hands_dataset/Subject1/Subject1/1007_color.png")
# process_input("data/raw/hands_dataset/Subject1/Subject1/2296_color.png")
# process_input("data/raw/hands_dataset/Subject1/Subject1/3379_color.png")
# process_input("data/raw/hands_dataset/Subject2/Subject2/2445_color.png")
# process_input("data/raw/ph2014-handshape/test/images/final_phoenix_noPause_noCompound_lefthandtag_noClean/30July_2010_Friday_tagesschau_default-7/1/.png_fn000135-0.png")  # Use video or image file