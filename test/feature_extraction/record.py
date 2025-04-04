import cv2
from pathlib import Path
import time

def record_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define the codec and create VideoWriter object
    output_dir = Path('outputs/vids')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    i = 1
    while (output_path := output_dir / f'{i}.mp4').exists():
        i += 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 20.0, 
                         (int(cap.get(3)), int(cap.get(4))))

    recording = False
    print("Controls:")
    print("'r' - Start/Stop recording")
    print("'q' - Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add recording indicator
        if recording:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            out.write(frame)

        cv2.imshow('Recording', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
            if recording:
                print("Recording started...")
            else:
                print("Recording stopped.")
                print(f"Video saved to {output_path}")
        elif key == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    record_video()