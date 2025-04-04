import os
import cv2
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import argparse
sys.path.append('.')

from scripts.datasets.feature_extractor import extract_features_from_clipped_region

class VideoParser:
    def __init__(self):
        self.landmarks = []
        
    def extract_keypoints(self, frame):
        """Extract landmarks from a frame."""
        features, _, _ = extract_features_from_clipped_region(frame)
        landmarks = features[3]
        return landmarks.get('pose', np.zeros((0,2))), landmarks['hands'][0], landmarks['hands'][1]

    def parse_video(self, video_path):
        """Parse video file and extract landmarks."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {total_frames} frames...")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lmks = self.extract_keypoints(frame_rgb)
            self.landmarks.append([lmk.copy() for lmk in lmks])
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        return True

    def visualize(self):
        """Create and save 3D animation of landmarks."""
        if not self.landmarks:
            print("No landmarks to visualize!")
            return

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]

        def update(frame):
            ax.clear()
            # Plot pose landmarks
            pose = self.landmarks[frame][0]
            if pose.shape[0] > 0:
                ax.scatter(pose[:, 0], pose[:, 1], 0, c='g')
                for start, end in connections:
                    if start < pose.shape[0] and end < pose.shape[0]:
                        ax.plot([pose[start, 0], pose[end, 0]],
                                [pose[start, 1], pose[end, 1]],
                                [0, 0], 'g-')
            # Plot both hands
            for hand_idx in range(1, 3):  # Plot hand0 and hand1
                hand = self.landmarks[frame][hand_idx]
                if hand.shape[0] > 0:
                    # Scale Z coordinates for better visualization
                    hand_scaled = hand.copy()
                    
                    ax.scatter(hand_scaled[:, 0], hand_scaled[:, 1], hand_scaled[:, 2],
                             c='r' if hand_idx == 1 else 'b')
                    
                    # Draw connections between joints
                    for start, end in connections:
                        if start < hand.shape[0] and end < hand.shape[0]:
                            ax.plot([hand_scaled[start, 0], hand_scaled[end, 0]],
                                  [hand_scaled[start, 1], hand_scaled[end, 1]],
                                  [hand_scaled[start, 2], hand_scaled[end, 2]],
                                  'r-' if hand_idx == 1 else 'b-')

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_zlim([-0.5, 0.5])
            ax.set_title(f'Frame {frame}')

        anim = FuncAnimation(fig, update, frames=len(self.landmarks),
                           interval=50, blit=False)
        
        # Save animation
        output_dir = Path('outputs/hand_animations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f'hand_animation_{len(list(output_dir.glob("*.mp4")))}.mp4'
        writer = animation.FFMpegWriter(fps=20)
        anim.save(output_path, writer=writer)
        print(f"Animation saved to {output_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Parse video file and extract hand landmarks')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    args = parser.parse_args()

    video_path = Path(os.path.join("outputs/vids",args.video_path))
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    parser = VideoParser()
    if parser.parse_video(video_path):
        parser.visualize()

if __name__ == '__main__':
    main()