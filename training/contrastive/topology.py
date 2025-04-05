import numpy as np

# Define hand topology
fingers = np.array(
    [
        [0, 1,  2,  3,  4 ],    # Thumb
        [0, 5,  6,  7,  8 ],    # Index
        [0, 9,  10, 11, 12],    # Middle
        [0, 13, 14, 15, 16],    # Ring
        [0, 17, 18, 19, 20]     # Pinky
    ]
)
wrist_face = [0, 5, 9, 13, 17]  # Palm face
palm_nodes = {0,1,5,9,13,17}

edge_indices = [(finger[i],finger[i+1]) for finger in fingers for i in range(len(finger)-1)]
edge_indices += [(2,5),(5,9),(9,13),(13,17)]
edge_indices = np.array(edge_indices)