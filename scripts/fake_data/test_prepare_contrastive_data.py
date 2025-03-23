"""
This is only a test file.
"""

import torch
import numpy as np
import mano
from mano.utils import Mesh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from time import time_ns

model_path = 'model/mano'
n_comps = 45
batch_size = 10

rh_model = mano.load(model_path=model_path,
                     is_rhand= True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=False)

variance_new = dict(
    betas=.5,
    pose=.5,
    global_orient=5
)
# variance_new = dict(
#     betas=0,
#     pose=0,
#     global_orient=5
# )
variance = variance_new
vpow = 1

mmap = [
    16, 15, 14, 13, 0, # thumb
    1, 4, 10, 7, # index
    2, 5, 11, 8, # middle
    3, 6, 12, 9, # ring
    17, 18, 19, 20 # pinky
]

# Mediapipe hand gesture connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Plot the 3D joints for all samples in the batch
flag_3d = False
plot = False

# ====================

def gen(batch_size, flag_3d=False, plot=False):
    betas = torch.rand(batch_size, 10)*variance['betas']
    pose = (torch.rand(batch_size, n_comps) - 0.5) * 2 # + pose_bias
    pose = (torch.rand(batch_size, 1)**vpow * variance['pose']) * pose
    global_orient = torch.rand(batch_size, 3)*variance['global_orient']
    transl        = torch.zeros(batch_size, 3)# torch.rand(batch_size, 3), translation and scaling

    output = rh_model(betas=betas,
                    global_orient=global_orient,
                    hand_pose=pose,
                    transl=transl,
                    return_verts=True,
                    return_tips = True)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') if flag_3d else fig.add_subplot(111)

        colors = plt.cm.jet(torch.linspace(0, 1, batch_size))  # Generate distinct colors for each sample

    for j in range(batch_size):
        joints = output.joints[j]
        # joints = j_meshes[0].vertices  # shape = [21, 3]
        joints = np.array([joints[mmap[i]] for i in range(21)])
        joints[:, [0, 1]] = joints[:, [1, 0]]
        joints -= joints[0]
        # ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=np.full((21,),j/batch_size), marker='o', label=f'Hand {j}')

        # Annotate each point with its ID
        # for idx, (x, y, z) in enumerate(joints):
        #     ax.text(x, y, z, f'{idx}', color='black', fontsize=8)
        
        # Draw edges between connected joints
        if plot:
            for connection in connections:
                start, end = connection
                if flag_3d:
                    ax.plot(
                        [joints[start, 0], joints[end, 0]],
                        [joints[start, 1], joints[end, 1]],
                        [joints[start, 2], joints[end, 2]],
                        color=colors[j]
                    )
                else:
                    ax.plot(
                        [joints[start, 0], joints[end, 0]],
                        [joints[start, 1], joints[end, 1]],
                        color=colors[j]
                    )

    if plot:
        ax.set_xlim(-.5, .5)
        ax.set_ylim(-.5, .5)

        if flag_3d:
            ax.set_zlim(-.5, .5)

            ax.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='X-axis')  # X-axis in red
            ax.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Y-axis')  # Y-axis in green
            ax.quiver(0, 0, 0, 0, 0, 0.5, color='b', label='Z-axis')  # Z-axis in blue
            ax.text(0.5, 0, 0, 'X', color='r')
            ax.text(0, 0.5, 0, 'Y', color='g')
            ax.text(0, 0, 0.5, 'Z', color='b')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        plt.title('3D Hand Joints with Connections for All Samples')
        plt.legend()
        plt.show()

# ======

if __name__ == '__main__':
    G = 1000000000
    M = 1000000
    num_batchs = 100
    batch_size = 256
    start_time = time_ns()
    for _ in range(num_batchs):
        gen(batch_size=batch_size)
    end_time = time_ns()
    total_time = end_time-start_time
    print(f"Total Time: {total_time/G}s, Average Time per item {total_time/num_batchs/batch_size/M}ms")