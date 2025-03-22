
import torch
import numpy as np
import mano
from mano.utils import Mesh
from mpl_toolkits.mplot3d import Axes3D
model_path = 'model/mano'
n_comps = 45
batch_size = 10

rh_model = mano.load(model_path=model_path,
                     is_rhand= True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=False)

variance_default = dict(
    betas=.1, # .1
    pose=.1, # .1
    global_orient=1 # 1
)
variance_new = dict(
    betas=1.,
    pose=.5,
    global_orient=1
)
variance = variance_new

betas = torch.rand(batch_size, 10)*variance['betas']
pose = torch.rand(batch_size, n_comps)*variance['pose']
global_orient = torch.rand(batch_size, 3)*variance['global_orient']
transl        = torch.zeros(batch_size, 3)# torch.rand(batch_size, 3), translation and scaling

output = rh_model(betas=betas,
                  global_orient=global_orient,
                  hand_pose=pose,
                  transl=transl,
                  return_verts=True,
                  return_tips = True)

flag = True

if not flag:
    h_meshes = rh_model.hand_meshes(output)
    j_meshes = rh_model.joint_meshes(output, radius=0 if flag else .001)

# print(dir(h_meshes[0]))
# print(j_meshes[0]) # vertices shape = [21,3] when radius=0. 
# print(dir(j_meshes[0]))
# print(j_meshes[0].vertices)
# print(dir(output))

# print(output.joints.shape) # shape = [10,21,3]
# UNDERSTOOD:
# output.joints: randomly generated keypoints based on rotations etc
# j_meshes: fixed landmarks

if flag:
    import matplotlib.pyplot as plt

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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.jet(torch.linspace(0, 1, batch_size))  # Generate distinct colors for each sample

    for j in range(batch_size):
        joints = output.joints[j]
        # joints = j_meshes[0].vertices  # shape = [21, 3]
        joints = np.array([joints[mmap[i]] for i in range(21)])
        joints -= joints[0]
        # ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=np.full((21,),j/batch_size), marker='o', label=f'Hand {j}')

        # Annotate each point with its ID
        # for idx, (x, y, z) in enumerate(joints):
        #     ax.text(x, y, z, f'{idx}', color='black', fontsize=8)
        
        # Draw edges between connected joints
        for connection in connections:
            start, end = connection
            ax.plot(
                [joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
                [joints[start, 2], joints[end, 2]],
                color=colors[j]
            )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Hand Joints with Connections for All Samples')
    plt.legend()
    plt.show()

if not flag:
    #visualize hand mesh only
    h_meshes[0].show()  # Set display image size

    #visualize joints mesh only
    j_meshes[0].show()  # Set display image size

    # #visualize hand and joint meshes
    # hj_meshes = Mesh.concatenate_meshes([h_meshes[0], j_meshes[0]])
    # hj_meshes.show()  # Set display image size
