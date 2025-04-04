import torch
import numpy as np
import mano
from mano.utils import Mesh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

model_path = 'model/mano'
n_comps = 45
size = 3
node_size = 10

pose_fb_var = 2
fb_fingers = {i for i in range(9*4) if i % 9 in {2,5,8}}
fb_fingers.add(36)
pose_non_fb_var = .3
variance_new = dict(
    betas=2,
    pose=np.array([pose_fb_var if i in fb_fingers else pose_non_fb_var for i in range(45)],dtype=np.float32), # front and back
    global_orient=0
)
bias = dict(
    pose=np.array([1.5 if i == 36 else .5 if i in fb_fingers else 0 for i in range(45)],dtype=np.float32)
)
# variance_new = dict(
#     betas=.3,
#     pose=1,
#     global_orient=0
# )
variance = variance_new

pose_bias = torch.zeros(n_comps)
vpow = 1

batch_size = size**2
rh_model = mano.load(model_path=model_path,
                     is_rhand=True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=False)

betas = (torch.rand(batch_size, 10)*2-1) * variance['betas']
pose = torch.randn(batch_size, n_comps) * variance['pose'] * 0
pose[:, list(fb_fingers)] = torch.rand(batch_size, len(fb_fingers))**2 * variance['pose'][list(fb_fingers)]
pose -= torch.ones(batch_size, n_comps) * bias['pose']


global_orient = torch.rand(batch_size, 3) * variance['global_orient']
transl = torch.zeros(batch_size, 3)

output = rh_model(betas=betas,
                  global_orient=global_orient,
                  hand_pose=pose,
                  transl=transl,
                  flat_hand_mean=True,
                  return_verts=True,
                  return_tips=True)

# Keypoint mapping (same as in prepare_contrastive_data.py)
mmap = [
    0, 13, 14, 15, 16,  # thumb
    1, 2, 3, 17,        # index
    4, 5, 6, 18,        # middle
    10, 11, 12, 19,        # ring
    7, 8, 9, 20      # pinky
]
# mmap = [0]*21
# for i, v in enumerate(mmap1):
#     mmap[v] = i
# mmap = [i for i in range(21)]

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

h_meshes = rh_model.hand_meshes(output)
joints = output.joints.detach().cpu().numpy()[:,mmap]

fig, axes = plt.subplots(size, size, figsize=(15, 15), subplot_kw={'projection': '3d'})
for i, ax in enumerate(axes.flat):
    if i >= batch_size:
        break
    verts = h_meshes[i].vertices
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=node_size)
    
    # Apply keypoint mapping
    joint = joints[i]
    joint[1:3] = (joint[0:2]+joint[2:4]).copy()/2
    ax.scatter(joint[:, 0], joint[:, 1], joint[:, 2], s=5)
    
    # Draw connections
    for start, end in connections:
        ax.plot(
            [joint[start, 0], joint[end, 0]],
            [joint[start, 1], joint[end, 1]],
            [joint[start, 2], joint[end, 2]],
            color='r', linewidth=0.5
        )
    
    # Label each node with its index
    for idx, (x, y, z) in enumerate(joint):
        ax.text(x, y, z, str(idx), color='black', fontsize=6)
    
    ax.set_title(f'Hand {i+1}', fontsize=6)
    ax.axis('off')

plt.subplots_adjust(wspace=-.1, hspace=-.1)
plt.show()
