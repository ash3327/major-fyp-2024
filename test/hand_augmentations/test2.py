import torch
import numpy as np
import mano
from mano.utils import Mesh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
model_path = 'model/mano'
n_comps = 45
size = 5

variance_new = dict(
    betas=.5,
    pose=1,
    global_orient=1
)
variance = variance_new

# Parameters to control pose variance and bias
pose_variance_magnitude = .5  # Adjust the magnitude of the variance
pose_bias = torch.zeros(n_comps)  # Bias towards a specific pose (e.g., flat pose)

# Generate 100 hand meshes
batch_size = size**2
rh_model = mano.load(model_path=model_path,
                     is_rhand=True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=False)

betas = torch.rand(batch_size, 10) * variance['betas']
pose = (torch.rand(batch_size, n_comps) - 0.5) * 2 # + pose_bias
# print((torch.rand(1, batch_size) * pose_variance_magnitude).shape, pose.shape)
pose = (torch.rand(batch_size, 1)**.5 * pose_variance_magnitude) * pose
# print(pose.shape)
global_orient = torch.rand(batch_size, 3) * variance['global_orient']
transl = torch.zeros(batch_size, 3)

output = rh_model(betas=betas,
                  global_orient=global_orient,
                  hand_pose=pose,
                  transl=transl,
                  return_verts=True,
                  return_tips=True)

h_meshes = rh_model.hand_meshes(output)

# Plot the hand meshes in a 10x10 grid with compact subplots
fig, axes = plt.subplots(size, size, figsize=(15, 15), subplot_kw={'projection': '3d'})

for i, ax in enumerate(axes.flat):
    if i >= batch_size:
        break
    verts = h_meshes[i].vertices
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=1)
    ax.set_title(f'Hand {i+1}', fontsize=6)
    ax.axis('off')

plt.subplots_adjust(wspace=-.1, hspace=-.1)  # Minimize spacing between subplots
plt.show()