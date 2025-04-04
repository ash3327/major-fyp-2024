import torch
import numpy as np
import mano
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import torch
import numpy as np
import mano
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_hand_joints_and_mesh(ax, joints, vertices, base_joints, connections, component_idx, threshold=0.0001):
    """Helper function to plot hand joints, mesh vertices, and connections with difference highlighting."""
    # Plot mesh vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, color='gray', alpha=0.3)
    
    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=5, color='blue')
    
    # Calculate differences from base pose
    differences = np.linalg.norm(joints - base_joints, axis=1)
    max_diff = np.max(differences)
    
    # Create custom colormap: blue -> yellow -> red
    colors = ['blue', 'yellow', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Draw connections with color based on endpoint differences
    for start, end in connections:
        diff = max(differences[start], differences[end])
        if diff > threshold:
            # Normalize difference for color mapping
            color = cmap(min(diff / (max_diff + 1e-6), 1.0))
            width = 1.5  # Thicker line for emphasized segments
            ax.plot(
                [base_joints[start, 0], base_joints[end, 0]],
                [base_joints[start, 1], base_joints[end, 1]],
                [base_joints[start, 2], base_joints[end, 2]],
                color='gray', linewidth=1
            )
        else:
            color = 'gray'
            width = 0.5
            
        ax.plot(
            [joints[start, 0], joints[end, 0]],
            [joints[start, 1], joints[end, 1]],
            [joints[start, 2], joints[end, 2]],
            color=color, linewidth=width
        )
    
    ax.set_title(f'Component {component_idx}', fontsize=8)
    ax.axis('off')
    # ax.view_init(elev=120, azim=60)
    ax.view_init(elev=-30, azim=0)

# Model setup
model_path = 'model/mano'
n_comps = 45
batch_size = n_comps

# Initialize MANO model
rh_model = mano.load(model_path=model_path,
                     is_rhand=True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=False)

# ==== Mapping =====
# Keypoint mapping
mmap = [
    0, 13, 14, 15, 16,  # thumb
    1, 2, 3, 17,        # index
    4, 5, 6, 18,        # middle
    10, 11, 12, 19,     # ring
    7, 8, 9, 20         # pinky
]

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# ======= Base ========
# Create base pose (all zeros)
base_pose = torch.zeros(1, n_comps)
base_output = rh_model(
    betas=torch.zeros(1, 10),
    global_orient=torch.zeros(1, 3),
    hand_pose=base_pose,
    transl=torch.zeros(1, 3),
    flat_hand_mean=True,
    return_verts=True,
    return_tips=True
)
base_joints = base_output.joints.detach().cpu().numpy()[0, mmap]

# ======= Other poses ====
# Create poses: each row activates one component
pose = torch.zeros(batch_size, n_comps)
for i in range(n_comps):
    pose[i, i] = 0.5

# Other parameters
betas = torch.zeros(batch_size, 10)
global_orient = torch.zeros(batch_size, 3)
transl = torch.zeros(batch_size, 3)

# Generate hand meshes
output = rh_model(betas=betas,
                 global_orient=global_orient,
                 hand_pose=pose,
                 transl=transl,
                 flat_hand_mean=True,
                 return_verts=True,
                 return_tips=True)

# Get vertices
h_meshes = rh_model.hand_meshes(output)

joints = output.joints.detach().cpu().numpy()[:, mmap]

# === Figure ===
# Create two 6x4 grids
fig1 = plt.figure(figsize=(20, 30))
fig2 = plt.figure(figsize=(20, 30))

# Modify the plotting loop
for i in range(n_comps):
    # Determine which figure and position to use
    if i < 24:  # First 24 components
        fig = fig1
        pos = i + 1
    else:  # Remaining components
        fig = fig2
        pos = i - 23
    
    ax = fig.add_subplot(4, 6, pos, projection='3d')
    vertices = h_meshes[i].vertices
    plot_hand_joints_and_mesh(ax, joints[i], vertices, base_joints, connections, i)


plt.figure(fig1.number)
plt.suptitle('PCA Components 0-23', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.figure(fig2.number)
plt.suptitle('PCA Components 24-44', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()