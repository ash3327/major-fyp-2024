import torch
import numpy as np
import mano
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LinearSegmentedColormap

def plot_hand_joints_and_mesh(ax, joints, vertices, base_joints, connections, threshold=0.0001):
    """Plot hand joints, mesh, and highlight differences from base pose."""
    # Clear previous plot
    ax.clear()
    
    # Plot mesh vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, color='gray', alpha=0.3)
    
    # Plot joints and connections with difference highlighting
    differences = np.linalg.norm(joints - base_joints, axis=1)
    max_diff = np.max(differences)
    
    # Create custom colormap
    colors = ['blue', 'yellow', 'red']
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)

    joints[1:3] = (joints[0:2]+joints[2:4]).copy()/2
    
    # Draw connections
    for start, end in connections:
        diff = max(differences[start], differences[end])
        if diff > threshold:
            color = cmap(min(diff / (max_diff + 1e-6), 1.0))
            width = 1.5
        else:
            color = 'gray'
            width = 0.5
            
        ax.plot([joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
                [joints[start, 2], joints[end, 2]],
                color=color, linewidth=width)
    
    # Set view properties
    # ax.view_init(elev=30, azim=45)
    ax.set_box_aspect([1,1,1])
    # bound = 0.5
    # ax.set_xlim([-bound, bound])
    # ax.set_ylim([-bound, bound])
    # ax.set_zlim([-bound, bound])
    ax.axis('off')

def main():
    # Setup
    model_path = 'model/mano'
    n_comps = 45
    
    # Initialize MANO model
    rh_model = mano.load(model_path=model_path,
                        is_rhand=True,
                        num_pca_comps=n_comps,
                        batch_size=1,
                        flat_hand_mean=False)
    
    # Keypoint mapping
    mmap = [0, 13, 14, 15, 16,  # thumb
            1, 2, 3, 17,         # index
            4, 5, 6, 18,         # middle
            10, 11, 12, 19,      # ring
            7, 8, 9, 20]         # pinky
    
    connections = [(0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
                  (0, 5), (5, 6), (6, 7), (7, 8),      # Index
                  (0, 9), (9, 10), (10, 11), (11, 12), # Middle
                  (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                  (0, 17), (17, 18), (18, 19), (19, 20)] # Pinky
    
    # Create base pose
    base_output = rh_model(
        betas=torch.zeros(1, 10),
        global_orient=torch.zeros(1, 3),
        hand_pose=torch.zeros(1, n_comps),
        transl=torch.zeros(1, 3),
        return_verts=True,
        return_tips=True
    )
    base_joints = base_output.joints.detach().cpu().numpy()[0, mmap]
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.35, top=0.95)  # Make room for sliders
    
    # Create sliders
    slider_axes = []
    sliders = []
    n_rows = 5
    n_cols = 9
    for i in range(n_comps):
        row = i // n_cols
        col = i % n_cols
        ax_pos = [0.1 + col * 0.1, 0.05 + row * 0.05, 0.08, 0.03]
        slider_ax = plt.axes(ax_pos)
        slider = Slider(slider_ax, f'C{i}', -1.5, 1.5, valinit=0.0)
        slider_axes.append(slider_ax)
        sliders.append(slider)
    
    # Update function for sliders
    def update(val):
        pose = torch.zeros(1, n_comps)
        for i, slider in enumerate(sliders):
            pose[0, i] = slider.val
            
        output = rh_model(
            betas=torch.zeros(1, 10),
            global_orient=torch.zeros(1, 3),
            hand_pose=pose,
            transl=torch.zeros(1, 3),
            return_verts=True,
            return_tips=True
        )
        
        joints = output.joints.detach().cpu().numpy()[0, mmap]
        vertices = output.vertices.detach().cpu().numpy()[0]
        
        plot_hand_joints_and_mesh(ax, joints, vertices, base_joints, connections)
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    for slider in sliders:
        slider.on_changed(update)
    
    # Add reset button
    reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset')
    
    def reset(event):
        for slider in sliders:
            slider.reset()
    reset_button.on_clicked(reset)

    # Add open hand button
    open_hand_ax = plt.axes([0.65, 0.025, 0.1, 0.04])
    open_hand_button = Button(open_hand_ax, 'Open Hand')
    
    def open_hand(event):
        ar = [0,0,-.3,0,0,-.8,0,0,0, 0,0,-.3,0,0,-.8,0,0,0, 0,0,-.8,0,0,-.8,0,0,0, 0,0,-.5,0,0,-.8,0,0,0, .2,0,-1,.1,.1,0,-.3,0,0]
        for i, slider in enumerate(sliders):
            slider.set_val(ar[i])
        # for i, slider in enumerate(sliders):
        #     if i % 9 in {2, 5}:  # Example logic for opening hand
        #         slider.set_val(-0.5)
        #     else:
        #         slider.set_val(0.0)

    open_hand_button.on_clicked(open_hand)

    # Initial plot
    update(None)
    
    plt.show()

if __name__ == "__main__":
    main()