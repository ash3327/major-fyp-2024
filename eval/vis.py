import matplotlib.pyplot as plt
import numpy as np

# Data from the table
methods = [
    "No Model",
    "GAT-A (SupCon)",
    "GCN-A (SupCon)",
    "GAT-A (Unsup, pink)",
    "GAT-A (Unsup, grey)",
    "GCN-A (Unsup, blue)"
]
# Inter-class means and standard deviations
handshape_inter_mean = [0.829, 0.996, 0.997, 0.721, 0.824, 0.216]
handshape_inter_std = [0.111, 0.003, 0.002, 0.165, 0.094, 0.421]
senz3d_inter_mean = [0.917, 0.993, 0.994, 0.471, 0.651, 0.043]
senz3d_inter_std = [0.038, 0.004, 0.003, 0.261, 0.156, 0.461]
# Intra-class means and standard deviations
handshape_intra_mean = [0.888, 0.998, 0.9987, 0.903, 0.930, 0.747]
handshape_intra_std = [0.085, 0.0006, 0.0007, 0.039, 0.035, 0.124]
senz3d_intra_mean = [0.955, 0.999, 0.999, 0.938, 0.956, 0.872]
senz3d_intra_std = [0.015, 0.001, 0.0004, 0.026, 0.020, 0.038]

# Categories for grouping
categories = [
    "No Model",
    "SupCon only, No Curriculum Scheduling",
    "SupCon only, No Curriculum Scheduling",
    "With Unsupervised Dataset and Curriculum Scheduling",
    "With Unsupervised Dataset and Curriculum Scheduling",
    "With Unsupervised Dataset and Curriculum Scheduling"
]

# Set up the figure and axis
plt.figure(figsize=(12, 4))

# Plot horizontal error bars for each method
y_pos = np.arange(len(methods))  # Y positions for methods
for i, (method, h_inter_mean, h_inter_std, s_inter_mean, s_inter_std, h_intra_mean, h_intra_std, s_intra_mean, s_intra_std, cat) in enumerate(
    zip(methods, handshape_inter_mean, handshape_inter_std, senz3d_inter_mean, senz3d_inter_std, 
        handshape_intra_mean, handshape_intra_std, senz3d_intra_mean, senz3d_intra_std, categories)
):
    if i in {1,2}:
        # Plot Handshape inter-class (blue, bold)
        plt.errorbar(h_inter_mean, y_pos[i] - 0.3, xerr=h_inter_std, fmt='o', color='blue', ecolor='blue', capsize=3, linewidth=2)
        # Plot Handshape intra-class (red, bold)
        plt.errorbar(h_intra_mean, y_pos[i] - 0.1, xerr=h_intra_std, fmt='^', color='red', ecolor='red', capsize=3, linewidth=2)
        # Plot Senz3d inter-class (blue, thin)
        plt.errorbar(s_inter_mean, y_pos[i] + 0.1, xerr=s_inter_std, fmt='s', color='blue', ecolor='blue', capsize=3, linewidth=1)
        # Plot Senz3d intra-class (red, thin)
        plt.errorbar(s_intra_mean, y_pos[i] + 0.3, xerr=s_intra_std, fmt='v', color='red', ecolor='red', capsize=3, linewidth=1)

# Customize the plot
plt.yticks(y_pos[1:3], [f"{method}" for i, (method, cat) in enumerate(zip(methods, categories)) if i in {1,2}], fontsize=10)
plt.xlabel("Cosine Similarity (1 = Match, 0 = Mismatch)", fontsize=12)
plt.title("Comparison of Inter-class and Intra-class Cosine Similarities Across Methods", fontsize=14)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
# plt.xlim(0, 1.1)  # Set x-axis limit to cover cosine similarity range

# Custom legend for inter-class/intra-class and dataset (linewidth)
plt.legend(handles=[
    plt.Line2D([0], [0], color='blue', marker='o', linestyle='-', linewidth=2, label='Inter-class (Handshape, bold)'),
    plt.Line2D([0], [0], color='red', marker='^', linestyle='-', linewidth=2, label='Intra-class (Handshape, bold)'),
    plt.Line2D([0], [0], color='blue', marker='s', linestyle='-', linewidth=1, label='Inter-class (Senz3d, thin)'),
    plt.Line2D([0], [0], color='red', marker='v', linestyle='-', linewidth=1, label='Intra-class (Senz3d, thin)')
], loc='upper left', fontsize=9, ncol=2)

plt.tight_layout()

# Save the plot
plt.savefig("cosine_similarity_comparison_plot.png")