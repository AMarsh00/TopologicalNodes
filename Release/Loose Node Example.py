"""
Loose Node Example.py
Alexander Marsh
Last Edit: 10 September 2025

GNU Affero General Public License

Simple example of usage for our LooseTopologicalNode module.
Must be run in Python 13.3 due to our compiler instructions.
If you would like to run in an older version, simply change the version you are compiling with and re-compile the module yourself.
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.nn import MSELoss

# Import the C++ extension module
import LooseTopologicalNode

# --- Device setup ---
device = torch.device("cpu")

# --- Hyperparameters ---
HP = {
    'n_points_per_knot': 50,
    'epochs': 2000,
    'lr': 1e-3,
    'snapshot_interval': 200,
    'animation_interval_ms': 200,
    'latent_dim': 4,          # Should match the 4D bottleneck of the node
    'hidden_dim': 20,
    'SAVE_RESULTS': True,
}

# --- Data Generation ---
def generate_trefoil_knot_tensor(n_points=50):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    knot = np.stack([x, y, z], axis=1)
    np.random.shuffle(knot)
    return torch.tensor(knot, dtype=torch.float32), "trefoil"

# --- Module Grabbed From C++ ---
class PyLooseTopologicalNode(torch.nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        super().__init__()
        self.cpp_node = LooseTopologicalNode.LooseTopologicalNode(input_dim, output_dim)
        # Note: do NOT do self.add_module("cpp_node", self.cpp_node)

    def forward(self, x):
        return self.cpp_node.forward(x)

    def parameters(self):
        return self.cpp_node.parameters()

# --- Training Loop with Snapshots ---
def train_with_snapshots(model, data, epochs, lr, snapshot_interval):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    snapshots = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, z_torus = model(data)
        loss = loss_fn(recon, data)
        loss.backward()
        optimizer.step()

        if epoch % snapshot_interval == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:05d} | Loss: {loss.item():.6f}")
            snapshots.append((recon.detach().cpu(), z_torus.detach().cpu()))

    return snapshots

# --- Torus Mapping for Visualization ---
def torus_from_latent(z, R=3.0, r=1.0):
    theta = z[:, 0]
    phi = z[:, 1]

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z_ = r * np.sin(phi)
    return x, y, z_

# --- Animation Utility ---
def animate_all(data_xyz, snapshots, snapshot_interval, interval, save_path=None):
    fig = plt.figure(figsize=(18, 5))
    axs = [
        fig.add_subplot(1, 3, 1, projection='3d'),
        fig.add_subplot(1, 3, 2),
        fig.add_subplot(1, 3, 3, projection='3d')
    ]
    titles = ["Original vs Reconstruction", "Latent Angles", "Torus Embedding"]
    for ax, title in zip(axs, titles):
        ax.set_title(title)

    axs[0].scatter(*data_xyz.T, color='gray', s=10, alpha=0.5)

    # Initialize with first snapshot data to set correct shape
    first_recon, first_z_torus = snapshots[0]
    recon_scatter = axs[0].scatter(*first_recon.T, color='red', s=10)
    angles = first_z_torus
    angle_scatter = axs[1].scatter(angles[:, 0], angles[:, 1], color='green', s=10)
    x_t, y_t, z_t = torus_from_latent(first_z_torus)
    torus_scatter = axs[2].scatter(x_t, y_t, z_t, color='blue', s=10)

    epoch_text = fig.text(0.5, 0.01, "", ha='center')

    def update(idx):
        recon, z_torus = snapshots[idx]

        # Update reconstruction scatter (3D)
        xs, ys, zs = recon.T
        recon_scatter._offsets3d = (xs, ys, zs)

        # Update angle scatter (2D)
        angles = z_torus.numpy()
        angle_scatter.set_offsets(angles)

        # Update torus scatter (3D)
        x_t, y_t, z_t = torus_from_latent(z_torus)
        torus_scatter._offsets3d = (x_t, y_t, z_t)

        epoch_text.set_text(f"Epoch {idx * snapshot_interval}")
        return recon_scatter, angle_scatter, torus_scatter, epoch_text

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=interval, blit=False)

    if save_path:
        anim.save(save_path, writer='pillow', fps=1000 // interval)

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    data, knot_name = generate_trefoil_knot_tensor(HP['n_points_per_knot'])
    data = data.to(device)

    if HP['SAVE_RESULTS']:
        os.makedirs(f"HomotopyResults/{knot_name}", exist_ok=True)
        save_path = f"HomotopyResults/{knot_name}/Animation.gif"
    else:
        save_path = None

    model = PyLooseTopologicalNode().to(device)
    snapshots = train_with_snapshots(
        model, data, HP['epochs'], HP['lr'], HP['snapshot_interval']
    )

    animate_all(
        data.cpu().numpy(),
        snapshots,
        HP['snapshot_interval'],
        HP['animation_interval_ms'],
        save_path
    )
