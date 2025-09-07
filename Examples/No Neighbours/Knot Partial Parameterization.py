"""
Knot Partial Parameterization.py
Alexander Marsh
Last Edit 07 September 2025

GNU Affero General Public License

An autoencoder that uses a non-neighbours-based topological node to find a loose parameterization of knots.
You can choose which knot to use easily by uncommenting the one you want to use and commenting out the rest.
If you sample the knot more frequently, the loose parameterization becomes worse and you may have to run it more times to find one.
Examples have already been generated for all of the knot datasets and are viewable on GitHub.

It is not guaranteed that every run finds a good looking parameterization.
"""

import os
import math
import random
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D

# Ensure matplotlib animations save correctly
os.environ["XDG_RUNTIME_DIR"] = f"/tmp/runtime-{os.getenv('USER')}"

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Hyperparameters ---
HP = {
    'n_points_per_knot': 50,
    'epochs': 10000,
    'lr': 1e-3,
    'snapshot_interval': 200,
    'animation_interval_ms': 200,
    'latent_dim': 7,
    'hidden_dim': 20,
    'SAVE_RESULTS': True,
}

# --- Data Generation ---
def generate_trefoil_knot_tensor(n_points=50):
    """
    Generate points on a 3D trefoil knot.
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    knot = np.stack([x, y, z], axis=1)
    np.random.shuffle(knot)
    return torch.tensor(knot, dtype=torch.float32), "trefoil"

def generate_figure_eight_knot_tensor(n_points=50):
    """Generate points on a figure-eight knot."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = (2 + np.cos(2 * t)) * np.cos(3 * t)
    y = (2 + np.cos(2 * t)) * np.sin(3 * t)
    z = np.sin(4 * t)
    knot = np.stack([x, y, z], axis=1)
    np.random.shuffle(knot)
    return torch.tensor(knot, dtype=torch.float32), "figure_eight"

def generate_pentafoil_knot_tensor(n_points=50):
    """Generate points on a pentafoil (5-crossing) knot."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = (2 + np.cos(5 * t)) * np.cos(2 * t)
    y = (2 + np.cos(5 * t)) * np.sin(2 * t)
    z = np.sin(5 * t)
    knot = np.stack([x, y, z], axis=1)
    np.random.shuffle(knot)
    return torch.tensor(knot, dtype=torch.float32), "pentafoil"

def generate_sixfoil_knot_tensor(n_points=50):
    """Generate points on a 6-crossing knot."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = (3 + np.cos(6 * t)) * np.cos(t)
    y = (3 + np.cos(6 * t)) * np.sin(t)
    z = np.sin(6 * t)
    knot = np.stack([x, y, z], axis=1)
    np.random.shuffle(knot)
    return torch.tensor(knot, dtype=torch.float32), "sixfoil"

def generate_hopf_link_tensor(n_points=50):
    """Generate points on a Hopf link (two linked circles)."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # First circle in xy-plane
    x1 = np.cos(t)
    y1 = np.sin(t)
    z1 = np.zeros_like(t)
    circle1 = np.stack([x1, y1, z1], axis=1)

    # Second circle in xz-plane, shifted in y
    x2 = np.cos(t)
    y2 = np.ones_like(t) * 1.5
    z2 = np.sin(t)
    circle2 = np.stack([x2, y2, z2], axis=1)

    link = np.concatenate([circle1, circle2], axis=0)
    np.random.shuffle(link)
    return torch.tensor(link, dtype=torch.float32), "hopf"

def generate_torus_knot_3_2_tensor(n_points=50):
    """Generate points on a (3,2) torus knot."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    p, q = 3, 2
    x = (2 + np.cos(q * t)) * np.cos(p * t)
    y = (2 + np.cos(q * t)) * np.sin(p * t)
    z = np.sin(q * t)
    knot = np.stack([x, y, z], axis=1)
    np.random.shuffle(knot)
    return torch.tensor(knot, dtype=torch.float32), "3_2_torus"

def generate_torus_knot_5_3_tensor(n_points=50):
    """Generate points on a (5,3) torus knot."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    p, q = 5, 3
    x = (2 + np.cos(q * t)) * np.cos(p * t)
    y = (2 + np.cos(q * t)) * np.sin(p * t)
    z = np.sin(q * t)
    knot = np.stack([x, y, z], axis=1)
    np.random.shuffle(knot)
    return torch.tensor(knot, dtype=torch.float32), "5_3_torus"

def generate_unlinked_circles_tensor(n_points=50):
    """Generate points on two unlinked circles."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # First circle in xy-plane
    x1 = np.cos(t)
    y1 = np.sin(t)
    z1 = np.zeros_like(t)
    circle1 = np.stack([x1, y1, z1], axis=1)

    # Second circle in xz-plane, shifted in y
    x2 = np.cos(t)
    y2 = np.ones_like(t) * 5
    z2 = np.sin(t)
    circle2 = np.stack([x2, y2, z2], axis=1)

    circles = np.concatenate([circle1, circle2], axis=0)
    np.random.shuffle(circles)
    return torch.tensor(circles, dtype=torch.float32), "unlinked_circles"

def generate_simple_circle_tensor(n_points=50):
    """Generate points on a simple circle in the xy-plane."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros_like(t)
    circle = np.stack([x, y, z], axis=1)
    np.random.shuffle(circle)
    return torch.tensor(circle, dtype=torch.float32), "circle"

def generate_spiral_tensor(n_points=50):
    """Generate points on a 3D spiral."""
    t = np.linspace(0, 4 * np.pi, n_points, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (2 * np.pi)  # linearly increasing height
    spiral = np.stack([x, y, z], axis=1)
    np.random.shuffle(spiral)
    return torch.tensor(spiral, dtype=torch.float32), "spiral"


# --- Model Definition ---
class TorusAutoencoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=7, hidden_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.5),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.5),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LeakyReLU(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.5),
            nn.Linear(hidden_dim, input_dim)
        )
        self.high_d_layer = nn.Linear(latent_dim, hidden_dim)
        self.return_layer = nn.Linear(hidden_dim, latent_dim)
        self.weights = nn.Parameter(torch.zeros(latent_dim))

    # Set target geometry of a cylinder
    def normalize_z2(self, z2):
        z2 = z2 - z2.mean(dim=0, keepdim=True)
        vec2d = F.normalize(z2[:, :2], dim=1)
        vec3d = F.normalize(z2[:, 2:], dim=1)
        z2_normed = z2.clone()
        z2_normed[:, :2] = vec2d
        z2_normed[:, 2:] = vec3d
        return z2_normed

    def angle_layer(self, z2):
        angle_2d = torch.atan2(z2[:, 1], z2[:, 0])
        angle_3d = torch.atan2(z2[:, 3], z2[:, 2])
        return torch.stack([angle_2d, angle_3d], dim=1)

    def rewrap_layer_from_angles(self, angles):
        cos2d, sin2d = torch.cos(angles[:, 0]), torch.sin(angles[:, 0])
        cos3d, sin3d = torch.cos(angles[:, 1]), torch.sin(angles[:, 1])
        return torch.stack([cos2d, sin2d, cos3d, sin3d], dim=1)

    def forward(self, x):
        z1 = self.encoder(x)
        z2 = z1 + z1.norm(dim=1, keepdim=True) * self.weights
        z2 = self.normalize_z2(z2)
        angles = self.angle_layer(z2)
        out = self.decoder(z2[:, :4])
        return z1, z2, angles, out


# --- Training Loop with Snapshots ---
def train_with_snapshots(model, data, epochs=1000, lr=1e-3, snapshot_interval=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    snapshots_angles, snapshots_recon = [], []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z1, z2, angles, recon = model(data)
        loss = loss_fn(recon, data)

        if epoch > epochs / 2:
            loss += 0.1 * torch.norm(z1[:, 3:])

        loss.backward()
        optimizer.step()

        if epoch % snapshot_interval == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f}")
            snapshots_angles.append(angles.detach().cpu().numpy())
            snapshots_recon.append(recon.detach().cpu().numpy())

    return snapshots_angles, snapshots_recon


# --- Torus Mapping ---
def torus_from_angles(angles, R=3.0, r=1.0):
    theta, phi = angles[:, 0], angles[:, 1]
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z


# --- Animation ---
def animate_all(data_xyz, snapshots_angles, snapshots_recon, snapshot_interval=5, interval=200, save_path=None):
    fig = plt.figure(figsize=(18, 5))
    
    axs = [
        fig.add_subplot(1, 3, 1, projection='3d'),  # Original vs Reconstructed
        fig.add_subplot(1, 3, 2),                   # Angle Representation
        fig.add_subplot(1, 3, 3, projection='3d')   # Torus Embedding
    ]
    titles = [
        "Original vs Reconstruction (3D)",
        "Angle Representation (θ vs φ)",
        "Torus Embedding (from angles)"
    ]
    
    for ax, title in zip(axs, titles):
        ax.set_title(title)

    # --- Set axis limits ---
    axs[0].set_xlim(-6, 6)
    axs[0].set_ylim(-6, 6)
    axs[0].set_zlim(-6, 6)
    axs[0].set_box_aspect([1, 1, 1])
    
    axs[1].set_xlim(-np.pi, np.pi)
    axs[1].set_ylim(-np.pi, np.pi)
    axs[1].set_aspect("equal")
    
    axs[2].set_xlim(-5, 5)
    axs[2].set_ylim(-5, 5)
    axs[2].set_zlim(-2, 2)
    axs[2].set_box_aspect([1, 1, 1])

    # --- Static original data (gray) ---
    axs[0].scatter(data_xyz[:, 0], data_xyz[:, 1], data_xyz[:, 2], s=10, c='gray', alpha=0.5, label='Original')

    # --- Animated objects ---
    recon_scatter = axs[0].scatter([], [], [], s=10, c='red', alpha=0.7, label='Reconstruction')
    angle_scatter = axs[1].scatter([], [], s=10, c='green', alpha=0.7)
    torus_scatter = axs[2].scatter([], [], [], s=10, c='cyan', alpha=0.7)

    axs[0].legend(loc='upper right')
    epoch_text = fig.text(0.5, 0, "", ha='center', fontsize=14)

    def update(frame):
        angles = snapshots_angles[frame]
        recon = snapshots_recon[frame]

        # Update reconstructed data in 3D
        recon_scatter._offsets3d = (recon[:, 0], recon[:, 1], recon[:, 2])

        # Angle representation
        angle_scatter.set_offsets(angles)

        # Torus coordinates from angles
        x_tor, y_tor, z_tor = torus_from_angles(angles)
        torus_scatter._offsets3d = (x_tor, y_tor, z_tor)

        # Epoch label
        epoch_text.set_text(f"Epoch {frame * snapshot_interval}")
        return [recon_scatter, angle_scatter, torus_scatter, epoch_text]

    anim = FuncAnimation(
        fig, update, frames=len(snapshots_angles),
        interval=interval, blit=False, repeat=False
    )

    if save_path:
        print(f"Saving animation to {save_path} ...")
        from matplotlib.animation import PillowWriter
        anim.save(save_path, writer=PillowWriter(fps=1000 // interval))

    plt.tight_layout()
    plt.show()


# --- Main ---
if __name__ == "__main__":
    # Generate Data
    data, knot_name = generate_trefoil_knot_tensor(HP['n_points_per_knot'])
    #data, knot_name = generate_figure_eight_knot_tensor(HP['n_points_per_knot'])
    #data, knot_name = generate_pentafoil_knot_tensor(HP['n_points_per_knot'])
    #data, knot_name = generate_sixfoil_knot_tensor(HP['n_points_per_knot'])
    #data, knot_name = generate_hopf_link_tensor(HP['n_points_per_knot'])
    #data, knot_name = generate_torus_knot_3_2_tensor(HP['n_points_per_knot'])
    #data, knot_name = generate_torus_knot_5_3_tensor(HP['n_points_per_knot'])
    #data, knot_name = generate_unlinked_circles_tensor(HP['n_points_per_knot'])
    #data, knot_name = generate_simple_circle_tensor(HP['n_points_per_knot'])
    #data, knot_name = generate_spiral_tensor(HP['n_points_per_knot'])
    
    data = data.to(device)

    if HP['SAVE_RESULTS'] == True:
        # Ensure the directory we want to save results to exists
        os.makedirs(f"HomotopyResults/{knot_name}", exist_ok=True)

        path = "HomotopyResults\\{knot_name}\\Animation.gif"
    else:
        path = None

    # Initialize Model
    model = TorusAutoencoder(
        input_dim=3,
        latent_dim=HP['latent_dim'],
        hidden_dim=HP['hidden_dim']
    ).to(device)

    # Train Model
    snapshots = train_with_snapshots(
        model,
        data,
        epochs=HP['epochs'],
        lr=HP['lr'],
        snapshot_interval=HP['snapshot_interval']
    )

    # Prepare for animation
    data_xy = data.detach().cpu().numpy()
    animate_all(
        data_xy,
        *snapshots,
        snapshot_interval=HP['snapshot_interval'],
        interval=HP['animation_interval_ms'],
        save_path = path
    )
