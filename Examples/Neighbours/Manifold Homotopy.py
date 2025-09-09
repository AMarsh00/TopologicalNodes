"""
Manifold Homotopy.py
Alexander Marsh
Last Edit 09 September 2025

GNU Affero General Public License

A pytorch implementation of a topology-based autoencoder that learns parameterizations of manifolds via a uniformly-sampled sphere (S2).
It is a little finnicky and takes quite a while to train.
Choose which dataset to run by uncommenting that one and commenting out the rest.
Results for all of our sample datasets can be found on our GitHub, but the .gif animation files were not uploaded due to size.
"""

import os
import torch
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random

from itertools import combinations
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# === Hyperparameters ===
EPOCHS = 100000
INDIM = 3   # Input dimension (3D)
HDIM = 20   # Hidden size
MDIM = 6    # Pre-latent dimension
BDIM = 3    # Latent dimension (3D for visualization)
NH = 0      # Hidden layers
LR = 0.01
ALPHA = 1.0
K_NEIGHBORS = 5
SAVE_RESULTS = True

# === Geometry Generators ===
def generate_cylinder_with_caps(radius, height, surface_pts, cap_pts):
    theta_surface = torch.rand(surface_pts) * 2 * torch.pi
    z_surface = torch.rand(surface_pts) * height - height / 2
    x_surface = radius * torch.cos(theta_surface)
    y_surface = radius * torch.sin(theta_surface)

    theta_caps = torch.rand(cap_pts) * 2 * torch.pi
    x_top = torch.rand(cap_pts)*radius * torch.cos(theta_caps)
    y_top = torch.rand(cap_pts)*radius * torch.sin(theta_caps)
    z_top = torch.full_like(x_top, height / 2)

    x_bottom = torch.rand(cap_pts)*radius * torch.cos(theta_caps)
    y_bottom = torch.rand(cap_pts)*radius * torch.sin(theta_caps)
    z_bottom = torch.full_like(x_bottom, -height / 2)

    surface = torch.stack((x_surface, y_surface, z_surface), dim=1)
    top = torch.stack((x_top, y_top, z_top), dim=1)
    bottom = torch.stack((x_bottom, y_bottom, z_bottom), dim=1)

    return torch.cat((surface, top, bottom), dim=0), "cylinder"

def generate_bent_cylinder(N, radius, height, bend=1.0):
    theta = torch.rand(N) * 2 * torch.pi
    z = torch.rand(N) * height - height / 2
    x = radius * torch.cos(theta) + bend * torch.sin(z / height * torch.pi)
    y = radius * torch.sin(theta) + bend * torch.cos(z / height * torch.pi)
    return torch.stack((x, y, z), dim=1), "bent_cylinder"

def generate_nonuniform_sphere(n, r=1.0):
    theta = 2 * torch.pi * torch.rand(n)
    phi = torch.acos(2 * torch.rand(n) - 1)
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.stack((x, y, z), dim=1), "nonuniform_sphere"

def generate_uniform_sphere(n_lat, n_lon, r=1.0):
    theta = torch.linspace(0, 2 * torch.pi, n_lon + 1)[:-1]
    phi = torch.linspace(0, torch.pi, n_lat)
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing='ij')
    x = r * torch.sin(phi_grid) * torch.cos(theta_grid)
    y = r * torch.sin(phi_grid) * torch.sin(theta_grid)
    z = r * torch.cos(phi_grid)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1), "uniform_sphere"

def generate_alexander_horned_sphere(n, r=1.0, horns=4, strength=1.0):
    theta = torch.linspace(0, 2 * torch.pi, n // 2)
    phi = torch.linspace(0, torch.pi, n // 2)
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing='ij')
    x = r * torch.sin(phi_grid) * torch.cos(theta_grid)
    y = r * torch.sin(phi_grid) * torch.sin(theta_grid)
    z = r * torch.cos(phi_grid)
    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

    for i in range(horns):
        horn_r = strength * r * torch.sin(torch.tensor(2 * torch.pi * (i + 1) / horns))
        deform = horn_r * torch.sin(points[:, 1] * (i + 1))
        points[:, 2] += deform
    return points, "Alexander_Horn"

# === Model and Loss Function ===
class Encoder(torch.nn.Module):
    def __init__(self, indim, hdim, mdim, nh):
        super().__init__()
        self.f1 = torch.nn.Linear(indim, hdim)
        self.fmid = torch.nn.Linear(hdim, hdim)
        self.flast = torch.nn.Linear(hdim, mdim)
        self.nh = nh
        self.activation = torch.nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for layer in [self.f1, self.fmid, self.flast]:
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        h = self.f1(x)
        for _ in range(max(0, self.nh - 1)):
            h = self.activation(h)
            h = self.fmid(h)
        h = self.activation(h)
        return self.flast(h)

class Central(torch.nn.Module):
    def forward(self, x):
        return x[:, :3]  # Identity transform (could normalize)

class Decoder(torch.nn.Module):
    def __init__(self, indim, hdim, bdim, nh):
        super().__init__()
        self.f1 = torch.nn.Linear(bdim, hdim)
        self.fmid = torch.nn.Linear(hdim, hdim)
        self.flast = torch.nn.Linear(hdim, indim)
        self.nh = nh
        self.activation = torch.nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for layer in [self.f1, self.fmid, self.flast]:
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        h = self.f1(x)
        for _ in range(max(0, self.nh - 1)):
            h = self.activation(h)
            h = self.fmid(h)
        h = self.activation(h)
        return self.flast(h)

def apply_coordinate_order_penalty_nearest_neighbors(points, pre_latent, k=5, alpha=1.0):
    """
    Apply a penalty based on the order of coordinates (x, y, z) for each point and its nearest neighbors.
    The penalty increases when the coordinate order differs between a point and its nearest neighbors in the latent space.

    Args:
        points: Tensor of shape [num_points, 3] representing the 3D points (e.g., on the sphere).
        pre_latent: Tensor of shape [num_points, 3] representing the corresponding points in latent space.
        k: Number of nearest neighbors to consider.
        alpha: Hyperparameter to control the penalty strength.

    Returns:
        penalty: The calculated penalty value.
    """
    penalty = 0.0

    # Should we move points to sphere or not?
    points = points - torch.mean(points)
    for i in range(len(points)):
        points[i,:] = points[i,:] / torch.norm(points[i,:])
        
    # Compute nearest neighbors using sklearn's NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points.detach().numpy())
    distances, indices = nbrs.kneighbors(points.detach().numpy())

    for i in range(len(points)):
        # Get the current point and its nearest neighbors
        p0 = points[i]
        l0 = pre_latent[i]

        # Loop through the nearest neighbors
        for neighbor_idx in indices[i][1:]:  # Skip the first index (the point itself)
            p1 = points[neighbor_idx]
            l1 = pre_latent[neighbor_idx]

            # Compare each coordinate (x, y, z) for the order of the points
            for j in range(3):  # Loop through coordinates (x=0, y=1, z=2)
                # Compare the order of coordinates between the current point and the nearest neighbor
                order_sphere = p1[j] - p0[j]
                order_latent = l1[j] - l0[j]

                # If the orders are not the same, apply a penalty
                if order_sphere != order_latent:
                    penalty += torch.exp(-alpha * order_sphere * order_latent)

    return penalty

# === Data Selection (Choose one dataset) ===
data_train, dataset_name = generate_cylinder_with_caps(1, 2, 40, 20)
data_test, dataset_name = generate_cylinder_with_caps(1, 2, 40, 20)
#data_train, dataset_name = generate_bent_cylinder(100, 1, 2)
#data_test, dataset_name = generate_bent_cylinder(100, 1, 2)
#data_train, dataset_name = generate_nonuniform_sphere(n=40)
#data_test, dataset_name = generate_nonuniform_sphere(n=40)
#data_train, dataset_name = generate_uniform_sphere(1, 1)
#data_test, dataset_name = generate_uniform_sphere(1, 1)
#data_train, dataset_name = generate_alexander_horned_sphere(n=40)
#data_test, dataset_name = generate_alexander_horned_sphere(n=40)

path = f"ManifoldHomotopyResults/{dataset_name}"
if SAVE_RESULTS == True:
    # === Directory Setup ===
    os.makedirs(path, exist_ok=True)

device = torch.device("cpu")

# Initialize models
encoder = Encoder(INDIM, HDIM, MDIM, NH).to(device)
central = Central().to(device)
decoder = Decoder(INDIM, HDIM, BDIM, NH).to(device)

# Define loss and optimizers
criterion = torch.nn.MSELoss()
opt_enc = torch.optim.Adam(encoder.parameters(), lr=LR)
opt_dec = torch.optim.Adam(decoder.parameters(), lr=LR)

# Storage for visualizations
recon_history = []
latent_history = []

# === Training Loop ===
for epoch in range(1, EPOCHS + 1):
    encoder.train()
    decoder.train()

    # Forward pass
    pre_lat = encoder(data_train)
    lat = central(pre_lat)
    recon = decoder(lat)

    # Loss calculation (MSE + optional order penalty)
    loss = criterion(recon, data_train)
    loss += apply_coordinate_order_penalty_nearest_neighbors(data_train, pre_lat, k=K_NEIGHBORS, alpha=ALPHA)

    # Backprop + optimization
    opt_enc.zero_grad(); opt_dec.zero_grad()
    loss.backward()
    opt_enc.step(); opt_dec.step()

    # Logging
    if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"[Epoch {epoch}/{EPOCHS}] Loss: {loss.item():.6f}")

    # Save for animation
    recon_history.append(recon.detach().cpu())
    latent_history.append(lat.detach().cpu())

# === After Training: Evaluation ===
encoder.eval(); central.eval(); decoder.eval()
with torch.no_grad():
    pre_lat_test = encoder(data_test)
    lat_test = central(pre_lat_test)
    recon_test = decoder(lat_test)

# === Visualization: Static Plot ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*data_test.cpu().T, c='blue', label='Original Data', s=50)
ax.scatter(*recon_test.cpu().T, c='red', marker='^', label='Reconstructed', s=50)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
ax.set_title('Test Data vs Reconstructed at Final Epoch')
plt.tight_layout()
if SAVE_RESULTS == True:
    print("Saving Results...")
    static_path = f"{path}/final_reconstructed.png"
    plt.savefig(static_path)
plt.close()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*lat_test.cpu().T, c='red', marker='^', label='Latent', s=50)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
ax.set_title('Latent at Final Epoch')
plt.tight_layout()
if SAVE_RESULTS == True:
    static_path = f"{path}/final_latent.png"
    plt.savefig(static_path)
plt.close()

# === Animation Helpers ===
def make_animation_recon(points_history, original_data, title, out_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_idx):
        ax.cla()
        if original_data != None:
            ax.scatter(*original_data.T, c='green', label='Original', s=30)
        pts = points_history[frame_idx]
        ax.scatter(*pts.T, c='red', label=f'{title} Epoch {frame_idx * 10}', s=30)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f"{title} at Epoch {frame_idx * 10}")
        ax.legend()

    frames = list(range(0, len(points_history), 10))
    anim = FuncAnimation(fig, update, frames=frames, interval=100, repeat=True)
    if SAVE_RESULTS == True:
        anim.save(out_path, fps=15)
    plt.close()
def make_animation_latent(points_history, title, out_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_idx):
        ax.cla()
        pts = points_history[frame_idx]
        ax.scatter(*pts.T, c='red', label=f'{title} Epoch {frame_idx * 10}', s=30)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f"{title} at Epoch {frame_idx * 10}")
        ax.legend()

    frames = list(range(0, len(points_history), 10))
    anim = FuncAnimation(fig, update, frames=frames, interval=100, repeat=True)
    if SAVE_RESULTS == True:
        anim.save(out_path, fps=15)
    plt.close()

# === Save Animations ===
make_animation_latent(latent_history, "Latent Embedding", f"{path}/latent.gif")
make_animation_recon(recon_history, data_train.cpu(), "Reconstruction", f"{path}/reconstruction.gif")
if SAVE_RESULTS == True:
    print("Results Saved!")
