"""
Data Cloud Topology.py
Alexander Marsh
Last Edit 08 September 2025

GNU Affero General Public License

A data clustering algorithm that tries to pick out topological features of data clouds via forcing the data on to a torus (S1 cross S1) geometry.
It is not guaranteed to produce anything meaningful, but is likely to pick out vague circles if there are any (remember the plot is mod 2pi, so lines are alos circles)
Choose which dataset to run by uncommenting it and commenting out the rest.
Results for all of our example datasets can be found on our GitHub.
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
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D

# Ensure matplotlib animations save correctly
os.environ["XDG_RUNTIME_DIR"] = f"/tmp/runtime-{os.getenv('USER')}"

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Hyperparameters ---
HP = {
    'n_points_per_cloud': 1000,
    'epochs': 50000,
    'lr': 1e-3,
    'snapshot_interval': 200,
    'animation_interval_ms': 200,
    'latent_dim': 7,
    'hidden_dim': 20,
    'SAVE_RESULTS': True,
}

# --- Data Generation ---
def generate_unlinked_circles_tensor(n_points=50):
    """Two unlinked circles."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x1 = np.cos(t); y1 = np.sin(t); z1 = np.zeros_like(t)
    circle1 = np.stack([x1, y1, z1], axis=1)

    x2 = np.cos(t); y2 = np.ones_like(t)*5; z2 = np.sin(t)
    circle2 = np.stack([x2, y2, z2], axis=1)

    points = np.concatenate([circle1, circle2], axis=0)
    np.random.shuffle(points)
    return torch.tensor(points, dtype=torch.float32), "unlinked_circles"

def generate_random_circlish_cloud(n_points=1000, n_rings=3, noise_std=0.15):
    """
    Generate a random 'circlish' 3D point cloud composed of noisy rings.

    Parameters:
        n_points (int): Total number of points.
        n_rings (int): Number of rings to generate.
        noise_std (float): Standard deviation of Gaussian noise added to ring points.

    Returns:
        torch.Tensor: (n_points, 3) point cloud.
        str: label "random_circlish_cloud"
    """
    points = []

    points_per_ring = n_points // n_rings

    for _ in range(n_rings):
        t = np.linspace(0, 2 * np.pi, points_per_ring, endpoint=False)

        # Base ring in XY plane
        x = np.cos(t)
        y = np.sin(t)
        z = np.zeros_like(t)

        ring = np.stack([x, y, z], axis=1)

        # Add Gaussian noise
        ring += np.random.normal(scale=noise_std, size=ring.shape)

        # Random rotation matrix
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        psi = np.random.uniform(0, 2 * np.pi)

        def rot_x(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        def rot_y(a): return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        def rot_z(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

        R = rot_z(psi) @ rot_y(phi) @ rot_x(theta)
        ring = ring @ R.T  # Rotate the ring

        # Random offset to spread rings apart
        offset = np.random.uniform(-1, 1, size=(1, 3))
        ring += offset

        points.append(ring)

    points = np.concatenate(points, axis=0)

    # Shuffle to avoid structure in order
    np.random.shuffle(points)

    return torch.tensor(points, dtype=torch.float32), "random_circlish_cloud"

def generate_random_spiral_cloud(n_points=1000, n_arms=3, noise_std=0.1, vertical_twist=True):
    """
    Generate a random 'spiralish' 3D point cloud with multiple noisy spiral arms.

    Parameters:
        n_points (int): Total number of points.
        n_arms (int): Number of spiral arms.
        noise_std (float): Standard deviation of Gaussian noise added to points.
        vertical_twist (bool): Whether to add height variation (z-axis twist).

    Returns:
        torch.Tensor: (n_points, 3) point cloud.
        str: label "random_spiral_cloud"
    """
    points = []

    points_per_arm = n_points // n_arms
    t = np.linspace(0, 4 * np.pi, points_per_arm)

    for i in range(n_arms):
        angle_offset = (2 * np.pi / n_arms) * i

        # Spiral in polar coordinates
        r = t
        theta = t + angle_offset

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = t if vertical_twist else np.zeros_like(t)

        spiral = np.stack([x, y, z], axis=1)

        # Normalize scale
        spiral /= np.max(np.abs(spiral))

        # Add noise
        spiral += np.random.normal(scale=noise_std, size=spiral.shape)

        # Random rotation
        def rot_x(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        def rot_y(a): return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        def rot_z(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

        theta_r = np.random.uniform(0, 2 * np.pi)
        phi_r = np.random.uniform(0, 2 * np.pi)
        psi_r = np.random.uniform(0, 2 * np.pi)

        R = rot_z(psi_r) @ rot_y(phi_r) @ rot_x(theta_r)
        spiral = spiral @ R.T

        # Random offset to spread arms apart
        offset = np.random.uniform(-1, 1, size=(1, 3))
        spiral += offset

        points.append(spiral)

    points = np.concatenate(points, axis=0)
    np.random.shuffle(points)

    return torch.tensor(points, dtype=torch.float32), "random_spiral_cloud"

def generate_random_blob_cloud(n_points=1000, n_blobs=5, blob_std=0.1):
    """
    Generate a 3D point cloud made of several Gaussian blobs.

    Parameters:
        n_points (int): Total number of points.
        n_blobs (int): Number of blob clusters.
        blob_std (float): Standard deviation of each blob.

    Returns:
        torch.Tensor: (n_points, 3) point cloud.
        str: label "random_blob_cloud"
    """
    points = []
    points_per_blob = n_points // n_blobs

    for _ in range(n_blobs):
        # Random center in 3D space (uniform in [-1, 1]^3)
        center = np.random.uniform(-1, 1, size=(1, 3))
        
        # Generate Gaussian blob around center
        blob = np.random.normal(loc=center, scale=blob_std, size=(points_per_blob, 3))
        points.append(blob)

    # Concatenate all blobs
    points = np.concatenate(points, axis=0)

    # If there's any remainder, fill in with extra points from the first blob
    if points.shape[0] < n_points:
        extra = n_points - points.shape[0]
        points = np.concatenate([points, points[:extra]], axis=0)

    np.random.shuffle(points)
    return torch.tensor(points, dtype=torch.float32), "random_blob_cloud"

def generate_helix_column_cloud(n_points=1000, n_turns=5, radius=1.0, height=4.0, noise_std=0.05):
    """
    Generate a 3D point cloud in the shape of a noisy vertical helix column.

    Parameters:
        n_points (int): Number of points in the helix.
        n_turns (int): How many full spiral turns.
        radius (float): Radius of the helix.
        height (float): Total height of the helix column.
        noise_std (float): Gaussian noise added to points.

    Returns:
        torch.Tensor: (n_points, 3) point cloud.
        str: label "helix_column_cloud"
    """
    t = np.linspace(0, 2 * np.pi * n_turns, n_points)
    
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.linspace(0, height, n_points)

    helix = np.stack([x, y, z], axis=1)

    # Add Gaussian noise
    helix += np.random.normal(scale=noise_std, size=helix.shape)

    # Optional: Random rotation
    theta = np.random.uniform(0, 2 * np.pi)
    rot_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0,              0,             1]])
    
    helix = helix @ rot_z.T

    return torch.tensor(helix, dtype=torch.float32), "helix_column_cloud"

def generate_chaotic_swirl_cloud(n_points=1000, n_tendrils=5, swirliness=3.0, noise_std=0.1):
    """
    Generate a chaotic-looking 3D cloud made of multiple swirling noisy tendrils.

    Parameters:
        n_points (int): Total number of points.
        n_tendrils (int): Number of swirling structures.
        swirliness (float): Frequency multiplier for twists.
        noise_std (float): Standard deviation of added noise.

    Returns:
        torch.Tensor: (n_points, 3) chaotic point cloud.
        str: label "chaotic_swirl_cloud"
    """
    points = []
    points_per_tendril = n_points // n_tendrils

    for _ in range(n_tendrils):
        t = np.linspace(0, 2 * np.pi, points_per_tendril)

        # Generate a wild swirling pattern in 3D
        x = np.sin(t * swirliness) * np.cos(t)
        y = np.sin(t * swirliness) * np.sin(t)
        z = np.cos(t * swirliness)

        swirl = np.stack([x, y, z], axis=1)

        # Apply some distortions
        distortion = np.sin(t * swirliness * 1.5).reshape(-1, 1)
        swirl += 0.3 * distortion * np.random.randn(*swirl.shape)

        # Random rotation
        def random_rotation_matrix():
            a, b, c = np.random.uniform(0, 2*np.pi, size=3)
            Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
            Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
            Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
            return Rz @ Ry @ Rx

        R = random_rotation_matrix()
        swirl = swirl @ R.T

        # Random offset to scatter tendrils
        offset = np.random.uniform(-1.5, 1.5, size=(1, 3))
        swirl += offset

        # Add final Gaussian noise
        swirl += np.random.normal(scale=noise_std, size=swirl.shape)

        points.append(swirl)

    points = np.concatenate(points, axis=0)
    np.random.shuffle(points)

    return torch.tensor(points, dtype=torch.float32), "chaotic_swirl_cloud"

def generate_fractal_tree_cloud(n_points=1000, depth=4, branch_factor=3, branch_length=0.4, noise_std=0.02):
    """
    Generate a fractal-like branching 3D point cloud (like a tree or neuron).

    Parameters:
        n_points (int): Target number of total points.
        depth (int): Recursion depth (levels of branching).
        branch_factor (int): Number of child branches per node.
        branch_length (float): Length scale of each branch.
        noise_std (float): Jitter to make it look organic.

    Returns:
        torch.Tensor: (n_points, 3) fractal point cloud.
        str: label "fractal_tree_cloud"
    """
    points = []

    def recurse_branch(origin, direction, current_depth):
        if current_depth > depth or len(points) >= n_points:
            return

        # Compute end point of current branch
        end = origin + direction * branch_length
        segment = np.linspace(origin, end, num=5)
        segment += np.random.normal(scale=noise_std, size=segment.shape)
        points.extend(segment.tolist())

        # Recurse with child branches
        for _ in range(branch_factor):
            if len(points) >= n_points:
                break

            # Small random angular deviation from current direction
            rand_dir = direction + np.random.normal(scale=0.5, size=3)
            rand_dir /= np.linalg.norm(rand_dir)

            recurse_branch(end, rand_dir, current_depth + 1)

    # Start from the origin with an upward direction
    recurse_branch(origin=np.zeros(3), direction=np.array([0, 0, 1]), current_depth=0)

    points_array = np.array(points)

    # If we undershot the number of points, duplicate a few to pad
    if len(points_array) < n_points:
        pad = points_array[np.random.choice(len(points_array), size=n_points - len(points_array))]
        points_array = np.concatenate([points_array, pad], axis=0)

    np.random.shuffle(points_array)
    return torch.tensor(points_array[:n_points], dtype=torch.float32), "fractal_tree_cloud"

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

    # Set target geometry of a torus
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

    return model

# --- Main ---
if __name__ == "__main__":
    # Generate Data
    data, dataset_name = generate_unlinked_circles_tensor(HP['n_points_per_cloud'])
    #data, dataset_name = generate_random_circlish_cloud(HP['n_points_per_cloud'])
    #data, dataset_name = generate_random_spiral_cloud(HP['n_points_per_cloud'])
    #data, dataset_name = generate_random_blob_cloud(HP['n_points_per_cloud'])
    #data, dataset_name = generate_helix_column_cloud(HP['n_points_per_cloud'])
    #data, dataset_name = generate_chaotic_swirl_cloud(HP['n_points_per_cloud'])
    #data, dataset_name = generate_fractal_tree_cloud(HP['n_points_per_cloud'])
    
    data = data.to(device)

    if HP['SAVE_RESULTS'] == True:
        # Ensure the directory we want to save results to exists
        os.makedirs(f"DataCloudResults/{dataset_name}", exist_ok=True)

        path = f"DataCloudResults/{dataset_name}"
    else:
        path = None

    # Initialize Model
    model = TorusAutoencoder(
        input_dim=3,
        latent_dim=HP['latent_dim'],
        hidden_dim=HP['hidden_dim']
    ).to(device)

    # Train Model
    model = train_with_snapshots(
        model,
        data,
        epochs=HP['epochs'],
        lr=HP['lr'],
        snapshot_interval=HP['snapshot_interval']
    )

    model.eval()
    with torch.no_grad():
        z1, z2, angles, recon = model(data)

    # angles shape: (N, 2), where angles[:, 0] = theta, angles[:, 1] = phi
    theta = angles[:, 0].cpu().numpy()
    phi = angles[:, 1].cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.scatter(theta, phi, s=10, alpha=0.7, c='blue')
    plt.xlabel(r'θ')
    plt.ylabel(r'φ')
    plt.title('Post-Training Angle Representation: θ vs φ')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.grid(True)
    
    if HP['SAVE_RESULTS'] == True:
        plt.tight_layout()
        plt.savefig(f"{path}/AngleRepresentation.png")
    plt.show()

    points = recon.detach().cpu().numpy()
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    data = data.cpu().numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=10, alpha=0.7, c='royalblue')
    ax.scatter(data[:,0], data[:,1], data[:,2], s=10, alpha=0.7, c='green')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reconstructed Data")

    if HP['SAVE_RESULTS'] == True:
        plt.tight_layout()
        plt.savefig(f"{path}/Reconstruction.png")
        
    plt.show()
