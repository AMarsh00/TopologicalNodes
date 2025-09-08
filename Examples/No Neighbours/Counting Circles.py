"""
Counting Circles.py
Alexander Marsh
Last Edit 07 September 2025

GNU Affero General Public License

An implementation of topological node-based circle counting. This method is a little iffy - it will induce discontinuities into the circles then try to count how many, so it is really counting how many circles plus how many discontinuities.
If you pass in a knot (see our trefoil knot example), it will break the knot to try to unknot it, and will not be able to tell that it was originally connected.
Results for all datasets can be found on our GitHub.
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
    'n_points_per_circle': 50,
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


def generate_hopf_link_tensor(n_points=50):
    """Hopf link: two linked circles orthogonal."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Circle 1 in xy-plane
    x1 = np.cos(t)
    y1 = np.sin(t)
    z1 = np.zeros_like(t)

    # Circle 2 in yz-plane, shifted in y
    x2 = np.zeros_like(t)
    y2 = np.cos(t) + 1.0
    z2 = np.sin(t)

    circle1 = np.stack([x1, y1, z1], axis=1)
    circle2 = np.stack([x2, y2, z2], axis=1)

    points = np.concatenate([circle1, circle2], axis=0)
    np.random.shuffle(points)
    return torch.tensor(points, dtype=torch.float32), "hopf_link"


def generate_linked_circles_chain_tensor(n_points=50, n_links=3):
    """
    Generate a chain of n_links interlocked circles (like chain links).
    Each circle alternates between lying in the XY-plane and the YZ-plane.
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    spacing = 2.2  # Enough spacing to allow linking

    points_list = []

    for i in range(n_links):
        if i % 2 == 0:
            # Circle in XY-plane
            x = spacing * i + np.cos(t)
            y = np.sin(t)
            z = np.zeros_like(t)
        else:
            # Circle in YZ-plane, shifted in x slightly
            x = spacing * i + 1.5*np.cos(t)
            y = np.zeros_like(t)
            z = np.sin(t)
        
        circle = np.stack([x, y, z], axis=1)
        points_list.append(circle)

    points = np.concatenate(points_list, axis=0)
    np.random.shuffle(points)

    return torch.tensor(points, dtype=torch.float32), f"linked_chain_{n_links}"


def generate_trefoil_knot_tensor(n_points=50):
    """Trefoil knot."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = np.sin(t) + 2*np.sin(2*t)
    y = np.cos(t) - 2*np.cos(2*t)
    z = -np.sin(3*t)
    knot = np.stack([x, y, z], axis=1)
    np.random.shuffle(knot)
    return torch.tensor(knot, dtype=torch.float32), "trefoil_knot"


def generate_unlinked_two_ellipses_tensor(n_points=50):
    """Two unlinked ellipses in different planes."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Ellipse 1 in xy-plane
    x1 = 2 * np.cos(t)
    y1 = np.sin(t)
    z1 = np.zeros_like(t)

    # Ellipse 2 in xz-plane, shifted in y
    x2 = 1.5 * np.cos(t)
    y2 = np.ones_like(t) * 4
    z2 = 0.5 * np.sin(t)

    ellipse1 = np.stack([x1, y1, z1], axis=1)
    ellipse2 = np.stack([x2, y2, z2], axis=1)

    points = np.concatenate([ellipse1, ellipse2], axis=0)
    np.random.shuffle(points)
    return torch.tensor(points, dtype=torch.float32), "unlinked_ellipses"


def generate_linked_two_ellipses_tensor(n_points=50):
    """Two linked ellipses (Hopf-like) orthogonal."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Ellipse 1 in xy-plane
    x1 = 2 * np.cos(t)
    y1 = np.sin(t)
    z1 = np.zeros_like(t)

    # Ellipse 2 in yz-plane, shifted in x
    x2 = np.ones_like(t) * 2.5
    y2 = 1.5 * np.cos(t)
    z2 = 0.75 * np.sin(t)

    ellipse1 = np.stack([x1, y1, z1], axis=1)
    ellipse2 = np.stack([x2, y2, z2], axis=1)

    points = np.concatenate([ellipse1, ellipse2], axis=0)
    np.random.shuffle(points)
    return torch.tensor(points, dtype=torch.float32), "linked_ellipses"


def generate_unlinked_circles_far_tensor(n_points=50):
    """Two far apart unlinked circles."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Circle 1 in xy-plane at origin
    x1 = np.cos(t)
    y1 = np.sin(t)
    z1 = np.zeros_like(t)

    # Circle 2 in xy-plane, shifted far in x and y
    x2 = np.cos(t) + 10
    y2 = np.sin(t) + 10
    z2 = np.zeros_like(t)

    circle1 = np.stack([x1, y1, z1], axis=1)
    circle2 = np.stack([x2, y2, z2], axis=1)

    points = np.concatenate([circle1, circle2], axis=0)
    np.random.shuffle(points)
    return torch.tensor(points, dtype=torch.float32), "unlinked_circles_far"


def generate_linked_two_circles_tilted_tensor(n_points=50):
    """Two linked circles with one tilted in space."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Circle 1 in xy-plane
    x1 = np.cos(t)
    y1 = np.sin(t)
    z1 = np.zeros_like(t)

    # Circle 2 tilted 45 deg around x axis and shifted in y
    angle = np.pi / 4
    x2 = np.cos(t)
    y2 = np.cos(angle)*np.ones_like(t)*2 - np.sin(angle)*np.sin(t)
    z2 = np.sin(angle)*np.ones_like(t)*2 + np.cos(angle)*np.sin(t)

    circle1 = np.stack([x1, y1, z1], axis=1)
    circle2 = np.stack([x2, y2, z2], axis=1)

    points = np.concatenate([circle1, circle2], axis=0)
    np.random.shuffle(points)
    return torch.tensor(points, dtype=torch.float32), "linked_circles_tilted"


def generate_trefoil_with_linked_circle_tensor(n_points=50):
    """Trefoil knot with a linked circle nearby."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Trefoil
    x_t = np.sin(t) + 2*np.sin(2*t)
    y_t = np.cos(t) - 2*np.cos(2*t)
    z_t = -np.sin(3*t)
    trefoil = np.stack([x_t, y_t, z_t], axis=1)

    # Circle linked near trefoil center (shifted)
    x_c = np.cos(t) + 3
    y_c = np.sin(t) + 3
    z_c = np.zeros_like(t)
    circle = np.stack([x_c, y_c, z_c], axis=1)

    points = np.concatenate([trefoil, circle], axis=0)
    np.random.shuffle(points)
    return torch.tensor(points, dtype=torch.float32), "trefoil_with_linked_circle"


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

def torus_distance_matrix(points, L=2*torch.pi):
    """
    Compute pairwise torus distance matrix with periodic boundary.
    
    points: tensor of shape (N, 2) with coordinates in [0, L)
    L: length of domain in each dimension (default 2pi)
    
    Returns:
        dist_mat: (N, N) matrix of torus distances
    """
    N = points.shape[0]
    diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N, N, 2)
    diff = diff.abs()
    diff = torch.min(diff, L - diff)  # wrap around distances
    dist_mat = torch.norm(diff, dim=2)  # Euclidean norm along last dim
    
    return dist_mat

def adjacency_from_torus(points, threshold, L=2*torch.pi):
    """
    Compute adjacency matrix for points on torus with threshold distance.
    
    points: (N, 2) tensor of coordinates in [0, L)
    threshold: scalar, max distance to consider connected
    L: domain size (default 2pi)
    
    Returns:
        adjacency matrix (N, N) of 0/1 connections (torch.bool)
    """
    dist_mat = torus_distance_matrix(points, L)
    adj = (dist_mat < threshold).to(torch.float32)
    # Remove self connections
    adj.fill_diagonal_(0)
    return adj

def get_connected_components(adj):
    """
    adj: (N, N) adjacency matrix as a torch tensor (0/1)
    
    Returns:
        num_components: int, number of connected components
        labels: np.array of shape (N,) assigning component labels to each node
    """
    adj_np = adj.cpu().numpy()
    graph = csr_matrix(adj_np)
    num_components = connected_components(csgraph=graph, directed=False, return_labels=False)
    return num_components

def find_connected_components(adj):
    # Convert adjacency matrix to NetworkX graph
    G = nx.from_numpy_array(adj.cpu().numpy())
    components = list(nx.connected_components(G))
    return components

def plot_angles_and_original_data(angles, original_data, components, path):
    colors = plt.colormaps['tab20']

    # Move tensors to CPU and numpy for plotting
    angles_np = angles.cpu().numpy() if hasattr(angles, 'cpu') else angles
    original_np = original_data.cpu().numpy() if hasattr(original_data, 'cpu') else original_data

    fig = plt.figure(figsize=(14, 6))

    # Plot angles (2D)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Angles (Theta vs Z or Phi)")
    ax1.set_xlabel("Theta")
    ax1.set_ylabel("Phi or Z coordinate")
    for i, comp in enumerate(components):
        comp_indices = list(comp)
        pts = angles_np[comp_indices]
        ax1.scatter(pts[:, 0], pts[:, 1], s=30, color=colors(i), label=f"Component {i+1}")
    ax1.legend()

    # Plot original data (3D)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("Original Data (3D)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    for i, comp in enumerate(components):
        comp_indices = list(comp)
        pts = original_np[comp_indices]
        ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=30, color=colors(i), label=f"Component {i+1}")
    ax2.legend()

    plt.tight_layout()
    if HP.get('SAVE_RESULTS', False):
        plt.savefig(f"{path}/AnglesAndOriginalData.png")
    plt.show()

# --- Main ---
if __name__ == "__main__":
    # Generate Data
    data, dataset_name = generate_unlinked_circles_tensor(HP['n_points_per_circle'])
    #data, dataset_name = generate_hopf_link_tensor(HP['n_points_per_circle'])
    #data, dataset_name = generate_linked_circles_chain_tensor(HP['n_points_per_circle'], n_links=3)
    #data, dataset_name = generate_linked_circles_chain_tensor(HP['n_points_per_circle'], n_links=5) # Fail example because circles are too close and sampled too slowly
    #data, dataset_name = generate_trefoil_knot_tensor(HP['n_points_per_circle'])
    #data, dataset_name = generate_unlinked_two_ellipses_tensor(HP['n_points_per_circle'])
    #data, dataset_name = generate_linked_two_ellipses_tensor(HP['n_points_per_circle'])
    #data, dataset_name = generate_unlinked_circles_far_tensor(HP['n_points_per_circle'])
    #data, dataset_name = generate_linked_two_circles_tilted_tensor(HP['n_points_per_circle'])
    #data, dataset_name = generate_trefoil_with_linked_circle_tensor(HP['n_points_per_circle'])
    
    data = data.to(device)

    if HP['SAVE_RESULTS'] == True:
        # Ensure the directory we want to save results to exists
        os.makedirs(f"CountingCirclesResults/{dataset_name}", exist_ok=True)

        path = f"CountingCirclesResults/{dataset_name}"
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

    # Adjacency matrix to separate lines
    adj = adjacency_from_torus(angles, threshold=0.25) # If your dataset is larger, you want a smaller threshold
    num_components = get_connected_components(adj)

    print(f"Number of connected components: {num_components}")

    components = find_connected_components(adj)
    plot_angles_and_original_data(angles, data, components, path)
