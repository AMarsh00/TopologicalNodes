"""
Strict Node Penalties.py
Alexander Marsh
Last Edit 10 September 2025

File containing both of my strict node custom loss functions. Can easily be copied into a custom net.
"""

import os
import numpy as np
import torch
import torch.nn.init as init
import torchplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def coordinate_wise_order_penalty(pre_latent, alpha=ALPHA):
    N, D = pre_latent.shape

    first_coord = pre_latent[:, 0]
    second_coord = pre_latent[:, 1]

    first_angle = torch.atan2(second_coord[0], first_coord[0]) + torch.pi
    cos_a, sin_a = torch.cos(-first_angle), torch.sin(-first_angle)

    # Rotate points by -first_angle
    new_first = cos_a * first_coord - sin_a * second_coord
    new_second = sin_a * first_coord + cos_a * second_coord
    first_coord, second_coord = new_first, new_second

    three_eighths = (3 * N) // 8
    half = N // 2
    seven_eighths = (7 * N) // 8

    penalty = 0

    # Enforce increasing and decreasing behavior as per problem statement
    penalty += torch.sum(torch.exp(-alpha * (first_coord[1:three_eighths] - first_coord[:three_eighths - 1])))
    penalty += torch.sum(torch.exp(alpha * (first_coord[three_eighths + 1:seven_eighths] - first_coord[three_eighths:seven_eighths - 1])))
    penalty += torch.sum(torch.exp(-alpha * (first_coord[seven_eighths + 1:] - first_coord[seven_eighths:-1])))

    penalty += torch.sum(torch.exp(-alpha * (second_coord[1:half] - second_coord[:half - 1])))
    penalty += torch.sum(torch.exp(alpha * (second_coord[half + 1:] - second_coord[half:-1])))

    return penalty

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
