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
