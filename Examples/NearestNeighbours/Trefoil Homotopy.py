"""
Trefoil Homotopy.py
Alexander Marsh
Last Modified 03 September 2025

A pytorch implementation of an autoencoder that learns a parameterization of a trefoil knot by a circle via a nearest neighbours based penalty.
"""

import os
import numpy as np
import torch
import torch.nn.init as init
import torchplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Ensure output directory exists
os.makedirs("HomotopyResults/Trefoil", exist_ok=True)

# --- Hyperparameters ---
EPOCHS = 4000
INPUT_DIM = 3      # 3D points input
HIDDEN_DIM = 20    # Hidden layer size
PRE_LATENT_DIM = 4 # Pre-latent dimension (4D)
LATENT_DIM = 2     # Latent dimension (2D)
NUM_HIDDEN_LAYERS = 0

# --- Prepare Data (Knot points) ---
tparam = np.arange(-np.pi, np.pi, 0.05)

x = np.cos(tparam) + 2 * np.cos(2 * tparam)
y = np.sin(tparam) - 2 * np.sin(2 * tparam)
z = -3 * np.sin(3 * tparam)

tparam_shifted = tparam + 0.025
xt = np.cos(tparam_shifted) + 2 * np.cos(2 * tparam_shifted)
yt = np.sin(tparam_shifted) - 2 * np.sin(2 * tparam_shifted)
zt = -3 * np.sin(3 * tparam_shifted)

npts = x.shape[0]
print(f"Number of data points: {npts}")

data_train = torch.FloatTensor(np.vstack((x, y, z)).T)
data_test = torch.FloatTensor(np.vstack((xt, yt, zt)).T)

# --- Model Definitions ---

class Encoder(torch.nn.Module):
    def __init__(self, indim, hdim, mdim, nh):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.nh = nh
        self.f1 = torch.nn.Linear(indim, hdim)
        self.fmid = torch.nn.Linear(hdim, hdim)
        self.flast = torch.nn.Linear(hdim, mdim)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.f1, self.fmid, self.flast]:
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        out = self.f1(x)
        for _ in range(1, self.nh - 1):
            out = self.sigmoid(out)
            out = self.fmid(out)
        out = self.sigmoid(out)
        pre_latent = self.flast(out)
        return pre_latent


class Central(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Take first two dimensions
        points_2d = x[:, :2]

        def scale_to_unit_circle(points):
            x_min, x_max = torch.min(points[:, 0]), torch.max(points[:, 0])
            y_min, y_max = torch.min(points[:, 1]), torch.max(points[:, 1])

            scaled = torch.empty_like(points)
            scaled[:, 0] = 2 * (points[:, 0] - x_min) / (x_max - x_min) - 1
            scaled[:, 1] = 2 * (points[:, 1] - y_min) / (y_max - y_min) - 1
            return scaled

        return scale_to_unit_circle(points_2d)


class CircularNode(torch.nn.Module):
    def __init__(self, bdim):
        super().__init__()
        self.f = torch.nn.Linear(bdim, bdim)
        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.f.weight)
        init.zeros_(self.f.bias)

    def forward(self, x, epoch):
        out = self.f(x)

        # We want to normalize the circular node to a circle after it is unknotted
        if epoch > 3000:
            out = out - out.mean(dim=0, keepdim=True)  # center data

            # Normalize each vector individually to unit length:
            norms = torch.norm(out, dim=1, keepdim=True) + 1e-8
            out = out / norms

        return out


class Decoder(torch.nn.Module):
    def __init__(self, indim, hdim, bdim, nh):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.nh = nh

        self.f1 = torch.nn.Linear(bdim, hdim)
        self.fmid = torch.nn.Linear(hdim, hdim)
        self.flast = torch.nn.Linear(hdim, indim)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.f1, self.fmid, self.flast]:
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        out = self.f1(x)
        for _ in range(1, self.nh - 1):
            out = self.sigmoid(out)
            out = self.fmid(out)
        out = self.sigmoid(out)
        reconstruction = self.flast(out)
        return reconstruction

# --- Penalty Functions ---

def coordinate_wise_order_penalty(pre_latent, alpha=10.0):
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

def LatentCirclePenalty(latent):
    norms = torch.norm(latent, dim=1, keepdim=True)
    targets = latent / (norms + 1e-8)
    penalty = torch.sum((latent - targets) ** 2)
    return penalty * 100

def even_angle_spacing_penalty(X):
    vecs = X[1:] - X[:-1]
    vecs_norm = vecs / (torch.norm(vecs, dim=1, keepdim=True) + 1e-8)
    dot_prods = (vecs_norm[1:] * vecs_norm[:-1]).sum(dim=1).clamp(-0.99, 0.99)
    angles = torch.acos(dot_prods)
    mean_angle = angles.mean()
    return torch.mean((angles - mean_angle) ** 2)

# --- Model Initialization ---
encoder = Encoder(INPUT_DIM, HIDDEN_DIM, PRE_LATENT_DIM, NUM_HIDDEN_LAYERS)
central = Central()
circular_node = CircularNode(LATENT_DIM)
decoder = Decoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, NUM_HIDDEN_LAYERS)

device = torch.device("cpu")
criterion = torch.nn.MSELoss()

optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=0.01)
optimizer_circular = torch.optim.Adam(circular_node.parameters(), lr=0.01)
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=0.01)

# --- Training Loop ---
keepdata, keeplatentnc, keeplatent = [], [], []

for epoch in range(EPOCHS):
    encoder.train()
    circular_node.train()
    decoder.train()

    pre_latent = encoder(data_train)
    non_circular_latent = central(pre_latent)
    circular_latent = circular_node(non_circular_latent, epoch)
    reconstructed = decoder(circular_latent)

    loss = criterion(reconstructed, data_train)
    loss += coordinate_wise_order_penalty(pre_latent)

    if epoch >= 2000:
        loss += min(1, (epoch - 2000) / 1000) * LatentCirclePenalty(circular_latent)
    if epoch >= 4000:
        loss += even_angle_spacing_penalty(circular_latent)

    optimizer_encoder.zero_grad()
    optimizer_circular.zero_grad()
    optimizer_decoder.zero_grad()
    loss.backward()
    optimizer_encoder.step()
    optimizer_circular.step()
    optimizer_decoder.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")

    keepdata.append(reconstructed)
    keeplatentnc.append(non_circular_latent)
    keeplatent.append(circular_latent)

# --- Evaluation ---
encoder.eval()
central.eval()
circular_node.eval()
decoder.eval()

with torch.no_grad():
    pre_latent = encoder(data_test)
    non_circular_latent = central(pre_latent)
    circular_latent = circular_node(non_circular_latent, 10000)
    reconstructed = decoder(circular_latent)

# --- Plot Non-Circular Latent Space ---
plt.figure()
plt.plot(non_circular_latent[:, 0].numpy(), non_circular_latent[:, 1].numpy())
plt.title('Non-Circular Latent Space Representation - Test Data')
plt.savefig('HomotopyResults/Trefoil/NonCircularLatentTestData.png')
plt.show()

# --- Plot Circular Latent Space ---
plt.figure()
plt.scatter(circular_latent[:, 0].numpy(), circular_latent[:, 1].numpy())
plt.title('Circular Latent Space Representation - Test Data')
plt.savefig('HomotopyResults/Trefoil/CircularLatentTestData.png')
plt.show()

# --- 3D Plot of Original vs Reconstructed Data ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot lines connecting reconstructed points in 3D
for start in range(0, reconstructed.shape[0], 2):
    ax.plot(reconstructed[start:start+2, 0], reconstructed[start:start+2, 1], reconstructed[start:start+2, 2], 'ro-')
for start in range(1, reconstructed.shape[0] - 1, 2):
    ax.plot(reconstructed[start:start+2, 0], reconstructed[start:start+2, 1], reconstructed[start:start+2, 2], 'ro-')

# Scatter plot reconstructed and original test data points
ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], c='r', label='Reconstructed')
ax.scatter(data_test[:, 0], data_test[:, 1], data_test[:, 2], c='g', label='Test Data')

ax.set_title('Original vs Reconstructed 3D Data (Test Data)')
ax.legend()
plt.savefig('HomotopyResults/Trefoil/ReconstructedTestData.png')
plt.show()

print("Saving Results...")

# --- 3D Animation of Reconstructed Data ---
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.set_xlim([-5, 5])
ax_3d.set_ylim([-5, 5])
ax_3d.set_zlim([-5, 5])

def update_3d(epoch):
    ax_3d.cla()  # Clear axes for each frame
    ax_3d.scatter(data_train.detach().numpy()[:, 0], data_train.detach().numpy()[:, 1], data_train.detach().numpy()[:, 2], c='g', label='Original 3D Data')

    reconstructed_data = keepdata[epoch].detach().numpy()
    for start in range(0, reconstructed_data.shape[0], 2):
        ax_3d.plot(reconstructed_data[start:start+2, 0], reconstructed_data[start:start+2, 1], reconstructed_data[start:start+2, 2], 'ro-')
    for start in range(1, reconstructed_data.shape[0] - 1, 2):
        ax_3d.plot(reconstructed_data[start:start+2, 0], reconstructed_data[start:start+2, 1], reconstructed_data[start:start+2, 2], 'ro-')

    ax_3d.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], reconstructed_data[:, 2], c='r', label=f'Reconstructed at Epoch {epoch}')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('Original vs Reconstructed 3D Data')
    ax_3d.legend()
    ax_3d.set_xlim([-5, 5])
    ax_3d.set_ylim([-5, 5])
    ax_3d.set_zlim([-5, 5])

ani_3d = FuncAnimation(fig_3d, update_3d, frames=range(0, len(keeplatent), 10), interval=100, repeat=True)
ani_3d.save('HomotopyResults/Trefoil/reconstruction.gif', fps=15)

# --- 2D Animation Utility Function ---
def animate_latent_space(fig, ax, latent_data_seq, title):
    def update(epoch):
        ax.cla()
        latent_data = latent_data_seq[epoch].detach().numpy()

        for start in range(0, latent_data.shape[0], 2):
            ax.plot(latent_data[start:start+2, 0], latent_data[start:start+2, 1], 'ro-')
        for start in range(1, latent_data.shape[0] - 1, 2):
            ax.plot(latent_data[start:start+2, 0], latent_data[start:start+2, 1], 'ro-')

        ax.scatter(latent_data[:, 0], latent_data[:, 1], c='r', label=f'Latent at Epoch {epoch}')
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_title(title)
        ax.legend()

    return FuncAnimation(fig, update, frames=range(0, len(latent_data_seq), 10), interval=100, repeat=True)

# --- Non-Circular Latent Space Animation ---
fig_nc_latent, ax_nc_latent = plt.subplots()
ani_nc_latent = animate_latent_space(fig_nc_latent, ax_nc_latent, keeplatentnc, 'Non-Circular Latent Space')
ani_nc_latent.save('HomotopyResults/Trefoil/non_circular_latent.gif', fps=15)

# --- Circular Latent Space Animation ---
fig_latent, ax_latent = plt.subplots()
ani_latent = animate_latent_space(fig_latent, ax_latent, keeplatent, 'Circular Latent Space')
ani_latent.save('HomotopyResults/Trefoil/circular_latent.gif', fps=15)

print("Results Saved!")

plt.show()
