"""
Knot Homotopy.py
Alexander Marsh
Last Edit 07 September 2025

GNU Affero General Public License

A pytorch implementation of a topology-based autoencoder that learns parameterizations of knots via grouops isomorphic to circles.
You can choose which knot to run easily by uncommenting whichever one you want and commenting out the rest.
Caveat Emptor - you will have to adjust hyperparameters for all knots besides the trefoil. Generally more ugly knots require more epochs and possibly a lower alpha.
Examples have already been generated for all of the knot datasets and are viewable on GitHub.

This takes ordered data as the input, so we are not computing nearest neighbours at each step.
"""

import os
import numpy as np
import torch
import torch.nn.init as init
import torchplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- Hyperparameters ---
EPOCHS = 5000
INPUT_DIM = 3         # 3D points input
HIDDEN_DIM = 20       # Hidden layer size
PRE_LATENT_DIM = 4    # Pre-latent dimension (4D)
LATENT_DIM = 2        # Latent dimension (2D)
NUM_HIDDEN_LAYERS = 0 # Number of Hidden Layers (none are needed)
SAVE_RESULTS = True   # Should we save results to the disk? (not advised for small amounts of epochs)
SHOW_RESULTS = True   # Should we animate our results? (not advised for large amounts of epochs)
ALPHA = 10            # Scaling factor for our exponential penalty - higher alpha means faster convergence, but more risk of exploding gradients

# --- Knots ---

def Trefoil(npts=126):
    t = np.linspace(-np.pi, np.pi, npts, endpoint=False)
    x = np.cos(t) + 2 * np.cos(2 * t)
    y = np.sin(t) - 2 * np.sin(2 * t)
    z = -3 * np.sin(3 * t)
    shift = np.pi * 2 / npts / 2
    t_shift = t + shift
    xt = np.cos(t_shift) + 2 * np.cos(2 * t_shift)
    yt = np.sin(t_shift) - 2 * np.sin(2 * t_shift)
    zt = -3 * np.sin(3 * t_shift)
    return torch.FloatTensor(np.vstack((x, y, z)).T), torch.FloatTensor(np.vstack((xt, yt, zt)).T), "Trefoil"

def TorusKnot(npts=126, p=3, q=4):
    t = np.linspace(0, 2*np.pi, npts, endpoint=False)
    x = np.cos(q * t) * (2 + np.cos(p * t))
    y = np.sin(q * t) * (2 + np.cos(p * t))
    z = np.sin(p * t)
    t_shift = t + np.pi * 2 / npts / 2
    xt = np.cos(q * t_shift) * (2 + np.cos(p * t_shift))
    yt = np.sin(q * t_shift) * (2 + np.cos(p * t_shift))
    zt = np.sin(p * t_shift)
    return torch.FloatTensor(np.vstack((x, y, z)).T), torch.FloatTensor(np.vstack((xt, yt, zt)).T), "TorusKnot"

def LissajousKnot(npts=6283, a=3, b=4, c=5, delta=np.pi/2):
    t = np.linspace(-np.pi, np.pi, npts, endpoint=False)
    x = np.sin(a * t + delta)
    y = np.sin(b * t)
    z = np.sin(c * t)
    t_shift = t + np.pi * 2 / npts / 2
    xt = np.sin(a * t_shift + delta)
    yt = np.sin(b * t_shift)
    zt = np.sin(c * t_shift)
    return torch.FloatTensor(np.vstack((x, y, z)).T), torch.FloatTensor(np.vstack((xt, yt, zt)).T), "LissajousKnot"

def FigureEight(npts=6283):
    t = np.linspace(0, 2*np.pi, npts, endpoint=False)
    x = (2 + np.cos(2 * t)) * np.cos(3 * t)
    y = (2 + np.cos(2 * t)) * np.sin(3 * t)
    z = np.sin(4 * t)
    t_shift = t + np.pi * 2 / npts / 2
    xt = (2 + np.cos(2 * t_shift)) * np.cos(3 * t_shift)
    yt = (2 + np.cos(2 * t_shift)) * np.sin(3 * t_shift)
    zt = np.sin(4 * t_shift)
    return torch.FloatTensor(np.vstack((x, y, z)).T), torch.FloatTensor(np.vstack((xt, yt, zt)).T), "FigureEight"

def Cinquefoil(npts=126):
    t = np.linspace(0, 2*np.pi, npts, endpoint=False)
    x = (2 + np.cos(5*t)) * np.cos(2*t)
    y = (2 + np.cos(5*t)) * np.sin(2*t)
    z = np.sin(5*t)
    t_shift = t + np.pi * 2 / npts / 2
    xt = (2 + np.cos(5*t_shift)) * np.cos(2*t_shift)
    yt = (2 + np.cos(5*t_shift)) * np.sin(2*t_shift)
    zt = np.sin(5*t_shift)
    return torch.FloatTensor(np.vstack((x, y, z)).T), torch.FloatTensor(np.vstack((xt, yt, zt)).T), "Cinquefoil"

def PretzelKnot(npts=126):
    t = np.linspace(0, 2*np.pi, npts, endpoint=False)
    x = np.sin(2*t) + 2*np.sin(3*t)
    y = np.cos(3*t) - 2*np.cos(2*t)
    z = np.sin(5*t)
    t_shift = t + np.pi * 2 / npts / 2
    xt = np.sin(2*t_shift) + 2*np.sin(3*t_shift)
    yt = np.cos(3*t_shift) - 2*np.cos(2*t_shift)
    zt = np.sin(5*t_shift)
    return torch.FloatTensor(np.vstack((x, y, z)).T), torch.FloatTensor(np.vstack((xt, yt, zt)).T), "PretzelKnot"

def Helix(npts=126):
    t = np.linspace(0, 4*np.pi, npts, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (2 * np.pi)
    t_shift = t + np.pi * 2 / npts / 2
    xt = np.cos(t_shift)
    yt = np.sin(t_shift)
    zt = t_shift / (2 * np.pi)
    return torch.FloatTensor(np.vstack((x, y, z)).T), torch.FloatTensor(np.vstack((xt, yt, zt)).T), "Helix"

def SimpleCircle(npts=126):
    t = np.linspace(0, 2*np.pi, npts, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros_like(t)
    t_shift = t + np.pi * 2 / npts / 2
    xt = np.cos(t_shift)
    yt = np.sin(t_shift)
    zt = np.zeros_like(t_shift)
    return torch.FloatTensor(np.vstack((x, y, z)).T), torch.FloatTensor(np.vstack((xt, yt, zt)).T), "Circle"

data_train, data_test, knot_name = Trefoil(126)
#data_train, data_test, knot_name = TorusKnot(126)
#data_train, data_test, knot_name = LissajousKnot(6283) # The ugliest dataset here, would only recommend with a very powerful machine
#data_train, data_test, knot_name = FigureEight(6283)
#data_train, data_test, knot_name = Cinquefoil(6283)
#data_train, data_test, knot_name = PretzelKnot(126)
#data_train, data_test, knot_name = Helix(126) # The Helix dataset is not actually a knot, but it has a cool visualization
#data_train, data_test, knot_name = SimpleCircle(126) # More interesting than you think in how it reconstructs

if SAVE_RESULTS == True:
    # Ensure the directory we want to save results to exists
    os.makedirs(f"HomotopyResults/{knot_name}", exist_ok=True)

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

# --- Model Initialization ---
encoder = Encoder(INPUT_DIM, HIDDEN_DIM, PRE_LATENT_DIM, NUM_HIDDEN_LAYERS)
central = Central()
decoder = Decoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, NUM_HIDDEN_LAYERS)

device = torch.device("cpu")
criterion = torch.nn.MSELoss()

optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=0.01)
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=0.01)

# --- Training Loop ---
keepdata, keeplatent = [], []

for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()

    pre_latent = encoder(data_train)
    latent = central(pre_latent)
    reconstructed = decoder(latent)

    loss = criterion(reconstructed, data_train)
    loss += coordinate_wise_order_penalty(pre_latent)

    optimizer_encoder.zero_grad()
    optimizer_decoder.zero_grad()
    loss.backward()
    optimizer_encoder.step()
    optimizer_decoder.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")

    keepdata.append(reconstructed)
    keeplatent.append(latent)

# --- Evaluation ---
encoder.eval()
central.eval()
decoder.eval()

with torch.no_grad():
    pre_latent = encoder(data_test)
    latent = central(pre_latent)
    reconstructed = decoder(latent)

# --- Plot Non-Circular Latent Space ---
plt.figure()
plt.plot(latent[:, 0].numpy(), latent[:, 1].numpy())
plt.title('Latent Space Representation - Test Data')
if SAVE_RESULTS == True:
    plt.savefig('HomotopyResults/{knot_name}/LatentTestData.png')
if SHOW_RESULTS == True:
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
if SAVE_RESULTS == True:
    plt.savefig('HomotopyResults/{knot_name}/ReconstructedTestData.png')
if SHOW_RESULTS == True:
    plt.show()

if SAVE_RESULTS == True:
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
if SAVE_RESULTS == True:
    ani_3d.save('HomotopyResults/{knot_name}/reconstruction.gif', fps=15)

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

# --- Circular Latent Space Animation ---
fig_latent, ax_latent = plt.subplots()
ani_latent = animate_latent_space(fig_latent, ax_latent, keeplatent, 'Latent Space')
if SAVE_RESULTS == True:
    ani_latent.save('HomotopyResults/{knot_name}/latent.gif', fps=15)

    print("Results Saved!")

if SHOW_RESULTS == True:
    plt.show()
