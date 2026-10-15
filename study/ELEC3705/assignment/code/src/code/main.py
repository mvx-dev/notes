from numpy import linalg as LA

import matplotlib.pyplot as plt
import numpy as np

# meta
OUT_DIR = "figures"

# constants
gamma_n = 42.48e6
gaussian_std = 10

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

sigma = np.array([sigma_x, sigma_y, sigma_z])

# global configuration
# plt.rcParams["font.size"] = 18
# plt.rcParams["lines.linewidth"] = 2

# === Question 1.1 ===
img = plt.imread("image.jpg").astype(float)
nrows, ncols = img.shape


# === Question 1.2 ===
def B_z(c, mu):
    G = np.exp(-((c - mu) ** 2) / (2 * gaussian_std**2)) / (
        gaussian_std * np.sqrt(2 * np.pi)
    )
    G_max = G.max()
    return 1 - G / (2 * G_max)


mu_1 = 0  # case a)
mu_2 = 20  # case b)
c = np.arange(ncols) - ncols // 2

# mu = 0
B1_z = B_z(c, mu_1)
E1_z = gamma_n * B1_z

# mu = 20
B2_z = B_z(c, mu_2)
E2_z = gamma_n * B2_z

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.plot(c, E1_z)
ax2.plot(c, E2_z)

ax1.set_title("Centred on middle pixel")
ax1.set_xlabel("Column Number")
ax1.set_ylabel("Frequency (MHz)")
ax2.set_title("Centred on right pixel")
ax2.set_xlabel("Column Number")
fig.suptitle("Zeeman energies")
fig.savefig(f"{OUT_DIR}/zeeman_energies.png", dpi=160)
# plt.show()

# === Question 1.3 ===
# NOTE assuming 64 refers to the centre pixel, however this should be mu=0
# or c=127, not 64 (image shape = 255x255)
B_vec = np.stack([np.zeros(ncols), np.zeros(ncols), B_z(c, 0)], axis=1)
I_vec = 0.5 * sigma
hamiltonian = gamma_n * np.einsum("ci,ijk->cjk", B_vec, I_vec)
print(hamiltonian[ncols // 2])
print(hamiltonian[ncols - 1])

# === Question 1.4 ===
B_amp = 50e-6
f_drive = gamma_n * 0.5
omega_R = gamma_n * B_amp / 2
delta = gamma_n * B1_z - f_drive
H_tilde = delta[:, None, None] * sigma_z + omega_R * sigma_x
print(H_tilde[ncols // 2])
print(H_tilde[ncols - 1])


# === Question 1.5 ===
def expectations(states):
    return np.array([np.real(np.einsum("ti,ij,tj->t", states.conj(), sigma_z, states))])


expect = delta**2 / (delta**2 + f_drive**2)
plt.clf()
plt.plot(c, expect)
plt.show()
