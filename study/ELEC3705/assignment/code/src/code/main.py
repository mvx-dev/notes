from numpy import linalg as LA
from scipy.linalg import expm

import matplotlib.pyplot as plt
import numpy as np

# meta
OUT_DIR = "figures"

# constants
gamma_n = 42.48e6
gaussian_std = 10

d_ket = np.array([0, 1])
u_ket = np.array([1, 0])

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

I_x = sigma_x / 2
I_y = sigma_y / 2
I_z = sigma_z / 2

sigma = np.array([sigma_x, sigma_y, sigma_z])
I_vec = np.array([I_x, I_y, I_z])

# global configuration
plt.rcParams["font.size"] = 18
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.figsize"] = (8, 5.5)
# plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["savefig.dpi"] = 150

# === Question 1.1 ===
img = plt.imread("image.jpg").astype(float)
nrows, ncols = img.shape

print(f"Rows: {nrows}, Cols: {ncols}")


# === Question 1.2 ===
def B_z(c, mu):
    G = np.exp(-((c - mu) ** 2) / (2 * gaussian_std**2)) / (
        gaussian_std * np.sqrt(2 * np.pi)
    )
    G_max = G.max()
    return 1 - G / (2 * G_max)


mu_1 = 0
mu_2 = 20
c = np.arange(ncols) - ncols // 2

B1_z = B_z(c, mu_1)
E1_z = gamma_n * B1_z

B2_z = B_z(c, mu_2)
E2_z = gamma_n * B2_z

print(f"Columns: 0 -> {ncols//2}, 20 -> {ncols//2 + 20}")

for energies, label, file in zip(
    [E1_z, E2_z], [r"$\mu=0$", r"$\mu=20$"], ["zeeman-mu00.png", "zeeman-mu20.png"]
):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(c, energies / 1e6, color="blue")
    ax.set_title(f"Zeeman Energies at {label}")
    ax.set_xlabel("Column Number")
    ax.set_ylabel("Frequency (MHz)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{file}")


# === Question 1.3 ===
B_vec = np.stack([np.zeros(ncols), np.zeros(ncols), B_z(c, 0)], axis=1)
hamiltonian = gamma_n * np.einsum("ci,ijk->cjk", B_vec, I_vec)
print(hamiltonian[ncols // 2])
print(hamiltonian[ncols - 1])

# === Question 1.4 ===
B_amp = 0.05
f_drive = gamma_n * 0.5
omega_R = gamma_n * B_amp / 2
delta = gamma_n * B1_z - f_drive
H_tilde = delta[:, None, None] * I_z + omega_R * I_x
print(H_tilde[ncols // 2])
print(H_tilde[ncols - 1])


# === Question 1.5 ===
states = np.tile(d_ket, (ncols, 1))  # set one spin per pixel
expect = np.real(np.einsum("...i,ij,...j->...", states.conj(), I_z, states))

fig, ax = plt.subplots()
ax.plot(c, expect)

ax.set_title("Prepared State Expectation")
ax.set_xlabel("Column Number")
ax.set_ylabel(r"Expectation Value $\langle I_z \rangle$")
ax.set_ylim(-1, 1)  # otherwise a flat line autoscales to a misleading range
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/single-spin-exp.png")

# === Question 1.6 ===
T_rabi = 1 / omega_R
# t = np.arange(0, 2 * T_rabi + T_rabi / 200, T_rabi / 100)
t = np.linspace(0, 2 * T_rabi, 201)

cols = [ncols // 2, ncols // 2 + 5, ncols // 2 + 10, ncols - 1]

psi_0 = d_ket

for col in cols:
    psi = np.array([expm(-1j * 2 * np.pi * H_tilde[col] * tk) @ psi_0 for tk in t])

    Ix_exp = np.real(np.einsum("...i,ij,...j->...", psi.conj(), I_x, psi))
    Iy_exp = np.real(np.einsum("...i,ij,...j->...", psi.conj(), I_y, psi))
    Iz_exp = np.real(np.einsum("...i,ij,...j->...", psi.conj(), I_z, psi))

    fig = plt.figure()
    ax = fig.add_subplot()
    label_col = f"{col - ncols // 2:03}"
    ax.plot(t / T_rabi, Ix_exp, label=r"$I_x$ ", color="red")
    ax.plot(t / T_rabi, Iy_exp, label=r"$I_y$ ", color="green")
    ax.plot(t / T_rabi, Iz_exp, label=r"$I_z$ ", color="blue")

    ax.set_title(f"Rabi Oscillation in Static Field")
    ax.set_xlabel(r"Time ($T_{rabi}$)")
    ax.set_ylabel(r"Expectation value")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/single-spin-rabi-{label_col}.png")

# === Question 1.7 ===
t_pi2 = T_rabi / 4

psi0 = np.array([0, 1], dtype=complex)
psi_pi2 = np.array(
    [expm(-1j * 2 * np.pi * H_tilde[col] * t_pi2) @ psi0 for col in range(ncols)]
)  # (ncols, 2)

Ix_exp = np.real(np.einsum("ci,ij,cj->c", psi_pi2.conj(), I_x, psi_pi2))
Iy_exp = np.real(np.einsum("ci,ij,cj->c", psi_pi2.conj(), I_y, psi_pi2))
Iz_exp = np.real(np.einsum("ci,ij,cj->c", psi_pi2.conj(), I_z, psi_pi2))

fig, ax = plt.subplots()
ax.plot(c, Ix_exp, color="red", linewidth=2, label=r"$\langle I_x\rangle$")
ax.plot(c, Iy_exp, color="green", linewidth=2, label=r"$\langle I_y\rangle$")
ax.plot(c, Iz_exp, color="blue", linewidth=2, label=r"$\langle I_z\rangle$")
ax.set_title("Short Pulse Expectation Values")
ax.set_xlabel("Column c")
ax.set_ylabel("Expectation value")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/single-spin-pulse.png")

# == Question 1.8 ===
T_larmor = 1 / (gamma_n * 0.5)
t = np.linspace(0, 2 * T_larmor, 201)

cols = [ncols // 2, ncols // 2 + 10, ncols - 1]

print(hamiltonian[cols[0]])

for col in cols:
    psi = np.array(
        [expm(-1j * 2 * np.pi * hamiltonian[col] * tk) @ psi_pi2[col] for tk in t]
    )

    Ix_exp = np.real(np.einsum("...i,ij,...j->...", psi.conj(), I_x, psi))
    Iy_exp = np.real(np.einsum("...i,ij,...j->...", psi.conj(), I_y, psi))
    Iz_exp = np.real(np.einsum("...i,ij,...j->...", psi.conj(), I_z, psi))

    fig = plt.figure()
    ax = fig.add_subplot()
    label_col = f"{col - ncols // 2:03}"
    ax.plot(t / T_larmor, Ix_exp, label=r"$I_x$ ", color="red")
    ax.plot(t / T_larmor, Iy_exp, label=r"$I_y$ ", color="green")
    ax.plot(t / T_larmor, Iz_exp, label=r"$I_z$ ", color="blue")

    ax.set_title(f"Prepared State with Static Field")
    ax.set_xlabel(r"Time ($T_{larmor}$)")
    ax.set_ylabel(r"Expectation value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/single-spin-larmor-{label_col}.png")

# === Question 1.9 ===
r = nrows // 2
t = np.linspace(0, 2 * T_larmor, 201)

psi = np.empty((len(t), ncols, 2), dtype=complex)
for col in range(ncols):
    for i, tk in enumerate(t):
        psi[i, col] = expm(-1j * 2 * np.pi * hamiltonian[col] * tk) @ psi_pi2[col]

Ix_all = np.real(np.einsum("tci,ij,tcj->tc", psi.conj(), I_x, psi))

P = Ix_all @ img[r, :]

fig, ax = plt.subplots()
ax.plot(t / T_larmor, P, color="blue")
ax.set_title(f"Aggregated Signal (Row {r})")
ax.set_xlabel(r"Time ($T_{larmor}$)")
ax.set_ylabel(r"Intensity")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/aggregate-spin-row.png")

t_max = t[np.argmax(P)]

# === Question 1.10 ===
t_read = t_max
psi0 = d_ket

recon = np.zeros((nrows, ncols))

for k, mu in enumerate(c):
    Bz = B_z(c, mu)
    delta = gamma_n * Bz - f_drive
    H_rot = delta[:, None, None] * I_z + omega_R * I_x

    U_pulse = expm(-1j * 2 * np.pi * H_rot * t_pi2)
    psi = np.einsum("cij,j->ci", U_pulse, psi0)

    H_lab = gamma_n * Bz[:, None, None] * I_z
    U_lab = expm(-1j * 2 * np.pi * H_lab * t_read)
    psi = np.einsum("cij,cj->ci", U_lab, psi)

    Ix = np.real(np.einsum("ci,ij,cj->c", psi.conj(), I_x, psi))
    # w = np.abs(c - mu) <= 10
    # recon[:, k] = img[:, w] @ Ix[w]
    recon[:, k] = img @ Ix

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(img, cmap="gray")
axs[0].set_title("Original")
axs[1].imshow(recon, cmap="gray")
axs[1].set_title("Reconstructed")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/reconstruction.png")

print(
    Ix[0], Ix[ncols // 2]
)  # far columns: ~1e-3. Last-loop dip is at the last column, so Ix[-1] should be large
print(np.allclose(recon[r, ncols // 2], P.max()))  # consistency with 1.9 (r = nrows//2)
print(
    recon.std(axis=0).mean(), recon.std(axis=1).mean()
)  # both should now be comparable

# print(Bz.min(), Bz.max())  # expect ~0.5 and ~1.0
# print(delta[0], delta[ncols // 2], delta[-1])  # expect ~2.124e7, ~2.124e7, ~0
# print(gaussian_std, omega_R, f_drive)  # expect 10, ~1.062e6, 2.124e7
# print(c[:3], c[-3:], mu)  # expect -127.. 127, mu = 127
#
# print(t_max / T_larmor)  # readout time
# print(Ix[ncols // 2], Ix[0])
# print(recon.std(axis=0).mean(), recon.std(axis=1).mean())
