import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm
from constants import *

import matplotlib.pyplot as plt

omega = 2 * np.pi * gamma_e


def magnetic_field(dOmega, t, B_perp, B_parallel):
    B_x = B_perp * np.outer(np.cos((omega + dOmega) * t), x_hat)
    B_y = B_perp * np.outer(np.sin((omega + dOmega) * t), y_hat)
    B_z = B_parallel * np.outer(t / t, z_hat)

    return B_x + B_y + B_z


def ham_rot(dOmega, omega_1):
    return hbar / 2 * np.array([[-dOmega, omega_1], [omega_1, dOmega]])


def ham_lab(t):
    t = np.asarray(t)
    H = np.zeros(t.shape + (2, 2), dtype=complex)
    H[..., 0, 0] = 1
    H[..., 1, 1] = -1
    H[..., 0, 1] = 20e-3 * np.exp(-1j * omega * t)
    H[..., 1, 0] = np.conj(H[..., 0, 1])
    return np.pi * gamma_e * H


def time_operator(H, t):
    return expm(-1j * H * t / hbar)


def evolve_lab(t, psi_0, H):  # H has shape (N, 2, 2)
    dt = t[1] - t[0]
    psi = np.empty((len(t), 2), dtype=complex)
    psi[0] = psi_0.ravel()
    for k in range(len(t) - 1):
        Hmid = 0.5 * (H[k] + H[k + 1])
        psi[k + 1] = expm(-1j * Hmid * dt) @ psi[k]
    return psi


def evolve(t, psi_0, H):
    vals, vecs = LA.eigh(H)
    c0 = vecs.conj().T @ psi_0.ravel()
    phases = np.exp(-1j * np.outer(t, vals))
    return (phases * c0) @ vecs.T


def expectations(states):
    return np.array(
        [np.real(np.einsum("ti,ij,tj->t", states.conj(), s, states)) for s in sigma]
    )


B_perp = 20e-3
B_parallel = 1

psi_0 = d_ket

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
for f, ax in zip([0.00000001, 10, 200], axes):  # MHz
    freq = f * 1e6
    period = 1 / freq
    dOmega = 2 * np.pi * freq
    t = np.linspace(0, period, 10000)

    B_T = magnetic_field(dOmega, t, B_perp, B_parallel)
    omega_1 = gamma_e * B_T[:, 0]

    H = ham_lab(t)

    states = evolve_lab(t, psi_0, H)
    exp = expectations(states)

    for val, label in zip(exp, ["a", "b", "c"]):
        ax.plot(t * 1e6, val, label=label)

    ax.set_title(f"Frequency: {f}MHz")
    ax.legend(loc="upper right")

plt.savefig("output.svg")
plt.show()
