from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm

from constants import *

print("=" * 3, "Question 1", "=" * 3)

gamma_e = 28e9

B = np.array([0, 0, 1])
h1 = np.pi * gamma_e * (B[0] * sigma_x + B[1] * sigma_y + B[2] * sigma_z)

print(h1)
print(LA.eigh(h1))

print()
print("=" * 3, "Question 2", "=" * 3)


def psi(t, psi_0, hamiltonian):
    sum_ = 0
    vals, vecs = LA.eigh(hamiltonian)
    for index in range(len(vecs)):
        c_n = bra(vecs[:, index]) @ psi_0
        c_n = c_n * np.exp(-1j * vals[index] * t)
        sum_ += c_n * vecs[index]

    return sum_


np.set_printoptions(precision=3)

times = np.linspace(0, 1, 11)
psi_0 = u_ket
h2 = 2 * np.pi * sigma_z
for t in times:
    psi_t = psi(t, psi_0, h2)
    # print(psi_t)
    print(f"Time:    {t:.2f}")
    print(f"Overlap: {probability(psi_t, psi_0)}")
    print(f"Vector:  {psi_t}")

print()
print("=" * 6, "Task", "=" * 6)

b_mag = 57e-6
b_the = np.deg2rad(90 - 64)
b_phi = np.pi / 2
b_vec = b_mag * np.array(
    [np.sin(b_the) * np.cos(b_phi), np.sin(b_the) * np.sin(b_phi), np.cos(b_the)]
)


def evolve(t, psi_0, H):
    vals, vecs = LA.eigh(H)
    c0 = vecs.conj().T @ psi_0.ravel()
    phases = np.exp(-1j * np.outer(t, vals))
    return (phases * c0) @ vecs.T


def expectations(states):
    return np.array(
        [np.real(np.einsum("ti,ij,tj->t", states.conj(), s, states)) for s in sigma]
    )


h3 = np.pi * gamma_e * np.tensordot(b_vec, sigma, axes=1)

larmor_freq = gamma_e * magnitude(b_vec)
larmor_period = 1 / larmor_freq

psi_a = 1 / np.sqrt(2) * (u_ket + d_ket)
psi_b = 1 / np.sqrt(2) * (u_ket + d_ket * 1j)
psi_c = u_ket

x = np.linspace(0, larmor_period, 10000)
states_a = evolve(x, psi_a, h3)
states_b = evolve(x, psi_b, h3)
states_c = evolve(x, psi_c, h3)

exp_a = expectations(states_a)
exp_b = expectations(states_b)
exp_c = expectations(states_c)

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
for ax, exp, title in zip(
    axes,
    [exp_a, exp_b, exp_c],
    [r"a)$|+x\rangle$", r"b) $|+y\rangle$", r"c) $|\uparrow\rangle$"],
):
    for comp, label in zip(
        exp,
        [
            r"$\langle\sigma_x\rangle$",
            r"$\langle\sigma_y\rangle$",
            r"$\langle\sigma_z\rangle$",
        ],
    ):
        ax.plot(x * 1e6, comp, label=label)
    ax.set_title(title)
    ax.legend(loc="upper right")
axes[-1].set_xlabel("t (µs)")
plt.savefig("output.svg")
# plt.show()
plt.clf()

print()
print("=", "Extended Task", "=")

theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 200)
th, ph = np.meshgrid(theta, phi, indexing="ij")

coherent = np.stack([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)], axis=-1)

x = np.sin(th) * np.cos(ph)
y = np.sin(th) * np.sin(ph)
z = np.cos(th)


def husimi(psi):
    amplitude = np.einsum("abi,i->ab", coherent.conj(), psi.ravel())
    return np.abs(amplitude) ** 2


def plot_bloch_q(psi, ax):
    Q = husimi(psi)
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=cm.viridis(Q),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
plot_bloch_q(states_c[0], ax)
plt.show()
