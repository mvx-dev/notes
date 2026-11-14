import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

I2 = np.eye(2)
sx = np.array([[0, 1], [1, 0]]) / 2
sy = np.array([[0, -1j], [1j, 0]]) / 2
sz = np.array([[1, 0], [0, -1]]) / 2
Sx, Sy, Sz = [np.kron(s, I2) + np.kron(I2, s) for s in (sx, sy, sz)]
S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz

omega_H = 2 * np.pi * 10e6
H = (
    omega_H
    / 2
    * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
)

T = 2 * np.pi / omega_H
N = 1000
t = np.linspace(0, T, N)
U = expm(-1j * H * (t[1] - t[0]))


def density(psi):
    return np.outer(psi, psi.conj())


def rho_first(rho):
    return np.einsum("ijkj->ik", rho.reshape(2, 2, 2, 2))


def entropy(rho):
    lam = np.linalg.eigvalsh(rho_first(rho))
    lam = lam[lam > 1e-12]
    return -np.sum(lam * np.log(lam))


def ev(O, psi):
    return np.real(psi.conj() @ O @ psi)


def run(psi0, title):
    psi = psi0.astype(complex)
    out = np.zeros((5, N))
    for i in range(N):
        out[:, i] = [
            ev(Sx, psi),
            ev(Sy, psi),
            ev(Sz, psi),
            ev(S2, psi),
            entropy(density(psi)),
        ]
        psi = U @ psi
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    for row, name in zip(out[:4], ["<Sx>", "<Sy>", "<Sz>", "<S²>"]):
        ax[0].plot(t * 1e9, row, label=name)
    ax[0].legend()
    ax[0].set_title(title)
    ax[1].plot(t * 1e9, out[4])
    ax[1].set_ylabel("entropy (nats)")
    ax[1].set_xlabel("t (ns)")
    plt.tight_layout()
    plt.show()


uu = np.array([1, 0, 0, 0])
dd = np.array([0, 0, 0, 1])
run(dd, "a)")
run((uu + dd) / np.sqrt(2), "b)")
