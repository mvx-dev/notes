from scipy.linalg import expm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from constants import *

N = 10
a = 1e-6

x = np.linspace(0, a, 1000)
n = np.arange(1, N + 1)
k = n * np.pi / a
E = hbar**2 * n**2 * np.pi**2 / (2 * m_e * a**2)

T_rev = 4 * m_e * a**2 / (np.pi * hbar)
t = np.linspace(0, T_rev, 2001)

# a) c_n = 1, 2
c_n = np.zeros(10)
np.put(
    c_n,
    [
        0,
    ],
    1,
)
psi = (np.exp(-1j * t[:, None] * E[None, :] / hbar) * c_n[None, :]) @ (
    np.sqrt(2 / a) * np.sin(k[:, None] * x[None, :])
)

psi = np.asarray(psi, dtype=complex)
psi = psi / np.linalg.norm(psi)

rho = np.abs(psi) ** 2
# image = plt.pcolormesh(x / a, t / T_rev, a * rho, shading="auto", cmap="viridis")

fig, ax = plt.subplots()

lines = ax.plot(x / a, psi[0])[0]


def evolve(i):
    lines.set_data(x / a, psi[int(i)])
    return lines


ani = animation.FuncAnimation(fig, evolve, range(2000), interval=20)

plt.show()
