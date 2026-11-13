import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import matplotlib.animation as animation

N = 400
L = 110
dx = L / (N - 1)
x = np.linspace(0, L, N)

# barrier
b_height = 1
b_width = 10
b_start = L / 2

U = np.heaviside(x - b_start, 0.5)
U = U - np.heaviside(x - (b_start + b_width), 0.5)

# plt.plot(x, U)
# plt.show()

laplace = (
    -1
    / (2 * dx**2)
    * (
        (-2 * np.diag(np.ones(N), 0))
        + np.diag(np.ones(N - 1), -1)
        + np.diag(np.ones(N - 1), 1)
    )
)
hamiltonian = laplace + np.diag(U, 0)

k0 = 2
sigma_k = 0.2
dk = 2 * np.pi / L
k = np.linspace(0, N * dk, N)
phi = np.exp(-((k - k0) ** 2) / (2 * sigma_k**2))

psi = np.fft.ifft(phi)
psi = psi / np.linalg.norm(psi)
psi = np.fft.fftshift(psi)
psi = np.roll(psi, int(-25 * (N / L)))

psi_density = np.real(np.conj(psi.T) * psi)

dt = 0.5
T = np.arange(0, 30, 0.5)
Udt = expm(-1j * hamiltonian * dt)

densities = [np.abs(psi) ** 2]

for t in range(len(T)):
    psi = Udt @ psi
    densities.append(np.abs(psi) ** 2)

fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(x, U, color="k", label="Potential")
(line,) = ax.plot(x, densities[0], label="Probability density")
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=14)
ax.set_xlim(0, L)
ax.set_ylim(0, 0.2)
ax.set_xlabel("x (m)")
ax.set_ylabel("Probability density [1/m]")
ax.legend(loc="upper right")


def update(frame):
    line.set_ydata(densities[frame])
    time_text.set_text(f"t = {frame*dt:.1f}")
    return line, time_text


anim = animation.FuncAnimation(
    fig, update, frames=len(densities), interval=60, blit=True
)
plt.show()
