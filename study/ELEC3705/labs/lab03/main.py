import numpy as np
from numpy import linalg as LA

from constants import *

print("="*3, "Question 1", "="*3)

gamma_e = 24e9

B = np.array([[0], [0], [1]])
hamiltonian = np.pi * gamma_e * \
    (B[0] * sigma_x + B[1] * sigma_y + B[2] * sigma_z)

print(hamiltonian)
print(LA.eigh(hamiltonian))

print()
print("="*3, "Question 2", "="*3)


def psi(t, psi_0, hamiltonian):
    sum_ = 0
    _, vecs = LA.eigh(hamiltonian)
    for vec in vecs:
        c_n = bra(vec) @ psi_0
        sum_ += c_n * np.exp(-1j * 2*np.pi/hbar * sigma_z*t) @ vec

    return sum_


np.set_printoptions(precision=3)

times = np.array(range(0, 10))/10
psi_0 = u_ket
for t in times:
    psi_t = psi(t, psi_0, hamiltonian)
    print(f"Time:    {t}")
    print(f"Overlap: {probability(psi_t, psi_0)}")
    print(f"Vector:  {psi_t}")
