from constants import *
from numpy import linalg as LA
import numpy as np
import scipy as sc


def stern_gerlach(psi, theta, phi):
    basis = sigma_u(theta, phi)
    p_ket = normalise(get_eigenvector(basis, 1))
    n_ket = normalise(get_eigenvector(basis, -1))

    P_pos = probability(p_ket, psi)
    P_neg = probability(n_ket, psi)
    # P_pos = LA.norm(bra(p_ket) @ psi)**2
    # P_neg = LA.norm(bra(n_ket) @ psi)**2
    expectation = P_pos - P_neg

    return (basis, (P_pos, P_neg, expectation))


def print_row(P_pos, P_neg, expectation, stage=None):
    if stage is not None:
        print("="*3, f"Stage {stage}", "="*3)
    else:
        print("="*15)

    print(f"P(+):   {P_pos}")
    print(f"P(-):   {P_neg}")
    print(f"Expect: {expectation}")
    print("="*15, end="\n\n")


# X
theta = np.pi/2
phi = 2*np.pi
state_0 = sigma_z
psi_1 = get_eigenvector(state_0, 1)
state_1, row = stern_gerlach(psi_1, theta, phi)
print_row(*row, stage=1)
# X
psi_2 = get_eigenvector(state_1, 1)
state_2, row = stern_gerlach(psi_2, theta, phi)
print_row(*row, stage=2)
# Z
theta = 2*np.pi
phi = np.pi
psi_3 = get_eigenvector(state_2, 1)
state_3, row = stern_gerlach(psi_3, theta, phi)
print_row(*row, stage=3)
