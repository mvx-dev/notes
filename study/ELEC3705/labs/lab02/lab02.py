from constants import *
from numpy import linalg as LA
import numpy as np
import scipy as sc

theta = 0
phi = 0

psi_0 = 1/np.sqrt(2)*(u_ket + d_ket)

state_0 = sigma_z
psi_1 = state_0 @ u_ket

exp_1u = u_bra @ state_0 @ u_ket
exp_1d = d_bra @ state_0 @ d_ket
exp_1 = exp_1u + exp_1d

P_1neg = np.abs(d_bra @ psi_1)**2
P_1pos = np.abs(u_bra @ psi_1)**2

print("="*3, "State 1", "="*3)
print(state_0)
print(psi_1)
print("="*15)
print("Expect:", exp_1)
print("P(-1): ", P_1neg)
print("P(+1): ", P_1pos)
print()

# psi_2
state_1 = state_0 @ sigma_x
exp_2u = u_bra @ state_1 @ u_ket
exp_2d = d_bra @ state_1 @ d_ket
exp_2 = exp_2u + exp_2d

P_2neg = np.abs(bra(sigma_x_neg) @ psi_1)**2
P_2pos = np.abs(bra(sigma_x_pos) @ psi_1)**2

psi_2 = P_2neg * d_ket + P_2pos * u_ket
psi_2 = psi_2 / LA.norm(psi_2)

print("="*3, "State 2", "="*3)
print(state_1)
print(psi_2)
print("="*15)
print("Expect:", exp_2)
print("P(-1): ", P_2neg)
print("P(+1): ", P_2pos)
print()

# psi_3
state_2 = state_1 @ sigma_x
exp_3u = u_bra @ state_2 @ u_ket
exp_3d = d_bra @ state_2 @ d_ket
exp_3 = exp_3u + exp_3d

P_3neg = np.abs(bra(sigma_x_neg) @ psi_2)**2
P_3pos = np.abs(bra(sigma_x_pos) @ psi_2)**2

psi_3 = P_3neg * d_ket + P_3pos * u_ket
psi_3 = psi_3 / LA.norm(psi_3)

print("="*3, "State 3", "="*3)
print(state_2)
print(psi_3)
print("="*15)
print("Expect:", exp_3)
print("P(-1): ", P_3neg)
print("P(+1): ", P_3pos)
print()
