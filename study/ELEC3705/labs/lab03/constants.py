from numpy import linalg as LA
import numpy as np

hbar = (6.626*10**(-34))/(2*np.pi)

u_ket = np.array([[1], [0]])
d_ket = np.array([[0], [1]])
u_bra = u_ket.T
d_bra = d_ket.T

# spin matrices
sigma_x = np.array([[0,  1],  [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1,  0],  [0, -1]])

# eigenvectors
sigma_x_pos = 1/np.sqrt(2)*(u_ket + d_ket)
sigma_x_neg = 1/np.sqrt(2)*(u_ket - d_ket)
sigma_y_pos = 1/np.sqrt(2)*(u_ket + d_ket * 1j)
sigma_y_neg = 1/np.sqrt(2)*(u_ket - d_ket * 1j)
sigma_z_pos = u_ket
sigma_z_neg = d_ket


def sigma_u(theta, phi):
    return np.array(
        [[np.cos(theta),  np.sin(theta)*(np.cos(phi) - np.sin(phi)*1j)],
         [np.sin(theta)*(np.cos(phi) + np.sin(phi)*1j), -np.cos(theta)]])


def get_eigenvector(matrix, eigenvalue):
    val, vec = LA.eigh(matrix)
    i = np.where(np.isclose(val, eigenvalue))[0][0]
    return vec[:, [i]]


def get_eigenvalue(matrix, eigenvector):
    lam = np.vdot(eigenvector, matrix @ eigenvector) / \
        np.vdot(eigenvector, eigenvector)
    if not np.allclose(matrix @ eigenvector, lam * eigenvector):
        raise ValueError("vector is not an eigenvector of this matrix")
    return lam.real


def bra(ket):
    return ket.conj().T


def ket(bra):
    return bra.conj().T


def normalise(vector):
    return vector / LA.norm(vector)


def probability(ket1, ket2):
    return abs((bra(ket1) @ ket2).item())**2
