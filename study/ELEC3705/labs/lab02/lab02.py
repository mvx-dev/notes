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


def print_row(P_pos, P_neg, expectation, meas_no=None):
    title = ""
    if meas_no is not None:
        title = f"Measurement {meas_no}"
        print("="*3, title, "="*3)
    else:
        print("="*15)

    print(f"P(+):   {P_pos:.5f}")
    print(f"P(-):   {P_neg:.5f}")
    print(f"Expect: {expectation:.5f}")
    if meas_no is not None:
        print("="*(len(title) + 8))
    else:
        print("="*15, end="\n\n")


def run_1():
    # X
    theta = np.pi/2
    phi = 2*np.pi

    measurement_0 = sigma_z
    psi_1 = get_eigenvector(measurement_0, 1)
    measurement_1, row = stern_gerlach(psi_1, theta, phi)
    print_row(*row, meas_no=1.1)

    # X
    theta = np.pi/2
    phi = 2*np.pi

    psi_2 = get_eigenvector(measurement_1, 1)
    measurement_2, row = stern_gerlach(psi_2, theta, phi)
    print_row(*row, meas_no=1.2)

    # Z
    theta = 2*np.pi
    phi = np.pi

    psi_3 = get_eigenvector(measurement_2, 1)
    measurement_3, row = stern_gerlach(psi_3, theta, phi)
    print_row(*row, meas_no=1.3)


def run_2():
    # X
    theta = np.pi/2
    phi = 2*np.pi

    psi_1 = 1/np.sqrt(2)*u_ket + (1+1j)/2*d_ket
    measurement_1, row = stern_gerlach(psi_1, theta, phi)
    print_row(*row, meas_no=2.1)

    # X
    theta = np.pi/2
    phi = 2*np.pi

    psi_2 = get_eigenvector(measurement_1, 1)
    measurement_2, row = stern_gerlach(psi_2, theta, phi)
    print_row(*row, meas_no=2.2)

    # Z
    theta = 2*np.pi
    phi = np.pi

    psi_3 = get_eigenvector(measurement_2, 1)
    measurement_3, row = stern_gerlach(psi_3, theta, phi)
    print_row(*row, meas_no=2.3)


def run_3():
    # X
    theta = np.pi/2
    phi = 2*np.pi

    measurement_0 = sigma_z
    psi_1 = get_eigenvector(measurement_0, -1)
    measurement_1, row = stern_gerlach(psi_1, theta, phi)
    print_row(*row, meas_no=3.1)

    # U (theta=35, phi=20 - degrees)
    theta = np.deg2rad(35)
    phi = np.deg2rad(20)
    psi_2 = get_eigenvector(measurement_1, 1)
    measurement_2, row = stern_gerlach(psi_2, theta, phi)
    print_row(*row, meas_no=3.2)

    # Z
    theta = 2*np.pi
    phi = np.pi
    psi_3 = get_eigenvector(measurement_2, 1)
    _, row = stern_gerlach(psi_3, theta, phi)
    print_row(*row, meas_no=3.3)


if __name__ == "__main__":
    print("Run 1:")
    run_1()
    print("\nRun 2:")
    run_2()
    print("\nRun 3:")
    run_3()
