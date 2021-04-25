# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 16:10:45 2021

@author: ryuhei
"""

import christoffel
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint  # type: ignore


def compute_connection(q0, q1):
    r"""\Gamma_{ij}^k == connection[i,j,k]

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    return christoffel.compute_christoffel_2d_to_2d(
        joint_to_cartesian, [q0, q1])


def geodesic_equation(state, t):
    u = state[:2]
    v = state[2:]

    # dtdu
    dtdu = v

    # dtdv
    q0, q1 = u
    christoffel = compute_connection(q0, q1)
    dtdv = -np.einsum('ijk,i,j', christoffel, v, v)

    return np.hstack((dtdu, dtdv))


def joint_to_cartesian(u, l0=1.0, l1=1.0):
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u)

    u_transposed = u.T
    q0 = u_transposed[0]
    q1 = u_transposed[1]

    x0 = l0 * torch.cos(q0)
    x1 = x0 + l1 * torch.cos(q0 + q1)

    y0 = l0 * torch.sin(q0)
    y1 = y0 + l1 * torch.sin(q0 + q1)
    return torch.vstack((x1, y1)).T


if __name__ == '__main__':
    l0 = 1.0
    l1 = 1.0
    q_deg = [45, -90]

    q0 = np.deg2rad(q_deg[0])
    q1 = np.deg2rad(q_deg[1])

    u0 = np.deg2rad(q_deg)
    v0 = np.array([0.5, -1.])
    t = np.linspace(0, 10, 100)

    # Solve initial value problem
    s0 = np.hstack((u0, v0))
    s, infodict = odeint(geodesic_equation, s0, t, full_output=True)
    u = s[:, :2]

    # Plot the geodegic
    xy = joint_to_cartesian(u).numpy()

    for u_t in u:
        q0, q1 = u_t
        x0 = l0 * np.cos(q0)
        x1 = x0 + l1 * np.cos(q0 + q1)
        y0 = l0 * np.sin(q0)
        y1 = y0 + l1 * np.sin(q0 + q1)

        plt.plot([0, x0], [0, y0])
        plt.plot([x0, x1], [y0, y1])
        plt.plot(0, 0, '.k')
        plt.plot(x0, y0, '.k')
        plt.plot(x1, y1, '.k')
        plt.xlim(-2.2, 2.2)
        plt.ylim(-2.2, 2.2)
        plt.grid()
        plt.gca().set_aspect('equal')
        plt.show()
