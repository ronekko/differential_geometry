# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:59:08 2021

@author: ryuhei
"""

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import torch


def compute_christoffel(f, u):
    r"""Compute Christoffel symbol.

    Parameters
    ----------
    f : function: torch.tensor of shape (2,) -> torch.tensor of shape (3,)
        Coordinate transform from U to X.
    u : An array-like of length 2.
        Point u.

    Returns
    -------
    Christoffel symbol $\Gamma_{ij}^k$ at point u.

    """
    u0, u1 = u[0], u[1]
    u0 = torch.tensor([u0], requires_grad=True)
    u1 = torch.tensor([u1], requires_grad=True)
    u = torch.cat((u0, u1))
    x = f(u).reshape(3, 1)

    x0_0, x0_1 = torch.autograd.grad(x[0], (u0, u1),
                                     retain_graph=True, create_graph=True)
    x1_0, x1_1 = torch.autograd.grad(x[1], (u0, u1),
                                     retain_graph=True, create_graph=True)
    x2_0, x2_1 = torch.autograd.grad(x[2], (u0, u1),
                                     retain_graph=True, create_graph=True)

    e0 = torch.cat((x0_0, x1_0, x2_0)).requires_grad_(True)  # = x_0
    e1 = torch.cat((x0_1, x1_1, x2_1)).requires_grad_(True)  # = x_1

    g00 = e0.dot(e0)
    g01 = e0.dot(e1)
    g10 = e1.dot(e0)
    g11 = e1.dot(e1)
    g0 = torch.hstack((g00, g01))
    g1 = torch.hstack((g10, g11))
    g = torch.vstack((g0, g1))
    g_inv = g.inverse()

    g00_0, g00_1 = torch.autograd.grad(g00, (u0, u1),
                                       retain_graph=True, allow_unused=True)
    g01_0, g01_1 = torch.autograd.grad(g01, (u0, u1),
                                       retain_graph=True, allow_unused=True)
    g10_0, g10_1 = torch.autograd.grad(g10, (u0, u1),
                                       retain_graph=True, allow_unused=True)
    g11_0, g11_1 = torch.autograd.grad(g11, (u0, u1),
                                       retain_graph=True, allow_unused=True)

    gl0_0 = torch.vstack((g00_0, g10_0))
    g0l_0 = torch.vstack((g00_0, g01_0))
    g00_l = torch.vstack((g00_0, g00_1))
    gamma00k = 0.5 * g_inv.matmul(gl0_0 + g0l_0 - g00_l)

    gl1_0 = torch.vstack((g01_0, g11_0))
    g0l_1 = torch.vstack((g00_1, g01_1))
    g01_l = torch.vstack((g01_0, g01_1))
    gamma01k = 0.5 * g_inv.matmul(gl1_0 + g0l_1 - g01_l)

    gl0_1 = torch.vstack((g00_1, g10_1))
    g1l_0 = torch.vstack((g10_0, g11_0))
    g10_l = torch.vstack((g10_0, g10_1))
    gamma10k = 0.5 * g_inv.matmul(gl0_1 + g1l_0 - g10_l)

    gl1_1 = torch.vstack((g01_1, g11_1))
    g1l_1 = torch.vstack((g10_1, g11_1))
    g11_l = torch.vstack((g11_0, g11_1))
    gamma11k = 0.5 * g_inv.matmul(gl1_1 + g1l_1 - g11_l)

    chirstoffel = np.concatenate((
        gamma00k.detach().numpy().T,
        gamma01k.detach().numpy().T,
        gamma10k.detach().numpy().T,
        gamma11k.detach().numpy().T)).reshape(2, 2, 2)
    return chirstoffel


def compute_christoffel_2d_to_2d(f, u):
    r"""Compute Christoffel symbol.

    Parameters
    ----------
    f : function: torch.tensor of shape (2,) -> torch.tensor of shape (2,)
        Coordinate transform from U to X.
    u : An array-like of length 2.
        Point u.

    Returns
    -------
    Christoffel symbol $\Gamma_{ij}^k$ at point u.

    """
    u0, u1 = u[0], u[1]
    u0 = torch.tensor([u0], requires_grad=True)
    u1 = torch.tensor([u1], requires_grad=True)
    u = torch.cat((u0, u1))
    x = f(u).reshape(2, 1)

    x0_0, x0_1 = torch.autograd.grad(x[0], (u0, u1),
                                     retain_graph=True, create_graph=True)
    x1_0, x1_1 = torch.autograd.grad(x[1], (u0, u1),
                                     retain_graph=True, create_graph=True)

    e0 = torch.cat((x0_0, x1_0)).requires_grad_(True)  # = x_0
    e1 = torch.cat((x0_1, x1_1)).requires_grad_(True)  # = x_1

    g00 = e0.dot(e0)
    g01 = e0.dot(e1)
    g10 = e1.dot(e0)
    g11 = e1.dot(e1)
    g0 = torch.hstack((g00, g01))
    g1 = torch.hstack((g10, g11))
    g = torch.vstack((g0, g1))
    g_inv = g.inverse()

    g00_0, g00_1 = torch.autograd.grad(g00, (u0, u1),
                                       retain_graph=True, allow_unused=True)
    g01_0, g01_1 = torch.autograd.grad(g01, (u0, u1),
                                       retain_graph=True, allow_unused=True)
    g10_0, g10_1 = torch.autograd.grad(g10, (u0, u1),
                                       retain_graph=True, allow_unused=True)
    g11_0, g11_1 = torch.autograd.grad(g11, (u0, u1),
                                       retain_graph=True, allow_unused=True)

    gl0_0 = torch.vstack((g00_0, g10_0))
    g0l_0 = torch.vstack((g00_0, g01_0))
    g00_l = torch.vstack((g00_0, g00_1))
    gamma00k = 0.5 * g_inv.matmul(gl0_0 + g0l_0 - g00_l)

    gl1_0 = torch.vstack((g01_0, g11_0))
    g0l_1 = torch.vstack((g00_1, g01_1))
    g01_l = torch.vstack((g01_0, g01_1))
    gamma01k = 0.5 * g_inv.matmul(gl1_0 + g0l_1 - g01_l)

    gl0_1 = torch.vstack((g00_1, g10_1))
    g1l_0 = torch.vstack((g10_0, g11_0))
    g10_l = torch.vstack((g10_0, g10_1))
    gamma10k = 0.5 * g_inv.matmul(gl0_1 + g1l_0 - g10_l)

    gl1_1 = torch.vstack((g01_1, g11_1))
    g1l_1 = torch.vstack((g10_1, g11_1))
    g11_l = torch.vstack((g11_0, g11_1))
    gamma11k = 0.5 * g_inv.matmul(gl1_1 + g1l_1 - g11_l)

    chirstoffel = np.concatenate((
        gamma00k.detach().numpy().T,
        gamma01k.detach().numpy().T,
        gamma10k.detach().numpy().T,
        gamma11k.detach().numpy().T)).reshape(2, 2, 2)
    return chirstoffel


def spherical_to_cartesian(u, radius=1.0):
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u)

    u_transposed = u.T
    theta = u_transposed[0]
    phi = u_transposed[1]
    sin_theta = torch.sin(theta)
    x = radius * sin_theta * torch.cos(phi)
    y = radius * sin_theta * torch.sin(phi)
    z = radius * torch.cos(theta)
    return torch.vstack((x, y, z)).T


if __name__ == '__main__':
    u = np.array([0.4, 0], dtype=np.float32)
    christoffel = compute_christoffel(spherical_to_cartesian, u)
    print(christoffel)
