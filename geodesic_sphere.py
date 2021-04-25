# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 11:17:18 2021

@author: ryuhei
"""

import christoffel
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import open3d as o3d  # type: ignore
from scipy.integrate import odeint  # type: ignore


def compute_connection(theta, phi):
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
    return christoffel.compute_christoffel(
        christoffel.spherical_to_cartesian, [theta, phi])


def geodesic_equation(state, t):
    u = state[:2]
    v = state[2:]

    # dtdu
    dtdu = v

    # dtdv
    theta, phi = u
    christoffel = compute_connection(theta, phi)
    dtdv = -np.einsum('ijk,i,j', christoffel, v, v)

    return np.hstack((dtdu, dtdv))


def spherical_to_cartesian(u, radius=1.0):
    theta = u.T[0]
    phi = u.T[1]
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.vstack((x, y, z)).T


if __name__ == '__main__':
    theta = 0.0
    phi = 0.0

    u0 = np.array([np.pi/4, np.pi/6])
    v0 = np.array([0.3, 0.3])
    t = np.linspace(0, 15, 100)

    # Solve initial value problem
    s0 = np.hstack((u0, v0))
    s, infodict = odeint(geodesic_equation, s0, t, full_output=True)
    u = s[:, :2]

    plt.title('theta-phi')
    plt.plot(*u.T)
    plt.xlabel('theta')
    plt.xlim(0.0, np.pi)
    plt.ylabel('phi')
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.show()

    ############################
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot the unit sphere
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(-np.pi, np.pi, 100)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones(np.size(phi)))
    ax.plot_surface(x, y, z, alpha=0.3)

    # Plot the geodegic
    xyz = christoffel.spherical_to_cartesian(u).numpy()
    ax.plot(*xyz.T, label='Geodesic')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot the initial point
    xyz0 = spherical_to_cartesian(u0).reshape(3)
    ax.plot(*xyz0.reshape(3, 1), 'ok', label='u_0')

    # Plot Cartesian coordinate frame (XYZ=RGB)
    ax.plot([0, 1], [0, 0], [0, 0], '-r')
    ax.plot([0, 0], [0, 1],  [0, 0], '-g')
    ax.plot([0, 0], [0, 0], [0, 1],   '-b')

    plt.legend()
    plt.show()

    # Open3D visualization
    sphere = o3d.geometry.TriangleMesh.create_sphere()
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.5, 0.5, 0.5])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
    frame.compute_vertex_normals()

    # Geodesic (orange curve)
    lines = np.array([[i, i+1] for i in range(len(xyz) - 1)])
    colors = [[0.6, 0.2, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(xyz),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Initial point of geodesic integration (black dot)
    dot_sphere = o3d.geometry.TriangleMesh.create_sphere(0.03)
    dot_sphere.compute_vertex_normals()
    dot_sphere.translate(xyz0)
    dot_sphere.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw(
        [frame, sphere, dot_sphere, line_set], show_ui=True)
