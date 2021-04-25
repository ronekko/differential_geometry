# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:02:03 2021

@author: ryuhei
"""

import christoffel
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import open3d as o3d  # type: ignore
import torus
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
        torus.torus_to_cartesian, [theta, phi])


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


if __name__ == '__main__':
    major_radius = 2.0
    minor_radius = 1.0
    theta = 0.0
    phi = 0.0

    # u0 = np.array([np.pi/4, np.pi/6])
    u0 = np.array([0.0, 0.0])
    v0 = np.array([1.0, 0.3])
    t = np.linspace(0, 30, 300)

    # Solve initial value problem
    s0 = np.hstack((u0, v0))
    s, infodict = odeint(geodesic_equation, s0, t, full_output=True)
    u = s[:, :2]

    # Plot the geodegic
    xyz = torus.torus_to_cartesian(u).numpy()

    # Plot the initial point
    xyz0 = torus.torus_to_cartesian(u0).reshape(3)

    # Open3D visualization
    # torus = torus.create_torus(major_radius, minor_radius)
    torus = o3d.geometry.TriangleMesh.create_torus(major_radius, minor_radius)
    torus.compute_vertex_normals()
    torus.paint_uniform_color([0.5, 0.5, 0.5])

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
        [frame, torus, dot_sphere, line_set], show_ui=True)
