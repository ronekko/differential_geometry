# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 20:58:03 2021

@author: ryuhei
"""

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import open3d as o3d  # type: ignore
import torch


def spherical_to_cartesian(u, radius=1.0):
    theta = u.T[0]
    phi = u.T[1]
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.vstack((x, y, z)).T


def torus_to_cartesian(u, R=2.0, r=1.0):
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u)

    u_transposed = u.T
    theta = u_transposed[0]
    phi = u_transposed[1]
    sin_theta = torch.sin(theta)
    x = (R + r * sin_theta) * torch.cos(phi)
    y = (R + r * sin_theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.vstack((x, y, z)).T


def generate_points_on_torus(N, R=2.0, r=1.0):
    acceptance_rate = R / (R + r)
    N_proposal = int(N / acceptance_rate) // 2

    theta_minus = np.random.uniform(-np.pi, 0, N_proposal)
    theta_plus = np.random.uniform(0, np.pi, N_proposal)

    threashold_minus = theta_minus * (2 * r) / np.pi + R + r
    mask = np.random.uniform(0, R + r, N_proposal) < threashold_minus
    theta_minus = theta_minus[mask]

    threashold_plus = -theta_plus * (2 * r) / np.pi + R + r
    mask = np.random.uniform(0, R + r, N_proposal) < threashold_plus
    theta_plus = theta_plus[mask]

    theta = np.concatenate((theta_minus, theta_plus)) + np.pi / 2.0
    phi = np.random.uniform(-np.pi, np.pi, len(theta))
    u = np.vstack((theta, phi)).T.copy()

    x = torus_to_cartesian(u, R, r)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    pcd.estimate_normals()

    normals = spherical_to_cartesian(u)
    pcd_normals_array = np.asarray(pcd.normals)
    pcd_normals_array[:] = normals[:]

    return pcd


def create_torus(R=2.0, r=1.0, num_vertices=3000):
    """Create a torus mesh.


    Parameters
    ----------
    R : float, optional
        Major radius. The default is 2.0.
    r : TYPE, optional
        Minor radius. The default is 1.0.
    num_vertices : TYPE, optional
        Approximate number of vertices of the mesh. The default is 3000.

    Returns
    -------
    mesh : o3d.geometry.TriangleMesh
        DESCRIPTION.

    """
    # Generate random points uniformly on torus.
    pcd = generate_points_on_torus(num_vertices, R, r)

    # Create mesh from sample points.
    radii = [0.5]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))

    return mesh


if __name__ == '__main__':
    R = 2.0
    r = 1.0
    N = 3000

    mesh = create_torus(R, r, N)
    mesh.paint_uniform_color([0.7, 0.5, 0.5])

    mesh2 = o3d.geometry.TriangleMesh.create_torus(R, r)
    mesh2.compute_vertex_normals()
    mesh2.paint_uniform_color([0.5, 0.5, 0.7])

    mesh.translate([0, 0, 2])
    mesh2.translate([0, 0, -2])

    o3d.visualization.draw([mesh, mesh2], show_ui=True)
    # o3d.visualization.draw_geometries([mesh, mesh2])
