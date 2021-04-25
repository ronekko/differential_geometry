# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 11:53:34 2021

@author: ryuhei
"""


import open3d as o3d  # type: ignore

if __name__ == '__main__':
    sphere = o3d.geometry.TriangleMesh.create_sphere()
    sphere.compute_vertex_normals()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
    o3d.visualization.draw([frame, sphere], show_ui=True)
