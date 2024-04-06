import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d
from utils import superellipsoid_points

# Parameters
a, b, c = 1, 1.5, 1  # Semi-axes
eps1, eps2 = 0.5, 1.0  # Shape parameters

# Generate points
x, y, z = superellipsoid_points(a, b, c, eps1, eps2, 300)

points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
np.save('super_ellipsoid_points.npy', points)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])

# print('points shape:', points.shape)