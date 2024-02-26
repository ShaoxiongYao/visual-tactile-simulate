import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as skn
import torch
import open3d as o3d

import context
from vis_tac_sim.o3d_utils import create_arrow_lst

np.set_printoptions(precision=3, suppress=True)

num_nns = 5

# num_pts = 8
# rest_pts = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 
#                      [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0],
#                      [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

num_pts = 20
rest_pts = np.random.rand(num_pts, 3)

neigh = skn.NearestNeighbors(n_neighbors=num_nns)
neigh.fit(rest_pts)
neighbor_ary = neigh.kneighbors(rest_pts, return_distance=False)

handle_idx = np.array([0, 1])
handle_pts = rest_pts[handle_idx].copy()
handle_pts[0, 0] += 0.1

arrow_lst = create_arrow_lst(rest_pts[handle_idx], handle_pts)


free_idx = np.setdiff1d(np.arange(num_pts), handle_idx)

# GOAL: solve new locations for free points

curr_pts = rest_pts.copy()
curr_pts[handle_idx] = handle_pts

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

rest_pcd = o3d.geometry.PointCloud()
rest_pcd.points = o3d.utility.Vector3dVector(rest_pts)
rest_pcd.paint_uniform_color([0.0, 0.0, 1.0])

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(rest_pts)
connect_lst = []
for j in range(num_nns):
    connect_lst.extend(np.stack([np.arange(num_pts), neighbor_ary[:, j]], axis=1).tolist())
line_set.lines = o3d.utility.Vector2iVector(connect_lst)
print('number of lines:', np.array(line_set.lines).shape)
print('number of lines:', len(connect_lst))
print('colors shape:', np.array([[0.0, 0.0, 1.0]]*len(connect_lst)).shape)
line_set.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]]*len(connect_lst)))

curr_pcd = o3d.geometry.PointCloud()
curr_pcd.points = o3d.utility.Vector3dVector(curr_pts)
curr_pcd.paint_uniform_color([1.0, 0.0, 0.0])

line_set_curr = o3d.geometry.LineSet()
line_set_curr.points = o3d.utility.Vector3dVector(curr_pts)
line_set_curr.lines = o3d.utility.Vector2iVector(connect_lst)
line_set_curr.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]*len(connect_lst)))

o3d.visualization.draw_geometries([coord_frame, rest_pcd, curr_pcd, 
                                   line_set, line_set_curr] + arrow_lst)

# setup the differentiable function
W = np.zeros((num_pts, num_pts))
for i in range(num_pts):
    W[i, neighbor_ary[i]] = 1

W = (W + W.T) / 2

A_mat = np.zeros((num_pts, num_pts))

rot_mat = np.stack([np.eye(3) for _ in range(num_pts)], axis=0)

for i in range(num_pts):
    for j in neighbor_ary[i]:
        A_mat[i, i] += W[i, j]
        A_mat[i, j] -= W[i, j]

delta_pts_lst = []
for iter_idx in range(100):
    rhs = np.zeros((num_pts, 3))
    for i in range(num_pts):
        for j in neighbor_ary[i]:
            avg_rot = (rot_mat[i] + rot_mat[j]) / 2
            print('avg_rot:', avg_rot)
            rhs[i, :] += W[i, j] * avg_rot @ (rest_pts[i] - rest_pts[j])

    next_pts = np.zeros((num_pts, 3))
    next_pts[handle_idx] = handle_pts
    for axis_idx in range(3):
        print(A_mat[free_idx][:, free_idx])
        f_int = -A_mat[free_idx][:, handle_idx] @ handle_pts[:, axis_idx] + rhs[free_idx, axis_idx]
        axis_out = np.linalg.solve(A_mat[free_idx][:, free_idx], f_int)
        next_pts[free_idx, axis_idx] = axis_out

    delta_pts_lst.append(np.linalg.norm(next_pts - curr_pts).item())
    curr_pts = next_pts

    curr_pcd.points = o3d.utility.Vector3dVector(next_pts)
    line_set_curr.points = o3d.utility.Vector3dVector(next_pts)
    o3d.visualization.draw_geometries([coord_frame, rest_pcd, curr_pcd, 
                                       line_set, line_set_curr] + arrow_lst)

    P = curr_pts[neighbor_ary] - curr_pts[:, None]
    Q = rest_pts[neighbor_ary] - rest_pts[:, None]

    S = P.transpose(0, 2, 1) @ Q
    out = np.linalg.svd(S)

    R = out[0] @ out[2]

    RT_R = R.transpose(0, 2, 1) @ R
    assert np.allclose(RT_R, np.eye(3)[None, :, :])

    rot_mat = R

plt.plot(delta_pts_lst)
plt.show()
