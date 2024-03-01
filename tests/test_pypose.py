import torch, pypose as pp
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
import sklearn.neighbors as skn
from pypose.optim.strategy import Constant

import context
from vis_tac_sim.o3d_utils import select_points

view_params = {
    "front" : [ 0.97746411351068019, -0.20982603939514938, -0.023171965616350335 ],
    "lookat" : [ 0.041919476468845619, 1.5261680477612984, -2.1117606444984363 ],
    "up" : [ 0.02182415861010388, -0.0087369048962646322, 0.99972364811171432 ],
    "zoom" : 0.71999999999999997
}

num_pts = 3000
# num_nns = 20
nn_radius = 0.15

bar_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=10)
bar_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([bar_mesh])

bar_pcd = bar_mesh.sample_points_uniformly(num_pts)
rest_pts = np.asarray(bar_pcd.points)

neigh = skn.radius_neighbors_graph(rest_pts, nn_radius, mode='distance')
connect_ary = np.array(neigh.nonzero()).T

# neigh = skn.NearestNeighbors(n_neighbors=num_nns)
# neigh.fit(rest_pts)
# neighbor_ary = neigh.kneighbors(rest_pts, return_distance=False)

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(rest_pts)
# connect_lst = []
connect_lst = connect_ary.tolist()
# for j in range(num_nns):
#     connect_lst.extend(np.stack([np.arange(num_pts), neighbor_ary[:, j]], axis=1).tolist())
# connect_ary = np.array(connect_lst)
# connect_ary = np.append(connect_ary, np.stack([connect_ary[:, 1], connect_ary[:, 0]], axis=1), axis=0)
line_set.lines = o3d.utility.Vector2iVector(connect_lst)
line_set.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]]*len(connect_lst)))

fix_idx = np.where(rest_pts[:, 2] < -3)[0]
handle_idx = np.concatenate([select_points(bar_pcd), fix_idx])
handle_pts = rest_pts[handle_idx].copy()
handle_pts[0, :] += np.array([0.0, 3., -5.0])

# rest_pts = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]])

# num_pts = 5
# bar_pcd = o3d.geometry.PointCloud()
# bar_pcd.points = o3d.utility.Vector3dVector(rest_pts)

# handle_idx = np.array([0, 3])
# handle_pts = rest_pts[handle_idx].copy()
# handle_pts[1, :] += np.array([0.0, 1., -1.0])

# line_set = o3d.geometry.LineSet()
# line_set.points = o3d.utility.Vector3dVector(rest_pts)
# connect_lst = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 0], [2, 1], [3, 2], [4, 3]]
# connect_ary = np.array(connect_lst)
# line_set.lines = o3d.utility.Vector2iVector(connect_lst)
# line_set.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]]*len(connect_lst)))

free_idx = np.setdiff1d(np.arange(num_pts), handle_idx)
pcd_colors = np.zeros_like(rest_pts)
pcd_colors[handle_idx] = [1.0, 0.0, 0.0]
pcd_colors[free_idx] = [0.0, 1.0, 0.0]
bar_pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
o3d.visualization.draw_geometries([bar_pcd, line_set])

rest_pts_tsr = torch.tensor(rest_pts, dtype=torch.float32)
free_pts_tsr = torch.tensor(rest_pts[free_idx], dtype=torch.float32, 
                            requires_grad=True)
handle_pts_tsr = torch.tensor(handle_pts, dtype=torch.float32)
rot_tsr = pp.randn_so3(num_pts, sigma=1e-5, dtype=torch.float32, requires_grad=True)

optimizer = torch.optim.Adam([free_pts_tsr, rot_tsr], lr=0.1)

energy_lst = []
for i in range(10000):
    curr_pts_tsr = torch.zeros_like(rest_pts_tsr)
    curr_pts_tsr[free_idx] = free_pts_tsr
    curr_pts_tsr[handle_idx] = handle_pts_tsr

    start_time = time.time()
    energy = 0.0
    edges_diff = curr_pts_tsr[connect_ary[:, 0], :] - curr_pts_tsr[connect_ary[:, 1], :]
    print('edges diff:', edges_diff)
    edges_rest = rest_pts_tsr[connect_ary[:, 0], :] - rest_pts_tsr[connect_ary[:, 1], :]
    print('edges rest:', edges_rest)
    raw_edge_len = torch.norm(edges_rest, dim=1)
    edges_rest = pp.Exp(rot_tsr[connect_ary[:, 1], :]) @ edges_rest
    rot_edge_len = torch.norm(edges_rest, dim=1)
    assert torch.allclose(raw_edge_len, rot_edge_len)
    print('edges rest:', edges_rest)
    energy = torch.sum((edges_diff - edges_rest) ** 2)
    print('rot difference:', edges_diff - edges_rest)
    print('energy value:', energy.item())
    print('energy time:', time.time()-start_time)

    optimizer.zero_grad()
    energy.backward()
    optimizer.step()
    energy_lst.append(energy.item())

    if i % 999 == 0:
        curr_bar_pcd = o3d.geometry.PointCloud()
        curr_bar_pcd.points = o3d.utility.Vector3dVector(curr_pts_tsr.detach().numpy())
        curr_bar_pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

        line_set.points = o3d.utility.Vector3dVector(curr_pts_tsr.detach().numpy())

        coord_frame_lst = []
        for pt_i in range(0, num_pts, 30):
            ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            pt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            rot_mat:torch.Tensor = pp.Exp(rot_tsr[pt_i, :]) @ torch.eye(3)

            pt_frame.rotate(rot_mat.detach().numpy())
            pt_frame.translate(curr_pts_tsr[pt_i, :].detach().numpy())
            pt_frame.paint_uniform_color([0.2, 0.9, 0.7])

            ref_frame.translate(curr_pts_tsr[pt_i, :].detach().numpy())
            ref_frame.paint_uniform_color([0.7, 0.2, 0.9])
            coord_frame_lst.append(pt_frame)
            coord_frame_lst.append(ref_frame)

        o3d.visualization.draw_geometries([curr_bar_pcd, line_set] + coord_frame_lst, **view_params)        

plt.plot(energy_lst)
plt.show()
