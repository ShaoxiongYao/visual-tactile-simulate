import torch, pypose as pp
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
import sklearn.neighbors as skn
from pypose.optim.strategy import Constant

import context
from vis_tac_sim.o3d_utils import select_points

num_pts = 2000
num_nns = 10

bar_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=10)
bar_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([bar_mesh])

bar_pcd = bar_mesh.sample_points_uniformly(num_pts)
rest_pts = np.asarray(bar_pcd.points)

neigh = skn.NearestNeighbors(n_neighbors=num_nns)
neigh.fit(rest_pts)
neighbor_ary = neigh.kneighbors(rest_pts, return_distance=False)

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(rest_pts)
connect_lst = []
for j in range(num_nns):
    connect_lst.extend(np.stack([np.arange(num_pts), neighbor_ary[:, j]], axis=1).tolist())
connect_ary = np.array(connect_lst)
line_set.lines = o3d.utility.Vector2iVector(connect_lst)
line_set.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]]*len(connect_lst)))

fix_idx = np.where(rest_pts[:, 2] < -3)[0]
handle_idx = np.concatenate([select_points(bar_pcd), fix_idx])
handle_pts = rest_pts[handle_idx].copy()
handle_pts[0, :] += np.array([0.0, 1., -1.0])

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
rot_tsr = pp.randn_so3(len(free_idx), dtype=torch.float32, 
                          requires_grad=True)

optimizer = torch.optim.Adam([free_pts_tsr, rot_tsr], lr=0.1)

# strategy = Constant(damping=1e-4)
# class DeformModel(torch.nn.Module):
#     def __init__(self, free_pts_ary):
#         super().__init__()
#         free_pts_tsr = torch.tensor(free_pts_ary, dtype=torch.float32)
#         self.free_pts_tsr = torch.nn.Parameter(free_pts_tsr)

#     def forward(self):
#         pass

# model = DeformModel(free_pts_tsr)
# optimizer = pp.optim.LM(model, strategy=strategy)

energy_lst = []
for i in range(1000):
    curr_pts_tsr = torch.zeros_like(rest_pts_tsr)
    curr_pts_tsr[free_idx] = free_pts_tsr
    curr_pts_tsr[handle_idx] = handle_pts_tsr

    start_time = time.time()
    energy = 0.0
    # for pt_idx in free_idx:
    #     edge_diff = curr_pts_tsr[pt_idx, :] - curr_pts_tsr[neighbor_ary[pt_idx], :]
    #     edge_rest = rest_pts_tsr[pt_idx, :] - rest_pts_tsr[neighbor_ary[pt_idx], :]
    #     energy += torch.sum((edge_diff - edge_rest) ** 2)
    edges_diff = curr_pts_tsr[connect_ary[:, 0], :] - curr_pts_tsr[connect_ary[:, 1], :]
    edges_rest = rest_pts_tsr[connect_ary[:, 0], :] - rest_pts_tsr[connect_ary[:, 1], :]
    energy = torch.sum((edges_diff - edges_rest) ** 2)
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
        o3d.visualization.draw_geometries([curr_bar_pcd, line_set])        

plt.plot(energy_lst)
plt.show()
