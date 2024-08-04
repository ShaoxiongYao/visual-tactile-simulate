import numpy as np
import torch
import time
import open3d as o3d
import pickle

import matplotlib.pyplot as plt
import pyvista as pv

import context
from vis_tac_sim.object_model import ObjectModel
from vis_tac_sim.sim_quasi_static import QuasiStaticSim
from vis_tac_sim.sim_utils import TouchSampler
from vis_tac_sim.o3d_utils import extract_surface_mesh, create_arrow_lst
from vis_tac_sim.material_model import LinearTetraModel
from vis_tac_sim.data import FullObsSeq
from vis_tac_sim.o3d_utils import select_points


if __name__ == '__main__':
    obj_name = '6polygon01'

    rest_points = np.load(f'assets/{obj_name}_points.npy')
    tetra_elements = np.load(f'assets/{obj_name}_tetra.npy')

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(rest_points)

    # top_pts_idx = select_points(obj_pcd)
    top_pts_idx = [1824, 1436, 992]
    top_u_dir = torch.tensor([0.0, 0.0, -0.01])

    # down_pts_idx = select_points(obj_pcd)
    down_pts_idx = [1132, 982, 1591]
    down_u_dir = torch.tensor([0.0, 0.0, 0.01])

    rest_points = torch.tensor(rest_points, dtype=torch.float32)
    deform = torch.zeros_like(rest_points)

    for idx in top_pts_idx:
        deform[idx] = top_u_dir
    for idx in down_pts_idx:
        deform[idx] = down_u_dir
    handle_idx = top_pts_idx + down_pts_idx
    handle_st_pts = rest_points[handle_idx].clone()
    handle_ed_pts = handle_st_pts + deform[handle_idx, :]
    arrow_lst = create_arrow_lst(handle_st_pts.numpy(), handle_ed_pts.numpy(), min_len=0.001)

    material_model = LinearTetraModel()

    num_pts = rest_points.shape[0]
    num_elements = tetra_elements.shape[0]

   

    tetra_tensor = torch.tensor(tetra_elements, dtype=torch.int64)

    m_gt = 10000*torch.ones((num_elements, 2), dtype=torch.float32)

    for _ in range(100):
        

        p = torch.zeros(num_elements, 4, 3)
        u = torch.zeros(num_elements, 4, 3)
        for i in range(num_elements):
            p[i] = rest_points[tetra_elements[i]]
            u[i] = deform[tetra_elements[i]]

        f_obs = material_model.element_forces_batch(p, u, m_gt)
        
        f_vertex = torch.zeros_like(rest_points)
        f_vertex.index_add_(0, tetra_tensor.flatten(), f_obs.reshape(-1, 3))

        print('f_vertex max:', torch.max(f_vertex.norm(dim=1)))

        obj_pcd.normals = o3d.utility.Vector3dVector(f_vertex.numpy())
        o3d.visualization.draw_geometries([obj_pcd] + arrow_lst, point_show_normal=True)



        # p += 0.001 * f_obs
        # obj_pcd.points = o3d.utility.Vector3dVector(p.flatten().detach().cpu().numpy())
        