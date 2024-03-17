import torch, pypose as pp
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
from pathlib import Path
import sklearn.neighbors as skn
from pypose.optim.strategy import Constant

from node_graph_utils import NodeGraph

import tqdm

import context
from vis_tac_sim.o3d_utils import select_points, create_arrow_lst

view_params = {
    "front" : [ 0.97746411351068019, -0.20982603939514938, -0.023171965616350335 ],
    "lookat" : [ 0.041919476468845619, 1.5261680477612984, -2.1117606444984363 ],
    "up" : [ 0.02182415861010388, -0.0087369048962646322, 0.99972364811171432 ],
    "zoom" : 0.71999999999999997
}

view_params = {	
    "front" : [ 0.98508056959279211, -0.13266337925160263, 0.10962070614754219 ],
    "lookat" : [ 0.062986389805103071, 1.3399728318050277, 0.0023437429118096098 ],
    "up" : [ -0.11069585443786382, -0.00073382464039204939, 0.99385405835649165 ],
    "zoom" : 0.61999999999999988
}

def prepare_sim_bar(num_pts, nn_radius):

    bar_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=10)
    bar_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([bar_mesh])

    bar_pcd = bar_mesh.sample_points_uniformly(num_pts)
    rest_pts = np.asarray(bar_pcd.points)

    neigh = skn.radius_neighbors_graph(rest_pts, nn_radius, mode='distance')
    connect_ary = np.array(neigh.nonzero()).T

    fix_idx = np.where(rest_pts[:, 2] < -3)[0]
    handle_idx = np.concatenate([select_points(bar_pcd), fix_idx])
    handle_dir = np.zeros((len(handle_idx), 3))
    handle_dir[0, :] = np.array([0.0, 3., -5.0])

    return rest_pts, connect_ary, handle_idx, handle_dir

def prepare_sim_plant(rest_pts_fn, nn_radius):

    rest_pts = np.load(rest_pts_fn)

    bar_pcd = o3d.geometry.PointCloud()
    bar_pcd.points = o3d.utility.Vector3dVector(rest_pts)
    bar_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    neigh = skn.radius_neighbors_graph(rest_pts, nn_radius, mode='distance')
    connect_ary = np.array(neigh.nonzero()).T

    fix_idx = np.where(rest_pts[:, 2] < 0.5)[0]
    handle_idx = np.concatenate([select_points(bar_pcd), fix_idx])
    handle_dir = np.zeros((len(handle_idx), 3))
    handle_dir[0, :] = np.array([0.0, -1.5, -0.5])

    return rest_pts, connect_ary, handle_idx, handle_dir

if __name__ == '__main__':

    # obj_name = 'bar_01'
    # num_pts = 3000
    # nn_radius = 0.15
    # rest_pts, connect_ary, handle_idx, handle_dir = prepare_sim_bar(num_pts, nn_radius)
    # rest_vis_pts = rest_pts.copy()
    # trans_rest_vis_pts = rest_vis_pts.copy()
    # H_mat = np.eye(4)
    # inv_H_mat = np.eye(4)

    obj_name = 'fiddle_tree_leaf_03'
    nn_radius = 0.10
    asset_dir = f'out_data/plant_assets/{obj_name}'
    Path(f'out_data/sim_{obj_name}').mkdir(parents=True, exist_ok=True)

    rest_vis_pts = np.load(f'{asset_dir}/all_pts.npy')
    H_mat = np.load(f'{asset_dir}/H_mat.npy')
    inv_H_mat = np.linalg.inv(H_mat)

    trans_rest_vis_pts = (H_mat[:3, :3] @ rest_vis_pts.T + H_mat[:3, 3:4]).T
    
    rest_pts_fn = f'{asset_dir}/rest_pts.npy'
    rest_pts, connect_ary, handle_idx, handle_dir = prepare_sim_plant(rest_pts_fn, nn_radius)

    sim_out_dir = f'out_data/sim_{obj_name}'
    np.save(f'{sim_out_dir}/handle_idx.npy', handle_idx)
    np.save(f'{sim_out_dir}/handle_dir.npy', handle_dir)
    
    node_graph = NodeGraph(rest_pts, connect_ary, corotate=False, device='cuda')
    vis_beta = node_graph.get_pts_beta(trans_rest_vis_pts)

    line_set = node_graph.get_line_set()

    optimizer = torch.optim.Adam(node_graph.deform_state.parameters(), lr=0.01)

    energy_lst = []
    for i in range(10000):
        if i % 100 == 0:
            print('iter:', i)
        handle_pts_tsr = torch.tensor(rest_pts[handle_idx, :] + (i/10000)*handle_dir, 
                                      dtype=torch.double, device='cuda')
        
        # for _ in range(3):
        energy = node_graph.energy(handle_idx, handle_pts_tsr)
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()

        energy_lst.append(energy.item())

        if i % 100 == 0:
            delta_vis_pts = node_graph.get_delta_pts().detach().cpu().numpy()
            curr_vis_pts = trans_rest_vis_pts + vis_beta @ delta_vis_pts
            curr_vis_pts = (inv_H_mat[:3, :3] @ curr_vis_pts.T + inv_H_mat[:3, 3:4]).T
            np.save(f'{sim_out_dir}/step_{i:05d}.npy', curr_vis_pts)

            curr_pcd = node_graph.get_pcd(handle_idx, handle_pts_tsr)

            line_set.points = curr_pcd.points
            arrow_lst = create_arrow_lst(rest_pts[handle_idx], handle_pts_tsr.detach().cpu().numpy())

            # rot_frames_lst = node_graph.get_rot_frames()

            # o3d.visualization.draw_geometries([curr_pcd, line_set] + arrow_lst, **view_params)        

    plt.plot(energy_lst)
    plt.show()
