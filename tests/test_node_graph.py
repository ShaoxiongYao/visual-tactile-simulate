import torch, pypose as pp
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
from pathlib import Path
import sklearn.neighbors as skn
from pypose.optim.strategy import Constant

import scipy
from node_graph_utils import NodeGraph

import tqdm
import kaolin

import context
from vis_tac_sim.o3d_utils import select_points, create_arrow_lst

view_params ={
    "front" : [ 0.95190160901372811, -0.00013963062843987974, 0.30640383036176833 ],
    "lookat" : [ 0.28297228105785405, 0.014716200386109213, 0.18298984389663137 ],
    "up" : [ -0.30400394310213302, 0.12448445490008581, 0.94450263264142564 ],
    "zoom" : 0.10000000000000001
}

view_params = {
    "front" : [ 0.89885173809442098, -0.20218870560331784, 0.38882551388906239 ],
    "lookat" : [ 0.28493994175726012, 0.015598617646948918, 0.17890003914085656 ],
    "up" : [ -0.37499329029617406, 0.10434726916404241, 0.92113608096244737 ],
    "zoom" : 0.50000000000000011
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

    fix_idx = np.where(rest_pts[:, 2] < 0.05)[0]
    handle_idx = np.concatenate([select_points(bar_pcd), fix_idx])
    handle_dir = np.zeros((len(handle_idx), 3))
    handle_dir[0, :] = np.array([0.0, -1.0, 0.0])

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

    obj_name = 'fiddle_tree_leaf_10'
    nn_radius = 0.03
    asset_dir = f'out_data/plant_assets/{obj_name}'
    Path(f'out_data/sim_{obj_name}').mkdir(parents=True, exist_ok=True)

    init_vis_pcd = o3d.io.read_point_cloud(f'{asset_dir}/step_000.pcd')
    rest_vis_pts = np.asarray(init_vis_pcd.points)

    # rest_vis_pts = np.load(f'{asset_dir}/all_pts.npy')
    # H_mat = np.load(f'{asset_dir}/H_mat.npy')
    # # scale = np.load(f'{asset_dir}/scale.npy')
    # inv_H_mat = np.linalg.inv(H_mat)

    # trans_rest_vis_pts = (H_mat[:3, :3] @ rest_vis_pts.T + H_mat[:3, 3:4]).T
    # # trans_rest_vis_pts *= scale
    
    rest_pts_fn = f'{asset_dir}/rest_pts.npy'
    rest_pts, connect_ary, handle_idx, handle_dir = prepare_sim_plant(rest_pts_fn, nn_radius)

    sim_out_dir = f'out_data/sim_{obj_name}'
    np.save(f'{sim_out_dir}/handle_idx.npy', handle_idx)
    np.save(f'{sim_out_dir}/handle_dir.npy', handle_dir)
    
    node_graph = NodeGraph(rest_pts, connect_ary, corotate=True, device='cuda')
    vis_beta = node_graph.get_pts_beta(rest_vis_pts)

    line_set = node_graph.get_line_set()

    optimizer = torch.optim.Adam(node_graph.deform_state.parameters(), lr=0.001)

    err_dist_list = []
    obs_dist_list = []
    h_err_dist_list = []
    h_obs_dist_list = []

    energy_lst = []
    for i in range(16):
        step_pcd = o3d.io.read_point_cloud(f'{asset_dir}/step_{i:03d}.pcd')
        step_pcd.paint_uniform_color([0.7, 0.0, 0.7])

        obs_pts_tsr = torch.tensor(np.asarray(step_pcd.points), 
                                   dtype=torch.double, device='cuda')

        if i % 100 == 0:
            print('iter:', i)
        
        step_move_vec = 0.005*i*handle_dir
        handle_pts_tsr = torch.tensor(rest_pts[handle_idx, :] + step_move_vec, 
                                      dtype=torch.double, device='cuda')
        
        for _ in range(100):
            start_time = time.time()
            energy = node_graph.energy(handle_idx, handle_pts_tsr)
            optimizer.zero_grad()
            energy.backward()
            optimizer.step()
            print('step time:', time.time() - start_time)

        energy_lst.append(energy.item())

        if i % 1 == 0:
            delta_vis_pts = node_graph.get_delta_pts().detach().cpu().numpy()
            curr_vis_pts = rest_vis_pts + vis_beta @ delta_vis_pts

            dist = scipy.spatial.distance.directed_hausdorff(np.asarray(step_pcd.points), 
                                                             rest_vis_pts)[0]
            h_obs_dist_list.append(dist)

            dist = scipy.spatial.distance.directed_hausdorff(np.asarray(step_pcd.points), 
                                                             curr_vis_pts)[0]
            h_err_dist_list.append(dist)

            rest_pts_tsr = torch.tensor(rest_vis_pts, dtype=torch.double, device='cuda')
            chamfer_dist = kaolin.metrics.pointcloud.chamfer_distance(rest_pts_tsr[None, :, :],
                                                                      obs_pts_tsr[None, :, :])
            obs_dist_list.append(chamfer_dist.item())

            sim_pts_tsr = torch.tensor(curr_vis_pts, dtype=torch.double, device='cuda')
            chamfer_dist = kaolin.metrics.pointcloud.chamfer_distance(sim_pts_tsr[None, :, :], 
                                                                      obs_pts_tsr[None, :, :])
            err_dist_list.append(chamfer_dist.item())

            # curr_vis_pts = (inv_H_mat[:3, :3] @ curr_vis_pts.T + inv_H_mat[:3, 3:4]).T
            # np.save(f'{sim_out_dir}/step_{i:05d}.npy', curr_vis_pts)

            curr_pcd = node_graph.get_pcd(handle_idx, handle_pts_tsr)

            colors = np.ones_like(curr_pcd.points)
            colors *= np.array([0.0, 0.7, 0.7])
            colors[handle_idx, :] *= np.array([0.7, 0.7, 0.0])
            curr_pcd.colors = o3d.utility.Vector3dVector(colors)

            line_set.points = curr_pcd.points
            arrow_lst = create_arrow_lst(rest_pts[handle_idx], handle_pts_tsr.detach().cpu().numpy())

            # rot_frames_lst = node_graph.get_rot_frames()

            curr_vis_pcd = o3d.geometry.PointCloud()
            curr_vis_pcd.points = o3d.utility.Vector3dVector(curr_vis_pts)
            curr_vis_pcd.paint_uniform_color([0.0, 0.7, 0.7])
            o3d.visualization.draw_geometries([curr_vis_pcd, step_pcd] + arrow_lst, **view_params)        
            # o3d.visualization.draw_geometries([curr_vis_pcd, line_set] + arrow_lst, **view_params)        

    plt.plot(energy_lst)
    plt.xlabel('time step')
    plt.title('Elasitc energy')
    plt.show()

    move_dist_ary = np.linspace(0, 8, 16)

    plt.plot(move_dist_ary, np.sqrt(obs_dist_list))
    plt.plot(move_dist_ary, np.sqrt(err_dist_list))
    plt.xlabel('y displacement (m)')
    plt.title('Chamfer distance (m)')
    plt.savefig(f'chamfer_dist_{obj_name}.png')
    plt.show()

    plt.plot(move_dist_ary, h_obs_dist_list)
    plt.plot(move_dist_ary, h_err_dist_list)
    plt.xlabel('y displacement (m)')
    plt.title('Hausdorff distance (m)')
    plt.savefig(f'hausdorff_dist_{obj_name}.png')
    plt.show()

