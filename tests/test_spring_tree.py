import numpy as np
import torch
import time
import open3d as o3d
import pickle

import matplotlib.pyplot as plt
import pyvista as pv
from scipy.sparse import save_npz, load_npz

import context
from vis_tac_sim.object_model import ObjectModel
from vis_tac_sim.geometry_model import NodeGraph
from vis_tac_sim.sim_quasi_static import QuasiStaticSim
from vis_tac_sim.sim_utils import TouchSampler
from vis_tac_sim.o3d_utils import select_points, create_arrow_lst
from vis_tac_sim.material_model import LinearSpringModel
from vis_tac_sim.data import FullObsSeq

view_params = {	
    "front" : [ -0.79972587949023577, -0.31308722154467494, 0.51226449162420451 ],
    "lookat" : [ -2.6916440980065777, 2.2527603830704903, 5.4091535379608633 ],
    "up" : [ 0.11550648616528994, -0.91755227398388794, -0.38046823278789499 ],
    "zoom" : 0.67999999999999994
}

all_pts_view = {	
    "front" : [ 0.84524418474615381, 0.093060768835465241, -0.52621474842174654 ],
    "lookat" : [ -9.2476893922056078, 2.199256716604217, 9.529552884491423 ],
    "up" : [ -0.045511976230804461, -0.9686053699084044, -0.24440191775878031 ],
    "zoom" : 0.21999999999999958
}

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(1)

    all_pts = np.load('/home/motion/gaussian-splatting/output/debug/points.npy')
    # all_pts = np.load('/home/motion/gaussian-splatting/output/debug/orange_tree_raw_pts.npy')

    rest_pts = np.load('assets/orange_tree_pts.npy')
    elements_lst = np.load('assets/orange_tree_edges.npy')
    fix_idx = np.load('assets/orange_tree_fix_idx.npy')

    node_graph:NodeGraph = pickle.load(open('assets/orange_tree_node_graph.pkl', 'rb'))
    obj_model = ObjectModel(rest_points=rest_pts, element_lst=elements_lst,
                            geometry_model=node_graph, material_model=LinearSpringModel())
    
    all_pts_beta = node_graph.get_sparse_beta(all_pts)

    start_time = time.time()
    material_values = obj_model.unit_material_values(dtype='list')

    sim = QuasiStaticSim(obj_model, curr_points=None, material_values=material_values)
    print('setup sim time:', time.time()-start_time)

    touch_num = 1
    touch_dir = [0.0, 0.0, -1.0]
    touch_sampler = TouchSampler(obj_model.num_pts(), pts_std=0.01)
    touch_sampler.set_fix_idx(fix_idx)

    allow_touch_idx = select_points(obj_model.get_obj_pcd(), view_params=view_params)
    # allow_touch_idx = [5358, 3816]
    touch_sampler.set_allow_touch_idx(allow_touch_idx)
    touch_idx, touch_u = touch_sampler.sample_touch(touch_num, touch_dir=touch_dir)

    rest_geom = obj_model.get_element_o3d()
    o3d.visualization.draw_geometries([rest_geom])

    num_obs = 15

    value_vec = o3d.utility.DoubleVector(np.ones(obj_model.num_pts()))

    u_lst = []
    f_lst = []

    full_obs_seq = FullObsSeq()
    for t in np.linspace(0.1, 2.0, num_obs):
        
        start_time = time.time()
        sim.move_points(touch_idx, t*touch_u)
        u_all = sim.solve_deform()
        u_lst.append(u_all)
        f_all = sim.get_f_int(u_all)
        print("Solve deform time:", time.time() - start_time)

        u_all_mat = u_all.reshape(-1, 3)

        u_gs_mat = np.zeros_like(all_pts)
        for i in [0, 1, 2]:
            u_gs_mat[:, i] = all_pts_beta @ u_all_mat[:, i]
        np.save(f'out_data/u_gs_mat_{t:05.02f}.npy', u_gs_mat)

        gt_pts = all_pts + u_gs_mat
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_pts)
        o3d.visualization.draw_geometries([pcd], **all_pts_view)

        sim.check_f_balance(u_all)

        curr_points = obj_model.rest_points + u_all_mat

        # sim.check_f_balance(u_all)
        # full_obs_seq.add_obs(u_all, f_all)

        """ Visualize object deformation """

        pts1 = obj_model.rest_points[touch_idx, :]
        pts2 = pts1 + t*touch_u.reshape(-1, 3)
        arrow_lst1 = create_arrow_lst(pts1, pts2, color=[0.7, 0.2, 0.0])

        f_touch = 1.0*f_all.reshape(-1, 3)[touch_idx, :]
        arrow_lst2 = create_arrow_lst(pts2, pts2 + f_touch, color=[0.0, 0.7, 0.2])

        f_lst.append(f_all.reshape(-1, 3)[touch_idx[-1], :])

        curr_geom = obj_model.get_element_o3d(curr_points)
        geom_lst = [curr_geom] + arrow_lst1

        o3d.visualization.draw_geometries(geom_lst, **view_params)

    # full_obs_seq.save('data/exp_sim_tree1')

    # obj_model.elements2edges()
    # with open('data/exp_sim_tree1/obj_model.pkl', 'wb') as f:
    #     pickle.dump(obj_model, f)

    # np.save(f'data/u_lst_mu_{mu_s:05.02f}_lam_{lam_s:05.02f}.npy', np.array(u_lst))
    # np.save(f'data/f_lst_mu_{mu_s:05.02f}_lam_{lam_s:05.02f}.npy', np.array(f_lst))
    # full_obs_seq.save("data/example_linear_fem")
