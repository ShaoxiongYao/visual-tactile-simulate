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

view_params = {
    "front" : [ 0.41634538001016846, -0.39068381489936144, 0.82098884360256086 ],
    "lookat" : [ 0.85919311809081156, 0.89302042215447397, 2.4507354194025086 ],
    "up" : [ -0.075146698770174505, -0.91466986839291886, -0.39715488857374692 ],
    "zoom" : 0.41999999999999971
}

all_pts_view = {	
    "front" : [ 0.84524418474615381, 0.093060768835465241, -0.52621474842174654 ],
    "lookat" : [ -9.2476893922056078, 2.199256716604217, 9.529552884491423 ],
    "up" : [ -0.045511976230804461, -0.9686053699084044, -0.24440191775878031 ],
    "zoom" : 0.21999999999999958
}

all_pts_view = {
    "front" : [ -0.39269290110359634, 0.061343612133974039, -0.91762151602564213 ],
    "lookat" : [ -0.0081280973560301471, 1.4773739719055841, 0.92505969280413958 ],
    "up" : [ 0.16282711958558563, -0.97737255797027411, -0.13501930252413807 ],
    "zoom" : 0.55999999999999983
}

all_pts_view = {	
    "front" : [ -0.15533664909224087, 0.29827835679396886, -0.94175397387910398 ],
    "lookat" : [ -0.0081280973560301471, 1.4773739719055841, 0.92505969280413958 ],
    "up" : [ 0.011459743267594439, -0.952717583930761, -0.30364103733417541 ],
    "zoom" : 0.059999999999999609
}

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(1)

    all_pcd = o3d.io.read_point_cloud('/home/motion/gaussian-splatting/output/9f74df28-d/point_cloud/iteration_30000/point_cloud.ply')
    all_pts = np.array(all_pcd.points)
    # all_pts = np.load('/home/motion/gaussian-splatting/output/debug/points.npy')
    # all_pts = np.load('/home/motion/gaussian-splatting/output/debug/orange_tree_raw_pts.npy')

    rest_pts = np.load('assets/new_orange_tree_pts.npy')
    elements_lst = np.load('assets/new_orange_tree_edges.npy')
    fix_idx = np.load('assets/new_orange_tree_fix_idx.npy')
    
    # colors = np.zeros_like(rest_pts)
    # colors[fix_idx] = [1.0, 0.0, 0.0]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(rest_pts)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])

    node_graph:NodeGraph = pickle.load(open('assets/new_orange_tree_node_graph.pkl', 'rb'))
    obj_model = ObjectModel(rest_points=rest_pts, element_lst=elements_lst,
                            geometry_model=node_graph, material_model=LinearSpringModel())
    
    all_pts_beta = node_graph.get_sparse_beta(all_pts, rbf_w_max=0.2)

    start_time = time.time()
    material_values = obj_model.unit_material_values(dtype='list')

    sim = QuasiStaticSim(obj_model, curr_points=None, material_values=material_values)
    print('setup sim time:', time.time()-start_time)

    touch_num = 1
    touch_dir = [0.0, 0.0, -1.0]
    touch_sampler = TouchSampler(obj_model.num_pts(), pts_std=0.01)
    touch_sampler.set_fix_idx(fix_idx)

    allow_touch_idx = select_points(obj_model.get_obj_pcd(), view_params=view_params)
    touch_sampler.set_allow_touch_idx(allow_touch_idx)
    touch_idx, touch_u = touch_sampler.sample_touch(touch_num, touch_dir=touch_dir)

    rest_geom = obj_model.get_element_o3d()
    o3d.visualization.draw_geometries([rest_geom])

    num_obs = 30

    u_lst = []
    f_lst = []

    full_obs_seq = FullObsSeq()
    for idx, t in enumerate(np.linspace(0.1, 2.0, num_obs)):
        
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
        # np.save(f'out_data/sim_tree_seq/u_new_gs_mat_{idx:03d}.npy', u_gs_mat)

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
