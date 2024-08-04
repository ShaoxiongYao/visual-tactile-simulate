import numpy as np
import torch
import time
import open3d as o3d
import pickle

import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path

import context
from vis_tac_sim.object_model import ObjectModel
from vis_tac_sim.sim_quasi_static import QuasiStaticSim
from vis_tac_sim.sim_utils import TouchSampler
from vis_tac_sim.o3d_utils import extract_surface_mesh, create_arrow_lst
from vis_tac_sim.material_model import LinearTetraModel, StVKTetraModel
from vis_tac_sim.data import FullObsSeq

view_params = {
    "front" : [ 0.98411964941290075, -0.06065860844546081, 0.16682040900588174 ],
    "lookat" : [ -2.0349855267147321, -0.50603127303317408, 5.0411290038031451 ],
    "up" : [ -0.16151026008620009, 0.083893585151037481, 0.98329868415318933 ],
    "zoom" : 0.71999999999999997
}

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(10)

    obj_name = '6polygon01'

    rest_pts = np.load(f'assets/{obj_name}_points.npy')
    elements_lst = np.load(f'assets/{obj_name}_tetra.npy')

    obj_model = ObjectModel(rest_points=rest_pts, element_lst=elements_lst,
                            material_model=LinearTetraModel())

    mu_s, lam_s = 0.0577 * 1e4, 0.0385 * 1e4

    start_time = time.time()
    material_values = obj_model.unit_material_values(dtype='list')
    material_values = [[mu_s*mu, lam_s*lam] for mu, lam in material_values]
    sim = QuasiStaticSim(obj_model, curr_points=None, material_values=material_values)
    print('setup sim time:', time.time()-start_time)

    # select_idx = select_points(obj_model.get_obj_pcd())
    # print("select index:", select_idx)
    # input()
    # select_idx = [1689, 393, 1251]

    sort_z_idx = np.argsort(obj_model.rest_points[:, 2])

    fix_idx_lst = sort_z_idx[-150:]
    allow_touch_idx = sort_z_idx[:150]

    touch_num = 1
    touch_sampler = TouchSampler(obj_model.num_pts(), pts_std=0.01)
    touch_sampler.set_fix_idx(fix_idx_lst)

    touch_sampler.set_allow_touch_idx(allow_touch_idx)
    for seq_idx in range(100):

        touch_idx, touch_u = touch_sampler.sample_touch(touch_num, touch_dir=[0.0, 0.0, 1.0])

        rest_geom = obj_model.get_element_o3d()
        rest_mesh = extract_surface_mesh(rest_geom)
        obj_pcd = obj_model.get_obj_pcd()
        colors = np.zeros_like(obj_model.rest_points)
        colors[fix_idx_lst] = [1.0, 0.0, 0.0]
        colors[allow_touch_idx] = [0.0, 1.0, 0.0]
        obj_pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([rest_geom, obj_pcd])

        num_obs = 15

        value_vec = o3d.utility.DoubleVector(np.ones(obj_model.num_pts()))

        u_lst = []
        f_lst = []

        full_obs_seq = FullObsSeq()
        for t in np.linspace(0.00001, 0.003, num_obs):
            
            start_time = time.time()
            sim.move_points(touch_idx, t*touch_u)
            u_all = sim.solve_deform()
            u_lst.append(u_all)
            f_all = sim.get_f_int(u_all)
            print("Solve deform time:", time.time() - start_time)

            sim.check_f_balance(u_all)

            curr_points = obj_model.rest_points + u_all.reshape(-1, 3)
            curr_tetra = obj_model.get_element_o3d(curr_points)

            # sim.check_f_balance(u_all)
            full_obs_seq.add_obs(u_all, f_all, fix_idx=fix_idx_lst)

            """ Visualize object deformation """

            pts1 = obj_model.rest_points[touch_idx, :]
            pts2 = pts1 + t*touch_u.reshape(-1, 3)
            arrow_lst1 = create_arrow_lst(pts1, pts2, min_len=1e-3, color=[0.7, 0.2, 0.0])

            f_touch = 1.0*f_all.reshape(-1, 3)[touch_idx, :]
            arrow_lst2 = create_arrow_lst(pts2, pts2 + f_touch, min_len=1e-3, color=[0.0, 0.7, 0.2])

            f_lst.append(f_all.reshape(-1, 3)[touch_idx[-1], :])

            curr_geom = obj_model.get_element_o3d(curr_points)
            curr_mesh = extract_surface_mesh(curr_tetra)

            geom_lst = [obj_pcd, curr_mesh] + arrow_lst2

            o3d.visualization.draw_geometries(geom_lst)

        Path(f'out_data/exp_sim_{obj_name}/seq_{seq_idx:03d}').mkdir(parents=True, exist_ok=True)
        full_obs_seq.save(f'out_data/exp_sim_{obj_name}/seq_{seq_idx:03d}')

        # obj_model.elements2edges()
        # with open(f'out_data/exp_sim_{obj_name}/seq_{seq_idx:03d}/obj_model.pkl', 'wb') as f:
        #     pickle.dump(obj_model, f)
