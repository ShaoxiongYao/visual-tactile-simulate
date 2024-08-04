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

# view_params = {
#     "front" : [ 0.98411964941290075, -0.06065860844546081, 0.16682040900588174 ],
#     "lookat" : [ -2.0349855267147321, -0.50603127303317408, 5.0411290038031451 ],
#     "up" : [ -0.16151026008620009, 0.083893585151037481, 0.98329868415318933 ],
#     "zoom" : 0.71999999999999997
# }

view_params = {	
    "front" : [ 0.95200153570221846, -0.23694605848506156, 0.19377729843563135 ],
    "lookat" : [ -1.391148932201788, 0.32579325574309886, 2.2778399976269745 ],
    "up" : [ -0.1860283253324265, 0.054858682851952827, 0.98101171608180182 ],
    "zoom" : 0.78000000000000003
}

num_seq = 100
num_obs = 15
touch_num = 10
exp_id = 'exp_multi_touch'

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(1)

    obj_name = 'test_small_tree_00'

    rest_pts:np.ndarray = np.load(f'assets/{obj_name}_points.npy')
    elements_lst:np.ndarray = np.load(f'assets/{obj_name}_tetra.npy')

    pv_tetra_mesh = pv.read(f'out_data/plant_assets/{obj_name}_.msh')

    obj_model = ObjectModel(rest_points=rest_pts, element_lst=elements_lst,
                            material_model=LinearTetraModel())

    elements_features = torch.zeros(obj_model.num_ele(), 3)
    for i in range(obj_model.num_ele()):
        elements_features[i] = torch.tensor(rest_pts[elements_lst[i]].mean(axis=0))

    # Uniform material assignment
    # mu_s, lam_s = 1.0, 10.0
    # material_values = obj_model.unit_material_values(dtype='list')
    # material_values = [[mu_s*mu, lam_s*lam] for mu, lam in material_values]
    
    gt_linear_weight = torch.tensor([[0.0, 0.0], [0.0, 0.4], [-1.0, -0.1]], dtype=torch.float32)
    material_values = elements_features @ gt_linear_weight + torch.tensor([8.0, 7.0])
    print('material_values shape:', material_values.shape)

    # Assign colors
    pv_tetra_mesh["mu"] = material_values[:, 0].numpy()
    pv_tetra_mesh["lam"] = material_values[:, 1].numpy()

    # Plot with the specified colors
    plotter = pv.Plotter()
    plotter.add_mesh(pv_tetra_mesh, scalars="mu", show_edges=False, cmap="YlOrBr")
    plotter.show()
    plotter = pv.Plotter()
    plotter.add_mesh(pv_tetra_mesh, scalars="lam", show_edges=False, cmap="YlOrBr")
    plotter.show()

    start_time = time.time()
    
    sim = QuasiStaticSim(obj_model, curr_points=None, material_values=material_values)
    print('setup sim time:', time.time()-start_time)

    arg_sorted_z = np.argsort(obj_model.rest_points[:, 2])
    fix_num = int(0.10*obj_model.num_pts())
    allow_touch_num = int(0.85*obj_model.num_pts())

    fix_idx = arg_sorted_z[:fix_num]
    allow_touch_idx = arg_sorted_z[-allow_touch_num:]

    touch_sampler = TouchSampler(obj_model.num_pts(), pts_std=0.01)
    touch_sampler.set_fix_idx(fix_idx)

    touch_sampler.set_allow_touch_idx(allow_touch_idx)

    rest_geom = obj_model.get_element_o3d()
    rest_mesh = extract_surface_mesh(rest_geom)

    obj_pcd = obj_model.get_obj_pcd()
    colors = np.zeros_like(obj_model.rest_points)
    colors[fix_idx] = [1.0, 0.0, 0.0]
    colors[allow_touch_idx] = [0.0, 1.0, 0.0]
    obj_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([obj_pcd])

    Path(f'out_data/{exp_id}_{obj_name}').mkdir(parents=True, exist_ok=True)
    with open(f'out_data/{exp_id}_{obj_name}/obj_model.pkl', 'wb') as f:
        pickle.dump(obj_model, f)

    for seq_idx in range(num_seq):
        touch_idx, touch_u = touch_sampler.sample_touch(touch_num, touch_dir=[0.0, 0.0, -1.0])


        u_lst = []
        f_lst = []

        full_obs_seq = FullObsSeq()
        for t in np.linspace(0.01, 0.15, num_obs):
            
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
            full_obs_seq.add_obs(u_all, f_all, fix_idx=touch_idx)

            """ Visualize object deformation """

            pts1 = obj_model.rest_points[touch_idx, :]
            pts2 = pts1 + t*touch_u.reshape(-1, 3)
            arrow_lst1 = create_arrow_lst(pts1, pts2, color=[0.7, 0.2, 0.0])

            f_touch = 1.0*f_all.reshape(-1, 3)[touch_idx, :]
            arrow_lst2 = create_arrow_lst(pts2, pts2 + f_touch, color=[0.0, 0.7, 0.2])

            f_lst.append(f_all.reshape(-1, 3)[touch_idx[-1], :])

            curr_geom = obj_model.get_element_o3d(curr_points)
            curr_mesh = extract_surface_mesh(curr_tetra)

            geom_lst = [rest_geom, curr_mesh] + arrow_lst1

            o3d.visualization.draw_geometries(geom_lst, **view_params)

        Path(f'out_data/{exp_id}_{obj_name}/seq_{seq_idx:03d}').mkdir(parents=True, exist_ok=True)
        full_obs_seq.save(f'out_data/{exp_id}_{obj_name}/seq_{seq_idx:03d}')
