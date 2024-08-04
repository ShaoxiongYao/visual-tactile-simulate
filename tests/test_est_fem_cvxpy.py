import numpy as np
import torch
import time
import open3d as o3d
import pickle

import matplotlib.pyplot as plt
import pyvista as pv
import cvxpy as cp

import context
from vis_tac_sim.object_model import ObjectModel
from vis_tac_sim.sim_quasi_static import QuasiStaticSim
from vis_tac_sim.sim_utils import TouchSampler
from vis_tac_sim.o3d_utils import extract_surface_mesh, create_arrow_lst
from vis_tac_sim.material_model import LinearTetraModel
from vis_tac_sim.data import FullObsSeq

dtype = torch.float32

if __name__ == '__main__':
    # np.set_printoptions(precision=3, suppress=True)
    obj_name = 'test_tree_00'

    rest_points = np.load(f'assets/{obj_name}_points.npy')
    tetra_elements = np.load(f'assets/{obj_name}_tetra.npy')
    tetra_tensor = torch.tensor(tetra_elements, dtype=torch.int64)

    with open(f"out_data/exp_sim_{obj_name}/obj_model.pkl", 'rb') as f:
        obj_model:ObjectModel = pickle.load(f)

    obs_seq = FullObsSeq(obj_idx=1)
    obs_seq.load(f"out_data/exp_sim_{obj_name}/seq_000")

    num_pts = rest_points.shape[0]
    num_elements = tetra_elements.shape[0]

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(rest_points)

    m_init_tsr = torch.ones((num_elements, 2), dtype=dtype)
    m_prior = np.array([[1.0, 10.0]])
    material_var = cp.Variable((num_elements, 2))

    loss = 0.0
    
    for i in range(len(obs_seq)):
        print('step:', i)
        u_obs, f_obs = obs_seq[i]
        fix_idx = obs_seq.get_fix_idx(i)

        curr_points = rest_points + u_obs

        # Naive version with direct computation
        # start_time = time.time()
        # fvm_jac_mat = obj_model.force_vs_material_jacobian(curr_points, m_tsr)
        # print('fvm_jac_mat time:', time.time()-start_time)

        start_time = time.time()
        fvm_jac_mat = obj_model.force_vs_material_jacobian_batch(curr_points, m_init_tsr)
        print('fvm_jac_mat time:', time.time()-start_time)

        f_hat = fvm_jac_mat @ material_var.flatten()
        loss += cp.norm(f_hat - f_obs.flatten(), 'fro')
        
        # obj_pcd.points = o3d.utility.Vector3dVector(rest_points + u_obs)
        # touch_points = (rest_points + u_obs)
        # touch_force = 1.0 * f_obs
        # arrow_lst2 = create_arrow_lst(touch_points, touch_points + touch_force, 
        #                               min_len=1e-3, color=[0.0, 0.7, 0.2])
        # o3d.visualization.draw_geometries([obj_pcd] + arrow_lst2, point_show_normal=True)

        # u_obs_tsr = torch.tensor(u_obs, dtype=dtype)
        # f_obs_tsr = torch.tensor(f_obs, dtype=dtype)

        # elem_u_tsr = torch.zeros((num_elements, 4, 3), dtype=dtype)
        # for i in range(4):
        #     elem_u_tsr[:, i] = u_obs_tsr[tetra_elements[:, i]]

        # elem_f_tsr = linear_tetra.element_forces_batch(elem_p_tsr, elem_u_tsr, m_tsr)

        # f_hat_tsr = torch.zeros((num_pts, 3), dtype=dtype)
        # f_hat_tsr.index_add_(0, tetra_tensor.flatten(), elem_f_tsr.reshape(-1, 3))

        # loss += torch.nn.functional.mse_loss(f_hat_tsr, f_obs_tsr, reduction='sum')
    
    prior_loss = cp.sum_squares(material_var - m_prior)

    prob = cp.Problem(cp.Minimize(loss + prior_loss), constraints=[material_var >= 0.0])

    start_time = time.time()
    prob.solve()
    print('solve time:', time.time()-start_time)

    print('loss:', loss.value)
    print('material_var:', material_var.value)
    

