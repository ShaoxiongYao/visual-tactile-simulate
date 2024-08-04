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

dtype = torch.float32
exp_id = 'exp_linear_gauss'

if __name__ == '__main__':
    # np.set_printoptions(precision=3, suppress=True)
    obj_name = 'test_small_tree_00'

    rest_points:np.ndarray = np.load(f'assets/{obj_name}_points.npy')
    tetra_elements:np.ndarray = np.load(f'assets/{obj_name}_tetra.npy')
    tetra_tensor = torch.tensor(tetra_elements, dtype=torch.int64)

    with open(f"out_data/{exp_id}_{obj_name}/obj_model.pkl", 'rb') as f:
        obj_model:ObjectModel = pickle.load(f)

    num_pts = rest_points.shape[0]
    num_elements = tetra_elements.shape[0]

    print('num_pts:', num_pts)
    print('num_elements:', num_elements)
    input()

    elements_features = torch.zeros(obj_model.num_ele(), 3)
    for i in range(obj_model.num_ele()):
        elements_features[i] = torch.tensor(rest_points[tetra_elements[i]].mean(axis=0))
    
    # linear_weight = 0.1*torch.randn((3, 2), dtype=dtype)
    # bias = 0.1*torch.randn((2,), dtype=dtype)

    linear_weight = torch.zeros((3, 2), dtype=dtype)
    # linear_weight = torch.tensor([[0.0, 0.0], [0.0, 0.4], [-1.0, -0.1]], dtype=dtype)
    bias = torch.tensor([8.0, 7.0], dtype=dtype)

    linear_weight.requires_grad = True
    bias.requires_grad = True

    optimizer = torch.optim.Adam([linear_weight, bias], lr=0.01)

    # mu_s, lam_s = 1.0, 10.0
    # m_tsr = torch.ones((num_elements, 2), dtype=dtype)
    # m_tsr[:, 0] *= mu_s
    # m_tsr[:, 1] *= lam_s

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(rest_points)

    obs_seq = FullObsSeq(obj_idx=0)

    loss_lst = []
    for epoch in range(1000):
        print('epoch:', epoch)

        seq_idx = np.random.randint(0, 100)
        obs_seq.clear()
        obs_seq.load(f"out_data/{exp_id}_{obj_name}/seq_{seq_idx:03d}")

        epoch_loss = 0.0
        for i in range(len(obs_seq)):
            # print('i:', i)
            u_obs, f_obs = obs_seq[i]
            fix_idx = obs_seq.get_fix_idx(i)

            curr_points = rest_points + u_obs

            # start_time = time.time()
            # fvm_jac_mat = obj_model.force_vs_material_jacobian(curr_points, m_tsr)
            # print('fvm_jac_mat time:', time.time()-start_time)

            m_tsr = elements_features @ linear_weight + bias
            # print('m_tsr:', m_tsr.shape)

            start_time = time.time()
            fvm_jac_mat = obj_model.force_vs_material_jacobian_batch(curr_points, m_tsr.detach())
            # print('fvm_jac_mat time:', time.time()-start_time)

            fvm_jac_mat = torch.tensor(fvm_jac_mat.toarray(), dtype=dtype)
            # print('fvm_jac_mat:', fvm_jac_mat.shape)
            
            # plt.spy(fvm_jac_mat.numpy())
            # plt.show()

            # start_time = time.time()
            # ret = torch.linalg.lstsq(fvm_jac_mat, torch.tensor(f_obs.flatten(), dtype=dtype))
            # print('lstsq time:', time.time()-start_time)

            # print('solution:', ret.solution[:10])
            # print('residual:', ret.residuals)
            # input()

            # expected force
            f_hat_tsr = fvm_jac_mat @ m_tsr.flatten()
            f_hat_tsr = f_hat_tsr.reshape(-1, 3)
            f_obs_tsr = torch.tensor(f_obs, dtype=dtype)

            # print('f_hat_tsr:', f_hat_tsr[:10])
            # print('f_obs_tsr:', f_obs_tsr[:10])
            # print('max force error:', torch.max(torch.abs(f_hat_tsr - f_obs_tsr)))
            # print('all close:', torch.allclose(f_hat_tsr, f_obs_tsr, atol=1e-5))

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

            start_time = time.time()
            loss = torch.nn.functional.mse_loss(f_hat_tsr, f_obs_tsr, reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # print('backward time:', time.time()-start_time)

            print('linear_weight:', linear_weight.data)
            print('bias:', bias.data)

        print('epoch loss:', epoch_loss)
        loss_lst.append(epoch_loss)
    
    plt.plot(loss_lst)
    plt.show()


    # plt.hist(m_tsr[:, 0].detach().numpy())
    # plt.show()

    # plt.hist(m_tsr[:, 1].detach().numpy())
    # plt.show()
