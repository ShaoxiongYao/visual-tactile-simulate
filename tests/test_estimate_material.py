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


if __name__ == '__main__':
    obj_name = '6polygon01'

    # rest_points = np.load(f'assets/{obj_name}_points.npy')
    # tetra_elements = np.load(f'assets/{obj_name}_tetra.npy')
    rest_points = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    tetra_elements = np.array([[0, 1, 2, 3]])

    rest_points = torch.tensor(rest_points, dtype=torch.float32)
    deform = torch.rand(rest_points.shape)

    material_model = LinearTetraModel()

    num_pts = rest_points.shape[0]
    num_elements = tetra_elements.shape[0]

    # compute Jacobian from single element
    p = torch.tensor(rest_points, dtype=torch.float32)
    p = p.flatten()
    u = torch.zeros_like(p)
    u[0] = 1.0
    print('u:', u)
    m = torch.ones((2,), dtype=torch.float32)
    jac = torch.func.jacrev(material_model.element_forces, argnums=2, has_aux=False, 
                            chunk_size=None, _preallocate_and_copy=False)
    jac_mat = jac(p, u, m)
    print('jac_mat:', jac_mat)
    print('jac_mat @ m', jac_mat @ m)
    print('out:', material_model.element_forces(p, u, m))
    input()

    p = torch.zeros(num_elements, 4, 3)
    u = torch.zeros(num_elements, 4, 3)
    for i in range(num_elements):
        p[i] = rest_points[tetra_elements[i]]
        u[i] = deform[tetra_elements[i]]
    
    batch_jac_mat = torch.zeros(num_elements, 12, 2)
    for i in range(num_elements):
        p_i = p[i].flatten()
        u_i = u[i].flatten()
        jac_mat = jac(p_i, u_i, m)
        batch_jac_mat[i] = jac_mat
    input()

    m_gt = torch.ones((num_elements, 2), dtype=torch.float32)
    start_time = time.time()
    f_obs = material_model.element_forces_batch(p, u, m_gt)
    print('time:', time.time()-start_time)

    # # manual iteration
    # for idx, tetra in enumerate(tetra_elements):
    #     p = rest_points[tetra]
    #     u = deform[tetra]
    #     m = torch.ones((2), dtype=torch.float32)
    #     f_element = material_model.element_forces(p.flatten(), u.flatten(), m)
        
    #     assert torch.allclose(f[idx], f_element)
    #     # print('f:', f[idx])
    #     # print('f_element:', f_element)
    #     # input()

    m_hat = torch.zeros_like(m_gt)
    m_hat.requires_grad = True
    optimizer = torch.optim.Adam([m_hat], lr=0.1)

    loss_lst = []
    for _ in range(1000):

        f_hat = material_model.element_forces_batch(p, u, m_hat)

        loss = torch.sum((f_hat - f_obs) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('loss:', loss.data)

        loss_lst.append(loss.item())

    plt.plot(loss_lst)
    plt.show()

    # p = rest_points.flatten()
    # u = deform.flatten()
    # m_gt = torch.tensor([100.0, 0.5])

    # f_obs = material_model.element_forces(p, u, m_gt)

    # m_hat = torch.zeros_like(m_gt)
    # m_hat.requires_grad = True
    # optimizer = torch.optim.Adam([m_hat], lr=0.1)

    # loss_lst = []
    # for _ in range(10000):

    #     f_hat = material_model.element_forces(p, u, m_hat)

    #     loss = torch.sum((f_hat - f_obs) ** 2)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     print('m_hat:', m_hat.data)

    #     loss_lst.append(loss.item())
    
    # plt.plot(loss_lst)
    # plt.show()
