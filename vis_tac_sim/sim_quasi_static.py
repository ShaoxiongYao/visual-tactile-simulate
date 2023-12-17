from functools import partial

import torch
from torch.autograd.functional import jacobian

import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from material_model import BaseMaterialModel
from object_model import ObjectModel
from sim_utils import solve_deform_sparse

class QuasiStaticSim:
    def __init__(self, object_model: ObjectModel, curr_points=None, material_values=None):
        self.object_model : ObjectModel = object_model

        if curr_points is None:
            self.curr_points = object_model.rest_points.copy()
        else:
            self.curr_points = curr_points
        if material_values is None:
            self.material_values = object_model.unit_material_values()
        else:
            self.material_values = material_values
        
        self.K_mat = self.object_model.object_stiffness(self.curr_points, self.material_values)
        self.f_ext = np.zeros_like(self.curr_points)

        self.move_idx : np.ndarray = np.zeros(0, dtype=int)
        self.move_u : np.ndarray = np.zeros((0, 3), dtype=float)
    
    def apply_forces(self, idx, f):
        self.f_ext[idx, :] += f
    
    def clear_forces(self):
        self.f_ext = np.zeros_like(self.curr_points)
    
    def move_points(self, idx, u):
        self.move_idx = idx
        self.move_u = u
    
    def get_free_mask(self):
        mask = np.ones(self.object_model.num_pts(), dtype=bool)
        mask[self.move_idx] = False
        return mask
    
    def get_f_int(self, u_all: np.ndarray, flat=True) -> np.ndarray:
        u_all = u_all.reshape(-1)
        f_all = self.K_mat @ u_all
        if flat:
            return f_all
        else:
            return f_all.reshape(-1, 3)
    
    def free_points(self, idx):
        # self.move_idx = np.setdiff1d(self.move_idx, idx)
        # mat_idx = idx_pts2mat(idx)
        # self.fix_idx = np.setdiff1d(self.fix_idx, mat_idx)
        raise NotImplementedError
    
    def save_curr_state(self):
        raise NotImplementedError
    
    def check_f_balance(self, u_all):
        f_all = self.get_f_int(u_all)
        f_all = f_all.reshape(-1, 3)

        free_mask = self.get_free_mask()
        
        balance = np.allclose(f_all[free_mask, :], 0.0)
        if not balance:
            print("residual forces:", f_all[free_mask, :])

    def solve_deform(self, num_iters=None):
        move_u_flat = self.move_u.reshape(-1)
        f_ext_flat = self.f_ext.reshape(-1)
        u_all = solve_deform_sparse(self.move_idx, move_u_flat, self.K_mat, 
                                    f_ext_flat, idx_mode='pts', perf=False)
        # else:
        #     for _ in range(num_iters):
        #         u_all = solve_deform_sparse(self.move_idx, move_u_flat, self.K_mat, 
        #                                     f_ext_flat, idx_mode='pts', perf=False)
        #         self.curr_points += u_all.reshape(-1, 3)
        #         self.K_mat = self.object_model.object_stiffness(self.curr_points, self.material_values)
        return u_all
