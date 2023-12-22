from typing import List, Union
import time
import torch
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse import linalg as splinalg

def solve_deform_sparse(fix_idx, fix_u, sp_K_mat, f_ext=None, idx_mode='pts', perf=False):
    u_dim = sp_K_mat.shape[0]

    fix_mask = np.zeros(u_dim, dtype=bool)
    if idx_mode == 'pts':
        fix_idx = np.stack([3*fix_idx, 3*fix_idx+1, 3*fix_idx+2]).T.flatten()
    else: 
        assert idx_mode == 'raw'
    
    fix_u = fix_u[np.argsort(fix_idx)] # sort index corresponds to touch_mask
    fix_idx = np.sort(fix_idx)
    fix_mask[fix_idx] = True

    start_time = time.time()
    K_ff = sp_K_mat[np.ix_(~fix_mask, ~fix_mask)]
    K_ft = sp_K_mat[np.ix_(~fix_mask,  fix_mask)]
    if perf:
        print("Index submatrix time:", time.time() - start_time)

    start_time = time.time()
    f_res = -K_ft @ fix_u
    if f_ext is not None:
        f_res += f_ext[~fix_mask]
    try:
        u_f = splinalg.spsolve(K_ff, f_res)
    except:
        raise ValueError("Singular matrix")
    if perf:
        print("Solve time:", time.time() - start_time)

    u_all = np.zeros(u_dim)
    u_all[fix_mask] = fix_u
    u_all[~fix_mask] = u_f

    return u_all


class TouchSampler:
    def __init__(self, num_pts, pts_std=1.0) -> None:
        self.num_pts = num_pts

        self.pts_std = pts_std
        self.fix_idx = np.zeros(0, dtype=int)
    
    def set_fix_idx(self, fix_idx: Union[np.ndarray, List[int]]):
        if isinstance(fix_idx, list):
            fix_idx = np.array(fix_idx)
        self.fix_idx = fix_idx
    
    def set_allow_touch_idx(self, allow_touch_idx):
        self.allow_touch_idx = allow_touch_idx
        
    def sample_touch(self, touch_num, touch_dir=None, flat_u=True):

        touch_idx = np.random.choice(self.allow_touch_idx, touch_num, replace=False)
        if touch_dir is None:
            touch_u = self.pts_std*np.random.randn(touch_num, 3)
        else:
            touch_u = np.tile(touch_dir, (touch_num, 1))

        touch_idx = np.concatenate([self.fix_idx, touch_idx])
        zero_u = np.zeros((self.fix_idx.shape[0], 3))
        touch_u = np.concatenate([zero_u, touch_u], axis=0)
        if not flat_u:
            return touch_idx, touch_u
        else:
            return touch_idx, touch_u.reshape(-1)
