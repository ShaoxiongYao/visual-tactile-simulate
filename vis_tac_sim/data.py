import os
from pathlib import Path
import json
from typing import List, Tuple

import numpy as np
import open3d as o3d

class BaseObsSeq:
    def __init__(self, obj_idx:int = None) -> None:
        self.obj_idx = obj_idx

    def seq_len(self):
        pass

class FullObsSeq(BaseObsSeq):
    def __init__(self, obj_idx:int = None) -> None:
        super().__init__(obj_idx)
        self.fix_idx_lst:List[np.ndarray] = []
        self.u_all_lst:List[np.ndarray] = []
        self.f_all_lst:List[np.ndarray] = []

    def check_obs_match(self):
        assert len(self.u_all_lst) == len(self.f_all_lst)

    def __len__(self):
        return self.seq_len()

    def seq_len(self):
        return len(self.u_all_lst)
    
    def save(self, path):
        Path(path).mkdir(exist_ok=True)

        seq_info = { 'obj_idx': self.obj_idx,
                     'seq_len': self.seq_len() } 
        with open(os.path.join(path, 'seq_info.json'), 'w') as f:
            json.dump(seq_info, f, indent=2)
        
        for i in range(len(self.u_all_lst)):
            np.save(os.path.join(path, f'u_all_{i:03d}.npy'), self.u_all_lst[i])
            np.save(os.path.join(path, f'f_all_{i:03d}.npy'), self.f_all_lst[i])
            np.save(os.path.join(path, f'fix_idx_{i:03d}.npy'), self.fix_idx_lst[i])
    
    def __getitem__(self, idx):
        return self.get_obs(idx)
    
    def load(self, path):
        with open(os.path.join(path, 'seq_info.json'), 'r') as f:
            seq_info = json.load(f)
            self.obj_idx = seq_info['obj_idx']

        assert self.seq_len() == 0
        for i in range(seq_info['seq_len']):
            self.u_all_lst.append(np.load(os.path.join(path, f'u_all_{i:03d}.npy')))
            self.f_all_lst.append(np.load(os.path.join(path, f'f_all_{i:03d}.npy')))
            self.fix_idx_lst.append(np.load(os.path.join(path, f'fix_idx_{i:03d}.npy')))

    def add_obs(self, u_all, f_all, fix_idx):
        self.u_all_lst.append(u_all)
        self.f_all_lst.append(f_all)
        self.fix_idx_lst.append(fix_idx)

    def get_obs(self, step_idx, flatten=False):
        if flatten:
            return self.u_all_lst[step_idx], self.f_all_lst[step_idx]
        else:
            return (self.u_all_lst[step_idx].reshape(-1, 3), 
                    self.f_all_lst[step_idx].reshape(-1, 3))


class PartialObsSeq:
    def __init__(self, obj_idx: int = None):
        self.obj_idx = obj_idx

        self.u_obs_lst : List = []
        self.u_idx_lst : List = []
        self.f_obs_lst : List = []
        self.f_idx_lst : List = []

    def check_obs_match(self):
        assert len(self.u_obs_lst) == len(self.u_idx_lst)
        assert len(self.f_obs_lst) == len(self.f_idx_lst)
        assert len(self.u_obs_lst) == len(self.f_obs_lst)

    def seq_len(self):
        return len(self.u_obs_lst)
    
    def save(self, path):
        Path(path).mkdir(exist_ok=True)

        seq_info = { 'obj_idx': self.obj_idx,
                     'seq_len': self.seq_len() } 
        with open(os.path.join(path, 'seq_info.json'), 'w') as f:
            json.dump(seq_info, f, indent=2)

        for i in range(self.seq_len()):
            np.save(os.path.join(path, f'u_obs_{i:03d}.npy'), self.u_obs_lst[i])
            np.save(os.path.join(path, f'u_idx_{i:03d}.npy'), self.u_idx_lst[i])
            np.save(os.path.join(path, f'f_obs_{i:03d}.npy'), self.f_obs_lst[i])
            np.save(os.path.join(path, f'f_idx_{i:03d}.npy'), self.f_idx_lst[i])

    def load(self, path):

        with open(os.path.join(path, 'seq_info.json'), 'r') as f:
            seq_info = json.load(f)
            self.obj_idx = seq_info['obj_idx']

        assert self.seq_len() == 0
        for i in range(seq_info['seq_len']):
            self.u_obs_lst.append(np.load(os.path.join(path, f'u_obs_{i:03d}.npy')))
            self.u_idx_lst.append(np.load(os.path.join(path, f'u_idx_{i:03d}.npy')))
            self.f_obs_lst.append(np.load(os.path.join(path, f'f_obs_{i:03d}.npy')))
            self.f_idx_lst.append(np.load(os.path.join(path, f'f_idx_{i:03d}.npy')))
    
    def add_u_obs(self, u_obs: np.ndarray, u_idx: np.ndarray):
        """
        Add partial visual observation to data sequence
        u_obs shape: num_vis_pts x 3
        u_idx shape: num_vis_pts
        """
        self.u_obs_lst.append(u_obs)
        self.u_idx_lst.append(u_idx)
    
    def add_f_obs(self, f_obs: np.ndarray, f_idx: np.ndarray):
        """
        Add partial tactile observation to data sequence
        f_obs shape: num_tac_pts x 3
        f_idx shape: num_tac_pts
        """
        self.f_obs_lst.append(f_obs)
        self.f_idx_lst.append(f_idx)

    def get_vis_obs(self, step_idx):
        return self.u_obs_lst[step_idx], self.u_idx_lst[step_idx]

    def get_tac_obs(self, step_idx):
        return self.f_obs_lst[step_idx], self.f_idx_lst[step_idx]

class RealObsSeq(BaseObsSeq):
    def __init__(self) -> None:

        self.pcd_lst : List[o3d.geometry.PointCloud] = []
        self.tau_lst : List[np.ndarray] = []

    def add_vis_obs(self, pcd_obs: o3d.geometry.PointCloud):
        """
        Add visual observation to data sequence
        pcd_obs shape: num_vis_pts x 3
        pts_idx shape: num_vis_pts
        """
        # NOTE: points correspondences may not be available
        self.pcd_lst.append(pcd_obs)

    def add_tac_obs(self, tau_obs: np.ndarray):
        """
        Add tactile observation to data sequence
        tac_obs shape: num_touch_pts x 3
        pts_idx shape: num_touch_pts
        """
        self.tau_lst.append(tau_obs)


# class MultiObjectDataset:
#     def __init__(self):
#         self.obj_path_lst = []
    
#     def load_obj_seq(self, obj_idx) -> VisTacObsSeq:
#         self.obj_path = self.obj_path_lst[obj_idx]
#         return VisTacObsSeq()
        

