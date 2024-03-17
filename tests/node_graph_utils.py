import numpy as np
import time
import open3d as o3d
import matplotlib.pyplot as plt
import pypose as pp
import scipy.sparse as scisp
import sklearn.neighbors as skn

import torch
from torch import nn

class DeformState(nn.Module):

    def __init__(self, num_pts, corotate=False) -> None:
        super().__init__()
        self.num_pts = num_pts
        self.delta_pts_tsr = nn.Parameter(torch.zeros(self.num_pts, 3), 
                                          requires_grad=True)
        self.corotate = corotate
        if corotate:
            self.rot_tsr = nn.Parameter(pp.randn_so3(self.num_pts, sigma=1e-5, 
                                                     requires_grad=True))
    
    def forward(self, rest_pts_tsr: torch.Tensor, 
                handle_idx:np.ndarray, handle_pts_tsr: torch.Tensor):
        curr_pts_tsr = rest_pts_tsr + self.delta_pts_tsr
        curr_pts_tsr[handle_idx] = handle_pts_tsr
        return curr_pts_tsr
    
    def compute_edge_rest(self, rest_pts_tsr: torch.Tensor, edges_ary: np.ndarray):
        edges_rest = rest_pts_tsr[edges_ary[:, 0], :] - rest_pts_tsr[edges_ary[:, 1], :]
        if self.corotate:
            edges_rest = pp.Exp(self.rot_tsr[edges_ary[:, 1], :]) @ edges_rest
        return edges_rest

    def compute_edge_diff(self, rest_pts_tsr: torch.Tensor, 
                          handle_idx:np.ndarray, handle_pts_tsr: torch.Tensor, 
                          edges_ary: np.ndarray):
        curr_pts_tsr = self.forward(rest_pts_tsr, handle_idx, handle_pts_tsr)
        edges_diff = curr_pts_tsr[edges_ary[:, 0], :] - curr_pts_tsr[edges_ary[:, 1], :]
        return edges_diff

class NodeGraph:
    def __init__(self, rest_pts:np.ndarray, edges:np.ndarray, corotate=False, 
                 num_nns:int=10, dtype=torch.double, device='cpu') -> None:

        self.tsr_params = {'dtype': dtype, 'device': device}

        self.rest_pts_tsr = torch.tensor(rest_pts, **self.tsr_params)
        self.edges_tsr = torch.tensor(edges, device=device, dtype=torch.long)

        self.num_pts = self.rest_pts_tsr.shape[0]
        self.num_edges = self.edges_tsr.shape[0]

        self.corotate = corotate
        self.deform_state = DeformState(self.num_pts, corotate=corotate)
        self.deform_state.to(**self.tsr_params)

        self.edges_weight_tsr = torch.ones(self.num_edges, **self.tsr_params)

        self.node_knn = skn.NearestNeighbors(n_neighbors=10)
        self.num_nns = num_nns
        self.node_knn.fit(rest_pts)
    
    def get_curr_pts(self, handle_idx, handle_pts_tsr: torch.Tensor) -> torch.Tensor:
        return self.deform_state.forward(self.rest_pts_tsr, handle_idx, handle_pts_tsr)
    
    def get_delta_pts(self) -> torch.Tensor:
        return self.deform_state.delta_pts_tsr

    def energy(self, handle_idx, handle_pts_tsr: torch.Tensor) -> torch.Tensor:
        edges_diff = self.deform_state.compute_edge_diff(self.rest_pts_tsr, 
                                                         handle_idx, handle_pts_tsr, 
                                                         self.edges_tsr)
        edges_rest = self.deform_state.compute_edge_rest(self.rest_pts_tsr, self.edges_tsr)

        edges_delta = (edges_diff - edges_rest).pow(2).sum(dim=1)
        energy = (edges_delta * self.edges_weight_tsr).sum()
        return energy
    
    def get_pts_beta(self, pts:np.ndarray, rbf_sig=0.5, rbf_w_max=0.2):
        num_pts = pts.shape[0]

        # pts shape: N x 3
        eud_ary, idx_ary = self.node_knn.kneighbors(pts)

        rbf_weights:np.ndarray = np.exp(-eud_ary/rbf_sig)
        rbf_weights[rbf_weights < rbf_w_max] = 0.0

        rbf_weights /= rbf_weights.sum(axis=1, keepdims=True) + 1e-5

        row_idx = np.repeat(np.arange(num_pts), self.num_nns)
        col_idx = idx_ary.reshape(-1)
        data = rbf_weights.reshape(-1)

        beta_tsr = scisp.csr_matrix((data, (row_idx, col_idx)), 
                                    shape=(num_pts, self.num_pts))
        return beta_tsr


    def get_line_set(self):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(self.rest_pts_tsr.cpu().numpy())

        edges_lst = self.edges_tsr.cpu().numpy().tolist()

        line_set.lines = o3d.utility.Vector2iVector(edges_lst)
        line_set.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]]*len(edges_lst)))
        return line_set

    def get_pcd(self, handle_idx=None, handle_pts_tsr:torch.Tensor=None):
        if handle_idx is None:
            curr_pts_tsr = self.rest_pts_tsr.clone()
        else:
            curr_pts_tsr = self.get_curr_pts(handle_idx, handle_pts_tsr)
        
        curr_pts_ary = curr_pts_tsr.detach().cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(curr_pts_ary)
        
        pcd_colors = np.zeros_like(curr_pts_ary)
        if handle_idx is not None:
            pcd_colors[handle_idx] = [1.0, 0.0, 0.0]
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

        return pcd

    def get_rot_frames(self, skip_rate = 100):
        coord_frame_lst = []
        rot_tsr = self.deform_state.rot_tsr
        curr_pts_tsr = self.rest_pts_tsr + self.deform_state.delta_pts_tsr

        for pt_i in range(0, self.num_pts, skip_rate):
            ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            pt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            rot_mat:torch.Tensor = pp.Exp(rot_tsr[pt_i, :]) @ torch.eye(3)

            pt_frame.rotate(rot_mat.detach().numpy())
            pt_frame.translate(curr_pts_tsr[pt_i, :].detach().numpy())
            pt_frame.paint_uniform_color([0.2, 0.9, 0.7])

            ref_frame.translate(curr_pts_tsr[pt_i, :].detach().numpy())
            ref_frame.paint_uniform_color([0.7, 0.2, 0.9])
            coord_frame_lst.append(pt_frame)
            coord_frame_lst.append(ref_frame)
        return coord_frame_lst

if __name__ == '__main__':
    rest_pts = torch.zeros(5, 3)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [1, 0], [2, 1], [3, 2], [4, 3]])

    ng = NodeGraph(rest_pts, edges, corotate=True)

    handle_idx = torch.tensor([0, 3], dtype=torch.long)
    handle_pts = torch.tensor([[0.0, 0.3, -0.3], [0.0, 0.3, -0.3]], dtype=torch.double)

    optimizer = torch.optim.Adam(ng.deform_state.parameters(), lr=0.1)

    energy_lst = []
    for _ in range(100):
        energy = ng.energy(handle_idx, handle_pts)

        optimizer.zero_grad()
        energy.backward()
        optimizer.step()

        energy_lst.append(energy.item())
        print('loss:', energy.item())
    
    plt.plot(energy_lst)
    plt.show()
