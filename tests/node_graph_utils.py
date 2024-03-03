import numpy as np
import time
import open3d as o3d
import matplotlib.pyplot as plt
import pypose as pp

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
    def __init__(self, rest_pts:np.ndarray, edges:np.ndarray, 
                 corotate=False, dtype=torch.double, device='cpu') -> None:

        self.tsr_params = {'dtype': dtype, 'device': device}

        self.rest_pts_tsr = torch.tensor(rest_pts, **self.tsr_params)
        self.edges_tsr = torch.tensor(edges, device=device, dtype=torch.long)

        self.num_pts = self.rest_pts_tsr.shape[0]
        self.num_edges = self.edges_tsr.shape[0]

        self.corotate = corotate
        self.deform_state = DeformState(self.num_pts, corotate=corotate)
        self.deform_state.to(**self.tsr_params)

        self.edges_weight_tsr = torch.ones(self.num_edges, **self.tsr_params)

    def energy(self, handle_idx, handle_pts_tsr: torch.Tensor) -> torch.Tensor:
        edges_diff = self.deform_state.compute_edge_diff(self.rest_pts_tsr, 
                                                         handle_idx, handle_pts_tsr, 
                                                         self.edges_tsr)
        edges_rest = self.deform_state.compute_edge_rest(self.rest_pts_tsr, self.edges_tsr)

        edges_delta = (edges_diff - edges_rest).pow(2).sum(dim=1)
        energy = (edges_delta * self.edges_weight_tsr).sum()
        return energy

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
