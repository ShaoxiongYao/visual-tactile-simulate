
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from o3d_utils import create_motion_lines

class GlobalDeformModel:
    def __init__(self) -> None:
        pass  

    def get_u_g_dim(self):
        """Return n_g"""
        raise NotImplementedError
    
    def get_control_pts(self):
        raise NotImplementedError

    def get_pts_beta(self, pts) -> np.ndarray:
        """Return beta_tsr, shape: num_pts x 3 x n_g"""
        raise NotImplementedError

    def get_o3d_obj(self, u_g):
        raise NotImplementedError

class SingleCage(GlobalDeformModel):

    def __init__(self, b_min, b_max) -> None:
        super().__init__()
        self.b_min = b_min
        self.b_max = b_max
    
    def get_u_g_dim(self):
        """Return n_g"""
        return 3
    
    def get_control_pts(self, flat=False):
        
        bound_lst = []
        for axis_dim in range(3):
            bound_lst.append([self.b_min[axis_dim], self.b_max[axis_dim]])

        pts_tsr = np.stack(np.meshgrid(*bound_lst, indexing='ij'), axis=-1)

        if not flat:
            return pts_tsr
        else:
            return pts_tsr.reshape(-1, 3)
    
    def get_pts_beta(self, pts):

        # deform w.r.t. z-axis
        z_min, z_max = self.b_min[2], self.b_max[2]
        beta_tsr = (pts[:, 2] - z_min)/(z_max - z_min)
        beta_tsr[beta_tsr < 0] = 0.0
        beta_tsr[beta_tsr > 1.0] = 1.0

        return np.kron(beta_tsr[:, None, None], np.eye(3))

    def get_o3d_obj(self, u_g, color=None):

        pts_tsr = self.get_control_pts(flat=False)
        pts_tsr[:, :, 1, :] += u_g
        pts = pts_tsr.reshape(-1, 3)

        lines = [ [0, 1], [0, 2], [1, 3], [2, 3], 
                  [4, 5], [4, 6], [5, 7], [6, 7],
                  [0, 4], [1, 5], [2, 6], [3, 7] ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        if color is not None:
            line_set.colors = color
        return line_set

class NodeGraph(GlobalDeformModel):

    def __init__(self, node_pts, num_nns=5, rbf_sig=1.0) -> None:
        super().__init__()
        self.node_pts = node_pts
        self.num_nodes = node_pts.shape[0]

        self.num_nns = num_nns
        self.rbf_sig = rbf_sig
        self.node_knn = NearestNeighbors(n_neighbors=num_nns).fit(node_pts)
    
    def get_u_g_dim(self):
        """Return n_g"""
        return np.prod(self.node_pts.shape)
    
    def get_control_pts(self, flat=False):
        if flat:
            return self.node_pts.reshape(-1, 3)
        else:
            return self.node_pts
    
    def get_pts_beta(self, pts):
        num_pts = pts.shape[0]

        # pts shape: N x 3
        eud_ary, idx_ary = self.node_knn.kneighbors(pts)

        rbf_weights = np.exp(-eud_ary/self.rbf_sig)
        rbf_weights /= rbf_weights.sum(axis=1, keepdims=True)

        # start with 
        beta_tsr = np.zeros((num_pts, self.num_nodes))

        for nn_idx in range(self.num_nns):
            idx_ary_col = idx_ary[:, nn_idx]
            beta_tsr[np.arange(num_pts), idx_ary_col] = rbf_weights[:, nn_idx]

        # return beta_tsr shape: N x 3 x (3*num_nodes)
        return np.kron(beta_tsr[:, None, :], np.eye(3))

    def compute_edges(self, connect_pattern='knn', self_loop=False, reversable=True):
        """ Compute edges, return np.array of shape (num_edges, 2) """
        if connect_pattern == 'knn':
            idx_ary = self.node_knn.kneighbors(self.node_pts)[1]
        else:
            raise NotImplementedError

        if not self_loop:
            idx_ary = idx_ary[:, 1:]
            edge_per_node = self.num_nns - 1
        else:
            edge_per_node = self.num_nns
        src_idx = np.repeat(np.arange(self.num_nodes), edge_per_node)
        dst_idx = idx_ary.reshape(-1)
                
        edges = np.stack([src_idx, dst_idx], axis=1)
        if not reversable:
            return edges
        else:
            sort_edges = np.sort(edges, axis=1)
            return np.unique(sort_edges, axis=0)

    def get_pts2nodes_graph(self, pts):
        num_pts = pts.shape[0]

        # pts shape: N x 3
        _, idx_ary = self.node_knn.kneighbors(pts)

        source_pts = np.tile(pts[:, None, :], (1, self.num_nns, 1))
        match_pts = np.zeros((num_pts, self.num_nns, 3))
        for nn_idx in range(self.num_nns):
            match_pts[:, nn_idx, :] = self.node_pts[idx_ary[:, nn_idx], :]
        
        source_pts = source_pts.reshape(-1, 3)
        match_pts = match_pts.reshape(-1, 3)

        match_geom = create_motion_lines(source_pts, match_pts, return_pcd=True)
        return match_geom

    def get_o3d_obj(self, u_g, color=None):

        u_g_mat = u_g.reshape(-1, 3)
        deform_pts = self.node_pts + u_g_mat

        node_pcd = o3d.geometry.PointCloud()
        node_pcd.points = o3d.utility.Vector3dVector(deform_pts)
        if color is not None:
            node_pcd.paint_uniform_color(color)
        return node_pcd


def get_knn_idx_lst(features, num_nns=5):
    knn_mat = kneighbors_graph(features, num_nns, 
                               mode='connectivity', 
                               include_self=False)
    knn_mat = knn_mat.tocoo()
    
    alledges = [ (r, l) if r <= l else (l, r)
                 for r, l in zip(knn_mat.row, knn_mat.col) ]
        
    edges = set(alledges)
    return edges


def get_knn_correspondence(source_vertices, target_vertices, 
                           target_knn=None, match_dist_thres=1.0):
    if target_knn is None:
        target_knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_vertices)

    # knn search: L-2 distance, matched indices
    l2d_ary, idx_ary = target_knn.kneighbors(source_vertices)
    l2d_ary = l2d_ary.squeeze()
    idx_ary = idx_ary.squeeze()

    # rigtnow setting threshold manualy, but if we have and landmark info we could set here
    source_match_idx = np.where(l2d_ary<match_dist_thres)[0]
    target_match_idx = idx_ary[l2d_ary<match_dist_thres]

    return source_match_idx, target_match_idx
