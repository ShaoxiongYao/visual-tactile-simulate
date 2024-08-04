import os
from typing import List, Dict, Union
import open3d as o3d
import torch
import numpy as np
from torch.func import vmap
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.spatial import Delaunay
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from itertools import combinations

from .geometry_model import GlobalDeformModel, NodeGraph
from .material_model import BaseMaterialModel, LinearSpringModel

ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets")

def idx_pts2mat(pts_idx):
    mat_idx = torch.tensor([[3*i, 3*i+1, 3*i+2] for i in pts_idx]).reshape(-1)
    return torch.cartesian_prod(mat_idx, mat_idx)

def create_object_model(obj_name, vis_obj=False):
    obj_model = ObjectModel()
    """ Setup object model """
    if obj_name == "toy":
        obj_model.rest_points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        obj_model.element_lst = np.array([[0, 1], [1, 2], [2, 3]])
    
    elif obj_name == "box":
        obj_model.rest_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 
                                          [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        obj_model.element_lst = np.array([[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], 
                                          [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]])

    elif obj_name == "orange_tree":
        model_pcd = o3d.io.read_point_cloud(os.path.join(ASSET_PATH, "orange_tree.ply"))

        voxel_size = 0.03 
        if voxel_size is not None:
            model_pcd = model_pcd.voxel_down_sample(voxel_size)

        node_pcd = model_pcd.farthest_point_down_sample(100)
        obj_model.rest_points = np.array(node_pcd.points)

        node_graph = NodeGraph(obj_model.rest_points, rbf_sig=0.1)
        match_geom = node_graph.get_pts2nodes_graph(obj_model.rest_points)

        if vis_obj:
            o3d.visualization.draw_geometries(match_geom + [model_pcd])

        # assemble K matrix
        obj_model.element_lst = node_graph.compute_edges()

    obj_model.material_model = LinearSpringModel()
    return obj_model


def assemble_K_mat(element_lst, material_lst, material_model: BaseMaterialModel, 
                   rest_points: np.ndarray, curr_points: np.ndarray, sp_type='coo'):
    rest_pts_tsr = torch.tensor(rest_points).double()
    curr_pts_tsr = torch.tensor(curr_points).double()

    num_pts = curr_points.shape[0]

    r_idx_lst = []
    c_idx_lst = []
    data = []
    for e, m in zip(element_lst, material_lst):

        p = rest_pts_tsr[e, :].flatten()
        u = (curr_pts_tsr[e, :]-rest_pts_tsr[e, :]).flatten()

        try:
            K = material_model.element_stiffness(p, u, m)
        except:
            raise ValueError('Failed to compute the stiffness')

        rc_idx = idx_pts2mat(e)
        r_idx_lst.extend(rc_idx[:, 0])
        c_idx_lst.extend(rc_idx[:, 1])
        data.extend(K.reshape(-1))

    sp_K_mat = coo_matrix((data, (r_idx_lst, c_idx_lst)), shape=(3*num_pts, 3*num_pts))
    if sp_type == "coo":
        return sp_K_mat
    elif sp_type == "csr":
        return sp_K_mat.tocsr()
    elif sp_type == "csc":
        return sp_K_mat.tocsc()

class ObjectModel:
    """
        Object model constains the material model and the geometry model.

        rest_points: (num_pts, 3) numpy array
        element_lst: (num_ele, ele_dim) numpy array
        geometry_model: object deformation basis functions
        material_model: defines the material properties
    """
    def __init__(self, rest_points=None, element_lst=None, 
                 geometry_model=None, material_model=None) -> None:
        
        self.rest_points: np.ndarray = rest_points
        self.element_lst: List[np.ndarray] = element_lst
        # element_mat works with flat vectors
        if self.element_lst is not None:
            self.element_mat: torch.Tensor = self.element_lst2mat()

        self.geometry_model : GlobalDeformModel = geometry_model
        self.material_model : BaseMaterialModel = material_model

        # auxiliary object geometry
        self.obj_bbox = None
    
    def sample_points(self, init_pcd: o3d.geometry.PointCloud, sample_params: Dict):
        """
            set object discretization: rest_points and element_lst
            initialize material values

            basic sampling methods: voxel down sampling, farthest point sampling
            occluded region sampling
        """
        if sample_params['method'] == 'voxel':
            pcd = init_pcd.voxel_down_sample(sample_params['voxel_size'])
        elif sample_params['method'] == 'farthest':
            pcd = init_pcd.farthest_point_down_sample(sample_params['num_pts'])
        self.rest_points = np.array(pcd.points)

    def generate_elements(self, connect_params={}):
        """Generate elements list"""
        if self.rest_points is None:
            raise ValueError("rest_points is not set")
        
        if self.material_model.geom_type == 'tetra':
            self.element_lst = Delaunay(self.rest_points).simplices
            # raise Warning("Delaunay tetrahedralization may produce bad elements")
        else:
            if connect_params['nn'] == 'kNN':
                A_mat = kneighbors_graph(self.rest_points, n_neighbors=connect_params['num_nns'], 
                                         include_self=connect_params['include_self'])
            elif connect_params['nn'] == 'radius':
                A_mat = radius_neighbors_graph(self.rest_points, radius=connect_params['radius'], 
                                               include_self=connect_params['include_self'])
            else:
                raise RuntimeWarning("Unknown nearest neighbor method:", connect_params['nn'])
            
            if self.material_model.geom_type == 'spring':
                row_counts = A_mat.indptr[1:] - A_mat.indptr[:-1]
                row_idx = np.repeat(np.arange(len(row_counts)), row_counts)
                col_idx = A_mat.indices

                # NOTE: remove duplicate edges
                edges = np.vstack([row_idx, col_idx]).T
                self.element_lst = np.unique(np.sort(edges, axis=1), axis=0)
                
            elif self.material_model.geom_type == 'particle':
                self.element_lst = np.split(A_mat.indices, A_mat.indptr[1:-1])
            
            else:
                raise RuntimeWarning("Unknown geometry type:", self.material_model.geom_type)

    def num_pts(self):
        return self.rest_points.shape[0]

    def num_ele(self):
        return len(self.element_lst)
    
    def zero_material_values(self):
        return [self.material_model.zero_material() for _ in range(self.num_ele())]
    
    def unit_material_values(self, dtype='list'):
        material_values = [self.material_model.unit_material() for _ in range(self.num_ele())]
        if dtype == 'list':
            return material_values
        elif dtype == 'torch':
            return torch.vstack(material_values)
        elif dtype == 'numpy':
            return torch.vstack(material_values).numpy()
    
    def sample_material_values(self, std=0.01):
        return [std*torch.randn(self.material_model.m_dim) for _ in range(self.num_ele())]

    def element_lst2mat(self):
        ii_ary, jj_ary = [], []
        c_row = 0
        for ele in self.element_lst:
            for ii, jj in enumerate(ele):
                ii_ary.extend([c_row+3*ii, c_row+3*ii+1, c_row+3*ii+2])
                jj_ary.extend([3*jj, 3*jj+1, 3*jj+2])
            c_row += 3*len(ele)
        idx_ary = [ii_ary, jj_ary]
        val_ary = np.ones(len(ii_ary))

        self.element_mat = torch.sparse_coo_tensor(idx_ary, val_ary)
        # self.element_mat = self.element_mat.to_sparse_csr()
        return self.element_mat

    def object_forces(self, curr_points, material_values, requires_grad=False, 
                      dtype='torch', use_sparse=False):        
        if requires_grad:
            assert isinstance(curr_points, torch.Tensor)
            assert isinstance(material_values, torch.Tensor) \
                   and material_values.requires_grad
            curr_pts_tsr = curr_points
            m_val_tsr = material_values
        else:
            curr_pts_tsr = torch.tensor(curr_points).double()
            m_val_tsr = torch.tensor(material_values).double()
        rest_pts_tsr = torch.tensor(self.rest_points).double()

        if use_sparse:
            p_all = self.element_mat @ rest_pts_tsr.flatten()
            p_all = p_all.reshape(-1, self.material_model.p_dim)
            u_all = self.element_mat @ (curr_pts_tsr - rest_pts_tsr).flatten()
            u_all = u_all.reshape(-1, self.material_model.u_dim)
            f_ele_all = vmap(self.material_model.element_forces)(p_all, u_all, m_val_tsr)
            f_ele_all = f_ele_all.flatten()
            f_all = (self.element_mat.T @ f_ele_all).reshape(-1, 3)
        else:
            f_all = torch.zeros((self.num_pts(), 3), 
                                requires_grad=requires_grad).double()
            for e, m in zip(self.element_lst, m_val_tsr):

                p = rest_pts_tsr[e, :].flatten()
                u = (curr_pts_tsr[e, :]-rest_pts_tsr[e, :]).flatten()

                f = self.material_model.element_forces(p, u, m)

                f_all.index_add_(0, torch.tensor(e), f.reshape(-1, 3))
        
        if dtype == 'torch':
            return f_all
        elif dtype == 'numpy':
            assert not requires_grad
            return f_all.numpy()

    def object_stiffness(self, curr_points, material_values) -> csc_matrix:
        if isinstance(curr_points, torch.Tensor):
            assert not curr_points.requires_grad
        if isinstance(material_values, torch.Tensor):
            assert not material_values.requires_grad

        sp_K_mat = assemble_K_mat(self.element_lst, material_values, self.material_model,
                                  self.rest_points, curr_points, sp_type='csc')
        return sp_K_mat
    
    def force_vs_material_jacobian(self, curr_points, material_values) -> csc_matrix:
        assert self.material_model.m_dim == 2

        # compute Jacobian from single element
        fvm_jac = torch.func.jacrev(self.material_model.element_forces, argnums=2, has_aux=False, 
                                    chunk_size=None, _preallocate_and_copy=False)
        
        num_pts = self.num_pts()
        num_elements = self.num_ele()

        rest_points = torch.tensor(self.rest_points, dtype=torch.float32)
        deform = torch.tensor(curr_points - self.rest_points, dtype=torch.float32)

        obj_jac_mat = torch.zeros(3*num_pts, 2*num_elements)
        for i in range(num_elements):
            tetra = self.element_lst[i]

            p_i = rest_points[tetra]
            u_i = deform[tetra]
            m_i = material_values[i]
            jac_mat = fvm_jac(p_i.flatten(), u_i.flatten(), m_i)

            for j in range(4):
                pt_idx = tetra[j]
                obj_jac_mat[3*pt_idx:3*(pt_idx+1), 2*i:2*(i+1)] += jac_mat[3*j:3*(j+1), :]

        return obj_jac_mat

    def get_obj_pcd(self):
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(self.rest_points)
        return obj_pcd
    
    def get_element_o3d(self, points=None, color=None):
        if points is None:
            points = self.rest_points

        if self.material_model.geom_type == 'tetra':
            element_o3d = o3d.geometry.TetraMesh(o3d.utility.Vector3dVector(points), 
                                                 o3d.utility.Vector4iVector(self.element_lst))
        elif self.material_model.geom_type == 'spring':
            element_o3d = o3d.geometry.LineSet()
            element_o3d.points = o3d.utility.Vector3dVector(points)
            element_o3d.lines = o3d.utility.Vector2iVector(self.element_lst)
        if color is not None:
            element_o3d.paint_uniform_color(color)
        return element_o3d
    
    def elements2edges(self):
        new_element_lst = []
        for element in self.element_lst:
            edge_lst = combinations(element, 2)
            new_element_lst.extend(edge_lst)

        edges = np.array(new_element_lst)
        sort_edges = np.sort(edges, axis=1)
        self.element_lst = np.unique(sort_edges, axis=0)
        return self.element_lst