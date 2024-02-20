import numpy as np
import open3d as o3d
import pickle

import context
from vis_tac_sim.o3d_utils import create_motion_lines, select_points
from vis_tac_sim.geometry_model import NodeGraph

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(1)

    rest_pts = np.load('assets/new_orange_tree_pts.npy')

    node_graph = NodeGraph(rest_pts, num_nns=8)
    # pickle.dump(node_graph, open('assets/new_orange_tree_node_graph.pkl', 'wb'))
    edges_lst = node_graph.compute_edges(connect_pattern='knn')
    np.save('assets/new_orange_tree_edges.npy', edges_lst)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    pcd = node_graph.get_o3d_obj(np.zeros(rest_pts.shape), color=[0.0, 0.0, 1.0])
    obb = pcd.get_oriented_bounding_box()
    obb.color = [1, 0, 0]
    o3d.visualization.draw_geometries([pcd, coord_frame, obb])

    pcd.rotate(np.linalg.inv(obb.R))
    new_obb = pcd.get_oriented_bounding_box()
    new_obb.color = [0, 1, 0]
    o3d.visualization.draw_geometries([pcd, coord_frame, new_obb])

    print('new_obb_center:', new_obb.center)

    fix_mask = np.array(pcd.points)[:, 0] < -1.0
    colors = np.zeros_like(rest_pts)
    colors[fix_mask] = [1.0, 0.0, 0.0]
    colors[~fix_mask] = [0.0, 1.0, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd, coord_frame])    

    np.save('assets/new_orange_tree_fix_idx.npy', np.where(fix_mask)[0])