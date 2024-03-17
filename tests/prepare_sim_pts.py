import numpy as np
import open3d as o3d
from pathlib import Path

import context
from vis_tac_sim.o3d_utils import select_points

def manual_prepare_points(pcd_fn):
    pcd = o3d.io.read_point_cloud(pcd_fn)
    pcd.normals = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, coord_frame])
    o3d.visualization.draw_geometries_with_editing([pcd])
    return pcd

import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_centroids(source, target):
    """Compute the centroids of the source and target point sets."""
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    return centroid_source, centroid_target

def compute_correspondence_matrix(source, target):
    """Compute the correspondence matrix (H) from source to target points."""
    source_centroid, target_centroid = compute_centroids(source, target)
    H = np.dot((source - source_centroid).T, (target - target_centroid))
    return H

def estimate_transformation(source, target):
    """Estimate the rotation and translation using SVD."""
    H = compute_correspondence_matrix(source, target)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Ensure we have a proper rotation matrix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    source_centroid, target_centroid = compute_centroids(source, target)
    t = -R.dot(source_centroid) + target_centroid

    return R, t

def apply_transformation(source, R, t):
    """Apply the estimated rotation and translation to the source points."""
    transformed_source = np.dot(source, R.T) + t
    return transformed_source

if __name__ == '__main__':
    exp_id = 'e34fe1a5-f'
    obj_id = 'fiddle_tree_leaf_03'
    pts_src = 'all' # 'vis' or 'all'

    out_dir = f'out_data/plant_assets/{obj_id}'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    data_dir = '/home/motion/gaussian-splatting/output'

    # b_min = np.array([0.5, -1.0, 0.05])
    # b_max = np.array([2.5,  1.0,  3.0])

    b_min = np.array([0.5, -1.0, 0.05])*4
    b_max = np.array([2.5,  1.0,  3.0])*4

    pcd_fn = f'{data_dir}/{exp_id}/point_cloud/iteration_30000/point_cloud.ply'
    all_pcd = manual_prepare_points(pcd_fn)

    np.save(f'{out_dir}/all_pts.npy', np.array(all_pcd.points))

    pcd_fn = f'{data_dir}/{exp_id}/cropped_pcd.ply'
    vis_pcd = o3d.io.read_point_cloud(pcd_fn)
    vis_pcd.normals = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
    print('vis number points:', len(vis_pcd.points))

    vis_pcd, ind = vis_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    vis_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([vis_pcd])

    if pts_src == 'all':
        select_idx = select_points(all_pcd)
        select_pts = np.asarray(all_pcd.points)[select_idx]
    elif pts_src == 'vis':
        select_idx = select_points(vis_pcd)
        select_pts = np.asarray(vis_pcd.points)[select_idx]
    
    select_pcd = o3d.geometry.PointCloud()
    select_pcd.points = o3d.utility.Vector3dVector(select_pts)

    fix_pts = np.array([
        [ 1.2, 1.2, -1.2, -1.2],
        [-1.1, 1.1,  1.1, -1.1],
        [ 0.0, 0.0,  0.0,  0.0],
    ])

    # Estimate the transformation
    R, t = estimate_transformation(select_pts, fix_pts.T)

    # Apply the transformation to the source points
    transformed_source = apply_transformation(select_pts, R, t)

    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)
    print("Transformed Source Points:\n", transformed_source)

    H_mat = np.block([[R, t.reshape(-1, 1)],
                      [np.zeros((1, 3)), 1]])
    np.save(f'{out_dir}/H_mat.npy', H_mat)

    vis_pcd.transform(H_mat)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([vis_pcd, coord_frame])

    bbox = o3d.geometry.AxisAlignedBoundingBox(b_min, b_max)
    bbox.color = (1, 0, 0)
    in_bound_idx = bbox.get_point_indices_within_bounding_box(vis_pcd.points)
    crop_pcd = vis_pcd.select_by_index(in_bound_idx)
    crop_pcd = crop_pcd.voxel_down_sample(voxel_size=0.03)

    print('number of points:', len(crop_pcd.points))

    o3d.visualization.draw_geometries([crop_pcd, coord_frame, bbox])

    np.save(f'{out_dir}/rest_pts.npy', np.array(crop_pcd.points))
