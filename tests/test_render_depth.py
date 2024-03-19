import cupoch as cph
import numpy as np
import time
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

import context
from vis_tac_sim.o3d_utils import set_view_params

view_params = {	
    "front" : [ 0.92533671872582346, -0.00977042279659638, 0.37902044247784583 ],
    "lookat" : [ 0.50709886611279531, 0.2256379128268034, 0.99633593149129163 ],
    "up" : [ -0.37856288616066625, 0.031629292431272216, 0.92503498803126549 ],
    "zoom" : 0.079999999999999613
}

def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 4]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans


def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)], [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0], [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)


def voxel_carving(
    sim_out_dir,
    cubic_size,
    voxel_resolution,
    w=300,
    h=300,
):

    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=np.array([-cube_size/2, -cube_size/2, 0.0]),
        color=np.array([0.0, 0.7, 1.0])
    )
    print('number of voxels:', len(voxel_carving.get_voxels()))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    
    num_voxels_lst = []
    fn_lst = sorted(Path(sim_out_dir).glob('step*.npy'))
    for fn in fn_lst:    
        pcd = o3d.geometry.PointCloud()
        step_pts = np.load(fn)
        pcd.points = o3d.utility.Vector3dVector(step_pts)

        vis.add_geometry(pcd)
        # vis.add_geometry(voxel_carving)
        set_view_params(vis, view_params)

        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()

        # capture depth image and make a point cloud
        start_time = time.time()
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)
        print('capture depth time:', time.time() - start_time)

        # plt.imshow(depth)
        # plt.axis('off')
        # plt.savefig(f'{fn.stem}.png', bbox_inches='tight', pad_inches=0)
        # plt.show()

        start_time = time.time()
        voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
        print('voxel carving time:', time.time() - start_time)
        print('number of voxels:', len(voxel_carving.get_voxels()))
        num_voxels_lst.append(len(voxel_carving.get_voxels()))

        vis.remove_geometry(pcd)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([voxel_carving, pcd, coord_frame], **view_params)

    plt.clf()
    plt.plot(num_voxels_lst)
    plt.xlabel('step')
    plt.ylabel('number of voxels')
    plt.show()

    vis.destroy_window()

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([voxel_carving, pcd, coord_frame], **view_params)

if __name__ == '__main__':

    cube_size = 2.0
    voxel_resolution = 100
    sim_out_dir = 'out_data/sim_fiddle_tree_leaf_04'
    voxel_carving(sim_out_dir, cube_size, voxel_resolution)