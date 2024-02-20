import torch
import numpy as np
import open3d as o3d


def create_o3dMesh_ball(radius=0.01, color=[1, 0, 0], center=None):
    ball = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    ball.compute_vertex_normals()
    ball.paint_uniform_color(color)
    if center is not None:
        ball.translate(center)
    return ball

def create_motion_lines(prev_pts, curr_pts, return_pcd=False):
    assert(prev_pts.shape == curr_pts.shape)
    prev_pcd = o3d.geometry.PointCloud()
    prev_pcd.points = o3d.utility.Vector3dVector(prev_pts)
    prev_pcd.paint_uniform_color([0, 0, 1])

    curr_pcd = o3d.geometry.PointCloud()
    curr_pcd.points = o3d.utility.Vector3dVector(curr_pts)
    curr_pcd.paint_uniform_color([1, 0, 0])

    pcd_correspondence = [[i, i] for i in range(curr_pts.shape[0])]
    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(prev_pcd, curr_pcd, pcd_correspondence)
    if return_pcd:
        return prev_pcd, curr_pcd, line_set
    else:
        return line_set

def create_graph_lines(points, edges):
    """ Visualize line set """
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    return line_set

def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def create_vector_arrow(end, origin=np.array([0, 0, 0]), scale=1, color=[0.707, 0.707, 0.0]):
    assert(not np.all(end == origin))
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return(mesh)


def create_arrow_lst(p1_ary, p2_ary, **args):
    arrow_lst = []
    for p1, p2 in zip(p1_ary, p2_ary):
        if np.linalg.norm(p2-p1) > 0.01:
            arrow_lst.append(create_vector_arrow(p2, origin=p1, **args))
    return arrow_lst

def set_view_params(o3d_vis, view_params={}):
    ctr = o3d_vis.get_view_control()
    if "zoom" in view_params.keys():
        ctr.set_zoom(view_params["zoom"])
    if "front" in view_params.keys():
        ctr.set_front(view_params["front"])
    if "lookat" in view_params.keys():
        ctr.set_lookat(view_params["lookat"])
    if "up" in view_params.keys():
        ctr.set_up(view_params["up"])

def select_points(pcd, view_params=None):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    if view_params is not None:
        set_view_params(vis, view_params)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()

def extract_surface_mesh(tetra_mesh: o3d.geometry.TetraMesh):
    # ref: https://stackoverflow.com/questions/66607716/how-to-extract-surface-triangles-from-a-tetrahedral-mesh

    envelope = set()
    for tet in np.array(tetra_mesh.tetras):
        for face in ( (tet[0], tet[1], tet[2]), 
                      (tet[0], tet[2], tet[3]), 
                      (tet[1], tet[3], tet[0]),
                      (tet[1], tet[3], tet[2]) ):
            # if face has already been encountered, then it's not on the envelope
            # the magic of hashsets makes that check O(1) (eg. extremely fast)
            face = tuple(sorted(face))
            if face in envelope:
                envelope.remove(face)
            else:
                envelope.add(face)

    face_lst = list(envelope)
    face_lst.extend([list(reversed(face)) for face in face_lst])

    surface_mesh = o3d.geometry.TriangleMesh()
    surface_mesh.vertices = tetra_mesh.vertices
    surface_mesh.triangles = o3d.utility.Vector3iVector(face_lst)
    surface_mesh.compute_vertex_normals()

    return surface_mesh