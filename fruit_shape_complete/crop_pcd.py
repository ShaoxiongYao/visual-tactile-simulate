import open3d as o3d

for i in [1, 2, 3]:
    fn = f'/home/planck/Downloads/yellow_pepper_0{i}.ply'
    pcd = o3d.io.read_point_cloud(fn)
    o3d.visualization.draw_geometries_with_editing([pcd])