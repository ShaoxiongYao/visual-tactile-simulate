import numpy as np
from scipy.optimize import minimize
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import pypose as pp

from utils import superellipsoid_points

# Define the superellipsoid equation
def superellipsoid(points, params, xyz=None, rot=None):
    a, b, c, eps1, eps2 = params
    r0_norm = torch.linalg.norm(points, dim=1)

    if xyz is not None:
        points = points - xyz
    
    if rot is not None:
        points = pp.Exp(rot) @ points

    x_a = torch.abs(points[:, 0] / a)
    y_b = torch.abs(points[:, 1] / b)
    z_c = torch.abs(points[:, 2] / c)

    term1 = torch.float_power(x_a, 2 / eps2) + torch.float_power(y_b, 2 / eps2)
    term1 = torch.float_power(term1, eps2 / eps1)

    term2 = torch.float_power(z_c, 2 / eps1)
    return r0_norm * torch.abs(torch.float_power(term1 + term2, -eps1/2) - 1)
    # return r0_norm * torch.abs(term1 + term2 - 1)

# Example point cloud
# pcd = o3d.io.read_point_cloud('/home/planck/Downloads/yellow_pepper_01_cropped.ply')

pcd = o3d.geometry.PointCloud()
points = np.load('super_ellipsoid_points.npy')
pcd.points = o3d.utility.Vector3dVector(points)

pcd = o3d.io.read_point_cloud('/home/planck/Downloads/clouds/pointcloud3.pcd')
points = np.asarray(pcd.points)
points -= points.mean(axis=0)

# Normalize the points
# points /= points.max()
print('points shape:', points.shape)

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
o3d.visualization.draw_geometries([pcd, coord_frame])

pts_tsr = torch.tensor(points, dtype=torch.float32)

# Initial guess for parameters [a, b, c, eps1, eps2]
# initial_guess = [0.08, 0.08, 0.08, 0.6, 0.6]
initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0]
# ground_truth = [1, 1.5, 1, 0.5, 1.0]
shape_params = torch.tensor(initial_guess, requires_grad=True)

# Visualize loss change on eps1
# loss_list = []
# # check with ground truth parameters
# for eps1 in np.linspace(0.25, 1.25, 100):
#     ground_truth[3] = eps1
#     f = superellipsoid(pts_tsr, ground_truth)
#     print('max f value:', np.abs(f).max())
#     loss_list.append(torch.sum(f ** 2).item() / len(f))
# plt.plot(np.linspace(0.25, 1.25, 100), loss_list)
# plt.show()

sup_eps_xyz = torch.zeros(3, requires_grad=True)
sup_eps_rot = pp.randn_so3(1, sigma=1e-4, requires_grad=True, 
                           dtype=torch.float32)

optimizer = torch.optim.Adam([shape_params, sup_eps_xyz, sup_eps_rot], lr=0.001) # Learning rate

num_epochs = 10000

# bounds on synthetic data
# abc_min, abc_max = 0.1, 2.0 
# eps_min, eps_max = 0.1, 2.0

abc_min, abc_max = 0.02, 0.15 
eps_min, eps_max = 0.3, 0.9

loss_list = []
for epoch in range(num_epochs):
    print('epoch:', epoch)

    optimizer.zero_grad()
    f = superellipsoid(pts_tsr, shape_params, 
                       xyz=sup_eps_xyz, rot=sup_eps_rot)

    loss = torch.sum(torch.abs(f) ** 2) / len(f)
    loss.backward()
    optimizer.step()

    print('loss:', loss.item())
    loss_list.append(loss.item())

    with torch.no_grad():
        shape_params[:3] = torch.clamp(shape_params[:3], min=abc_min, max=abc_max)
        shape_params[3:] = torch.clamp(shape_params[3:], min=eps_min, max=eps_max)

    print('shape_params:', shape_params)
    print('sup_eps_xyz:', sup_eps_xyz)
    print('sup_eps_rot:', sup_eps_rot)

plt.plot(loss_list)
plt.show()

gt_pcd = o3d.geometry.PointCloud()
gt_pcd.points = o3d.utility.Vector3dVector(points)
gt_pcd.paint_uniform_color([0.0, 0.7, 0.9])

a, b, c, eps1, eps2 = shape_params.detach().numpy()
x, y, z = superellipsoid_points(a, b, c, eps1, eps2, 300)
fit_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

with torch.no_grad():
    fit_pts_tsr = torch.tensor(fit_points, dtype=torch.float32)
    fit_points = pp.Exp(pp.Inv(sup_eps_rot)) @ fit_pts_tsr + sup_eps_xyz

fit_pcd = o3d.geometry.PointCloud()
fit_pcd.points = o3d.utility.Vector3dVector(fit_points.numpy())
fit_pcd.paint_uniform_color([0.9, 0.7, 0.0])
o3d.visualization.draw_geometries([fit_pcd, gt_pcd])
