import numpy as np
from scipy.optimize import minimize
import open3d as o3d

# Define the superellipsoid equation
def superellipsoid(x, y, z, params):
    a, b, c, eps1, eps2 = params
    try:
        term1 = ((x / a) ** (2 / eps2) + (y / b) ** (2 / eps2)) ** (eps2 / eps1)
    except:
        print('eps1:', eps1)
        print('eps2:', eps2)
    term2 = (z / c) ** (2 / eps1)
    return term1 + term2 - 1

# Define the objective function
def objective_function(params, points):
    x, y, z = points.T  # Unpack point cloud
    distances = superellipsoid(x, y, z, params)
    return np.sum(np.abs(distances) ** 2)  # G_2(theta)

# Example point cloud
points = np.load('super_ellipsoid_points.npy')
print('points shape:', points.shape)

# check with ground truth parameters
f = superellipsoid(points[:, 0], points[:, 1], points[:, 2], [1, 1.5, 1, 0.5, 1.0])
print('max f value:', np.abs(f).max())

# Initial guess for parameters [a, b, c, eps1, eps2]
initial_guess = [1, 1, 1, 1, 1]

# Perform optimization
result = minimize(objective_function, initial_guess, args=(points,), method='L-BFGS-B')

# Optimized parameters
optimized_params = result.x
print("Optimized Parameters:", optimized_params)