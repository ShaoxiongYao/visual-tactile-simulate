import numpy as np

def superellipsoid_points(a, b, c, eps1, eps2, num_points=1000):

    # Generate angles
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(-np.pi / 2, np.pi / 2, num_points)
    u, v = np.meshgrid(u, v)
    
    # Calculate coordinates
    x = a * np.sign(np.cos(v)) * np.abs(np.cos(v)) ** eps2 * np.sign(np.cos(u)) * np.abs(np.cos(u)) ** eps1
    y = b * np.sign(np.sin(v)) * np.abs(np.sin(v)) ** eps2 * np.sign(np.cos(u)) * np.abs(np.cos(u)) ** eps1
    z = c * np.sign(np.sin(u)) * np.abs(np.sin(u)) ** eps1
    
    return x, y, z