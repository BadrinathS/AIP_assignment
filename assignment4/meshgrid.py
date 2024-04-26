import torch
import numpy as np
def min_dist(points, res):
    # Extract batch size and number of points
    batch_size, num_points, _ = points.shape
    
    # Create grid of coordinates for the output tensor
    x_coords = np.linspace(0, 1, res)
    y_coords = np.linspace(0, 1, res)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_coords = np.stack((grid_x, grid_y), axis=-1)  # Shape: (res, res, 2)
    
    # Expand dimensions for broadcasting
    print(points[...].shape, points.shape)
    exit()
    grid_coords_expanded = np.expand_dims(np.expand_dims(grid_coords, axis=0), axis=0)  # Shape: (1, 1, res, res, 2)
    points_expanded = np.expand_dims(points[..., :2], axis=2)  # Shape: (B, P, 1, 1, 2)
    
    # Compute Euclidean distance
    distances = np.linalg.norm(grid_coords_expanded - points_expanded, axis=-1)  # Shape: (B, P, res, res)
    
    # Compute minimum distance for each pixel
    min_distances = np.min(distances, axis=1)  # Shape: (B, res, res)
    
    return min_distances

# Example usage:
batch_size = 3
num_points = 4
resolution = 10
points = np.random.rand(batch_size, num_points, 3)  # Example input points
result = min_dist(points, resolution)
print(result.shape)