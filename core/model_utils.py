import torch
import numpy as np



def generate_gaussian_heatmap(points, keypoints, sigma):
    '''
    Generate point heatmap (3D Gaussian blob around a keypoint) for every keypoint.
    
    Input:
        points: (n, 3) torch.tensor - The input points
        keypoints: (k, 3) torch.tensor - The keypoints
        sigma: float - The standard deviation for the Gaussian distribution
    
    Output:
        heatmap: (n, k) torch.tensor - The generated heatmap
    '''
    # Reshape points and keypoints for broadcasting
    points_expanded = points.unsqueeze(1)  # Shape: (n, 1, 3)
    keypoints_expanded = keypoints.unsqueeze(0)  # Shape: (1, k, 3)
    
    squared_distances = torch.sum((points_expanded - keypoints_expanded) ** 2, dim=2)  # Shape: (n, k)
    heatmap = torch.exp(-squared_distances / (2 * sigma ** 2))
    
    # Set very small values (below machine epsilon) to zero
    eps = torch.finfo(heatmap.dtype).eps * heatmap.max()
    heatmap[heatmap < eps] = 0.0
    
    return heatmap


def generate_heatmap_and_vector_batch(points, keypoints, batch, sigma):
    '''
    Generate point heatmap (3D Gaussian blob around a keypoint) for every keypoint.
    The input is a concatenated batch

    Input:
        points: (N, 3) torch.tensor - The input points
        keypoints: (B, k, 3) torch.tensor - The keypoints
        batch: (N,) batch index of each point
        sigma: (k,) tensor - The standard deviation for the Gaussian distribution
    
    Output:
        heatmap: (n, k) torch.tensor - The generated heatmap
        vector: (n, k, 3)
    '''
    # Reshape points and keypoints for broadcasting
    points_expanded = points.unsqueeze(1)  # Shape: (N, 1, 3)
    # keypoints_expanded = keypoints.unsqueeze(0)  # Shape: (1, k, 3)
    keypoints_expanded = keypoints[batch]   # (N, k, 3)
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma).to(points.device)  # (k,)
    
    vector = keypoints_expanded - points_expanded  # (N, k, 3)
    squared_distances = torch.sum(vector ** 2, dim=2)  # Shape: (N, k)
    heatmap = torch.exp(-squared_distances / (2 * sigma ** 2))  # (N, k)
    
    # Set very small values (below machine epsilon) to zero
    eps = torch.finfo(heatmap.dtype).eps * heatmap.max()
    heatmap[heatmap < eps] = 0.0
    
    return heatmap, vector


def vector6d_to_rotation_matrix_numpy(poses):

    if isinstance(poses, torch.Tensor):
        if poses.is_cuda:
            poses = poses.cpu()
        poses = poses.numpy()
    
    # ensure the input is at least 2D array
    if poses.ndim == 1:
        poses = poses[np.newaxis, :]
    
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]
    
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    
    matrix = np.stack((x, y, z), axis=-1)
    
    return matrix


def vector6d_to_rotation_matrix(poses):
    
    if poses.dim() == 1:
        poses = poses.unsqueeze(0)
    
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]
    
    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    
    matrix = torch.stack((x, y, z), dim=-1)
    
    return matrix