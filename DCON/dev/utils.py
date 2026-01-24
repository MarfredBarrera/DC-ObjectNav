import torch


def rgb_to_sh(rgb):
    """
    Converts RGB values to spherical harmonics (SH) coefficients.
    
    Args:
        rgb: RGB values in range [0, 1]
    
    Returns:
        SH coefficients
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def eval_sh(coeffs, dirs):
    """
    Evaluates spherical harmonics for a batch of directions.
    
    Args:
        coeffs: Spherical harmonics coefficients
        dirs: Directions to evaluate (normalized vectors)
    
    Returns:
        Evaluated values for the given directions
    """
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    # Degree 0 (Constant)
    C0 = 0.28209479177387814
    result = C0 * coeffs[..., 0]
    
    if coeffs.shape[1] > 1:
        # Degree 1
        C1 = 0.4886025119029199
        result += C1 * (-y * coeffs[..., 1] + z * coeffs[..., 2] - x * coeffs[..., 3])
    return result


def unprojection(depth, intrinsics, c2w, device):
    """
    Unprojects depth image to camera-space 3D coordinates.
    
    Args:
        depth: Depth image (H x W)
        intrinsics: Tuple of (fx, fy, cx, cy, H, W)
        device: PyTorch device
    
    Returns:
        x_c, y_c, z: Camera-space coordinates (each of shape H x W)
    """
    fx, fy, cx, cy, H, W = intrinsics
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    z_c = depth
    x_c = (x - cx) * z_c / fx
    y_c = (y - cy) * z_c / fy

    mask = (depth > 0.1) & (depth < 10.0)
        
    cam_points = torch.stack([
        x_c[mask],
        y_c[mask],
        z_c[mask],
        torch.ones_like(z_c[mask])
    ], dim=1)
    
    # Transform to world
    world_points = (c2w @ cam_points.T).T[:, :3]


    return world_points