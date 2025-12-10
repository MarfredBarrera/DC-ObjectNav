import os
import json
import math
import time
import torch
import numpy as np
import imageio.v2 as imageio
import cv2
import torch.nn.functional as F

# NEW: v1.0+ import
from gsplat import rasterization

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCENE_DIR = "/workspace/DCON/output/current_scene"
ITERATIONS = 10000
DEVICE = "cuda"

# -----------------------------------------------------------------------------
# 1. Data Loader & Depth Unprojection
# -----------------------------------------------------------------------------

def load_scene_data(data_dir, device="cuda"):
    """Loads images, depth, and camera poses from Habitat transforms.json"""
    json_path = os.path.join(data_dir, "transforms.json")
    with open(json_path, 'r') as f:
        meta = json.load(f)

    frames = meta['frames']
    img_0 = imageio.imread(os.path.join(data_dir, frames[0]['file_path']))
    H, W = img_0.shape[:2] # This is 720, 720
    
    fov_x = meta['camera_angle_x']
    fx = 0.5 * W / math.tan(0.5 * fov_x)
    fy = fx
    cx, cy = W / 2.0, H / 2.0

    gt_images = []
    gt_depths = []
    c2w_matrices = []

    print(f"Loading {len(frames)} frames...")
    for frame in frames:
        # Load RGB
        rgb_path = os.path.join(data_dir, frame['file_path'])
        rgb = imageio.imread(rgb_path)
        gt_images.append(torch.from_numpy(rgb).float().to(device) / 255.0)

        # Load Depth
        depth_name = os.path.basename(frame['file_path']).replace("rgb", "depth").replace(".png", ".npy")
        depth_path = os.path.join(data_dir, "depth_data", depth_name)
        depth = np.load(depth_path)
        
        # --- FIX: Resize depth if it doesn't match RGB ---
        if depth.shape[:2] != (H, W):
            # depth is (H, W) or (H, W, 1), cv2.resize expects (W, H)
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        # -------------------------------------------------

        gt_depths.append(torch.from_numpy(depth).float().to(device))

        # Load Pose
        c2w = np.array(frame['transform_matrix'])
        c2w_matrices.append(torch.from_numpy(c2w).float().to(device))

    return torch.stack(gt_images), torch.stack(gt_depths), torch.stack(c2w_matrices), (fx, fy, cx, cy, H, W)

def init_point_cloud_from_depth(gt_images, gt_depths, c2ws, intrinsics, num_init_frames=4):
    fx, fy, cx, cy, H, W = intrinsics
    points_list = []
    colors_list = []

    indices = np.linspace(0, len(gt_images)-1, num_init_frames, dtype=int)
    y, x = torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing='ij')
    
    print("Initializing point cloud from depth data...")
    for idx in indices:
        depth = gt_depths[idx]
        color = gt_images[idx]
        c2w = c2ws[idx]
        
        mask = (depth > 0.1) & (depth < 10.0)
        
        # Unproject
        z_c = -depth[mask]
        x_c = (x[mask] - cx) * depth[mask] / fx
        y_c = -(y[mask] - cy) * depth[mask] / fy 

        ones = torch.ones_like(z_c)
        cam_points = torch.stack([x_c, y_c, z_c, ones], dim=1)
        world_points = (c2w @ cam_points.T).T
        
        points_list.append(world_points[:, :3])
        colors_list.append(color[mask])

    full_points = torch.cat(points_list, dim=0)
    full_colors = torch.cat(colors_list, dim=0)
    
    # Random subsample if too large
    if full_points.shape[0] > 100000:
        indices = torch.randperm(full_points.shape[0])[:100000]
        full_points = full_points[indices]
        full_colors = full_colors[indices]

    print(f"Initialized with {full_points.shape[0]} points.")
    return full_points, full_colors

# -----------------------------------------------------------------------------
# 2. Main Training Script (v1.0 API)
# -----------------------------------------------------------------------------
def main():
    gt_images, gt_depths, c2ws, intrinsics = load_scene_data(SCENE_DIR, DEVICE)
    fx, fy, cx, cy, H, W = intrinsics
    num_cameras = len(gt_images)

    init_means, init_colors = init_point_cloud_from_depth(gt_images, gt_depths, c2ws, intrinsics)

    # Parameters
    means = torch.nn.Parameter(init_means)
    quats = torch.nn.Parameter(torch.zeros(init_means.shape[0], 4, device=DEVICE)) 
    with torch.no_grad():
        quats[:, 0] = 1.0
    scales = torch.nn.Parameter(torch.ones_like(init_means) * -4.0) # start small
    opacities = torch.nn.Parameter(torch.zeros(init_means.shape[0], device=DEVICE)) # logit(0.5)
    
    safe_colors = torch.clamp(init_colors, 0.01, 0.99)
    colors = torch.nn.Parameter(torch.logit(safe_colors))

    optimizer = torch.optim.Adam([
        {'params': [means], 'lr': 1e-3, 'name': 'means'},
        {'params': [quats], 'lr': 1e-3, 'name': 'quats'},
        {'params': [scales], 'lr': 5e-3, 'name': 'scales'},
        {'params': [opacities], 'lr': 5e-2, 'name': 'opacities'},
        {'params': [colors], 'lr': 1e-2, 'name': 'colors'},
    ])

    # Pre-compute Intrinsics Matrix K [3, 3]
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], device=DEVICE)

    print(f"Starting training for {ITERATIONS} iterations...")
    start_time = time.time()

    for step in range(ITERATIONS):
        cam_idx = torch.randint(0, num_cameras, (1,)).item()
        gt_image = gt_images[cam_idx]
        c2w = c2ws[cam_idx]

        # World-to-Camera Matrix
        w2c = torch.inverse(c2w)
        
        # NEW v1.0 API Call
        # Everything is handled in one function.
        # It takes raw parameters and returns the rendered image.
        renders, alphas, meta = rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=torch.sigmoid(colors),
            viewmats=w2c[None, ...],  # [1, 4, 4]
            Ks=K[None, ...],          # [1, 3, 3]
            width=W,
            height=H
        )
        
        # Remove batch dimension [1, H, W, 3] -> [H, W, 3]
        rgb_render = renders[0]

        loss = F.mse_loss(rgb_render, gt_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step:04d} | Loss: {loss.item():.5f} | Time: {time.time()-start_time:.1f}s")

    # Save Result
    print("Rendering final view...")
    with torch.no_grad():
        c2w = c2ws[0]
        w2c = torch.inverse(c2w)
        
        renders, _, _ = rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=torch.sigmoid(colors),
            viewmats=w2c[None, ...],
            Ks=K[None, ...],
            width=W,
            height=H
        )
        final_img = (renders[0].cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite("render_view_0.png", final_img)
        print("Saved render_view_0.png")

    print("Saving model parameters to model.pt...")
    torch.save({
        'means': means,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
        'colors': colors,
        'focal': (fx, fy),
        'center': (cx, cy)
    }, os.path.join(SCENE_DIR, "model.pt"))
    print("Model saved.")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
    main()