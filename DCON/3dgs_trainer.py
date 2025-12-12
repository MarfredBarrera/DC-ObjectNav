import os
import json
import math
import time
import torch
import numpy as np
import imageio.v2 as imageio
import cv2
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCENE_DIR = "/workspace/DCON/output/current_scene"
ITERATIONS = 15000 
DEVICE = "cuda"

def rgb_to_sh(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

# -----------------------------------------------------------------------------
# 1. Data Loader with Coordinate Conversion
# -----------------------------------------------------------------------------
def load_scene_data(data_dir, device="cuda"):
    json_path = os.path.join(data_dir, "transforms.json")
    with open(json_path, 'r') as f:
        meta = json.load(f)

    frames = meta['frames']
    img_0 = imageio.imread(os.path.join(data_dir, frames[0]['file_path']))
    H, W = img_0.shape[:2]
    
    # TODO: put all camera intrinsics into config file and get parameters directly instead of computing from fov
    fov_x = meta['camera_angle_x']
    fx = 0.5 * W / math.tan(0.5 * fov_x)
    fy = fx
    cx, cy = W / 2.0, H / 2.0

    gt_images = []
    gt_depths = []
    c2w_matrices = []

    # --- COORDINATE CONVERTER MATRIX ---
    # OpenGL (Habitat) -> OpenCV (GSplat)
    # Rotate 180 around X: (x, y, z) -> (x, -y, -z)
    # This makes -Z forward become +Z forward.
    start_c2w_habitat = np.array(frames[0]['transform_matrix'])
    
    # We apply this to the LOCAL camera frame
    # C2W_opencv = C2W_habitat @ Rot_x_180
    convert_mat = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])

    print(f"Loading {len(frames)} frames...")
    for frame in frames:
        rgb_path = os.path.join(data_dir, frame['file_path'])
        rgb = imageio.imread(rgb_path)
        gt_images.append(torch.from_numpy(rgb).float().to(device) / 255.0)

        depth_name = os.path.basename(frame['file_path']).replace("rgb", "depth").replace(".png", ".npy")
        depth_path = os.path.join(data_dir, "depth_data", depth_name)
        depth = np.load(depth_path)
        if depth.shape[:2] != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        gt_depths.append(torch.from_numpy(depth).float().to(device))

        # --- FIX: Convert Pose to OpenCV Convention ---
        c2w_hab = np.array(frame['transform_matrix'])
        c2w_cv = c2w_hab @ convert_mat
        c2w_matrices.append(torch.from_numpy(c2w_cv).float().to(device))

    return torch.stack(gt_images), torch.stack(gt_depths), torch.stack(c2w_matrices), (fx, fy, cx, cy, H, W)

def init_point_cloud_from_depth(gt_images, gt_depths, c2ws, intrinsics, num_init_frames=10):
    fx, fy, cx, cy, H, W = intrinsics
    points_list = []
    colors_list = []
    indices = np.linspace(0, len(gt_images)-1, num_init_frames, dtype=int)
    y, x = torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing='ij')
    
    print("Initializing point cloud from depth (OpenCV convention)...")
    for idx in indices:
        depth = gt_depths[idx]
        color = gt_images[idx]
        c2w = c2ws[idx]
        mask = (depth > 0.1) & (depth < 10.0)
        
        # --- FIX: Unproject to OpenCV Space (+Z Forward, +Y Down) ---
        # Z is positive depth
        z_c = depth[mask] 
        # X follows standard pinhole
        x_c = (x[mask] - cx) * depth[mask] / fx
        # Y is now positive down (matches pixel coordinates)
        y_c = (y[mask] - cy) * depth[mask] / fy 

        ones = torch.ones_like(z_c)
        cam_points = torch.stack([x_c, y_c, z_c, ones], dim=1)
        world_points = (c2w @ cam_points.T).T
        
        points_list.append(world_points[:, :3])
        colors_list.append(color[mask])

    full_points = torch.cat(points_list, dim=0)
    full_colors = torch.cat(colors_list, dim=0)
    
    if full_points.shape[0] > 100_000:
        indices = torch.randperm(full_points.shape[0])[:100_000]
        full_points = full_points[indices]
        full_colors = full_colors[indices]

    # # Add noise to force gradients
    # noise = (torch.rand_like(full_points) * 0.04) - 0.02
    # full_points = full_points + noise

    return full_points, full_colors

# -----------------------------------------------------------------------------
# 2. Main Training Logic
# -----------------------------------------------------------------------------
def main():
    gt_images, gt_depths, c2ws, intrinsics = load_scene_data(SCENE_DIR, DEVICE)
    fx, fy, cx, cy, H, W = intrinsics
    num_cameras = len(gt_images)

    init_means, init_colors = init_point_cloud_from_depth(
        gt_images, gt_depths, c2ws, intrinsics
    )

    N = len(init_means)
    print(f"Initialized point cloud with {N} points.")
    
    # Init Parameters (Safe values)
    params = {
        "means": torch.nn.Parameter(init_means.contiguous()),
        "scales": torch.nn.Parameter(torch.ones(N, 3, device=DEVICE).contiguous() * -2.5), # ~8cm
        "quats": torch.nn.Parameter(torch.zeros(N, 4, device=DEVICE).contiguous()),
        "opacities": torch.nn.Parameter(torch.ones(N, device=DEVICE).contiguous() * 0.5),
        "sh0": torch.nn.Parameter(rgb_to_sh(init_colors).unsqueeze(1).contiguous()), 
        "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device=DEVICE).contiguous())
    }
    
    with torch.no_grad():
        params["quats"][:, 0] = 1.0 

    splats = torch.nn.ParameterDict(params).to(DEVICE)

    # Optimizers (Standard Rates)
    optimizers = {
        "means": torch.optim.Adam([{"params": splats["means"], "lr": 1.6e-4, "name": "means"}], eps=1e-15),
        "scales": torch.optim.Adam([{"params": splats["scales"], "lr": 0.005, "name": "scales"}], eps=1e-15),
        "quats": torch.optim.Adam([{"params": splats["quats"], "lr": 0.001, "name": "quats"}], eps=1e-15),
        "opacities": torch.optim.Adam([{"params": splats["opacities"], "lr": 5e-2, "name": "opacities"}], eps=1e-15),
        "sh0": torch.optim.Adam([{"params": splats["sh0"], "lr": 2.5e-3, "name": "sh0"}], eps=1e-15),
        # Start shN at 0 learning rate
        "shN": torch.optim.Adam([{"params": splats["shN"], "lr": 0, "name": "shN"}], eps=1e-15),
    }

    strategy = DefaultStrategy(
        verbose=True,
        refine_start_iter=100,
        refine_stop_iter=ITERATIONS - 500,
        refine_every=100,
        reset_every=1000,
        grow_grad2d=0.0002, # Standard threshold should work now that coords are fixed
        prune_opa=0.005
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state()

    scheduler_means = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / ITERATIONS))
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=DEVICE)

    print(f"Starting training for {ITERATIONS} iterations...")
    start_time = time.time()

    for step in range(ITERATIONS):
        cam_idx = torch.randint(0, num_cameras, (1,)).item()
        gt_image = gt_images[cam_idx]
        c2w = c2ws[cam_idx]
        w2c = torch.inverse(c2w)

        if(step == 1000):
            for param_group in optimizers["shN"].param_groups:
                param_group['lr'] = 2.5e-3 / 20

        # Rasterize
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        sh_degree = min(step // 1000, 3)
        
        renders, alphas, info = rasterization(
            means=splats["means"],
            quats=splats["quats"] / splats["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=w2c[None, ...],
            Ks=K[None, ...],
            width=W,
            height=H,
            sh_degree=sh_degree,
            packed=False,
            # FIX: Set Near Plane to clip stuff behind camera
            near_plane=0.01 
        )
        
        rgb_render = renders[0]

        strategy.step_pre_backward(params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info)

        l1loss = F.l1_loss(rgb_render, gt_image)
        ssimloss = 1.0 - ssim_metric(rgb_render.permute(2, 0, 1).unsqueeze(0), gt_image.permute(2, 0, 1).unsqueeze(0))
        loss = l1loss * 0.8 + ssimloss * 0.2
        
        # Gentle scale regularization
        loss += 0.01 * torch.exp(splats["scales"]).mean()

        loss.backward()
        
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
            
        strategy.step_post_backward(params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info)
        scheduler_means.step()

        if step % 100 == 0:
            num_gs = len(splats["means"])
            print(f"Step {step:04d} | GS: {num_gs} | Loss: {loss.item():.5f} | Time: {time.time()-start_time:.1f}s")

    # Save as dict for simple_viewer
    print("Saving model...")
    # Detach everything and save as standard dictionary
    save_dict = {k: v.detach() for k, v in splats.items()}
    torch.save({"splats": save_dict}, os.path.join(SCENE_DIR, "model.pt"))
    print("Done.")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
    main()