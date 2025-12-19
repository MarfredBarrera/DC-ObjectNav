import os
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio.v2 as imageio
import cv2
import clip
import torchvision.transforms as T
import matplotlib.cm as cm
import tinycudann as tcnn
from transformers import CLIPProcessor, CLIPModel

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCENE_DIR = "/workspace/DCON/output/current_scene"
DEVICE = "cuda"
ITERATIONS = 3000

QUERY_TEXT = "kitchen" 

# -----------------------------------------------------------------------------
# 1. Semantic Components (CLIP & TCNN HashGrid)
# -----------------------------------------------------------------------------

from transformers import CLIPProcessor, CLIPModel

class DenseCLIPExtractor(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch16', device='cuda'):
            super().__init__()
            self.device = device
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()


    @torch.no_grad()
    def get_dense_features(self, images):
        """
        images: [B, 3, H, W] tensor (0-1 range) or list of numpy images
        """
        B, C, H_orig, W_orig = images.shape
        
        # 1. Process Inputs (Auto-Resizing & Normalization)
        # We must convert tensor [0-1] to list of images or handle inputs carefully.
        # HF Processor expects 0-255 inputs usually, but handles tensors if configured.
        # Simplest way: Convert tensor back to 0-255 uint8 for the processor
        images_uint8 = (images * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        
        # This handles resizing to (224, 224) automatically
        inputs = self.processor(images=list(images_uint8), return_tensors="pt").to(self.device)
        
        # 2. Forward Pass (Get Vision Transformer Outputs)
        vision_outputs = self.model.vision_model(**inputs)
        last_hidden = vision_outputs.last_hidden_state  # [B, 197, 768]
        
        # 3. Project to Shared Latent Space (768 -> 512)
        # We must project so it matches the text embeddings!
        projected_tokens = self.model.visual_projection(last_hidden) # [B, 197, 512]
        
        # 4. Remove CLS Token & Reshape
        patch_tokens = projected_tokens[:, 1:, :] # [B, 196, 512]
        
        # Calculate grid size (14x14 for ViT-B/16)
        n_patches = patch_tokens.shape[1]
        grid_size = int(math.sqrt(n_patches)) # Should be 14
        
        feat_map = patch_tokens.permute(0, 2, 1).view(B, 512, grid_size, grid_size)
        
        # 5. Upsample to Original Resolution
        feat_map_up = F.interpolate(
            feat_map, 
            size=(H_orig, W_orig), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize vectors (crucial for Cosine Similarity)
        feat_map_up = feat_map_up / feat_map_up.norm(dim=1, keepdim=True)
        
        return feat_map_up.permute(0, 2, 3, 1) # [B, H, W, 512]
    
    @torch.no_grad()
    def encode_text(self, text_list):
        """
        Encodes a list of text strings into 512-dim normalized vectors.
        """
        if isinstance(text_list, str):
            text_list = [text_list]
            
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True).to(self.device)
        
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

class TCNNSemanticField(nn.Module):
    def __init__(self, output_dim=512, aabb_min=None, aabb_max=None):
        super().__init__()
        
        # Default to a safe large box if not provided (-10m to +10m)
        if aabb_min is None: aabb_min = torch.tensor([-10, -10, -10])
        if aabb_max is None: aabb_max = torch.tensor([10, 10, 10])
        
        # Register buffers so they move to GPU automatically with the model
        self.register_buffer("aabb_min", aabb_min)
        self.register_buffer("aabb_max", aabb_max)
        self.register_buffer("aabb_size", aabb_max - aabb_min)
        
        # TCNN Config
        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        }
        
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=output_dim,
            encoding_config=config["encoding"],
            network_config=config["network"]
        )

    def forward(self, x):
        # x: [N, 3] in Habitat/OpenCV World Coordinates
        
        # 1. Normalize to [0, 1] based on Scene Bounds
        # x_norm = (x - min) / (max - min)
        x_norm = (x - self.aabb_min) / self.aabb_size
        
        # 2. Guard against outliers (points outside the box)
        # TCNN undefined behavior outside [0,1], so we clamp.
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        
        # 3. Query TCNN
        out = self.model(x_norm)
        
        # 4. Cast to float32 (TCNN returns float16)
        return out.float()

class SemanticReplayBuffer:
    def __init__(self, max_points=200_000, device="cuda"):
        self.max_points = max_points
        self.device = device
        self.ptr = 0
        self.size = 0
        self.coords = torch.zeros((max_points, 3), device=device)
        self.features = torch.zeros((max_points, 512), device=device)

    def add(self, new_coords, new_features):
        N = new_coords.shape[0]
        if N == 0: return
        indices = torch.arange(self.ptr, self.ptr + N, device=self.device) % self.max_points
        self.coords[indices] = new_coords
        self.features[indices] = new_features
        self.ptr = (self.ptr + N) % self.max_points
        self.size = min(self.size + N, self.max_points)

    def sample(self, batch_size=1024):
        if self.size == 0: return None, None
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.coords[idx], self.features[idx]

# -----------------------------------------------------------------------------
# 2. Data Loaders
# -----------------------------------------------------------------------------
def load_scene_data(data_dir, device="cuda"):
    json_path = os.path.join(data_dir, "transforms.json")
    with open(json_path, 'r') as f: meta = json.load(f)

    frames = meta['frames']
    img_0 = imageio.imread(os.path.join(data_dir, frames[0]['file_path']))
    H, W = img_0.shape[:2]
    
    fov_x = meta['camera_angle_x']
    fx = 0.5 * W / math.tan(0.5 * fov_x)
    fy = fx
    cx, cy = W / 2.0, H / 2.0

    gt_images, gt_depths, c2w_matrices = [], [], []
    convert_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    print(f"Loading {len(frames)} frames...")
    for frame in frames:
        rgb_path = os.path.join(data_dir, frame['file_path'])
        rgb = imageio.imread(rgb_path)
        gt_images.append(torch.from_numpy(rgb).float().to(device) / 255.0)

        depth_name = os.path.basename(frame['file_path']).replace("rgb", "depth").replace(".png", ".npy")
        depth = np.load(os.path.join(data_dir, "depth_data", depth_name))
        if depth.shape[:2] != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        gt_depths.append(torch.from_numpy(depth).float().to(device))

        c2w_hab = np.array(frame['transform_matrix'])
        c2w_cv = c2w_hab @ convert_mat
        c2w_matrices.append(torch.from_numpy(c2w_cv).float().to(device))

    return torch.stack(gt_images), torch.stack(gt_depths), torch.stack(c2w_matrices), (fx, fy, cx, cy, H, W)

def init_point_cloud_from_depth(gt_images, gt_depths, c2ws, intrinsics, num_init_frames=8):
    fx, fy, cx, cy, H, W = intrinsics
    points_list = []
    
    indices = np.linspace(0, len(gt_images)-1, num_init_frames, dtype=int)
    y, x = torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing='ij')
    
    print("Generating geometry context...")
    for idx in indices:
        depth = gt_depths[idx]
        c2w = c2ws[idx]

        points_list.append(unprojection(depth, c2w, intrinsics))

    full_points = torch.cat(points_list, dim=0)
    if full_points.shape[0] > 200_000:
        indices = torch.randperm(full_points.shape[0])[:200_000]
        full_points = full_points[indices]
        
    return full_points


def unprojection(depth, c2w, intrinsics):

    fx, fy, cx, cy, H, W = intrinsics
    y, x = torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing='ij')
    mask = (depth > 0.1) & (depth < 10.0)
    
    z_c = depth[mask]
    x_c = (x[mask] - cx) * depth[mask] / fx
    y_c = (y[mask] - cy) * depth[mask] / fy
    
    cam_points = torch.stack([x_c, y_c, z_c, torch.ones_like(z_c)], dim=1)
    world_points = (c2w @ cam_points.T).T
    return world_points[:, :3]

# -----------------------------------------------------------------------------
# 3. Visualization Helpers
# -----------------------------------------------------------------------------
# def get_text_embedding(text_query, clip_extractor, device):
#     # Use the processor and model from the extractor we just built
#     processor = clip_extractor.processor
#     model = clip_extractor.model
    
#     inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)
    
#     with torch.no_grad():
#         text_features = model.get_text_features(**inputs)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
#     return text_features

def apply_colormap(scores, cmap_name='turbo'):
    scores = torch.clamp(scores, 0.0, 1.0)
    scores_np = scores.cpu().numpy()
    colormap = cm.get_cmap(cmap_name)
    colors_np = colormap(scores_np)[:, :3]
    return torch.from_numpy(colors_np).float()

def compute_scene_bounds(c2ws, padding=2.0):
    """
    Computes a cubic bounding box around the camera trajectory.
    padding: Extra space (meters) to account for geometry visible beyond the camera center.
    """
    # Extract translation vectors (camera positions)
    cam_pos = c2ws[:, :3, 3] # [N, 3]
    
    min_xyz = cam_pos.min(dim=0)[0]
    max_xyz = cam_pos.max(dim=0)[0]
    
    # Add padding (e.g., 2m for walls visible from the camera)
    min_xyz -= padding
    max_xyz += padding
    
    # Make it a Cube (Uniform scaling)
    # This prevents distorting the aspect ratio of the voxels
    size = max_xyz - min_xyz
    max_dim = size.max()
    center = (min_xyz + max_xyz) / 2
    
    final_min = center - max_dim / 2
    final_max = center + max_dim / 2
    
    print(f"Computed Scene Bounds:\n Min: {final_min.tolist()}\n Max: {final_max.tolist()}")
    return final_min, final_max
# -----------------------------------------------------------------------------
# 4. Main Development Loop
# -----------------------------------------------------------------------------
def main():
    print("--- Starting Semantic Field Development Mode (TCNN Backend) ---")
    
    # 1. Load Data
    gt_images, gt_depths, c2ws, intrinsics = load_scene_data(SCENE_DIR, DEVICE)
    
    # 2. Compute Bounds from Data
    # We use the trajectory to define the "World" for the HashGrid
    scene_min, scene_max = compute_scene_bounds(c2ws, padding=4.0) # 4m padding is safe for indoor
    
    # 3. Init Geometry (Dummy)
    fixed_means = init_point_cloud_from_depth(
        gt_images, gt_depths, c2ws, intrinsics, num_init_frames=len(gt_images)
    )
    
    # 4. Init Components (PASS BOUNDS HERE)
    semantic_field = TCNNSemanticField(
        output_dim=512, 
        aabb_min=scene_min, 
        aabb_max=scene_max
    ).to(DEVICE)
    
    sem_optimizer = torch.optim.Adam(semantic_field.parameters(), lr=1e-3)
    replay_buffer = SemanticReplayBuffer(max_points=200_000, device=DEVICE)
    clip_extractor = DenseCLIPExtractor(device=DEVICE)
    
    # 4. Prepare Text Query
    print(f"--- Encoding Query: '{QUERY_TEXT}' ---")
    # target_emb = get_text_embedding(QUERY_TEXT, clip_extractor.model, DEVICE)
#
    target_emb = clip_extractor.encode_text(QUERY_TEXT)
    start_time = time.time()
    
    for step in range(ITERATIONS):
        
        # --- A. Online Data Gathering (Simulated) ---
        if step % 20 == 0:
            current_idx = (step // 20) % len(gt_images)
            
            with torch.no_grad():
                img = gt_images[current_idx]
                depth = gt_depths[current_idx]
                c2w = c2ws[current_idx]
                
                # Run CLIP
                img_input = img.permute(2, 0, 1).unsqueeze(0)
                clip_map = clip_extractor.get_dense_features(img_input).squeeze(0)
                
                # Unproject
                stride = 4
                d_sub = depth[::stride, ::stride]
                c_sub = clip_map[::stride, ::stride]

                unproj_points = unprojection(d_sub, c2w, intrinsics)
                unproj_feats = c_sub[(d_sub > 0.1) & (d_sub < 10.0)]

                replay_buffer.add(unproj_points, unproj_feats)


                # ys, xs = torch.meshgrid(torch.arange(0, H, stride, device=DEVICE), 
                #                         torch.arange(0, W, stride, device=DEVICE), indexing='ij')
                
                # valid = (d_sub > 0.1) & (d_sub < 10.0)
                # if valid.sum() > 0:
                #     z_p = d_sub[valid]
                #     x_p = (xs[valid] - cx) * z_p / fx
                #     y_p = (ys[valid] - cy) * z_p / fy
                #     cam_xyz = torch.stack([x_p, y_p, z_p], dim=1)
                    
                #     R, t = c2w[:3, :3], c2w[:3, 3]
                #     world_xyz = (cam_xyz @ R.T) + t
                #     features = c_sub[valid]
                    
                #     replay_buffer.add(world_xyz, features)

        # --- B. Continuous Training ---
        s_coords, s_targets = replay_buffer.sample(batch_size=4096) # Can handle larger batches with TCNN
        loss_val = 0.0
        
        if s_coords is not None:
            s_pred = semantic_field(s_coords)
            loss = 1.0 - F.cosine_similarity(s_pred, s_targets, dim=-1).mean()
            loss_val = loss.item()
            
            sem_optimizer.zero_grad()
            loss.backward()
            sem_optimizer.step()

        # --- C. Visualization (Save Splat) ---
        if step % 200 == 0 and s_coords is not None:
            print(f"Step {step:04d} | Loss: {loss_val:.4f} | Generating Heatmap...")
            
            with torch.no_grad():
                # Query field in chunks
                chunk_size = 100_000 # TCNN is fast, larger chunks ok
                colors_list = []
                
                for i in range(0, len(fixed_means), chunk_size):
                    batch_means = fixed_means[i:i+chunk_size]
                    feats = semantic_field(batch_means)
                    feats = F.normalize(feats, dim=-1)
                    
                    # Compute Cosine Similarity
                    similarity = (feats @ target_emb.T).squeeze(1)
                    
                    # Normalize for viz (0.20 - 0.35 is typical CLIP range for raw matches)
                    similarity = (similarity - 0.20) / (0.35 - 0.20)
                    rgb = apply_colormap(similarity, cmap_name='turbo').to(DEVICE)
                    colors_list.append(rgb)
                
                sem_colors = torch.cat(colors_list, dim=0)
                
                # Convert to SH (Degree 0)
                C0 = 0.28209479177387814
                sh0_sem = (sem_colors - 0.5) / C0
                
                # Save Splat Dictionary
                save_dict = {
                    "means": fixed_means,
                    "scales": torch.ones(len(fixed_means), 3, device=DEVICE) * -2.5,
                    "quats": torch.zeros(len(fixed_means), 4, device=DEVICE),
                    "opacities": torch.ones(len(fixed_means), device=DEVICE) * 0.8,
                    "sh0": sh0_sem.unsqueeze(1),
                    "shN": torch.zeros(len(fixed_means), 15, 3, device=DEVICE)
                }
                save_dict["quats"][:, 0] = 1.0
                
                filename = f"query_{QUERY_TEXT}.pt"
                torch.save({"splats": save_dict}, os.path.join(SCENE_DIR, filename))
                print(f"Saved {filename}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    main()