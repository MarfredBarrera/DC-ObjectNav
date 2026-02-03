import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np
import json
from utils import unprojection


class HashGrid(nn.Module):
    """
    HashGrid-based feature field that maps 3D positions to feature vectors.
    Uses tiny-cuda-nn for efficient hash encoding and MLP.
    """
    
    def __init__(self, config, device="cuda", transforms_json=None):
        super().__init__()
        self.cfg = config
        self.device = device
        
        # Scene bounds (will be updated during training or loaded from transforms.json)
        if transforms_json is not None:
            bounds_min, bounds_max = self.load_scene_bounds_from_json(transforms_json)
        else:
            bounds_min = [-5.0, -5.0, -5.0]
            bounds_max = [5.0, 5.0, 5.0]

        self.scene_bounds = torch.tensor([
            bounds_min,  # min
            bounds_max      # max
        ], device=device, dtype=torch.float32)
        
        # HashGrid encoding configuration
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": config.hash_n_levels,              # Number of resolution levels
            "n_features_per_level": config.hash_n_features_per_level,  # Features per level
            "log2_hashmap_size": config.hash_log2_hashmap_size,        # Hash table size
            "base_resolution": config.hash_base_resolution,            # Coarsest resolution
            "per_level_scale": config.hash_per_level_scale,            # Growth factor
        }
        
        # MLP network configuration
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": config.hash_n_neurons,
            "n_hidden_layers": config.hash_n_hidden_layers,
        }
        
        # Calculate encoding output dimension
        self.encoding_dim = encoding_config["n_levels"] * encoding_config["n_features_per_level"]
        
        # Feature output dimension (e.g., CLIP dimension)
        self.feature_dim = config.hash_feature_dim
        
        # Create the encoding + network model
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,  # 3D positions (x, y, z)
            n_output_dims=self.feature_dim,
            encoding_config=encoding_config,
            network_config=network_config
        )
        
        self.model.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.hash_lr,
            betas=(0.9, 0.99),
            eps=1e-15
        )
    
    def load_scene_bounds_from_json(self, json_path):
        """
        Load scene bounds from a transforms.json file.
        Args:
            json_path: Path to transforms.json file containing scene_bounds
        Returns:
            bounds_min: List of [x_min, y_min, z_min]
            bounds_max: List of [x_max, y_max, z_max]
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'scene_bounds' in data and data['scene_bounds'] is not None:
            bounds_min = data['scene_bounds']['min']
            bounds_max = data['scene_bounds']['max']
            print(f"Loaded scene bounds from {json_path}: min={bounds_min}, max={bounds_max}")
        else:
            print(f"Warning: No scene_bounds found in {json_path}, using default bounds")
            bounds_min = [-5.0, -5.0, -5.0]
            bounds_max = [5.0, 5.0, 5.0]

        return bounds_min, bounds_max
        
    def update_scene_bounds(self, points):
        """
        Update scene bounds based on observed 3D points.
        Args:
            points: (N, 3) tensor of 3D points
        """
        mins = points.min(dim=0)[0]
        maxs = points.max(dim=0)[0]
        
        # Add some padding
        padding = (maxs - mins) * 0.1
        mins = mins - padding
        maxs = maxs + padding
        
        # Update bounds
        self.scene_bounds[0] = torch.min(self.scene_bounds[0], mins)
        self.scene_bounds[1] = torch.max(self.scene_bounds[1], maxs)
        
    def normalize_positions(self, positions):
        """
        Normalize positions to [0, 1] based on scene bounds.
        Args:
            positions: (N, 3) tensor of 3D positions
        Returns:
            normalized positions in [0, 1]
        """
        normalized = (positions - self.scene_bounds[0]) / (self.scene_bounds[1] - self.scene_bounds[0])
        return torch.clamp(normalized, 0.0, 1.0)
    
    def forward(self, positions):
        """
        Query features at given 3D positions.
        Args:
            positions: (N, 3) tensor of 3D world positions
        Returns:
            features: (N, feature_dim) tensor of feature vectors
        """
        # Normalize positions to [0, 1]
        normalized_pos = self.normalize_positions(positions)
        
        # Query the network
        features = self.model(normalized_pos.float())
        
        # L2 normalize features (important for CLIP similarity)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        
        return features
    
    def train_step(self, depth, rgb, c2w, intrinsics, clip_features=None):
        """
        Single training step on an RGBD image.
        
        Args:
            depth: (H, W) depth map
            rgb: (H, W, 3) RGB image in [0, 1]
            c2w: (4, 4) camera-to-world transform
            intrinsics: tuple (fx, fy, cx, cy, H, W)
            clip_features: (H, W, D) CLIP feature map (optional, if None uses RGB)
            
        Returns:
            loss value
        """
        # 1. Unproject depth to 3D world points
        world_points = unprojection(depth, intrinsics, c2w, self.device)
        
        # 2. Get ground truth features
        fx, fy, cx, cy, H, W = intrinsics
        mask = (depth > 0.1) & (depth < 10.0)
        
        gt_features = clip_features[mask]
        
        # 3. Randomly sample points (to avoid OOM on full image)
        num_points = world_points.shape[0]
        if num_points > self.cfg.hash_train_batch_size:
            indices = torch.randperm(num_points, device=self.device)[:self.cfg.hash_train_batch_size]
            world_points = world_points[indices]
            gt_features = gt_features[indices]
        
        # 4. Forward pass
        pred_features = self.forward(world_points)
        
        # 5. Compute loss
        # Cosine similarity loss (1 - cosine similarity)
        # Normalize gt_features if using CLIP

        gt_features = gt_features / (gt_features.norm(dim=-1, keepdim=True) + 1e-8)
        loss = 1.0 - (pred_features * gt_features).sum(dim=-1).mean()

        
        # 6. Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def query_at_pixels(self, depth, c2w, intrinsics, pixel_coords=None):
        """
        Query features at specific pixel coordinates.
        
        Args:
            depth: (H, W) depth map
            c2w: (4, 4) camera-to-world transform
            intrinsics: tuple (fx, fy, cx, cy, H, W)
            pixel_coords: (N, 2) pixel coordinates (u, v), if None queries all valid pixels
            
        Returns:
            features: (N, feature_dim) feature vectors
            positions: (N, 3) corresponding 3D positions
        """
        fx, fy, cx, cy, H, W = intrinsics
        
        if pixel_coords is None:
            # Query all valid pixels
            world_points = unprojection(depth, intrinsics, c2w, self.device)
        else:
            # Query specific pixels
            u, v = pixel_coords[:, 0], pixel_coords[:, 1]
            z_c = depth[v, u]
            
            x_c = (u - cx) * z_c / fx
            y_c = (v - cy) * z_c / fy
            
            cam_points = torch.stack([x_c, y_c, z_c, torch.ones_like(z_c)], dim=1)
            world_points = (c2w @ cam_points.T).T[:, :3]
        
        # Query features
        with torch.no_grad():
            features = self.forward(world_points)
        
        return features, world_points
    
    def render_feature_map(self, depth, c2w, intrinsics):
        """
        Render a full feature map for visualization.
        
        Args:
            depth: (H, W) depth map
            c2w: (4, 4) camera-to-world transform  
            intrinsics: tuple (fx, fy, cx, cy, H, W)
            
        Returns:
            feature_map: (H, W, feature_dim) rendered feature map
        """
        fx, fy, cx, cy, H, W = intrinsics
        
        # Create output map
        feature_map = torch.zeros((H, W, self.feature_dim), device=self.device)
        
        # Get valid mask
        mask = (depth > 0.1) & (depth < 10.0)
        
        # Unproject valid points
        world_points = unprojection(depth, intrinsics, c2w, self.device)
        
        # Query features in batches
        batch_size = self.cfg.hash_inference_batch_size
        all_features = []
        
        for i in range(0, world_points.shape[0], batch_size):
            batch_points = world_points[i:i+batch_size]
            with torch.no_grad():
                batch_features = self.forward(batch_points)
            all_features.append(batch_features)
        
        all_features = torch.cat(all_features, dim=0)
        
        # Assign to feature map
        feature_map[mask] = all_features
        
        return feature_map
    
    def query_similarity(self, depth, c2w, intrinsics, text_query, clip_processor, clip_model):
        """
        Query semantic similarity using text prompt.
        
        Args:
            depth: (H, W) depth map
            c2w: (4, 4) camera-to-world transform
            intrinsics: tuple (fx, fy, cx, cy, H, W)
            text_query: text string for CLIP query
            clip_processor: CLIP processor
            clip_model: CLIP model
            
        Returns:
            similarity_map: (H, W) similarity scores
        """
        # Get feature map
        feature_map = self.render_feature_map(depth, c2w, intrinsics)
        
        # Get text embedding
        inputs = clip_processor(text=[text_query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embed = clip_model.get_text_features(**inputs)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        H, W, D = feature_map.shape
        flat_map = feature_map.view(-1, D)
        sim = torch.matmul(flat_map, text_embed.T).view(H, W)
        
        # Normalize to [0, 1]
        normalized = (sim + 1.0) / 2.0
        return torch.clamp(normalized, 0.0, 1.0)
    
    def save(self, path):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scene_bounds': self.scene_bounds,
        }, path)
        print(f"HashGrid saved to {path}")
    
    def load(self, path):
        """Load model state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scene_bounds = checkpoint['scene_bounds']
        print(f"HashGrid loaded from {path}")