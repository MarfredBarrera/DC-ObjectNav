import os
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import numpy as np
import json
from src.utils import unprojection


class FeatureField(nn.Module):
    """
    Feature field that maps 3D positions to feature vectors.
    Uses tiny-cuda-nn for efficient hash encoding and MLP.
    Updated to model aleatoric uncertainty using Negative Log-Likelihood (NLL) loss.
    """
    
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.cfg = config
        self.device = device

        bounds_min, bounds_max = self.load_scene_bounds(os.path.join(self.cfg.output_dir, "transforms.json"))
        self.scene_bounds = torch.tensor([
            bounds_min,  # min
            bounds_max   # max
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
        
        # Generate a unique random seed for this instance
        random_seed = int(time() * 1e9) % (2**32 - 1)
        
        # Calculate encoding output dimension
        self.encoding_dim = encoding_config["n_levels"] * encoding_config["n_features_per_level"]
        self.feature_dim = config.hash_feature_dim

        # Output dims: Mean (feature_dim) + Log Variance (1 scalar)
        self.output_dim = self.feature_dim + 1
        
        # Create the encoding + network model
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,  # 3D positions (x, y, z)
            n_output_dims=self.output_dim, 
            encoding_config=encoding_config,
            network_config=network_config,
            seed=random_seed 
        )
        
        self.model.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.hash_lr,
            betas=(0.9, 0.99),
            eps=1e-15,
            weight_decay=1e-6)
    
    def load_scene_bounds(self, json_path):
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
    
    def forward(self, positions, normalize=False):
        """
        Query features at given 3D positions.
        Args:
            positions: (N, 3) tensor of 3D world positions
            normalize: whether to L2 normalize output mean (use False during training)
        Returns:
            mean: (N, feature_dim) predicted feature vectors
            variance: (N, 1) predicted variance (positive, via softplus)
        """
        # Normalize positions to [0, 1]
        normalized_pos = self.normalize_positions(positions)
        
        # Query the network - returns (N, feature_dim + 1)
        raw_output = self.model(normalized_pos.float()).float()
        mean = raw_output[..., :-1]
        raw_var = raw_output[..., -1:]
        
        # Apply softplus to enforce var > 0 (following Kendall & Gal 2017)
        # softplus(x) = log(1 + exp(x)), is smooth and always positive
        # Add minimum variance for numerical stability
        variance = F.softplus(raw_var,beta=0.25) + 1e-4
        
        # Only normalize mean if requested (for inference/similarity computation)
        if normalize:
            mean = mean / (mean.norm(dim=-1, keepdim=True) + 1e-8)
        
        return mean, variance

    def safe_normalize(self, features, dim=-1, eps=1e-6):
        """Safely normalize features, handling zero-norm cases."""
        norms = features.norm(dim=dim, keepdim=True)
        # Only normalize if norm is above threshold, otherwise return small random vector
        mask = norms > eps
        
        safe_norms = torch.clamp(norms, min=eps)
        normalized = torch.where(
            mask,
            features / safe_norms,
            torch.zeros_like(features)
        )
        return normalized

    def train_step(self, batch_points, batch_gt_features):
        """
        Single training step using Negative Log-Likelihood (NLL) Loss with Isotropic Covariance.
        
        Args:
            batch_points: (N, 3) tensor of 3D points
            batch_gt_features: (N, feature_dim) tensor of ground truth features

        Output:
            loss: scalar tensor representing NLL loss
        """

        # Forward pass - returns mean (N, D) and variance (N, 1)
        pred_mean, variance = self.forward(batch_points, normalize=False)
        
        gt_norm = self.safe_normalize(batch_gt_features)
        
        # --- Robust Negative Log Likelihood (NLL) Loss ---
        
        # Shape: (N, 1)
        sse = ((gt_norm - pred_mean) ** 2).sum(dim=-1, keepdim=True)
        
        # 4. Compute NLL for Isotropic Gaussian
        # Loss = 0.5 * (log(2*pi) + log(sigma^2) + SSE / sigma^2)
        constant_term = np.log(2 * np.pi)
        log_var = torch.log(variance)
        
        nll = 0.5 * (constant_term + log_var + sse / variance)
        loss = nll.mean()

        torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(loss):
            print(f"NaN loss detected! Skipping step...")
            return None
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping 
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        if torch.isnan(total_norm):
            print(f"NaN gradients! Skipping step...")
            return None
        
        self.optimizer.step()
        
        return loss.item() 
    
    def get_hashgrid_features(self, depth, c2w, intrinsics, return_uncertainty=False):
        """
        Get a full feature map for the given depth and camera pose.

        Args: 
        depth: (H, W) depth map
        c2w: (4, 4) camera-to-world transform
        intrinsics: tuple (fx, fy, cx, cy, H, W)
        return_uncertainty: If True, returns (features, uncertainty)

        Returns:
        feature_map: (H, W, feature_dim) feature map
        uncertainty_map (optional): (H, W) scalar uncertainty map (variance)
        """
        fx, fy, cx, cy, H, W = intrinsics
        feature_map = torch.zeros((H, W, self.feature_dim), device=self.device)
        uncertainty_map = torch.zeros((H, W), device=self.device)
        
        mask = (depth > 0.1) & (depth < 10.0)
        world_points = unprojection(depth, intrinsics, c2w, self.device, mask=mask)
        
        batch_size = self.cfg.hash_inference_batch_size
        all_features = []
        all_variances = []
        
        for i in range(0, world_points.shape[0], batch_size):
            batch_points = world_points[i:i+batch_size]
            with torch.no_grad():
                # Get mean and variance (variance is already positive via softplus)
                batch_mean, batch_var = self.forward(batch_points, normalize=True) 
                
                all_features.append(batch_mean)
                if return_uncertainty:
                    # batch_var is (N, 1), we flatten to (N)
                    all_variances.append(batch_var.squeeze(-1))
        
        all_features = torch.cat(all_features, dim=0)
        feature_map[mask] = all_features.to(feature_map.dtype)
        
        if return_uncertainty:
            all_variances = torch.cat(all_variances, dim=0)
            uncertainty_map[mask] = all_variances.to(uncertainty_map.dtype)
            return feature_map, uncertainty_map
        
        return feature_map
    
    def query_similarity(self, depth, c2w, intrinsics, text_query, clip_processor, clip_model):
        """
        Query semantic similarity using text prompt.
        """
        # Get feature map (ignore uncertainty for similarity query)
        feature_map = self.get_hashgrid_features(depth, c2w, intrinsics, return_uncertainty=False)
        
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
        print(f"FeatureField saved to {path}")
    
    def load(self, path):
        """Load model state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scene_bounds = checkpoint['scene_bounds']
        print(f"FeatureField loaded from {path}")