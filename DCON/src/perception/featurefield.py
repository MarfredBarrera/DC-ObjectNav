import os
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import numpy as np
import json
from src.perception.utils import unprojection


class EvidentialFeatureField(nn.Module):
    """
    Feature field that maps 3D positions to feature vectors.
    Uses tiny-cuda-nn for efficient hash encoding and MLP.
    Updated to use Evidential Regression for single-pass Aleatoric and Epistemic Uncertainty.
    """

    # Sigmoid input shift for the scalar (feature_dim==1) CLIPSeg-score mode:
    # see the cold-start comment in forward().
    COLD_START_BIAS = 4.0

    def __init__(self, config, scene_bounds, device="cuda"):
        super().__init__()
        self.cfg = config
        self.device = device

        # Scene bounds are mandatory to ensure consistency
        bounds_min = scene_bounds[0]
        bounds_max = scene_bounds[1]
        self.scene_bounds = torch.tensor([
            bounds_min,  # min
            bounds_max   # max
        ], device=device, dtype=torch.float32)
        
        # HashGrid encoding configuration
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": config.hash_n_levels,              
            "n_features_per_level": config.hash_n_features_per_level,  
            "log2_hashmap_size": config.hash_log2_hashmap_size,        
            "base_resolution": config.hash_base_resolution,            
            "per_level_scale": config.hash_per_level_scale,            
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

        # Output dims: Mean (feature_dim) + Evidential Params (v, alpha, beta)
        # Using isotropic uncertainty (1 scalar set for the whole feature vector)
        self.output_dim = self.feature_dim + 3
        
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
            weight_decay=1e-5)

    def normalize_positions(self, positions):
        """Normalize positions to [0, 1] based on scene bounds."""
        normalized = (positions - self.scene_bounds[0]) / (self.scene_bounds[1] - self.scene_bounds[0])
        return torch.clamp(normalized, 0.0, 1.0)
    
    def forward(self, positions, normalize=False):
        """
        Query features at given 3D positions and predict evidential uncertainties.
        
        Returns:
            gamma: (N, feature_dim) predicted feature vectors (mean)
            aleatoric: (N, 1) predicted aleatoric uncertainty
            epistemic: (N, 1) predicted epistemic uncertainty
            evidential_params: Tuple of (v, alpha, beta) for loss calculation
        """
        # Normalize positions to [0, 1]
        normalized_pos = self.normalize_positions(positions)
        
        # Query the network - returns (N, feature_dim + 3)
        raw_output = self.model(normalized_pos.float()).float()
        
        # Extract parameters for Normal-Inverse-Gamma distribution
        gamma = raw_output[..., :-3]

        # Scalar CLIPSeg target is provably bounded in [0, 1] (it's a sigmoid
        # output) -- unlike the MaskCLIP embedding case, bound the prediction
        # to match. Without this, gamma has no output activation and can
        # drift to an arbitrarily large magnitude under sustained gradient
        # pressure (e.g. many steps over a near-uniform background batch),
        # which blows up error_squared = (gt - gamma)**2 and cascades into
        # NaN through the NLL's log terms. Bounding gamma caps error_squared
        # at 1 in the worst case and closes that instability off at the source.
        if self.feature_dim == 1:
            # Cold-start bias: an untrained region evaluates to whatever the
            # hash-grid encoding + MLP happen to produce with no gradient
            # signal, which is arbitrary -- observed empirically landing near
            # +1.0 similarity in unexplored territory, i.e. "everything is the
            # target" by default. Shifting the sigmoid's input means a raw
            # (cold) output of 0 maps to a low score instead of a coin-flip
            # 0.5, so a location reads "not relevant" until training
            # accumulates real positive evidence to push it up.
            gamma = torch.sigmoid(gamma - self.COLD_START_BIAS)

        # Apply constraints based on Evidential Deep Learning literature
        # v > 0, alpha > 1, beta > 0. v/beta floors are 1e-2, not the more
        # typical 1e-6: a scalar CLIPSeg target is frequently near-uniform
        # (e.g. background-only frames before the object is in view), which
        # drives beta toward 0 as the network claims near-zero residual
        # variance; omega = 2*beta*(1+v) then underflows and log(omega) blows
        # up in the NLL below. The larger floor keeps omega bounded away from
        # 0 and closes that instability off at the source.
        v = F.softplus(raw_output[..., -3:-2]) + 1e-2
        # alpha ceiling: unbounded "evidence" lets the network drive alpha
        # arbitrarily high while omega sits at its floor (common on a
        # near-uniform scalar batch, e.g. background-only frames) -- the
        # -alpha*log(omega) term's gradient w.r.t. omega scales as -alpha/omega,
        # which explodes as alpha grows with omega pinned near its floor.
        # Capping alpha bounds that ratio regardless of how long training
        # sees a low-variance batch.
        alpha = torch.clamp(F.softplus(raw_output[..., -2:-1]) + 1.0 + 1e-6, max=100.0)
        beta = F.softplus(raw_output[..., -1:]) + 1e-2
        
        # Calculate Uncertainties analytically 
        # Aleatoric (data uncertainty) = expected variance = beta / (alpha - 1)
        aleatoric = beta / (alpha - 1.0)
        
        # Epistemic (model uncertainty) = variance of the mean = beta / (v * (alpha - 1))
        epistemic = beta / (v * (alpha - 1.0))
        
        # Only normalize mean if requested
        if normalize:
            gamma = self.safe_normalize(gamma)
        
        return gamma, aleatoric, epistemic, (v, alpha, beta)

    def safe_normalize(self, features, dim=-1, eps=1e-6):
        """Safely normalize features, handling zero-norm cases.

        A no-op when the last dim is 1: a scalar target (e.g. a CLIPSeg
        relevance score) isn't a direction, and L2-normalizing it would
        collapse every value to its sign (+1.0, since sigmoid outputs are
        always >= 0), destroying the entire training signal.
        """
        if features.shape[-1] == 1:
            return features
        norms = features.norm(dim=dim, keepdim=True)
        mask = norms > eps
        safe_norms = torch.clamp(norms, min=eps)
        normalized = torch.where(
            mask,
            features / safe_norms,
            torch.zeros_like(features)
        )
        return normalized

    def train_step(self, batch_points, batch_gt_features, lambda_reg=0.01):
        """
        Single training step using the Evidential Regression Loss.
        """

        # Forward pass 
        gamma, _, _, (v, alpha, beta) = self.forward(batch_points, normalize=False)
        gt_norm = self.safe_normalize(batch_gt_features)
        
        # --- Evidential NLL Loss (Student-t Marginal Likelihood) ---
        # Squared error (N, D)
        error_squared = (gt_norm - gamma) ** 2
        
        # Omega helper term
        omega = 2 * beta * (1 + v)
        
        # Student-t NLL logic
        nll = (
            0.5 * torch.log(np.pi / v) 
            - alpha * torch.log(omega) 
            + (alpha + 0.5) * torch.log(omega + v * error_squared) 
            + torch.lgamma(alpha) 
            - torch.lgamma(alpha + 0.5)
        )
        
        # --- Evidential Regularizer ---
        # Penalize high evidence (low uncertainty) when the error is high
        error_l1 = torch.abs(gt_norm - gamma)
        reg = error_l1 * (2 * v + alpha)
        
        # Total Loss
        loss = (nll + lambda_reg * reg).mean()

        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

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
    
    def get_hashgrid_features(self, depth, c2w, intrinsics, return_uncertainties=False):
        """
        Get a full feature map for the given depth and camera pose.
        Now returns BOTH Aleatoric and Epistemic maps if requested.
        """
        fx, fy, cx, cy, H, W = intrinsics
        feature_map = torch.zeros((H, W, self.feature_dim), device=self.device)
        
        if return_uncertainties:
            aleatoric_map = torch.zeros((H, W), device=self.device)
            epistemic_map = torch.zeros((H, W), device=self.device)
        
        mask = (depth > self.cfg.min_sensor_dist) & (depth < self.cfg.max_sensor_dist)
        world_points = unprojection(depth, intrinsics, c2w, self.device, mask=mask)
        
        batch_size = self.cfg.hash_inference_batch_size
        all_features = []
        
        if return_uncertainties:
            all_aleatoric = []
            all_epistemic = []
        
        for i in range(0, world_points.shape[0], batch_size):
            batch_points = world_points[i:i+batch_size]
            with torch.no_grad():
                # Get mean, aleatoric, epistemic
                batch_mean, batch_alea, batch_epis, _ = self.forward(batch_points, normalize=True) 
                
                all_features.append(batch_mean)
                if return_uncertainties:
                    all_aleatoric.append(batch_alea.squeeze(-1))
                    all_epistemic.append(batch_epis.squeeze(-1))
        
        all_features = torch.cat(all_features, dim=0)
        feature_map[mask] = all_features.to(feature_map.dtype)
        
        if return_uncertainties:
            all_aleatoric = torch.cat(all_aleatoric, dim=0)
            all_epistemic = torch.cat(all_epistemic, dim=0)
            aleatoric_map[mask] = all_aleatoric.to(aleatoric_map.dtype)
            epistemic_map[mask] = all_epistemic.to(epistemic_map.dtype)
            return feature_map, aleatoric_map, epistemic_map
        
        return feature_map
    
    def save(self, path):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scene_bounds': self.scene_bounds,
        }, path)
        print(f"EvidentialFeatureField saved to {path}")
    
    def load(self, path, load_optimizer=True):
        """Load model state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scene_bounds = checkpoint['scene_bounds']
        print(f"EvidentialFeatureField loaded from {path} (optimizer={'restored' if load_optimizer else 'fresh'})")