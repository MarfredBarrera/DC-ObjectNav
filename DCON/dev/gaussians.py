import torch
import torch.nn.functional as F
import os
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from torchmetrics.image import StructuralSimilarityIndexMeasure
from dev.utils import rgb_to_sh, eval_sh

class GaussianSplatting:
    def __init__(self, config, init_points, init_colors, intrinsics_dict):
        self.cfg = config
        self.device = config.device
        
        # Unpack intrinsics
        self.H, self.W = intrinsics_dict['H'], intrinsics_dict['W']
        self.K = torch.tensor([
            [intrinsics_dict['fx'], 0, intrinsics_dict['cx']], 
            [0, intrinsics_dict['fy'], intrinsics_dict['cy']], 
            [0, 0, 1]
        ], device=self.device)

        # Initialize Parameters & Optimizers
        self.splats = torch.nn.ParameterDict()
        self.optimizers = {}
        self._init_parameters(init_points, init_colors)
        self._setup_optimizers_and_strategy()

        # Metrics
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def _init_parameters(self, points, colors):
        N = len(points)
        print(f"Initializing 3DGS model with {N} points...")
        
        params = {
            "means": torch.nn.Parameter(points.contiguous()),
            "scales": torch.nn.Parameter(torch.ones(N, 3, device=self.device).contiguous() * -2.5),
            "quats": torch.nn.Parameter(torch.zeros(N, 4, device=self.device).contiguous()),
            "opacities": torch.nn.Parameter(torch.ones(N, device=self.device).contiguous() * 0.5),
            "sh0": torch.nn.Parameter(rgb_to_sh(colors).unsqueeze(1).contiguous()), 
            "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device=self.device).contiguous()),
        }
        
        if self.cfg.train_uncertainty:
            params["uncertainty_sh"] = torch.nn.Parameter(
                torch.zeros(N, self.cfg.uncertainty_dim, device=self.device).contiguous()
            )
        
        with torch.no_grad():
            params["quats"][:, 0] = 1.0 

        self.splats = torch.nn.ParameterDict(params).to(self.device)

    def _setup_optimizers_and_strategy(self):
        self.optimizers = {
            "means": torch.optim.Adam([{"params": self.splats["means"], "lr": self.cfg.lr_means, "name": "means"}], eps=1e-15),
            "scales": torch.optim.Adam([{"params": self.splats["scales"], "lr": self.cfg.lr_scales, "name": "scales"}], eps=1e-15),
            "quats": torch.optim.Adam([{"params": self.splats["quats"], "lr": self.cfg.lr_quats, "name": "quats"}], eps=1e-15),
            "opacities": torch.optim.Adam([{"params": self.splats["opacities"], "lr": self.cfg.lr_opacities, "name": "opacities"}], eps=1e-15),
            "sh0": torch.optim.Adam([{"params": self.splats["sh0"], "lr": self.cfg.lr_sh0, "name": "sh0"}], eps=1e-15),
            "shN": torch.optim.Adam([{"params": self.splats["shN"], "lr": 0, "name": "shN"}], eps=1e-15),
        }
        
        if self.cfg.train_uncertainty:
            self.optimizers["uncertainty_sh"] = torch.optim.Adam(
                [{"params": self.splats["uncertainty_sh"], "lr": self.cfg.lr_uncertainty, "name": "uncertainty_sh"}], eps=1e-15
            )

        self.strategy = DefaultStrategy(
            verbose=False, # Reduce spam in main loop
            refine_start_iter=self.cfg.refine_start_iter,
            refine_stop_iter=self.cfg.refine_stop_iter,
            refine_every=self.cfg.refine_every,
            reset_every=self.cfg.reset_every,
            grow_grad2d=self.cfg.grow_grad2d,
            prune_opa=self.cfg.prune_opa
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()
        
        self.scheduler_means = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / self.cfg.iterations)
        )

    def _compute_uncertainty_loss(self, info, c2w):
        visible_mask = (info["radii"] > 0).squeeze(0)
        if not visible_mask.any(): return 0.0

        u_coeffs = self.splats["uncertainty_sh"][visible_mask]
        vis_means = self.splats["means"][visible_mask]
        
        cam_center = c2w[:3, 3] 
        dir_front = F.normalize(vis_means - cam_center, dim=-1)
        
        u_front = torch.sigmoid(eval_sh(u_coeffs, dir_front))
        u_back = torch.sigmoid(eval_sh(u_coeffs, -dir_front))
        
        lambda_u = 0.5
        return (1 - lambda_u) * u_front.mean() + lambda_u * (1.0 - u_back).mean()

    def step(self, step_idx, gt_image, c2w):
        """
        Performs one training step: Forward -> Loss -> Backward -> Optimizer -> Refinement
        """
        w2c = torch.inverse(c2w)

        # 1. LR Warmup
        if step_idx == 1000:
            for param_group in self.optimizers["shN"].param_groups:
                param_group['lr'] = self.cfg.lr_shN

        # 2. Rasterization
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        sh_degree = min(step_idx // 1000, 3)
        
        renders, alphas, info = rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"] / self.splats["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=w2c[None, ...],
            Ks=self.K[None, ...],
            width=self.W,
            height=self.H,
            sh_degree=sh_degree,
            packed=False,
            near_plane=self.cfg.near_plane
        )
        rgb_render = renders[0]

        # 3. Strategy Pre-Backward
        self.strategy.step_pre_backward(params=self.splats, optimizers=self.optimizers, state=self.strategy_state, step=step_idx, info=info)

        # 4. Loss Calculation
        l1loss = F.l1_loss(rgb_render, gt_image)
        ssimloss = 1.0 - self.ssim_metric(rgb_render.permute(2, 0, 1).unsqueeze(0), gt_image.permute(2, 0, 1).unsqueeze(0))
        loss = l1loss * self.cfg.l1_weight + ssimloss * self.cfg.ssim_weight

        if self.cfg.train_uncertainty:
            loss += self.cfg.uncertainty_weight * self._compute_uncertainty_loss(info, c2w)
        
        loss += self.cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

        # 5. Backward & Optimize
        loss.backward()
        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
            
        # 6. Strategy Post-Backward
        self.strategy.step_post_backward(params=self.splats, optimizers=self.optimizers, state=self.strategy_state, step=step_idx, info=info)
        self.scheduler_means.step()

        return loss.item(), len(self.splats["means"])

    def save(self, path):
        save_dict = {k: v.detach() for k, v in self.splats.items()}
        torch.save({"splats": save_dict}, path)