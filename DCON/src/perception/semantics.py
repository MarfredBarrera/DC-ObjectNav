import torch
import torch.nn.functional as F
import torchvision.transforms as T

from src.mask_clip.MaskCLIP import MaskCLIP


class MaskCLIPSemantics:
    """Dense per-pixel CLIP feature extraction using MaskCLIP value reparameterization.

    Replaces the old SAM+CLIP pipeline. Everything runs on GPU — no CPU transfers
    needed for feature extraction.

    The MaskCLIP approach extracts dense features by reparameterizing the last
    attention layer of the CLIP ViT to output value projections for every patch
    token instead of only the CLS token. These patch features live in the same
    embedding space as CLIP text features, enabling zero-shot dense similarity.

    Usage:
        sem = MaskCLIPSemantics(device="cuda")
        feats = sem.extract_dense_features(rgb_gpu)  # (H, W, 512)
        text  = sem.encode_text("a pillow")           # (1, 512)
    """

    # CLIP normalization constants (ImageNet-tuned for OpenAI CLIP)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, device="cuda", model_name="ViT-B/16", input_size=448):
        self.device = device
        self.input_size = input_size

        # Load the MaskCLIP model (downloads weights on first run)
        self.model = MaskCLIP(model_name=model_name, device=device)

        # GPU-only normalization transform (operates on CHW tensors, no PIL needed)
        self.normalize = T.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)

    @torch.inference_mode()
    def extract_dense_features(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Extract dense per-pixel CLIP features from an RGB image.

        Args:
            rgb_tensor: (H, W, 3) float32 tensor in [0, 1] range, on GPU.

        Returns:
            features: (H, W, 512) L2-normalized CLIP feature map on GPU.
        """
        H, W, _ = rgb_tensor.shape

        # (H, W, 3) → (3, H, W) → (1, 3, H, W)
        img = rgb_tensor.permute(2, 0, 1).unsqueeze(0)

        # Resize to MaskCLIP input size and apply CLIP normalization
        img = F.interpolate(img, size=(self.input_size, self.input_size),
                            mode="bilinear", align_corners=False)
        img = self.normalize(img.squeeze(0)).unsqueeze(0)  # normalize expects (C,H,W)

        # Extract dense patch features via value reparameterization
        # Returns (1, num_patches, 512) where num_patches = (input_size/16)^2
        patch_features = self.model.model.get_patch_encodings(img)

        # Reshape to spatial grid: (1, num_patches, 512) → (1, 512, grid, grid)
        num_patches = patch_features.shape[1]
        grid_size = int(num_patches ** 0.5)
        feature_grid = patch_features.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)

        # Bilinear upsample to original resolution: (1, 512, grid, grid) → (1, 512, H, W)
        feature_grid = F.interpolate(feature_grid, size=(H, W),
                                     mode="bilinear", align_corners=False)

        # (1, 512, H, W) → (H, W, 512)
        features = feature_grid.squeeze(0).permute(1, 2, 0)

        # L2 normalize per-pixel and cast to float32 (MaskCLIP runs in fp16 internally)
        features = features.float()
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)

        return features

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a text prompt into a CLIP embedding.

        Args:
            text: Natural language query string.

        Returns:
            text_features: (1, 512) L2-normalized text embedding on GPU.
        """
        text_tokens = self.model._tokenize(text)
        text_features = self.model.model.encode_text(text_tokens)
        text_features = text_features.float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        return text_features
