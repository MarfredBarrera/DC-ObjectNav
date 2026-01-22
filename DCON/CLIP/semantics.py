import torch
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel

def unprojection(depth, intrinsics, device):    
    fx, fy, cx, cy, H, W = intrinsics
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    z = depth
    x_c = (x - cx) * z / fx
    y_c = (y - cy) * z / fy
    return x_c, y_c, z

class SemanticFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch16", device="cuda"):
        """
        Initializes CLIP model and processor.
        We use CLIPVisionModel to access internal hidden states for dense features.
        """
        self.device = device
        self.model_name = model_name
        
        print(f"Loading CLIP model: {model_name}...")
        # We only need the Vision tower for image feature extraction
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # ViT specifics (patch size is usually 16 or 32)
        self.patch_size = 16 if "patch16" in model_name else 32
        
    def extract_dense_features(self, image_rgb):
        """
        Extracts per-pixel semantic features from an image.
        1. Resizes image to CLIP square input.
        2. Runs forward pass.
        3. Extracts patch tokens (local features) rather than the [CLS] token.
        4. Upsamples patch tokens back to original image dimensions.
        
        Args:
            image_rgb (np.array): HxWxC image (0-255, uint8)
        Returns:
            torch.Tensor: HxWxD dense feature map (D=512 or 768 usually)
        """
        orig_H, orig_W = image_rgb.shape[:2]
        
        # Prepare inputs (Processor handles resizing/norm to 224x224 usually)
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        
        # Get the pixel values that were actually fed to the network (usually 224x224)
        input_pixel_values = inputs.pixel_values
        
        with torch.no_grad():
            outputs = self.model(pixel_values=input_pixel_values, output_hidden_states=True)
            
            # Key Step: Access the last hidden state
            # Shape: (Batch, Sequence_Length, Hidden_Dim)
            # Sequence_Length = 1 (CLS token) + (H_clip/P * W_clip/P)
            last_hidden_state = outputs.last_hidden_state
            
            # Remove CLS token (index 0)
            patch_tokens = last_hidden_state[:, 1:, :] 
            
            # Reshape patches into a grid
            # For 224x224 image and patch size 16, grid is 14x14
            grid_size = int(np.sqrt(patch_tokens.shape[1]))
            feature_map = patch_tokens.view(1, grid_size, grid_size, -1) # (1, 14, 14, Dim)
            
            # Permute for grid_sample/interpolate: (N, C, H, W)
            feature_map = feature_map.permute(0, 3, 1, 2)
            
            # Upsample back to ORIGINAL image resolution
            # Bilinear interpolation is standard for feature maps
            dense_features = F.interpolate(
                feature_map, 
                size=(orig_H, orig_W), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Normalize features (Crucial for CLIP cosine similarity later)
            dense_features = F.normalize(dense_features, dim=1)
            
            # Reshape to (H, W, D) for easy indexing
            dense_features = dense_features.squeeze(0).permute(1, 2, 0)
            
        return dense_features

    def unproject_and_label(self, color_image, depth_image, intrinsic_matrix, c2w_matrix, depth_scale=1.0):
        """
        Unprojects depth to 3D and assigns the corresponding CLIP feature.
        
        Args:
            color_image (np.array): HxWx3
            depth_image (np.array): HxW
            intrinsic_matrix (torch.Tensor): 3x3
            c2w_matrix (torch.Tensor): 4x4
        Returns:
            points (torch.Tensor): (N, 3) World space coordinates
            features (torch.Tensor): (N, D) CLIP feature vectors
        """

        
        # 1. Extract Features
        # Shape: (H, W, Feature_Dim)
        dense_features = self.extract_dense_features(color_image)
        
        # 2. Unproject
        # Handle both numpy arrays and torch tensors
        if isinstance(depth_image, torch.Tensor):
            depth = depth_image.to(self.device) * depth_scale
        else:
            depth = torch.from_numpy(depth_image).to(self.device) * depth_scale

        x_cam, y_cam, z_cam = unprojection(depth, intrinsic_matrix, self.device)

        mask = (depth > 0.1) & (depth < 10.0) # Basic validity mask
        x_cam, y_cam, z_cam = x_cam[mask], y_cam[mask], z_cam[mask]
        
        
        cam_points = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=1)
        
        # 3. Transform to World Space
        c2w = c2w_matrix.to(self.device)
        world_points = (c2w @ cam_points.T).T[:, :3]
        
        # 4. Sample Features
        # Since we calculated features at H,W resolution, we can just index them using the mask
        valid_features = dense_features[mask]
        
        return world_points, valid_features

# -----------------------------------------------------------------------------
# Example Usage Block (Test this file directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import imageio.v2 as imageio
    
    # 1. Setup
    extractor = SemanticFeatureExtractor()
    
    # 2. Load Dummy Data (Replace with real paths)
    # H, W = 480, 640
    # rgb = imageio.imread("path/to/rgb.png")
    # depth = np.load("path/to/depth.npy")
    
    # Mock data for demonstration
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.rand(480, 640).astype(np.float32) * 5.0
    
    # Mock intrinsics/extrinsics
    intrinsics = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    c2w = torch.eye(4)
    
    # 3. Run Pipeline
    points, features = extractor.unproject_and_label(rgb, depth, intrinsics, c2w)
    
    print(f"Generated labeled point cloud:")
    print(f"Points Shape: {points.shape}")      # (N, 3)
    print(f"Features Shape: {features.shape}")  # (N, 768) for ViT-B/16
    
    # 4. (Optional) Save for HashGrid training
    # torch.save({"points": points.cpu(), "features": features.cpu()}, "labeled_pcd.pt")