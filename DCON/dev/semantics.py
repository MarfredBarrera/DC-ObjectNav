import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import CLIPProcessor, CLIPModel

class SAM_CLIP_Semantics:
    def __init__(self, 
                 config,
                 device="cuda"):
        """
        Implementation of the SAM + CLIP labeling
        """
        self.device = device
        self.cfg = config
        
        # 1. Load SAM
        self.sam = sam_model_registry[self.cfg.SAM_model_type](checkpoint=self.cfg.SAM_checkpoint_path)
        self.sam.to(device=self.device)
        
        # "We use the 'everything' prompt in SAM to get various proposed regions."
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.cfg.points_per_side,
            pred_iou_thresh=self.cfg.pred_iou_thresh,
            stability_score_thresh=self.cfg.stability_score_thresh,
            crop_n_layers=self.cfg.crop_n_layers,
            crop_n_points_downscale_factor=self.cfg.crop_n_points_downscale_factor,
            min_mask_region_area=self.cfg.min_mask_region_area,  # Filter very small noise
        )
        
        # 2. Load CLIP
        self.clip_model = CLIPModel.from_pretrained(self.cfg.CLIP_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.cfg.CLIP_model_name, use_fast=True)
        
    def _pad_and_crop(self, image, bbox, expand_ratio=1.25, min_size=10):
        """
        "Each region is padded, cropped, and resized..."
        This function handles the padding and cropping logic.
        """
        x, y, w, h = [int(v) for v in bbox]
        H, W, _ = image.shape
        
        # Skip invalid bounding boxes
        if w <= 0 or h <= 0:
            return None
        
        # Calculate padding to expand context slightly
        pad_w = int(w * (expand_ratio - 1) / 2)
        pad_h = int(h * (expand_ratio - 1) / 2)
        
        # coordinates with padding
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(W, x + w + pad_w)
        y2 = min(H, y + h + pad_h)
        
        # Check if crop is valid and large enough to avoid ambiguous dimensions
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w <= 0 or crop_h <= 0 or crop_w < min_size or crop_h < min_size:
            return None
        
        crop = image[y1:y2, x1:x2]
        
        # Convert float32 [0, 1] to uint8 [0, 255] if needed
        if crop.dtype == np.float32 or crop.dtype == np.float64:
            crop = (crop * 255).astype(np.uint8)

        return Image.fromarray(crop)

    def extract_dense_features(self, image_rgb):
        """
        Generates the dense feature map (H, W, D).
        """
        orig_H, orig_W = image_rgb.shape[:2]
        
        # 1. SAM: "Get various proposed regions"
        masks = self.mask_generator.generate(image_rgb)
        
        if len(masks) == 0:
            return None

        # Sort masks by area (Largest -> Smallest)
        # This ensures specific objects (eyes) overwrite general ones (face)
        # when we assign pixels later.
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # 2. Prepare Crops: "Padded, cropped, and resized to 224x224"
        # Note: The CLIPProcessor handles the resizing to 224x224 internally
        crop_images = []
        valid_masks = []  # Keep track of masks with valid crops
        for mask_data in masks:
            crop = self._pad_and_crop(image_rgb, mask_data['bbox'])
            if crop is not None:  # Skip invalid crops
                crop_images.append(crop)
                valid_masks.append(mask_data)
        
        # Check if we have any valid crops
        if len(crop_images) == 0:
            print("Warning: No valid crops generated from masks")
            return None
            
        # 3. Batch CLIP Encoding
        all_embeddings = []
        
        # Process in batches for efficiency
        for i in range(0, len(crop_images), self.cfg.CLIP_label_batch_size):
            batch_crops = crop_images[i : i + self.cfg.CLIP_label_batch_size]
            
            # Processor handles resizing and normalization
            inputs = self.clip_processor(
                images=batch_crops, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # Get embeddings
                batch_embeds = self.clip_model.get_image_features(**inputs)
                # Normalize (important for cosine similarity)
                batch_embeds = batch_embeds / batch_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(batch_embeds)
                
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0) # (Num_Valid_Masks, Dim)
        
        # 4. Pixel-wise Assignment
        embed_dim = all_embeddings.shape[1]
        feature_map = torch.zeros((orig_H, orig_W, embed_dim), device=self.device)
        
        # We iterate and paint. Since we sorted Largest -> Smallest, 
        # the smaller masks will be painted LAST, overwriting the larger ones.
        for i, mask_data in enumerate(valid_masks):  # Use valid_masks instead of masks
            # Get the binary mask
            binary_mask = torch.from_numpy(mask_data['segmentation']).to(self.device)
            
            # Get the corresponding embedding
            embedding = all_embeddings[i] # (Dim,)
            
            # Assign embedding to all True pixels in the mask
            # Broadcasting: (N_pixels, ) = (Dim,)
            feature_map[binary_mask] = embedding
            
        return feature_map

    def query(self, feature_map, text_query):
        """
        Standard Cosine Similarity query.
        """
        inputs = self.clip_processor(text=[text_query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embed = self.clip_model.get_text_features(**inputs)
            text_embed /= text_embed.norm(dim=-1, keepdim=True)
            
        # Reshape map for matmul: (H*W, D)
        H, W, D = feature_map.shape
        flat_map = feature_map.view(-1, D)
        
        # Similarity
        sim = torch.matmul(flat_map, text_embed.T).view(H, W)
        normalized = (sim + 1.0) / 2.0
    
        
        return torch.clamp(normalized, 0.0, 1.0)