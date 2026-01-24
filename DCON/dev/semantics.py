import torch
import torch.nn.functional as F
import numpy as np
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class SemanticFeatureExtractor:
    def __init__(self, model_name="CIDAS/clipseg-rd64-refined", device="cuda"):
        """
        Initialize CLIPSeg model for dense semantic features.
        
        Args:
            model_name: CLIPSeg model variant (default: refined version)
        """
        self.device = device
        print(f"Loading CLIPSeg model: {model_name}...")
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device)
        self.processor = CLIPSegProcessor.from_pretrained(model_name, use_fast=True)
        self.model.eval()
        print("CLIPSeg model loaded successfully!")
        
    def extract_dense_features_segmentation(self, image_rgb, text_query):
        """
        Extracts dense SEGMENTATION scores using CLIPSeg's decoder.
        This is trained for binary segmentation - it isolates ONLY the queried object
        and suppresses everything else (even semantically related objects).
        
        Use this when you want: "show me exactly where the pillow is"
        
        Args:
            image_rgb: torch.Tensor (H, W, 3) in range [0, 1] or numpy array (H, W, 3) uint8
            text_query: str - the semantic query (e.g., "a pillow")
        Returns:
            similarity_map: torch.Tensor (H, W) with segmentation scores in [0, 1]
        """
        # Convert to numpy uint8 for processor
        if isinstance(image_rgb, torch.Tensor):
            if image_rgb.max() <= 1.0:
                image_np = (image_rgb.cpu().numpy() * 255).astype(np.uint8)
            else:
                image_np = image_rgb.cpu().numpy().astype(np.uint8)
        else:
            image_np = image_rgb
            
        orig_H, orig_W = image_np.shape[:2]
        
        # Process with text query
        inputs = self.processor(
            images=image_np,
            text=text_query,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get logits and upsample to original size
            logits = outputs.logits.unsqueeze(1)
            upsampled = F.interpolate(
                logits,
                size=(orig_H, orig_W),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Apply sigmoid to get probabilities [0, 1]
            similarity_map = torch.sigmoid(upsampled)
            
        return similarity_map
    
    def extract_dense_features_similarity(self, image_rgb, text_query):
        """
        Extracts dense SEMANTIC SIMILARITY scores using CLIP embeddings.
        This computes cosine similarity, so semantically related objects also get
        high scores (e.g., "pillow" query will also activate "couch" and "bed").
        
        Use this when you want: "show me things semantically related to pillow"
        
        Args:
            image_rgb: torch.Tensor (H, W, 3) in range [0, 1] or numpy array (H, W, 3) uint8
            text_query: str - the semantic query
        Returns:
            similarity_map: torch.Tensor (H, W) with similarity scores in [0, 1]
        """
        # Convert to numpy uint8
        if isinstance(image_rgb, torch.Tensor):
            if image_rgb.max() <= 1.0:
                image_np = (image_rgb.cpu().numpy() * 255).astype(np.uint8)
            else:
                image_np = image_rgb.cpu().numpy().astype(np.uint8)
        else:
            image_np = image_rgb
            
        orig_H, orig_W = image_np.shape[:2]
        
        # Process image
        image_inputs = self.processor(images=image_np, return_tensors="pt").to(self.device)
        
        # Process text
        text_inputs = self.processor(text=[text_query], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            # Extract image features from CLIP vision encoder
            vision_outputs = self.model.clip.vision_model(
                pixel_values=image_inputs.pixel_values,
                output_hidden_states=True
            )
            
            # Get normalized patch features
            last_hidden = vision_outputs.last_hidden_state
            normalized_hidden = self.model.clip.vision_model.post_layernorm(last_hidden)
            patch_tokens = normalized_hidden[:, 1:, :]  # Remove CLS token
            
            # Project to joint embedding space
            visual_features = self.model.clip.visual_projection(patch_tokens)
            visual_features = F.normalize(visual_features, p=2, dim=-1)
            
            # Extract text features
            text_outputs = self.model.clip.text_model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
            text_embeds = text_outputs.pooler_output
            text_features = self.model.clip.text_projection(text_embeds)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            # Compute cosine similarity for each patch
            # THE BUG: For some reason, cosine similarity is inverted in CLIPSeg
            # Lower values = more similar, higher values = less similar
            # This is opposite of standard CLIP behavior
            # Solution: Negate the similarity to flip it
            similarity = torch.matmul(visual_features, text_features.T).squeeze()  # (num_patches,)
            similarity = -similarity  # FIX: Invert here at the source
            
            # Reshape to spatial grid
            num_patches = similarity.shape[0]
            grid_size = int(np.sqrt(num_patches))
            similarity_grid = similarity.reshape(1, 1, grid_size, grid_size)
            
            # Upsample to original size
            upsampled = F.interpolate(
                similarity_grid,
                size=(orig_H, orig_W),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Transform from [-1, 1] to [0, 1]
            normalized = (upsampled + 1.0) / 2.0
            normalized = torch.clamp(normalized, 0.0, 1.0)
            
        return normalized
