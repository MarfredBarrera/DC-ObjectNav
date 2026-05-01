import torch
import torch.nn.functional as F
import torchvision.transforms as T
import time

from model import build_model
import clip
from simple_tokenizer import SimpleTokenizer

class MaskCLIP:

    def __init__(self, device="cuda"):
        self.device = device
        self.tokenizer = SimpleTokenizer()

        #uses the downloaded weights and apply to model.py
        model_full, _ = clip.load("ViT-B/16", device=self.device)
        
        #reconstructs model from model.py for dense feature extraction
        self.model = build_model(model_full.state_dict()).to(self.device).eval()
        
        #standard CLIP processing
        self.preprocess = T.Compose([T.Resize((448, 448)), T.ToTensor(), T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    #tokenizes the prompt
    def _tokenize(self, prompt):
        tokens = [self.tokenizer.encoder["<|startoftext|>"]] + self.tokenizer.encode(prompt) + [self.tokenizer.encoder["<|endoftext|>"]]
        tokens += [0] * (77 - len(tokens))
        return torch.tensor([tokens]).to(self.device)

    #returns a float tensor [1, 512] and float tensor representing the heatmap
    def get_semantic_vector(self, process_image, prompt):
        w_orig, h_orig = process_image.size
        img_tensor = self.preprocess(process_image).unsqueeze(0).to(self.device)
        text_tensor = self._tokenize(prompt)

        #generate Mask, inference
        with torch.inference_mode():
            #extracting dense patch features and embeddings
            start_time = time.time()
            patch_features = self.model.get_patch_encodings(img_tensor)
            text_features = self.model.encode_text(text_tensor)
            end_time = time.time()
            print(f"Time taken for feature extraction: {end_time - start_time}")

            #normalizing embeddings for cosine similarity
            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            #calculating similarity grid
            heatmap = (patch_features @ text_features.T).reshape(1, 1, 28, 28)

            #apply the calculated surface on the original image
            mask = F.interpolate(heatmap, size=(h_orig, w_orig), mode='bilinear')[0, 0]
            
            #scales the image back to its original size and then sums up all the patches to get the weighted average --> [1, 512]
            weights = heatmap.view(1, 784, 1)
            weighted_features = patch_features * weights
            semantic_vector = weighted_features.sum(dim=1) / (weights. sum() + 1e-8)

        return semantic_vector, mask
