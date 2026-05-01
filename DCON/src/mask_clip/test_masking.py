import os
import torch

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F

from model import build_model
from simple_tokenizer import SimpleTokenizer

import clip

#use NVIDIA GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = SimpleTokenizer()

#loading the openAI ViT-B/16 weights
#uses the downloaded weights and apply to model.py
model_full, _ = clip.load("ViT-B/16", device=device)
state_dict = model_full.state_dict()

#reconstructs model from model.py for dense feature extraction
model = build_model(state_dict).to(device).eval()

#loading and preparing image
#img_path = "image_bear_mask.jpeg"
#img_path = "image_dog_hugging.jpg"
#img_path = "image_multiple_dogs.jpg"
img_path = "image_family_home.jpg"
img = Image.open(img_path).convert("RGB")
w_orig, h_orig = img.size

#uses standard CLIP processing
preprocess = T.Compose([T.Resize((448, 448)), T.ToTensor(), T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
img_tensor = preprocess(img).unsqueeze(0).to(device)

#tokenizing the text prompt, CLIP expects length 77
prompt = "a happy family consisting of 4 people"
tokens = [tokenizer.encoder["<|startoftext|>"]] + tokenizer.encode(prompt) + [tokenizer.encoder["<|endoftext|>"]]
tokens += [0] * (77 - len(tokens))
text_tensor = torch.tensor([tokens]).to(device)

#generate Mask, inference
with torch.inference_mode():
    #extracting dense patch features and text embeddings
    patch_features = model.get_patch_encodings(img_tensor)
    text_features = model.encode_text(text_tensor)

    #normalizing embeddings for cosine similarity
    patch_features /= patch_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    #calculating similarity grid
    heatmap = (patch_features @ text_features.T).reshape(1, 1, 28, 28)

    #apply the calculated surface on the original image
    mask = F.interpolate(heatmap, size=(h_orig, w_orig), mode='bilinear')[0, 0]

#visualization
plt.figure(figsize=(12, 6))

#creates heatmap
plt.subplot(1, 2, 1)
plt.title(f"Heatmap: {prompt}")
plt.imshow(mask.cpu().numpy(), cmap='jet')
plt.axis('off')

#heatmap applied on original photo to check that localization matches
plt.subplot(1, 2, 2)
plt.title("Overlay Result")
plt.imshow(img)
plt.imshow(mask.cpu().numpy(), cmap='jet', alpha=0.5) 
plt.axis('off')

plt.tight_layout()
print("Saving result to result.png...")
plt.savefig("result.png") # This saves the image to the 'clip copy' folder
print("Displaying image... close the window to finish.")

plt.show() 
