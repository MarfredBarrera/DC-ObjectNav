import os
import torch
import matplotlib.pyplot as plt
import time
from PIL import Image
from MaskCLIP import MaskCLIP

#use NVIDIA GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

#initialize MaskCLIP instance using the Vision Transformer (ViT-B/16)
mask_clip = MaskCLIP(model_name="ViT-B/16", device=device)

#defining path for all possible images. CHOOSE some image from the images folder
#img_path = os.path.join("images", "image_bear_mask.jpeg")
#img_path = os.path.join("images", "image_dog_hugging.jpg")
# img_path = os.path.join("images", "image_multiple_dogs.jpg")
#img_path = os.path.join("images", "image_family_home.jpg")
# img_path = os.path.join("images", "joker_batman.webp")

img_path = os.path.join("images", "image.png")


#UPDATE: prompt for desired image
prompt = "refridgerator"

#set a maximum dimension of 512 for image so that doesn't take too long to process
MAX_DIM = 512

if os.path.exists(img_path):
    img = Image.open(img_path).convert("RGB")
    w_orig, h_orig = img.size

    #resize image if either dimension w_orig, h_orig exceeds MAX_DIM
    if max(w_orig, h_orig) > MAX_DIM:
        scale = MAX_DIM / max(w_orig, h_orig)
        img = img.resize((int(w_orig * scale), int(h_orig * scale)), Image.LANCZOS)

else:
    print(f"Error: Could not find {img_path}.")

#creates the semantic vector given image and prompt, from MaskCLIP class
semantic_vec, mask = mask_clip.get_semantic_vector(img, prompt)
print(f"Semantic vector shape: {mask.shape}")

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

#saves image as .png file, compares heatmap to heatmap on original feature
plt.tight_layout()
print("Saving result to result.png...")
plt.savefig("result.png") # This saves the image to the 'clip copy' folder
print("Displaying image... close the window to finish.")

plt.show() 
