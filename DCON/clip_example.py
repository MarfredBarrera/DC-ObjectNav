import torch
import clip
import time
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("/workspace/DCON/output/current_scene/rgbs/rgb_010.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a kitchen", "a couch", "a table"]).to(device)

start_time = time.time()
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
end_time = time.time()
print("Inference time:", end_time - start_time, "seconds")

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]