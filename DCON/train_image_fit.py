#!/usr/bin/env python3
"""
Train a hash grid network to recreate an RGB image.
Maps 2D pixel coordinates (x, y) -> RGB values using tiny-cuda-nn.
"""
import tinycudann as tcnn
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

# Load the image
image_path = "/workspace/DCON/output/replica_cad_data/rgb_022.png"
img = Image.open(image_path).convert("RGB")
img_array = np.array(img) / 255.0  # Normalize to [0, 1]
height, width = img_array.shape[:2]

print(f"Image shape: {height}x{width}")

# Create coordinate grid: (x, y) in [0, 1]^2
y_coords, x_coords = np.meshgrid(
    np.linspace(0, 1, height),
    np.linspace(0, 1, width),
    indexing='ij'
)
coords = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 2)  # Shape: (H*W, 2)
rgb_values = img_array.reshape(-1, 3)  # Shape: (H*W, 3)

# Convert to PyTorch tensors
coords_tensor = torch.from_numpy(coords).float().cuda()
rgb_tensor = torch.from_numpy(rgb_values).float().cuda()

print(f"Coordinates shape: {coords_tensor.shape}")
print(f"RGB values shape: {rgb_tensor.shape}")

# Configure hash grid encoding for 2D input -> 3D RGB output
n_input_dims = 2  # (x, y)
n_output_dims = 3  # (R, G, B)

encoding_config = {
    "otype": "HashGrid",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "per_level_scale": 2.0,  # Higher scale for image details
}

network_config = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "Sigmoid",  # Sigmoid to keep output in [0, 1] for RGB
    "n_neurons": 64,
    "n_hidden_layers": 2,
}

# Create model
model = tcnn.NetworkWithInputEncoding(
    n_input_dims,
    n_output_dims,
    encoding_config,
    network_config
).cuda()

print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
batch_size = 8192  # Process in batches to avoid OOM
n_steps = 5000

print(f"\nTraining for {n_steps} steps with batch size {batch_size}...")

losses = []
for step in tqdm(range(n_steps)):
    # Random batch sampling
    indices = torch.randint(0, coords_tensor.shape[0], (batch_size,), device='cuda')
    batch_coords = coords_tensor[indices]
    batch_rgb = rgb_tensor[indices]
    
    # Forward pass
    pred_rgb = model(batch_coords)
    
    # MSE loss
    loss = ((pred_rgb - batch_rgb) ** 2).mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if step % 500 == 0:
        tqdm.write(f"Step {step:05d}, loss = {loss.item():.6f}")

print("\nTraining complete!")

# Reconstruct the full image
print("Reconstructing image...")
with torch.no_grad():
    # Process in batches to avoid OOM
    reconstructed = []
    for i in range(0, coords_tensor.shape[0], batch_size):
        batch = coords_tensor[i:i+batch_size]
        pred = model(batch)
        reconstructed.append(pred)
    reconstructed = torch.cat(reconstructed, dim=0)
    reconstructed_img = reconstructed.cpu().numpy().reshape(height, width, 3)

# Clip to [0, 1] range and ensure float32 for matplotlib
reconstructed_img = np.clip(reconstructed_img, 0, 1).astype(np.float32)
img_array = img_array.astype(np.float32)

# Calculate PSNR
mse = np.mean((img_array - reconstructed_img) ** 2)
psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
print(f"PSNR: {psnr:.2f} dB")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].imshow(img_array)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(reconstructed_img)
axes[1].set_title(f"Reconstructed (PSNR: {psnr:.2f} dB)")
axes[1].axis('off')

# # Difference map
# diff = np.abs(img_array - reconstructed_img)
# axes[2].imshow(diff)
# axes[2].set_title("Absolute Difference")
# axes[2].axis('off')

plt.tight_layout()
plt.savefig('/workspace/image_reconstruction.png', dpi=150, bbox_inches='tight')
print("Results saved to /workspace/image_reconstruction.png")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.yscale('log')
plt.grid(True)
plt.savefig('/workspace/training_loss.png', dpi=150, bbox_inches='tight')
print("Training loss plot saved to /workspace/training_loss.png")

plt.show()
