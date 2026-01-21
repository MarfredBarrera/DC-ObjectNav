"""
Simple ensemble: Train multiple hash grid networks with same config but different random seeds.
"""
import tinycudann as tcnn
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # Load the image
    image_path = "/workspace/rgb360/rgb_022.png"
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img) / 255.0
    height, width = img_array.shape[:2]

    print(f"Image shape: {height}x{width}")

    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(
        np.linspace(0, 1, height),
        np.linspace(0, 1, width),
        indexing='ij'
    )
    coords = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 2)
    rgb_values = img_array.reshape(-1, 3)

    coords_tensor = torch.from_numpy(coords).float().cuda()
    rgb_tensor = torch.from_numpy(rgb_values).float().cuda()

    # Ensemble configuration
    n_ensemble = 3  # Number of models
    n_input_dims = 2
    n_output_dims = 3

    encoding_config = {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 2.0,
    }

    network_config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "Sigmoid",
        "n_neurons": 64,
        "n_hidden_layers": 2,
    }

    # Create ensemble (same config, different random initializations)
    print(f"\nCreating ensemble of {n_ensemble} models...")
    models = []
    optimizers = []

    for i in range(n_ensemble):
        # Different random seed for each model
        torch.manual_seed(42 + i)
        
        model = tcnn.NetworkWithInputEncoding(
            n_input_dims,
            n_output_dims,
            encoding_config,
            network_config
        ).cuda()
        
        models.append(model)
        optimizers.append(torch.optim.Adam(model.parameters(), lr=1e-2))
        print(f"  Model {i+1} created")

    # Training
    batch_size = 8192
    n_steps = 5000

    print(f"\nTraining for {n_steps} steps...")
    all_losses = [[] for _ in range(n_ensemble)]

    for step in tqdm(range(n_steps)):
        # Same batch for all models
        indices = torch.randint(0, coords_tensor.shape[0], (batch_size,), device='cuda')
        batch_coords = coords_tensor[indices]
        batch_rgb = rgb_tensor[indices]
        
        # Train each model
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            pred_rgb = model(batch_coords)
            loss = ((pred_rgb - batch_rgb) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            all_losses[i].append(loss.item())
        
        if step % 500 == 0:
            avg_loss = np.mean([all_losses[i][-1] for i in range(n_ensemble)])
            tqdm.write(f"Step {step:05d}, avg loss = {avg_loss:.6f}")

    # Reconstruct with ensemble
    print("\nReconstructing with ensemble...")
    predictions = []

    for i, model in enumerate(models):
        with torch.no_grad():
            reconstructed = []
            for j in range(0, coords_tensor.shape[0], batch_size):
                batch = coords_tensor[j:j+batch_size]
                pred = model(batch)
                reconstructed.append(pred)
            reconstructed = torch.cat(reconstructed, dim=0).cpu().numpy()
            predictions.append(reconstructed.reshape(height, width, 3))

    predictions = [np.clip(p, 0, 1).astype(np.float32) for p in predictions]

    # Ensemble average
    ensemble_pred = np.mean(predictions, axis=0)
    img_array = img_array.astype(np.float32)

    # Calculate PSNR
    print("\nPSNR Results:")
    for i, pred in enumerate(predictions):
        mse = np.mean((img_array - pred) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        print(f"  Model {i+1}: {psnr:.2f} dB")

    ensemble_mse = np.mean((img_array - ensemble_pred) ** 2)
    ensemble_psnr = 10 * np.log10(1.0 / ensemble_mse) if ensemble_mse > 0 else float('inf')
    print(f"  Ensemble: {ensemble_psnr:.2f} dB")

    # Visualize
    fig, axes = plt.subplots(1, n_ensemble + 2, figsize=(5 * (n_ensemble + 2), 5))

    axes[0].imshow(img_array)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i, pred in enumerate(predictions):
        axes[i+1].imshow(pred)
        axes[i+1].set_title(f"Model {i+1}")
        axes[i+1].axis('off')

    axes[-1].imshow(ensemble_pred)
    axes[-1].set_title(f"Ensemble\nPSNR: {ensemble_psnr:.2f} dB")
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('/workspace/simple_ensemble.png', dpi=150, bbox_inches='tight')
    print("\nSaved to /workspace/simple_ensemble.png")

    plt.show()