import torch
import os
import argparse
import math

def convert(args):
    print(f"Loading custom model: {args.input}")
    # Load your Habitat model (flat dictionary)
    ckpt = torch.load(args.input, map_location="cpu")
    
    # 1. Prepare Data
    means = ckpt["means"]
    scales = ckpt["scales"]
    quats = ckpt["quats"]
    opacities = ckpt["opacities"]
    
    # 2. Convert RGB Logits -> Spherical Harmonics (Degree 0)
    # The viewer expects SH coefficients, not RGB colors.
    # We convert your RGB colors to the 0th SH coefficient.
    # SH_0 = RGB / 0.2820947917... (roughly)
    # But for simplicity, we calculate the RGB values and shape them as [N, 1, 3]
    
    print("Converting RGB to SH parameters...")
    rgb_logits = ckpt["colors"]
    rgb = torch.sigmoid(rgb_logits) # Convert logits to 0-1 RGB
    
    # Standard SH constant (C0)
    C0 = 0.28209479177387814
    
    # Inverse of render formula: color = sh * C0 + 0.5
    # So: sh = (color - 0.5) / C0
    sh0 = (rgb - 0.5) / C0
    sh0 = sh0.unsqueeze(1) # Shape [N, 1, 3]
    
    # Create empty higher-order SH (Degree 0 means no higher orders)
    shN = torch.zeros((means.shape[0], 0, 3))

    # 3. Wrap in "splats" dictionary format expected by simple_viewer.py
    new_ckpt = {
        "splats": {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh0": sh0,
            "shN": shN
        }
    }

    # 4. Save
    print(f"Saving converted model to: {args.output}")
    torch.save(new_ckpt, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to your model.pt")
    parser.add_argument("--output", type=str, default="converted_ckpt.pt", help="Output path")
    args = parser.parse_args()
    convert(args)