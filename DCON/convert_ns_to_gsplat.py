import torch
import argparse
import os

def convert(args):
    print(f"Loading Nerfstudio checkpoint: {args.input}")
    
    # 1. Load with weights_only=False (Bypasses PyTorch 2.6 security error)
    try:
        ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only
        ckpt = torch.load(args.input, map_location="cpu")
    
    if "pipeline" not in ckpt:
        print("Error: 'pipeline' key not found. Is this a valid Nerfstudio checkpoint?")
        return
        
    state_dict = ckpt["pipeline"]
    
    # 2. Define the parameter mapping
    # Key = gsplat name, Value = Nerfstudio name suffix
    param_map = {
        "means": "means",
        "scales": "scales",
        "quats": "quats",
        "opacities": "opacities",
        "sh0": "features_dc",   # Nerfstudio calls SH 0 "features_dc"
        "shN": "features_rest"  # Nerfstudio calls SH rest "features_rest"
    }
    
    splats = {}

    # 3. Helper to find keys with various prefixes
    def find_tensor(suffix):
        # List of possible prefixes in order of likelihood
        prefixes = [
            "_model.gauss_params.",  # Standard Splatfacto
            "_model.",               # Old Splatfacto
            "module._model.gauss_params.", # DDP training
            "gauss_params."          # Standalone model
        ]
        
        for prefix in prefixes:
            key = f"{prefix}{suffix}"
            if key in state_dict:
                return state_dict[key]
        
        return None

    print("Extracting parameters...")
    missing_keys = []
    
    for gsplat_name, ns_suffix in param_map.items():
        tensor = find_tensor(ns_suffix)
        if tensor is not None:
            splats[gsplat_name] = tensor
        else:
            missing_keys.append(ns_suffix)

    # 4. Error Handling & Debugging
    if missing_keys:
        print(f"\n❌ FAILED. Could not find these parameters: {missing_keys}")
        print("\nAvailable keys in checkpoint (First 20):")
        keys = list(state_dict.keys())
        for k in keys[:20]:
            print(f" - {k}")
        if len(keys) > 20: print(f" ... and {len(keys) - 20} more.")
        return

    # 5. Save
    new_ckpt = {"splats": splats}
    print(f"✅ Success! Saving compatible checkpoint to: {args.output}")
    torch.save(new_ckpt, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to Nerfstudio .ckpt")
    parser.add_argument("--output", type=str, default="converted_ns.pt", help="Output path")
    args = parser.parse_args()
    convert(args)