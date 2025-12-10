import json
import math
import os

# Path to your scene output
TRANSFORMS_PATH = "/workspace/DCON/output/current_scene/transforms.json"

# Resolution from your generation script
WIDTH = 720
HEIGHT = 720

def fix_transforms(path):
    print(f"Reading {path}...")
    with open(path, 'r') as f:
        data = json.load(f)

    # 1. Calculate Focal Length from Field of View (camera_angle_x)
    # Formula: focal_length = (Width / 2) / tan(FOV / 2)
    if "camera_angle_x" in data:
        fov_x = data["camera_angle_x"]
        fl_x = (WIDTH / 2) / math.tan(fov_x / 2)
        fl_y = fl_x  # Square pixels
    else:
        # Fallback if angle is missing (Habitat default is 90 deg = 1.5708 rad)
        print("Warning: camera_angle_x missing, assuming 90 degrees.")
        fl_x = (WIDTH / 2) / math.tan(1.570796 / 2)
        fl_y = fl_x

    # 2. Add Explicit Intrinsics to the Global JSON Object
    print(f"Patching intrinsics: fl_x={fl_x:.2f}, w={WIDTH}, h={HEIGHT}")
    
    data["fl_x"] = fl_x
    data["fl_y"] = fl_y
    data["cx"] = WIDTH / 2
    data["cy"] = HEIGHT / 2
    data["w"] = WIDTH
    data["h"] = HEIGHT
    
    # Optional: Nerfstudio standard also includes these often
    data["k1"] = 0.0
    data["k2"] = 0.0
    data["p1"] = 0.0
    data["p2"] = 0.0

    # 3. Save it back
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print("Success! transforms.json patched.")

if __name__ == "__main__":
    if os.path.exists(TRANSFORMS_PATH):
        fix_transforms(TRANSFORMS_PATH)
    else:
        print(f"Error: File not found at {TRANSFORMS_PATH}")