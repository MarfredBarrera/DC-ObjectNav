import os
import json
import habitat_sim
import habitat_sim.utils.common as utils
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Silence habitat-sim warnings and logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

# --------------------------------------------------------
# Create output directory
# --------------------------------------------------------
output_dir = "/workspace/DCON/output/current_scene"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/rgbs", exist_ok=True)
os.makedirs(f"{output_dir}/depth_data", exist_ok=True)
os.makedirs(f"{output_dir}/depth_vis", exist_ok=True)

# Define resolution once here to ensure consistency across config and JSON
IMG_WIDTH = 720
IMG_HEIGHT = 720
FOV_DEG = 90.0

def get_camera_matrix(agent):
    # 1. Get the state of the specific sensor 'rgb'
    # Note: agent.get_state() gives the *body* pose. 
    # We need .sensor_states['rgb'] for the actual camera pose (includes height offset)
    state = agent.get_state().sensor_states['rgb']
    
    # 2. Extract Rotation (Quaternion) and Translation (Vector)
    rot_quat = state.rotation
    translation = state.position

    # 3. Convert Quaternion to 3x3 Rotation Matrix
    # Habitat utils provides a clean conversion to Magnum types, then to numpy
    rot_mat = utils.quat_to_magnum(rot_quat).to_matrix()
    rot_mat = np.array(rot_mat) # Convert Magnum matrix to Numpy

    # 4. Build 4x4 Matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_mat
    transform_matrix[:3, 3] = translation
    
    return transform_matrix

# --------------------------------------------------------
# Habitat-Sim configuration
# --------------------------------------------------------
def make_cfg(scene_filepath):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_filepath
    sim_cfg.enable_physics = False
    sim_cfg.load_semantic_mesh = False

    # Define Sensors
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [720, 720] # Slightly smaller for smoother window display
    rgb_sensor.position = [0.0, 1.5, 0.0]
    rgb_sensor.orientation = [0.0, 0.0, 0.0]

    # Add depth sensor
    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [720, 720]
    depth_sensor.position = [0.0, 1.5, 0.0]  # Same position as RGB
    depth_sensor.orientation = [0.0, 0.0, 0.0]

    # Agent Configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
    # Explicitly register the action space to ensure controls work
    # You can adjust 'amount' to change step size (meters) or turn angle (degrees)
    agent_cfg.action_space = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward", habitat_sim.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left", habitat_sim.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right", habitat_sim.ActuationSpec(amount=10.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# --------------------------------------------------------
# Initialization
# --------------------------------------------------------
scene = "/workspace/DCON/gibson_scenes/Anaheim.glb"
cfg = make_cfg(scene)

try:
    sim = habitat_sim.Simulator(cfg)
except Exception as e:
    print(f"Error loading simulator: {e}")
    exit()

# Initialize agent
agent = sim.initialize_agent(0)

# Set initial position
if sim.pathfinder.is_loaded:
    nav_point = sim.pathfinder.get_random_navigable_point()
    agent_state = habitat_sim.AgentState()
    agent_state.position = nav_point
    agent.set_state(agent_state)
    print(f"Agent spawned at: {nav_point}")
else:
    print("Warning: No navmesh found. Agent spawned at origin.")

# --------------------------------------------------------
# Interactive Control Loop
# --------------------------------------------------------
print("\n" + "="*40)
print(" COMMANDS:")
print("  [w]    : Move Forward")
print("  [a]    : Turn Left")
print("  [d]    : Turn Right")
print("  [Q] or [ESC]  : Quit")
print("="*40 + "\n")

i = 0

rgb_imgs = []
depth_imgs = []
pose_matrices = []

while True:
    # get observations
    obs = sim.get_sensor_observations()    
    rgb = obs["rgb"]
    depth = obs["depth"]

    current_matrix = get_camera_matrix(agent)
    pose_matrices.append(current_matrix)

    # window display of RGB
    cv2_img = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    small_img = cv2.resize(cv2_img, (512, 512))
    cv2.imshow("Habitat Agent View", small_img)

    # save RGB
    rgb_imgs.append(cv2_img)
    # save depth as numpy array (meters)
    depth_imgs.append(depth)

    # save depth as visualization image
    # Normalize depth to 0-255 for visualization
    depth_vis = np.clip(depth * 255 / 10.0, 0, 255).astype(np.uint8)  # Assume max 10m range
    cv2.imwrite(f"{output_dir}/depth_vis/depth_vis_{i:03d}.png", depth_vis)

    print("Waiting for input...")
    key = cv2.waitKey(0)
    
    print(f"Key pressed: {key}")
    
    if key == ord('q'):
        break
    elif key == ord('w'):
        sim.step("move_forward")
        print("Action: Move Forward")

        
    # Left (Left Arrow or 'a')
    elif key == ord('a'):
        sim.step("turn_left")
        print("Action: Left")

    # Right (Right Arrow or 'd')
    elif key == ord('d'):
        sim.step("turn_right")
        print("Action: Right")

    i += 1

# --------------------------------------------------------
# Save Data & Transforms.json
# --------------------------------------------------------
print(f"Saving {len(rgb_imgs)} frames and transforms...")

# Extract scene bounds from habitat simulator
scene_bounds = None
if sim.pathfinder.is_loaded:
    # Get bounds from pathfinder
    bounds = sim.pathfinder.get_bounds()
    scene_bounds = {
        "min": np.array(bounds[0]).tolist(),  # [x_min, y_min, z_min]
        "max": np.array(bounds[1]).tolist()   # [x_max, y_max, z_max]
    }
    print(f"Scene bounds: min={bounds[0]}, max={bounds[1]}")
else:
    # Fallback: compute bounds from all agent positions
    all_positions = np.array([pose[:3, 3] for pose in pose_matrices])
    scene_min = all_positions.min(axis=0)
    scene_max = all_positions.max(axis=0)
    # Add padding
    padding = (scene_max - scene_min) * 0.2
    scene_min -= padding
    scene_max += padding
    scene_bounds = {
        "min": scene_min.tolist(),
        "max": scene_max.tolist()
    }
    print(f"Computed scene bounds from trajectory: min={scene_min}, max={scene_max}")

frames_data = []

# Calculate Intrinsics from FOV
# Convert FOV to radians
fov_rad = np.deg2rad(FOV_DEG)
# Formula: focal_length = (Width / 2) / tan(FOV / 2)
fl_x = (IMG_WIDTH / 2) / np.tan(fov_rad / 2)
fl_y = fl_x  # Square pixels

for idx, (cv2_img, depth, pose) in enumerate(zip(rgb_imgs, depth_imgs, pose_matrices)):
    # Filepaths relative to the transforms.json
    rgb_rel_path = f"rgbs/rgb_{idx:03d}.png"
    
    # Save Images
    cv2.imwrite(f"{output_dir}/{rgb_rel_path}", cv2_img)
    np.save(f"{output_dir}/depth_data/depth_{idx:03d}.npy", depth)

    # Add to JSON structure
    frames_data.append({
        "file_path": rgb_rel_path,
        "transform_matrix": pose.tolist() # Convert numpy -> list for JSON
    })

# 2. Construct final JSON with explicit intrinsics
json_data = {
    "camera_angle_x": fov_rad,
    "fl_x": fl_x,
    "fl_y": fl_y,
    "cx": IMG_WIDTH / 2,
    "cy": IMG_HEIGHT / 2,
    "w": IMG_WIDTH,
    "h": IMG_HEIGHT,
    # Standard pinhole model parameters
    "k1": 0.0,
    "k2": 0.0,
    "p1": 0.0,
    "p2": 0.0,
    "scene_bounds": scene_bounds,  # Add scene bounds
    "frames": frames_data
}

with open(f"{output_dir}/transforms.json", "w") as f:
    json.dump(json_data, f, indent=4)

print(f"Done! Transforms saved to {output_dir}/transforms.json")
# save all data
for i, (cv2_img, depth) in enumerate(zip(rgb_imgs, depth_imgs)):
    cv2.imwrite(f"{output_dir}/rgbs/rgb_{i:03d}.png", cv2_img)
    np.save(f"{output_dir}/depth_data/depth_{i:03d}.npy", depth)


# Cleanup
cv2.destroyAllWindows()
sim.close()