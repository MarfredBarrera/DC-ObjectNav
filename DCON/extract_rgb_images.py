import os
# Silence habitat-sim warnings and logs (set before importing habitat_sim)
os.environ['GLOG_minloglevel'] = '2'  # Suppress INFO and WARNING logs
os.environ['MAGNUM_LOG'] = 'quiet'     # Silence Magnum engine logs
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['CUDA_VISIBLE_DEVICES'] = '4' 

import habitat_sim
import habitat_sim.utils.common as utils
import numpy as np
import cv2

# --------------------------------------------------------
# Create output directory
# --------------------------------------------------------
output_dir = "/workspace/DCON/output/current_scene"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/rgbs", exist_ok=True)
os.makedirs(f"{output_dir}/depth_data", exist_ok=True)
os.makedirs(f"{output_dir}/depth_vis", exist_ok=True)

# --------------------------------------------------------
# Habitat-Sim configuration
# --------------------------------------------------------
def make_cfg(scene_filepath):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_filepath
    
    sim_cfg.enable_physics = False
    
    # Tell habitat-sim to load this as a simple scene without requiring dataset registration
    sim_cfg.load_semantic_mesh = False

    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    # Resolution is [height, width]
    rgb_sensor.resolution = [1024, 1024]
    rgb_sensor.position = [0.0, 1.5, 0.0]  # camera height relative to agent
    # Keep sensor pointing forward (no rotation relative to agent)
    rgb_sensor.orientation = [0.0, 0.0, 0.0]

    # Add depth sensor
    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [1024, 1024]
    depth_sensor.position = [0.0, 1.5, 0.0]  # Same position as RGB
    depth_sensor.orientation = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# Use a Gibson scene (.glb file) with absolute path
scene = "/workspace/DCON/gibson_scenes/Anaheim.glb"


cfg = make_cfg(scene)
sim = habitat_sim.Simulator(cfg)

print(f"Loaded scene: {scene}")

print(f"Scene has navmesh: {sim.pathfinder.is_loaded}")
if sim.pathfinder.is_loaded:
    print(f"Navigable area bounds: {sim.pathfinder.get_bounds()}")


# --------------------------------------------------------
# Initialize agent at a fixed, known-good location
# --------------------------------------------------------
agent = sim.initialize_agent(0)

# Set a safe pose
agent_state = habitat_sim.AgentState()

# If navmesh is available, use a navigable point
if sim.pathfinder.is_loaded:
    # Get a random navigable point on the floor
    nav_point = sim.pathfinder.get_random_navigable_point()
    agent_state.position = nav_point
    print(f"Agent position (navigable): {nav_point}")
else:
    # Fallback: manual position (ReplicaCAD uses Y-up coordinate system)
    # Y is the vertical axis in ReplicaCAD
    agent_state.position = np.array([0.0, 0.0, 0.0])  # Origin
    print(f"Agent position (manual): {agent_state.position}")
    print("Warning: No navmesh loaded, position may be invalid")

# --------------------------------------------------------
# Capture 360 degrees of RGB images
# --------------------------------------------------------
N = 36                  # 36 images = every 10°
step_deg = 360 / N

for i in range(N):
    yaw = np.deg2rad(i * step_deg)

    # ReplicaCAD uses Y-up coordinate system (like most 3D software)
    # Rotate around Y-axis for horizontal 360-degree view
    rot = utils.quat_from_angle_axis(yaw, np.array([0, 1, 0]))

    agent_state.rotation = rot
    agent.set_state(agent_state)

    obs = sim.get_sensor_observations()
    rgb = obs["rgb"]
    depth = obs["depth"]

    # Convert RGBA → BGR for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(f"{output_dir}/rgbs/rgb_{i:03d}.png", rgb)
    
    # Save depth as numpy array (meters)
    np.save(f"{output_dir}/depth_data/depth_{i:03d}.npy", depth)
    
    # Optional: Save depth as visualization image
    # Normalize depth to 0-255 for visualization
    depth_vis = np.clip(depth * 255 / 10.0, 0, 255).astype(np.uint8)  # Assume max 10m range
    cv2.imwrite(f"{output_dir}/depth_vis/depth_vis_{i:03d}.png", depth_vis)

    print(f"Saved view {i} (RGB + Depth)")

sim.close()
