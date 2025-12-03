import os
# Silence habitat-sim warnings and logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['CUDA_VISIBLE_DEVICES'] = '4' 

import habitat_sim
import habitat_sim.utils.common as utils
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

    # Agent Configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor]
    
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
print("  [Arrow Up]    : Move Forward")
print("  [Arrow Left]  : Turn Left")
print("  [Arrow Right] : Turn Right")
print("  [Q] or [ESC]  : Quit")
print("="*40 + "\n")


while True:
    print("DEBUG: Getting observations...")
    obs = sim.get_sensor_observations()
    
    print("DEBUG: Rendering image...")
    rgb = obs["rgb"]
    cv2_img = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    small_img = cv2.resize(cv2_img, (512, 512))
    cv2.imshow("Habitat Agent View", small_img)

    print("DEBUG: Waiting for input...")
    key = cv2.waitKey(0)
    
    print(f"DEBUG: Key pressed: {key}")
    
    if key == ord('q'):
        break
    elif key == ord('w'):
        print("DEBUG: Stepping Physics...")
        sim.step("move_forward")
        
    # Left (Left Arrow or 'a')
    elif key == ord('a'):
        sim.step("turn_left")
        print("Action: Left")

    # Right (Right Arrow or 'd')
    elif key == ord('d'):
        sim.step("turn_right")
        print("Action: Right")

# Cleanup
cv2.destroyAllWindows()
sim.close()