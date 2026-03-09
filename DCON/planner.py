import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
import numpy as np
import cv2
import habitat_sim
import habitat_sim.utils.common as utils
import habitat_sim.physics as physics
import magnum as mn

# Silence habitat-sim warnings and logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

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
    rgb_sensor.resolution = [720, 720] 
    rgb_sensor.position = [0.0, 1.5, 0.0]
    rgb_sensor.orientation = [0.0, 0.0, 0.0]

    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [720, 720]
    depth_sensor.position = [0.0, 1.5, 0.0]  
    depth_sensor.orientation = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
    
    # Notice we completely omitted the agent_cfg.action_space
    # because we are driving the state purely via VelocityControl

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
# Velocity Controller Setup
# --------------------------------------------------------
vel_control = physics.VelocityControl()
vel_control.controlling_lin_vel = True
vel_control.controlling_ang_vel = True

# Continuous control variables
v = 0.0      # Linear velocity (m/s)
omega = 0.0  # Angular velocity (rad/s)
dt = 1.0 / 30.0  # 30 FPS integration step

# --------------------------------------------------------
# Interactive Control Loop
# --------------------------------------------------------
print("\n" + "="*40)
print(" CONTINUOUS THROTTLE COMMANDS:")
print("  [w] / [s] : Increase / Decrease Forward Speed")
print("  [a] / [d] : Turn Left / Turn Right")
print("  [SPACE]   : Emergency Brake (Stop all movement)")
print("  [q]       : Quit")
print("="*40 + "\n")

while True:
    # 1. Get and display observations
    obs = sim.get_sensor_observations()    
    rgb = obs["rgb"]
    
    cv2_img = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    small_img = cv2.resize(cv2_img, (512, 512))
    
    # Overlay current velocity on the image
    cv2.putText(small_img, f"v: {v:.2f} m/s | w: {omega:.2f} rad/s", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Habitat Continuous Control", small_img)

    # 2. Handle continuous inputs
    # Wait 33ms (~30 FPS). The & 0xFF is required for cross-platform compatibility
    key = cv2.waitKey(33) & 0xFF 
    
    if key == ord('q'):
        break
    elif key == ord('w'):
        v += 0.1
    elif key == ord('s'):
        v -= 0.1
    elif key == ord('a'):
        omega += 0.1
    elif key == ord('d'):
        omega -= 0.1
    elif key == ord(' '):  # Spacebar to stop
        v = 0.0
        omega = 0.0

    # 3. Apply velocities
    # In Habitat, forward is typically the -Z axis
    vel_control.linear_velocity = np.array([0.0, 0.0, -v])
    # Rotation around the Y axis (up)
    vel_control.angular_velocity = np.array([0.0, omega, 0.0])

# 4. Integrate kinematics and update state
    agent_state = agent.get_state()
    
    # Convert python/numpy types to C++ Magnum types
    magnum_rotation = utils.quat_to_magnum(agent_state.rotation)
    magnum_translation = mn.Vector3(agent_state.position)
    
    # Create the RigidState
    rigid_state = habitat_sim.RigidState(magnum_rotation, magnum_translation)
    
    # Integrate the velocity over the dt
    new_rigid_state = vel_control.integrate_transform(dt, rigid_state)
    
    # Convert C++ Magnum types back to python/numpy types
    agent_state.position = np.array(new_rigid_state.translation)
    # FIX: Use quat_from_magnum here!
    agent_state.rotation = utils.quat_from_magnum(new_rigid_state.rotation) 
    
    # Set the agent to the newly calculated state
    agent.set_state(agent_state)

# Cleanup
cv2.destroyAllWindows()
sim.close()
print("Simulation closed.")