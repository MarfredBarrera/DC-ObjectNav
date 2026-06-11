import math
import numpy as np
import habitat_sim
import habitat_sim.utils.common as utils


def get_scene_bounds_from_pathfinder(sim) -> list:
    if not sim.pathfinder.is_loaded:
        raise RuntimeError("Pathfinder not loaded — cannot determine scene bounds.")
    bounds = sim.pathfinder.get_bounds()
    return [np.array(bounds[0]).tolist(), np.array(bounds[1]).tolist()]


def spawn_agent_at_random_navpoint(sim, agent) -> np.ndarray:
    if sim.pathfinder.is_loaded:
        nav_point = sim.pathfinder.get_random_navigable_point()
        initial_state = habitat_sim.AgentState()
        initial_state.position = nav_point
        agent.set_state(initial_state)
        print(f"Agent spawned at: {nav_point}")
        return np.array(nav_point)
    else:
        print("Warning: No navmesh found. Agent spawned at origin.")
        return np.zeros(3)

def spawn_agent_at_pos(sim, agent, pos: np.ndarray) -> np.ndarray:
    """Spawn agent at the navmesh point closest to `pos`. Falls back to origin if no navmesh."""
    if sim.pathfinder.is_loaded:
        nav_point = sim.pathfinder.snap_point(pos)
        if np.isnan(nav_point).any():
            print(f"Warning: snap_point({pos}) returned nan — no navigable point nearby.")
            return np.zeros(3)
        initial_state = habitat_sim.AgentState()
        initial_state.position = nav_point
        agent.set_state(initial_state)
        print(f"Agent spawned at: {nav_point}")
        return np.array(nav_point)
    else:
        print("Warning: No navmesh found. Agent spawned at origin.")
        return np.zeros(3)


def get_camera_matrix(agent) -> np.ndarray:
    """4×4 camera-to-world matrix from the 'rgb' sensor state (Habitat/OpenGL frame)."""
    state = agent.get_state().sensor_states["rgb"]
    rot_mat = np.array(utils.quat_to_magnum(state.rotation).to_matrix())
    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = state.position
    return transform


def make_cfg(
    scene_filepath: str,
    resolution: int = 512,
    fov_deg: float = 90.0,
    sensor_height: float = 1.5,
) -> habitat_sim.Configuration:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_filepath
    sim_cfg.enable_physics = False
    sim_cfg.load_semantic_mesh = False

    def _camera(uuid: str, sensor_type) -> habitat_sim.CameraSensorSpec:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [resolution, resolution]
        spec.position = [0.0, sensor_height, 0.0]
        spec.orientation = [0.0, 0.0, 0.0]
        spec.hfov = fov_deg
        return spec

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        _camera("rgb", habitat_sim.SensorType.COLOR),
        _camera("depth", habitat_sim.SensorType.DEPTH),
    ]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def init_simulator(scene_filepath: str, agent_radius: float = 0.1,
                   agent_height: float = 1.5, **make_cfg_kwargs):
    """Build config, create simulator, and initialise agent 0. Returns (sim, agent).

    When `agent_radius > 0`, recompute the navmesh with that radius instead of
    using the scene's prebaked .navmesh. The navmesh insets walls by this
    radius, so a smaller value lets `pathfinder.try_step` (which moves the
    agent in SimInterface) squeeze through narrower gaps. Must run before any
    navmesh-dependent setup (random spawn, scene-bounds query).
    """
    cfg = make_cfg(scene_filepath, **make_cfg_kwargs)
    try:
        sim = habitat_sim.Simulator(cfg)
    except Exception as exc:
        print(f"Error loading simulator: {exc}")
        raise SystemExit(1) from exc
    agent = sim.initialize_agent(0)

    if agent_radius and agent_radius > 0:
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = float(agent_radius)
        navmesh_settings.agent_height = float(agent_height)
        ok = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        if not ok:
            print(f"[init] navmesh recompute failed (r={agent_radius}); "
                  "keeping prebaked navmesh")
        else:
            print(f"[init] recomputed navmesh: agent_radius={agent_radius}m "
                  f"agent_height={agent_height}m")

    return sim, agent


# Habitat (OpenGL: -Z forward, Y up) -> OpenCV (+Z forward, -Y up)
# Usage: c2w_cv = c2w_hab @ HABITAT_TO_CV
HABITAT_TO_CV = np.array(
    [[1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, -1, 0],
     [0, 0, 0, 1]],
    dtype=np.float64,
)
