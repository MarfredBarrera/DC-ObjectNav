"""For each self-stopped OVON failure, measure the metric distance from the
agent's stopping pose to the surface in front of it, aiming the depth sensor at
the nearest annotated goal. Annotation-independent confirmation of whether a
'right object' near-miss actually reached the object.

Reports per episode:
  eucl   straight-line (x,z) to nearest annotated goal center  (= tree's 'm out')
  geo    final_geodesic from the run record (what success scores on)
  depth  median depth of the central image patch, sensor aimed at the goal
         bearing (metres to the surface the agent is facing) -- 0-miss removed
"""
import os, glob, json, math
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
from src.config import Config
from src.habitat.habitat_utils import init_simulator

BASE = "output/ovon_detector_pairwise_field_maxj"

def load_rows():
    rows = []
    for f in glob.glob(BASE + "/runs/*.json"):
        r = json.load(open(f))
        if r.get("error") or r.get("success") or not r.get("agent_stopped"):
            continue
        rows.append(r)
    return rows

rows = load_rows()
scene = os.path.abspath(rows[0]["scene"])
cfg = Config("config/config.yaml")
cfg.apply_yaml("config/agent_configs/agent_ovon_stretch.yaml")
sim, agent = init_simulator(scene, width=480, height=480, fov_deg=60.0,
                            sensor_height=1.31, agent_radius=0.0)

SENSOR_H = 1.31

def nearest_goal(fp, goals):
    fp = np.array(fp)
    best = min(goals, key=lambda g: np.linalg.norm(np.array(g)[[0, 2]] - fp[[0, 2]]))
    best = np.array(best)
    d = best - fp
    eucl_h = float(np.linalg.norm([d[0], d[2]]))       # horizontal to center
    eucl_3d = float(np.linalg.norm(best - fp))          # 3D to annotation center
    return best, eucl_h, eucl_3d

def aim_depth(fp, target):
    """Central-patch median depth with the sensor pointed in full 3D at the
    object center, so the ray lands on low furniture instead of overshooting."""
    cam = np.array(fp, dtype=np.float32) + np.array([0.0, SENSOR_H, 0.0])
    dirv = np.array(target, dtype=np.float32) - cam
    n = np.linalg.norm(dirv)
    if n < 1e-6:
        return float("nan")
    dirv = dirv / n
    st = habitat_sim.AgentState()
    st.position = np.array(fp, dtype=np.float32)
    # align camera forward (-z) with the direction to the object
    st.rotation = habitat_sim.utils.common.quat_from_two_vectors(
        np.array([0.0, 0.0, -1.0]), dirv)
    agent.set_state(st)
    d = sim.get_sensor_observations()["depth"]
    H, W = d.shape
    cy, cx = H // 2, W // 2
    ry, rx = int(H * 0.1), int(W * 0.1)
    patch = d[cy - ry:cy + ry, cx - rx:cx + rx]
    patch = patch[(patch > 0.05) & np.isfinite(patch)]
    if patch.size == 0:
        return float("nan")
    return float(np.median(patch))

out = []
for r in rows:
    fp = r["final_pos"]
    target, eucl_h, eucl_3d = nearest_goal(fp, r["goals"])
    depth = aim_depth(fp, target)
    out.append((r["id"], r["target"], eucl_h, r.get("final_geodesic"), depth))

sim.close()
out.sort(key=lambda t: (t[1], t[2]))
print(f"{'episode':40s} {'eucl':>6s} {'geo':>7s} {'depth->target':>13s}")
print("-" * 70)
for rid, tgt, eucl, geo, depth in out:
    short = rid.replace("__4ok3usBNeis", "")
    print(f"{short:40s} {eucl:6.2f} {geo:7.2f} {depth:13.2f}")
