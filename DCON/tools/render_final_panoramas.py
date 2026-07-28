"""Render a 4-view (0/90/180/270 deg) panorama at each self-stopped failure's
final pose, for visual adjudication of the OVON failure-mode tree.

All target episodes are in one scene, so the sim is loaded once. Run inside the
container with CUDA_VISIBLE_DEVICES pinned to an idle GPU.
"""
import os, glob, json, math
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image, ImageDraw, ImageFont

from src.config import Config
from src.habitat.habitat_utils import init_simulator

BASE = "output/ovon_detector_pairwise_field_maxj"
OUT = "output/_panoramas"
os.makedirs(OUT, exist_ok=True)

# --- collect the self-stopped failures (need adjudication) ---
def load_rows():
    rows = []
    for f in glob.glob(BASE + "/runs/*.json"):
        r = json.load(open(f))
        if r.get("error") or r.get("success"):
            continue
        if not r.get("agent_stopped"):
            continue  # timed-out failures are auto-classified; skip
        rows.append(r)
    return rows

rows = load_rows()
# skip already-rendered so refreshes only do the new episodes
rows = [r for r in rows if not os.path.exists(f"{OUT}/{r['id']}.png")]
assert rows, "no new self-stopped failures to render"
from collections import defaultdict
by_scene = defaultdict(list)
for r in rows:
    by_scene[os.path.abspath(r["scene"])].append(r)
print(f"[render] {len(rows)} new self-stopped failures across {len(by_scene)} scene(s)")

cfg = Config("config/config.yaml")
cfg.apply_yaml("config/agent_configs/agent_ovon_stretch.yaml")
SENSOR_H = 1.31   # OVON Stretch camera height

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except Exception:
    font = fsm = ImageFont.load_default()

YAWS = [0, 90, 180, 270]  # degrees, CCW about +y

def render_view(pos, yaw_deg):
    st = habitat_sim.AgentState()
    st.position = np.array(pos, dtype=np.float32)
    st.rotation = quat_from_angle_axis(math.radians(yaw_deg), np.array([0.0, 1.0, 0.0]))
    agent.set_state(st)
    obs = sim.get_sensor_observations()
    return Image.fromarray(obs["rgb"][..., :3])

def goal_bearing(final_pos, goals):
    """Compass yaw (deg, matching our YAW convention) toward nearest goal, + dist."""
    fp = np.array(final_pos)
    best = min(goals, key=lambda g: np.linalg.norm(np.array(g) - fp))
    d = np.array(best) - fp
    # habitat: -z is forward at yaw 0; +yaw rotates CCW (toward +x... we just
    # report the horizontal bearing in the same axis-angle convention we render)
    yaw = math.degrees(math.atan2(-d[0], -d[2]))  # 0 = -z, matches quat_from_angle_axis(+y)
    return (yaw % 360), float(np.linalg.norm([d[0], d[2]]))

for scene, srows in by_scene.items():
  print(f"[render] scene {os.path.basename(scene)}: {len(srows)} episodes")
  sim, agent = init_simulator(scene, width=480, height=480, fov_deg=90.0,
                              sensor_height=SENSOR_H, agent_radius=0.0)
  for r in srows:
    fp = r["final_pos"]
    gb_yaw, gb_dist = goal_bearing(fp, r["goals"])
    views = [render_view(fp, y) for y in YAWS]
    W, H = views[0].size
    pad, top = 8, 78
    canvas = Image.new("RGB", (W * 2 + pad, H * 2 + pad + top), (22, 26, 32))
    dr = ImageDraw.Draw(canvas)
    dr.text((10, 8), f"{r['target']}   [{r['id'].split('__')[-1]}]", font=font, fill=(240, 240, 240))
    dr.text((10, 36), f"query: {r['query']}    stopped {gb_dist:.1f} m from nearest goal "
                       f"(bearing {gb_yaw:.0f}deg)    #goals={len(r['goals'])}",
            font=fsm, fill=(170, 200, 210))
    # which quadrant points at the goal
    near_yaw = min(YAWS, key=lambda y: min((y - gb_yaw) % 360, (gb_yaw - y) % 360))
    for i, (y, v) in enumerate(zip(YAWS, views)):
        x0 = (i % 2) * (W + pad)
        y0 = top + (i // 2) * (H + pad)
        canvas.paste(v, (x0, y0))
        lbl = f"yaw {y}deg" + ("   <- toward goal" if y == near_yaw else "")
        col = (120, 230, 160) if y == near_yaw else (210, 210, 210)
        dr.rectangle([x0, y0, x0 + 150, y0 + 22], fill=(0, 0, 0))
        dr.text((x0 + 6, y0 + 3), lbl, font=fsm, fill=col)
    out = f"{OUT}/{r['id']}.png"
    canvas.save(out)
    print(f"  wrote {out}  ({r['target']}, {gb_dist:.1f}m out)")
  sim.close()
print("[render] done")
