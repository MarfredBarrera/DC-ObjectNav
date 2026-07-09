"""Multi-episode ObjectNav evaluation: Success Rate (SR) + SPL.

Reads a hand-annotated episodes file (YAML or JSON), runs each episode through
the same `main.run()` live loop used for a single navigation, and aggregates two
standard ObjectNav metrics:

  SR  = fraction of episodes where the agent self-stops within `success_radius_m`
        (geodesic) of a ground-truth target instance.
  SPL = mean over episodes of  S_i * l_i / max(p_i, l_i), where l_i is the
        geodesic start->nearest-goal distance and p_i is the distance the agent
        actually traveled. Efficient successes ~1; wandering ones lower; failures 0.

Episodes file format (see config/evaluation_configs/episodes.yaml):

    success_radius_m: 1.0          # optional; default 1.0
    episodes:
      - scene: gibson_scenes/Denmark.glb   # abs or repo-relative
        query: "a pillow"
        start: [0.0, -3.0, 2.5]            # spawn xyz (snapped to navmesh)
        goals: [[3.2, -3.0, -1.1]]         # >=1 real target instance xyz

Usage (inside the docker container):
    python benchmarks/evaluate.py --episodes config/evaluation_configs/episodes.yaml --gpu 0 --out output/eval_run1
"""

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import Config


def load_episodes(path: str):
    """Parse the episodes file. YAML if PyYAML is available and the extension is
    .yaml/.yml, otherwise JSON. Returns (success_radius_m, [episode dicts])."""
    with open(path, "r") as f:
        text = f.read()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        import yaml  # PyYAML; ships with the habitat stack
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if isinstance(data, list):
        # Bare list of episodes, no top-level radius.
        return 1.0, data
    return float(data.get("success_radius_m", 1.0)), data["episodes"]


def resolve_scene(scene: str) -> str:
    """Repo-relative scene paths -> absolute (run from the DCON dir)."""
    return scene if os.path.isabs(scene) else os.path.abspath(scene)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=str, required=True,
                        help="Path to the episodes YAML/JSON file")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device index")
    parser.add_argument("--out", type=str, default="output/eval",
                        help="Directory for results.json (and per-episode maps if --save)")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Base config YAML (overridden per-episode)")
    parser.add_argument("--save", action="store_true", default=False,
                        help="Save per-episode BEV maps/RGB (slow; off by default)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Import after CUDA_VISIBLE_DEVICES is set (init_simulator / torch pick it up).
    from main import run

    success_radius_m, episodes = load_episodes(args.episodes)
    os.makedirs(args.out, exist_ok=True)
    print(f"[eval] {len(episodes)} episode(s) | success_radius={success_radius_m}m")

    results = []
    for i, ep in enumerate(episodes):
        scene = resolve_scene(ep["scene"])
        query = ep["query"]
        start = ep["start"]
        goals = ep["goals"]
        print(f"\n[eval] === episode {i+1}/{len(episodes)} | "
              f"{os.path.basename(scene)} | '{query}' ===")

        cfg = Config(args.config)
        cfg.scene_path = scene
        cfg.target_query = query
        if args.save:
            cfg.output_dir = os.path.join(args.out, f"ep_{i:03d}")

        metrics = run(
            cfg, save_enabled=args.save, save_video=False,
            start_pos=start, goals=goals, success_radius_m=success_radius_m,
        )
        metrics["episode"] = i
        results.append(metrics)

        # Each run() builds a fresh ensemble + simulator and calls sim.close();
        # force cleanup so GPU memory doesn't creep across episodes.
        gc.collect()
        torch.cuda.empty_cache()

    n = len(results)
    sr = sum(1 for r in results if r["success"]) / n if n else 0.0
    spl = sum(r["spl"] for r in results) / n if n else 0.0

    summary = {
        "num_episodes": n,
        "success_rate": sr,
        "spl": spl,
        "success_radius_m": success_radius_m,
        "episodes": results,
    }
    out_path = os.path.join(args.out, "results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{'ep':>3} {'success':>8} {'spl':>6} {'l(m)':>7} {'path(m)':>8} {'scene/query'}")
    for r in results:
        print(f"{r['episode']:>3} {str(r['success']):>8} {r['spl']:>6.3f} "
              f"{r['l_geodesic']:>7.2f} {r['path_length']:>8.2f} "
              f"{os.path.basename(r['scene'])} / {r['query']}")
    print("-" * 60)
    print(f"SR  = {sr:.3f}   ({sum(1 for r in results if r['success'])}/{n})")
    print(f"SPL = {spl:.3f}")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
