import os
# os.environ['taskset'] = '-c 112-127'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
os.environ['GLOG_minloglevel'] = '2'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

import gc
import time
import torch
import numpy as np

from src.config import Config
from src.habitat.habitat_utils import (
    init_simulator,
    get_scene_bounds_from_pathfinder,
    spawn_agent_at_random_navpoint,
    spawn_agent_at_pos
)
from src.habitat.sim_interface import SimInterface
from src.perception.perception_stack import PerceptionStack


def spin_policy(perception: PerceptionStack) -> list:
    return [0.0, 5.0]


def run(cfg: Config, policy_fn=spin_policy, save_enabled: bool = True) -> None:
    sim, agent = init_simulator(cfg.scene_path, resolution=cfg.img_width, fov_deg=cfg.fov)
    spawn_agent_at_random_navpoint(sim, agent)
    target_point = np.array([0.0, 0.0, 1.0])
    # spawn_agent_at_pos(sim, agent, target_point)
    scene_bounds = get_scene_bounds_from_pathfinder(sim)

    sim_iface = SimInterface(cfg, sim, agent)
    perception = PerceptionStack(cfg, scene_bounds)

    perception.target_query = "a pillow"

    import cv2
    img = sim_iface.get_observations()[0]
    rgb_np = (img.numpy() * 255).astype(np.uint8)
    cv2.imwrite("debug_spawn.png", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

    # Seed the replay buffer before training starts
    min_frames = 3
    print(f"Seeding replay buffer with {min_frames} frames...")
    for i in range(min_frames):
        sim_iface.step(policy_fn(perception))
        pts, feats, depth, c2w = perception.observe(sim_iface)
        perception.update_replay_buffer(pts, feats)
        perception.update_occupancy(depth, c2w, sim_iface.intrinsics)
        print(f"  Buffered seed frame {i + 1}/{min_frames}")
        torch.cuda.empty_cache()

        # save current view 2d maps
        rgb, depth, c2w = sim_iface.get_observations()
        perception.save_2d_similarity(i, depth, c2w, sim_iface.intrinsics)
        # save rgb img
        rgb_np = (rgb.numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"{cfg.output_dir}/rgbs/rgb_{i:03d}.png", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

    refresh_interval = cfg.hash_buffer_refresh_interval
    super_pts, super_feats = perception.make_super_batch()
    start_time = time.time()

    for step in range(cfg.iterations + 1):

        if step % refresh_interval == 0:
            if step > 0:
                sim_iface.step(policy_fn(perception))
                pts, feats, depth, c2w = perception.observe(sim_iface)
                perception.update_replay_buffer(pts, feats)
                perception.update_occupancy(depth, c2w, sim_iface.intrinsics)
                print(f"Step {step}: buffer {perception.buffer_size}/{cfg.hash_replay_buffer_size}, "
                      f"frames {sim_iface.frames_processed}")
                gc.collect()
                torch.cuda.empty_cache()

            if super_pts is not None:
                del super_pts, super_feats
                torch.cuda.empty_cache()

            super_pts, super_feats = perception.make_super_batch()
            gc.collect()
            torch.cuda.empty_cache()

        avg_loss = perception.train_step(super_pts, super_feats)

        if step % 100 == 0:
            print(f"  Step {step:05d} | Loss: {avg_loss:.5f} | Time: {time.time() - start_time:.1f}s")

        if save_enabled and step % cfg.viz_interval == 0:
            perception.update_maps(step)
            # save current view 2d maps
            rgb, depth, c2w = sim_iface.get_observations()
            perception.save_2d_similarity(step, depth, c2w, sim_iface.intrinsics)
            # save rgb img
            rgb_np = (rgb.numpy() * 255).astype(np.uint8)
            cv2.imwrite(f"{cfg.output_dir}/rgbs/rgb_{step:03d}.png", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
            # start_time = time.time()

    perception.save_models()
    sim.close()
    print("Training complete.")


if __name__ == "__main__":
    runner_cfg = Config("config/config.yaml")
    run(runner_cfg, policy_fn=spin_policy, save_enabled=True)