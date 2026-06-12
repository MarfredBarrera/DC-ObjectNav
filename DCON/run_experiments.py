import os
import subprocess
import json
import numpy as np
import sys

# The user asked for "epistemic", "unseen", and "coverage"
ig_sources = ['epistemic', 'unseen', 'coverage']
num_trials = 3

def calculate_spl(log_path):
    with open(log_path, 'r') as f:
        lines = [line for line in f if line.strip()]
        
    if not lines: return 0.0, 0.0, False
    
    pos = []
    for line in lines:
        try:
            data = json.loads(line)
            pos.append(np.array(data['pos']))
        except:
            pass
            
    if len(pos) < 2: return 0.0, 0.0, False
        
    dist_taken = sum(np.linalg.norm(pos[i] - pos[i-1]) for i in range(1, len(pos)))
    
    last_data = json.loads(lines[-1])
    step = last_data.get('step', 0)
    
    # Check if target was logically reached
    # If the simulation terminated before max iterations (5000), it's a success
    # (since the agent breaks the loop upon target reached)
    success = step < 5000
    
    # Distance between start and actual reached target
    L = np.linalg.norm(pos[-1] - pos[0])
    
    spl = (1.0 if success else 0.0) * (L / max(dist_taken, L)) if dist_taken > 0 else 0.0
    return spl, dist_taken, success

results = {}

for ig in ig_sources:
    results[ig] = []
    for trial in range(num_trials):
        print(f"\n=========================================")
        print(f"=== Running IG: {ig}, Trial: {trial} ===")
        print(f"=========================================\n")
        
        mp4_path = f"./figs/nav_{ig}_trial{trial}.mp4"
        cmd = [
            "python", "main.py", "--query", "sink", 
            "--detector", "hybrid", "--ig-source", ig, 
            "--viz-output", mp4_path,
            "--no-save" # Skip saving to disk frequently during loop to save time, but it still saves the ending frames for visualize.py!
        ]
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = "4"
        
        # Flush to log
        sys.stdout.flush()
        
        # Run main.py
        subprocess.run(cmd, env=env)
        
        log_path = "/workspace/DCON/output/current_scene/traj_log.jsonl"
        if os.path.exists(log_path):
            spl, dist, success = calculate_spl(log_path)
            print(f"Result -> SPL: {spl:.3f}, Dist: {dist:.3f}, Success: {success}")
            results[ig].append({'trial': trial, 'spl': spl, 'dist': dist, 'success': success})
        else:
            print("traj_log.jsonl not found!")
            results[ig].append({'trial': trial, 'spl': 0.0, 'dist': 0.0, 'success': False})
        sys.stdout.flush()
        
print("\n=== FINAL RESULTS ===")
for ig, trials in results.items():
    spls = [t['spl'] for t in trials]
    successes = [t['success'] for t in trials]
    avg_spl = np.mean(spls) if spls else 0.0
    succ_rate = np.mean(successes) if successes else 0.0
    print(f"IG Source: {ig:10s} | Avg SPL: {avg_spl:.3f} | Success Rate: {succ_rate:.2f}")

    for t in trials:
        print(f"  Trial {t['trial']}: SPL {t['spl']:.3f}, Dist {t['dist']:.3f}, Success {t['success']}")
