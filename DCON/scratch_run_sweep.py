import sys, json
sys.path.insert(0, "/workspace/DCON")
import numpy as np
from PIL import Image
from src.perception.obj_detection import LLMDetDetector, CLIPSegDetector

cands = json.load(open("/workspace/DCON/scratch_candidates.json"))
llmdet = LLMDetDetector(device="cuda", threshold=0.05)
clipseg = CLIPSegDetector(device="cuda", threshold=0.5)

top_fracs = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

def process(label, c):
    path = c["frame_path"].replace("output/", "/workspace/DCON/output/", 1) if not c["frame_path"].startswith("/") else c["frame_path"]
    if not path.startswith("/workspace"):
        path = "/workspace/DCON/" + c["frame_path"]
    try:
        arr = np.asarray(Image.open(path).convert("RGB"))
    except Exception as e:
        print(f"  SKIP {c['id']}: cannot load frame {path}: {e}")
        return None
    dets = llmdet.detect_all(arr, c["query"])
    if not dets:
        print(f"  SKIP {c['id']}: no LLMDet box at all (even at 0.05)")
        return None
    score, box = dets[0]
    mask = clipseg.segment(arr, c["query"])
    scores = {f"top{int(tf*100)}": clipseg.score_in_box(mask, box, "topk", tf) for tf in top_fracs}
    scores["max"] = clipseg.score_in_box(mask, box, "max")
    result = {"id": c["id"], "label": label, "query": c["query"], "llmdet_score": score, "scores": scores}
    print(f"  {label} {c['id']:40s} llmdet={score:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    return result

results = []
print("=== TP ===")
for c in cands["tp"]:
    r = process("TP", c)
    if r:
        results.append(r)
print("=== FP ===")
for c in cands["fp"]:
    r = process("FP", c)
    if r:
        results.append(r)

json.dump(results, open("/workspace/DCON/scratch_sweep_results.json", "w"), indent=2)
print(f"\nSaved {len(results)} results")
