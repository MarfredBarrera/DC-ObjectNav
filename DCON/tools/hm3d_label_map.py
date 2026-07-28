"""Extract OVON val_seen query categories + the HM3D semantic label vocabulary,
and propose candidate HM3D labels per OVON category (for the category-complete
synonym map). Output is for human curation, not a final map.
"""
import gzip, json, glob, os, re
from collections import Counter

EPI = "benchmarks/episodes/hm3d_ovon/hm3d/val_seen/content"
SEM = "benchmarks/scene_datasets/hm3d/val"

# --- OVON val_seen query categories (with episode frequency) ---
cat_eps = Counter()
for f in glob.glob(EPI + "/*.json.gz"):
    d = json.load(gzip.open(f))
    for e in d["episodes"]:
        cat_eps[e["object_category"]] += 1

# --- HM3D semantic label vocabulary across val scenes (instance frequency) ---
label_freq = Counter()
label_line = re.compile(r'^\s*\d+\s*,\s*[0-9A-Fa-f]+\s*,\s*"(.*?)"')
for tf in glob.glob(SEM + "/*/*.semantic.txt"):
    for line in open(tf, errors="ignore"):
        m = label_line.match(line)
        if m:
            label_freq[m.group(1).strip().lower()] += 1

print(f"OVON val_seen: {len(cat_eps)} query categories, {sum(cat_eps.values())} episodes")
print(f"HM3D val labels: {len(label_freq)} unique labels, {sum(label_freq.values())} instances\n")

STOP = {"a", "an", "the", "of", "with", "and", "or"}
def toks(s):
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t and t not in STOP]

print("=" * 78)
print("CANDIDATE HM3D LABELS PER OVON CATEGORY  (label  ×instances)")
print("=" * 78)
for cat, n in sorted(cat_eps.items()):
    ct = set(toks(cat))
    cands = []
    for lbl, lf in label_freq.items():
        lt = set(toks(lbl))
        # candidate if any shared token, or category-substring-of-label / vice-versa
        if ct & lt or cat in lbl or lbl in cat:
            cands.append((lbl, lf, len(ct & lt)))
    # exact match count
    exact = label_freq.get(cat, 0)
    cands.sort(key=lambda x: (-x[2], -x[1]))
    shown = ", ".join(f"{l}×{c}" for l, c, _ in cands[:12])
    star = "  [EXACT label absent in HM3D]" if exact == 0 else f"  [exact '{cat}'×{exact}]"
    print(f"\n• {cat}  ({n} eps){star}")
    print(f"    candidates: {shown if shown else '(none by token overlap)'}")
