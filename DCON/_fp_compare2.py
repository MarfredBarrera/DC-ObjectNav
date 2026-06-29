import glob, numpy as np, torch
from PIL import Image
from src.config import Config
from src.perception.semantics import MaskCLIPSemantics
from src.perception.obj_detection import build_sink_bank, target_prob_from_sims

cfg=Config('config/config.yaml'); dev=cfg.device
mc=MaskCLIPSemantics(device=dev); q=cfg.target_query
temp=cfg.sink_softmax_temp; thr=cfg.sink_min_target_prob; top_pct=cfg.sink_top_pct
q_emb=mc.encode_text(q)
sinks=build_sink_bank(mc,init=cfg.sink_init,n_sinks=cfg.sink_num,special_str=cfg.sink_special_str,seed=cfg.sink_seed)
bank=torch.cat([q_emb,sinks],0)

def methods(crop):
    rgb=torch.from_numpy(crop).to(dev).float()/255.0
    P=mc.extract_dense_features(rgb).reshape(-1,512)
    # current: per-text topk cosine
    per=P@bank.T; k=max(1,int(round(top_pct*P.shape[0])))
    p_old,_=target_prob_from_sims(per.topk(k,dim=0).values.mean(0),temp)
    # mean
    ce=P.mean(0,keepdim=True); ce=ce/ce.norm(dim=-1,keepdim=True)
    p_mean,_=target_prob_from_sims((ce@bank.T).squeeze(0),temp)
    # NEW: per-patch target-vs-sink prob, then top-k% mean of those probs
    pp=torch.softmax(temp*(P@bank.T),dim=1)[:,0]      # (N,) per-patch target prob
    p_patchprob=pp.topk(k).values.mean().item()
    # NEW2: fraction of patches that robustly prefer target (prob>0.5)
    frac=(pp>0.5).float().mean().item()
    return p_mean,p_old,p_patchprob,frac

rows=[]
for f in sorted(glob.glob('output/current_scene/rgbs/rgb_*.png')):
    img=np.array(Image.open(f).convert('RGB')); H,W=img.shape[:2]
    for (x0,y0,x1,y1) in [(0,0,W//2,H//2),(W//2,0,W,H//2),(0,H//2,W//2,H),(W//2,H//2,W,H)]:
        rows.append(methods(img[y0:y1,x0:x1]))
a=np.array(rows)
print("background crops (N=%d) — want LOW for all:"%len(a))
print("  mean-pool prob:          mean=%.3f  pass(>=%.2f)=%.2f"%(a[:,0].mean(),thr,(a[:,0]>=thr).mean()))
print("  per-text topk (current): mean=%.3f  pass=%.2f"%(a[:,1].mean(),(a[:,1]>=thr).mean()))
print("  per-patch-prob topk:     mean=%.3f  pass=%.2f"%(a[:,2].mean(),(a[:,2]>=thr).mean()))
print("  frac patches prob>0.5:   mean=%.3f  pass(>=0.1)=%.2f"%(a[:,3].mean(),(a[:,3]>=0.1).mean()))
