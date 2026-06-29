"""Throwaway probe: does proper attention-sink surgery suppress FPs on LLMDet?"""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import os
m = os.environ.get('PROBE_MODEL', 'fushh7/llmdet_swin_tiny_hf')
proc = AutoProcessor.from_pretrained(m)
model = AutoModelForZeroShotObjectDetection.from_pretrained(m).eval()
tok = proc.tokenizer
emb = model.model.text_backbone.embeddings.word_embeddings.weight  # (V,768)
img = Image.open('output/current_scene/rgbs/rgb_000.png').convert('RGB')

N = 24
sink_tokens = [f'[unused{i}]' for i in range(N)]
sink_ids = tok.convert_tokens_to_ids(sink_tokens)
orig = emb[sink_ids].clone()


def set_init(kind):
    with torch.no_grad():
        if kind == 'none':
            v = orig                     # restore originals (literal [unused] emb)
            for j, sid in enumerate(sink_ids):
                emb[sid] = orig[j]
            return
        if kind == 'mean':
            v = emb.mean(dim=0)
        elif kind == 'special':
            sp = tok.convert_tokens_to_ids(tok.tokenize('[()]'))
            v = emb[sp].mean(dim=0)
        for sid in sink_ids:
            emb[sid] = v


def phrase_spans(ids):
    toks = tok.convert_ids_to_tokens(ids)
    spans, cur = [], []
    for i, t in enumerate(toks):
        if t in ('[CLS]', '[SEP]'):
            continue
        if t == '.':
            if cur:
                spans.append(cur); cur = []
        else:
            cur.append(i)
    if cur:
        spans.append(cur)
    return spans  # [query, sink0, sink1, ...]


@torch.no_grad()
def run(query, kind, use_sinks=True):
    if use_sinks:
        set_init(kind)
        text = query + '. ' + '. '.join(sink_tokens) + '.'
    else:
        text = query + '.'
    inp = proc(images=img, text=text, return_tensors='pt')
    out = model(**inp)
    prob = out.logits.sigmoid()[0]                # (Q,L)
    spans = phrase_spans(inp.input_ids[0])
    qspan = spans[0]
    qscore = prob[:, qspan].max(dim=1).values     # (Q,)
    if use_sinks and len(spans) > 1:
        sscore = torch.stack([prob[:, s].max(dim=1).values for s in spans[1:]], 0).max(0).values
        keep = qscore >= sscore
        best = float(qscore[keep].max()) if keep.any() else 0.0
        stolen = int((~keep & (qscore > 0.2)).sum())
    else:
        best = float(qscore.max()); stolen = 0
    return best, stolen


print(f"{'query':14s} {'no-sink':>8} {'mean':>8} {'special':>9}  (best query score after gating)")
for q in ['chair', 'a sofa', 'refrigerator', 'an airplane', 'a tractor', 'a polar bear']:
    b0, _ = run(q, 'none', use_sinks=False)
    bm, sm = run(q, 'mean')
    bs, ss = run(q, 'special')
    print(f"{q:14s} {b0:8.3f} {bm:8.3f} {bs:9.3f}   stolen(mean={sm},special={ss})")
