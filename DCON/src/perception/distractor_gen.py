"""Distractor vocabulary for the pairwise-logit CLIPSeg field.

The pairwise field (`cfg.clipseg_pairwise`, the chosen CLIPSeg verification
method) regresses one sigmoid channel per prompt and gates a detection by the
worst-case margin

    margin = presence(query) - max_i presence(distractor_i)

taken at verify time on the multi-view-converged channels. This is the standard
CLIP relevancy formulation: a raw "target-like?" similarity only becomes a
relevancy score once it is contrasted against a bank of competing classes. That
bank has two roles, and this module builds both:

  1. BACKGROUND (`cfg.background_terms`) — generic scene-background prompts
     (a wall, the floor, the ceiling, a window, ...). These are the
     negative/background prompts every CLIP-relevancy pipeline uses; they
     absorb the structurally-high background level that would otherwise tax
     every box's margin, and they are query-independent.
  2. OBJECT CONFUSERS — objects a detector visually MISTAKES for the target:
     structurally-similar but DISTINCT-MATERIAL look-alikes (a bench/armchair
     for a chair, a trash can for a potted plant, a framed picture for a tv).
     These are the dominant false-positive mode and the reason a competing-
     class gate exists — a single-frame look-alike box fails the margin because
     its confuser channel out-scores the query.

The governing principle for the object confusers: a good distractor fires high
on the FALSE POSITIVE but LOW on a real target. A same-material sibling (a
bathtub/sink for a toilet) is a RECALL-KILLER — a real toilet scores high on it
too, collapsing the margin below tau — so those are deliberately excluded (the
LLM prompt deletes them; the static list carries none). "Structurally similar"
means visually confusable, NOT same-surface.

`build_distractor_vocabulary` is the entry point: it prepends the background
bank to the object confusers (static `cfg.distractor_objects`, or a per-target
LLM set when `cfg.llm_distractors` is on) and de-duplicates. On ANY failure in
the LLM path (model load, OOM, JSON parse) it falls back to the static list —
the run must never crash on distractor generation.
"""

import gc
import json
import re

import torch


# --- LLM prompt: elicit distinct-material look-alike confusers -------------
# A distractor must fire high on the FALSE POSITIVE but LOW on a real target,
# so same-material siblings (a bathtub/sink for a toilet) are recall-killers
# and are deleted — the object confusers are the visually-similar but
# distinct-material objects. (Generic backgrounds are NOT elicited here; the
# cfg.background_terms bank is added separately by build_distractor_vocabulary.)

_SYSTEM_PROMPT = """\
You generate "distractor" object phrases for an open-vocabulary object detector
operating on indoor home RGB images. A distractor for a target object T is a
DIFFERENT object that the detector commonly mistakes FOR T when viewed from a
bad angle or in clutter. Your distractors are used as COMPETING classes: the
detector accepts a box as T only if T out-scores every distractor on that box.

THE ONE GOVERNING PRINCIPLE: a good distractor fires high on a false positive
but LOW on the true target. So a distractor is only useful if a REAL T does NOT
itself look like it. An object that shares T's own material and surface is a
TERRIBLE distractor — a real T scores high on it too, which suppresses real
detections. This is the single most important thing to get right.

Work in two steps, then output only the result of step 2:
  STEP 1 (think, do not output): list objects commonly mistaken for T in RGB.
  STEP 2 (output): DELETE from that list every object that shares T's primary
    material or surface, because a real T would score high on it. Bathroom
    fixtures (toilet, bathtub, sink, shower, bidet) all share white porcelain,
    so NONE of them may distract another — for target "a toilet" you MUST drop
    "a bathtub", "a sink", "a shower", "a bidet". Upholstered seating (couch,
    sofa, armchair, ottoman) shares fabric cushions; wooden casegoods (cabinet,
    dresser, chest of drawers) share wood panels. Keep at most 2 same-material
    siblings and only the strongest ones.

HARD RULES:
1. NEVER output a synonym, sub-part, or component of T (for "toilet": no
   "toilet seat", "toilet bowl", "bidet lid"; for "bed": no "mattress",
   "headboard", "bedding").
2. NEVER output an object that shares T's primary surface/material such that a
   real T would score high on it (for "toilet": NO "bathtub", NO "sink", NO
   "shower", NO "bidet" — they share porcelain-fixture appearance with real
   toilets). This is the deletion in STEP 2.
3. Prefer distinct-material confusers (a cabinet, a countertop, framed art, a
   trash can) over near-identical ones.
4. Return 8-12 phrases, each an indefinite noun phrase like "a cabinet".

Return ONLY a JSON array of strings, nothing else."""

# Few-shot exemplars grounded in the observed pairwise-calib FP frames.
# Anchoring the two classes (safe vs. intra-cluster) and, most importantly,
# the exclusions is what steers Qwen off the recall-killers.
_FEWSHOT = [
    ("a toilet",
     '["an armchair", "an upholstered chair", "an ottoman", "a footstool", '
     '"a trash can", "a bucket", "a cabinet"]'),
    ("a bed",
     '["a couch", "a sofa", "a kitchen island", "a countertop", "a cabinet", '
     '"a dresser", "a bench"]'),
    ("a couch",
     '["an armchair", "a chair", "an ottoman", "a bench", "a cabinet", '
     '"a chest of drawers", "a bed", "a framed picture"]'),
]

_USER_TEMPLATE = ('Target object: "{query}"    Scene domain: '
                  'indoor home / apartment')


def _build_messages(query: str):
    """Assemble the chat-template messages: system + few-shot turns + query."""
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for ex_query, ex_answer in _FEWSHOT:
        messages.append({"role": "user",
                         "content": _USER_TEMPLATE.format(query=ex_query)})
        messages.append({"role": "assistant", "content": ex_answer})
    messages.append({"role": "user",
                     "content": _USER_TEMPLATE.format(query=query)})
    return messages


def _parse_distractors(text: str):
    """Extract a list of short indefinite noun phrases from the completion.

    Tolerates markdown fences / trailing prose Qwen may wrap around the array:
    grabs the first bracketed span and json-loads it. Returns None if nothing
    usable is found so the caller can fall back to the static list.
    """
    # Strip code fences the model sometimes emits (```json ... ```).
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Grab the first [...] span — the JSON array.
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return None
    try:
        arr = json.loads(match.group(0))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(arr, list):
        return None
    phrases = []
    for item in arr:
        if not isinstance(item, str):
            continue
        phrase = item.strip().lower()
        # Reject empty / overly long junk (a real phrase is a few words).
        if phrase and len(phrase.split()) <= 5:
            phrases.append(phrase)
    return phrases or None


def _dedupe(phrases):
    """Case-insensitive de-dupe preserving first-seen order."""
    seen = set()
    out = []
    for p in phrases:
        p = str(p).strip()
        key = p.lower()
        if not p or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def generate_distractors(query: str, cfg) -> list:
    """Per-target OBJECT confuser phrases from a local Qwen instruct model.

    Returns only the structural object confusers (the background bank is added
    by `build_distractor_vocabulary`). Runs once per run: loads a local HF
    instruct model (`cfg.llm_distractor_model`), greedy-decodes a JSON array for
    `query`, then frees the model + empties the CUDA cache BEFORE the feature
    field / super-batch claim GPU memory. No network at inference (weights come
    from the HF hub cache) and no disk cache of the generated list.

    Falls back to `cfg.distractor_objects` on any error (load, OOM, parse) with
    a clear warning — the run must never crash here.
    """
    model = None
    tokenizer = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = cfg.llm_distractor_model
        print(f"[distractor_gen] loading {model_name} for per-target "
              f"distractors (query={query!r})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=cfg.device,
        )
        model.eval()

        messages = _build_messages(query)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,             # greedy => deterministic per prompt
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly-generated continuation.
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        phrases = _parse_distractors(completion)
        if not phrases:
            raise ValueError(f"could not parse a distractor list from "
                             f"completion: {completion!r}")
        # Cap to the configured upper bound (prompt asks for 8-12).
        phrases = phrases[:cfg.llm_distractor_count]
        print(f"[distractor_gen] generated {len(phrases)} object confusers for "
              f"{query!r}: {phrases}")
        return phrases
    except Exception as e:  # noqa: BLE001 — never crash the run on generation
        print(f"[distractor_gen] WARNING: LLM distractor generation failed "
              f"({type(e).__name__}: {e}); falling back to the static "
              f"cfg.distractor_objects list.")
        return list(cfg.distractor_objects)
    finally:
        # Free the instruct model before perception claims GPU memory.
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_distractor_vocabulary(query: str, cfg) -> list:
    """Assemble the full competing-class bank: background + object confusers.

    The generic `cfg.background_terms` bank is prepended to the object
    confusers (per-target LLM set when `cfg.llm_distractors`, else the static
    `cfg.distractor_objects`), then de-duplicated. The query's own category is
    NOT stripped here — `semantics.filter_distractors` does that per query
    inside CLIPSegSemantics, so this stays query-agnostic on the object side.
    """
    background = list(getattr(cfg, "background_terms", []) or [])
    objects = (generate_distractors(query, cfg)
               if cfg.llm_distractors else list(cfg.distractor_objects))
    return _dedupe(background + objects)
