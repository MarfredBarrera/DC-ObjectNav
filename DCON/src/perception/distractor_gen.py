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
  2. OBJECT CONFUSERS (`cfg.distractor_objects`) — objects a detector visually
     MISTAKES for the target: structurally-similar but DISTINCT-MATERIAL
     look-alikes (a bench/armchair for a chair, a trash can for a potted
     plant, a framed picture for a tv). These are the dominant false-positive
     mode and the reason a competing-class gate exists — a single-frame
     look-alike box fails the margin because its confuser channel out-scores
     the query.

The governing principle for the object confusers: a good distractor fires high
on the FALSE POSITIVE but LOW on a real target. A same-material sibling (a
bathtub/sink for a toilet) is a RECALL-KILLER — a real toilet scores high on it
too, collapsing the margin below tau — so those are deliberately excluded from
the static list. "Structurally similar" means visually confusable, NOT
same-surface.

(A per-target LLM-generated confuser variant existed and was removed — the
canonical maxj arm's τ_margin = 0.0 is calibrated against THIS static bank;
see git history if it needs to be resurrected. A materially different bank
warrants re-sweeping the margin threshold.)
"""


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


def build_distractor_vocabulary(cfg) -> list:
    """Assemble the full competing-class bank: background + object confusers.

    The generic `cfg.background_terms` bank is prepended to the static
    `cfg.distractor_objects` confusers, then de-duplicated. Query-agnostic:
    the query's own category is NOT stripped here — `semantics.
    filter_distractors` does that per query inside CLIPSegSemantics.
    """
    return _dedupe(list(cfg.background_terms) + list(cfg.distractor_objects))
