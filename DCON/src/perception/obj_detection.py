"""Object detectors. All share `detect(image, query) -> (score, box)`.

- `LLMDetDetector`: MM-Grounding-DINO with training-free attention sinks
  (Ruis et al., ICLR 2026) for background false-positive mitigation.
- `LocateAnythingDetector`: NVIDIA LocateAnything-3B generative grounding.
- `CascadeDetector`: LocateAnything proposes, LLMDet verifies — a frame counts
  only when both agree on the same region, cutting the geometric-look-alike
  false positives that survive LLMDet's sinks.

`make_detector(cfg)` picks LLMDet alone (default) or the cascade
(`cfg.detector_cascade`). The YOLO-World / COCO-YOLO / VLFM-hybrid backends were
removed earlier; see git history if one needs to be resurrected.
"""

import re
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def to_uint8_rgb(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> np.ndarray:
    """Coerce PIL / numpy / torch input to a uint8 (H, W, 3) RGB array.

    Float arrays are expected in [0, 1]; uint8 in [0, 255].
    """
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.0 + 1e-3:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got shape {arr.shape}")
    return arr


class LLMDetDetector:
    """Open-vocabulary detector using LLMDet (Fu et al., 2025) with the
    training-free attention-sink false-positive mitigation of Ruis et al.
    (ICLR 2026).

    LLMDet is an early-fusion grounding detector built on MM-Grounding-DINO with
    an LLM-supervised text backbone. Load the `iSEE-Laboratory/llmdet_*` weights,
    which declare `model_type="mm-grounding-dino"` and load natively (transformers
    >= 4.52) as `MMGroundingDinoForObjectDetection` through the standard
    `AutoModelForZeroShotObjectDetection` interface. NOTE: the `fushh7/*_hf`
    weights instead declare plain `grounding-dino`, whose contrastive head lacks
    MM-GDINO's bias + feature normalization — they load with 0 missing keys but
    produce non-discriminative (~0.5 everywhere) logits, so they are NOT usable.
    Per the paper, early-fusion detectors confidently hallucinate the prompted
    class on background images, because their vision-language fusion layers cannot
    select "no token" when nothing matches — irrelevant class signal smears across
    the vision tokens and the head picks the most prevalent one.

    The fix (paper Appendix A.1): append N semantically-neutral *attention sink*
    tokens to the prompt and treat them as competing classes. Excess attention
    routes to the sinks, so a box whose strongest phrase is a sink is "none of
    the above" and is dropped; the query box survives only if it out-scores every
    sink. We reuse the model's `[unused*]` vocabulary slots as sinks (no
    tokenizer resize) and re-initialise their word embeddings once at
    construction. NOTE: a local sweep on this HF port found the embedding init
    ("special" vs "mean") effectively INERT — the BERT text encoder
    recontextualizes the [unused] tokens, washing out the word-embedding init —
    and only `num_sinks=48` improves over no-sinks (8/24 over-suppress true
    positives). The paper's ~24/special-init optima are for the native mmdet
    LLMDet; they don't transfer to this port, so tune `num_sinks`/`threshold`
    here rather than trusting the paper's numbers.

    The sink mechanism here lives inside the detector's own vision-language
    fusion layers on the full image, which is exactly what the paper validated.

    NOTE (2026-07-12 probe): do NOT append semantic distractor phrases to the
    prompt alongside the sinks — the 48-sink suffix tokenizes to ~240
    wordpieces (each "[unusedN]" splits into ~5, so the re-initialized sink
    embeddings never actually appear in the tokenized prompt, which is why
    `sink_init` measured inert) and crushes every non-lead phrase's per-box
    score, silently disabling any inter-phrase comparison. A sink-free
    multi-phrase prompt discriminates cleanly (couch box: "a couch"=0.607 /
    "a bed"=0.036 in either phrase order) but changes the score distribution
    τ was calibrated on. A distractor-phrase gate built this way was
    implemented and then removed per user decision (2026-07-12; see
    handoff.md work package 3); look-alike suppression lives in the
    contrastive CLIPSeg field target instead (`cfg.clipseg_contrastive`,
    semantics.py).
    """

    name = "llmdet"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "iSEE-Laboratory/llmdet_large",
        threshold: float = 0.42,
        use_sinks: bool = True,
        num_sinks: int = 48,
        sink_init: str = "special",
        sink_special_str: str = "[()]",
    ):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.device = device
        self.threshold = float(threshold)
        self.use_sinks = bool(use_sinks)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer

        # Reuse [unused*] vocab slots as attention sinks (paper A.1): no resize,
        # and they carry no semantic meaning. Skipped entirely when use_sinks is
        # off (vanilla LLMDet, for A/B against the sink-gated variant).
        self.sink_tokens: list = []
        self.sink_ids: list = []
        if self.use_sinks and int(num_sinks) > 0:
            self._install_sinks(int(num_sinks), str(sink_init).lower(), sink_special_str)
        # The sinks never change per query, so build the prompt suffix once.
        self._sink_suffix = (
            (" " + ". ".join(self.sink_tokens) + ".") if self.sink_ids else ""
        )
        self._current_query: Optional[str] = None
        self._prompt: Optional[str] = None

    @classmethod
    def from_config(cls, cfg) -> "LLMDetDetector":
        """Construct from the global Config's llmdet_* knobs."""
        return cls(
            device=cfg.device,
            model_name=cfg.llmdet_model_name,
            threshold=cfg.llmdet_threshold,
            use_sinks=cfg.llmdet_use_sinks,
            num_sinks=cfg.llmdet_num_sinks,
            sink_init=cfg.llmdet_sink_init,
            sink_special_str=cfg.sink_special_str,
        )

    def _install_sinks(self, num_sinks: int, sink_init: str, special_str: str) -> None:
        emb = self.model.model.text_backbone.embeddings.word_embeddings.weight  # (V, D)
        toks = [f"[unused{i}]" for i in range(num_sinks)]
        ids = self.tokenizer.convert_tokens_to_ids(toks)
        unk = self.tokenizer.unk_token_id
        # Keep only sink tokens that exist as their own vocab id (not [UNK]).
        keep = [(t, i) for t, i in zip(toks, ids) if i is not None and i != unk]
        if not keep:
            print("[LLMDet] no [unused*] slots in vocab; running without sinks")
            return
        self.sink_tokens = [t for t, _ in keep]
        self.sink_ids = [i for _, i in keep]
        with torch.no_grad():
            if sink_init == "mean":
                v = emb.mean(dim=0)
            elif sink_init == "special":
                sp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(special_str))
                v = emb[sp].mean(dim=0)
            else:
                raise ValueError(
                    f"Unknown llmdet sink_init {sink_init!r} (expected 'mean' or 'special')")
            for sid in self.sink_ids:
                emb[sid] = v
        print(f"[LLMDet] installed {len(self.sink_ids)} attention sinks (init={sink_init})")

    @staticmethod
    def _phrase_spans(token_strs: list) -> list:
        """Group token indices into phrase spans separated by '.', skipping
        special tokens. Returns [[query idxs], [sink0 idxs], ...] in prompt
        order, so span 0 is always the query and the rest are sinks."""
        spans: list = []
        cur: list = []
        for i, t in enumerate(token_strs):
            if t in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                continue
            if t == ".":
                if cur:
                    spans.append(cur)
                    cur = []
            else:
                cur.append(i)
        if cur:
            spans.append(cur)
        return spans

    @torch.no_grad()
    def detect_all(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> list:
        """Every box that survives the sink gate + `self.threshold`, as
        `(score, (xmin, ymin, xmax, ymax))` in input-image pixels, sorted by
        score descending (best first). `detect` returns the top one; the
        verification cascade scans the list for a box overlapping a proposal, so
        a true positive isn't lost just because some look-alike elsewhere in the
        frame out-scored it. Empty list if nothing survives.
        """
        if query != self._current_query:
            self._current_query = query
            self._prompt = query.strip().rstrip(".") + "." + self._sink_suffix

        arr = to_uint8_rgb(image)
        pil = Image.fromarray(arr)
        h, w = arr.shape[:2]

        inputs = self.processor(images=pil, text=self._prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        prob = outputs.logits.sigmoid()[0]              # (num_boxes, num_tokens)
        boxes = outputs.pred_boxes[0]                   # (num_boxes, 4) cxcywh in [0, 1]
        token_strs = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        spans = self._phrase_spans(token_strs)
        if not spans:
            return []

        qscore = prob[:, spans[0]].max(dim=1).values    # (num_boxes,) per-box query score
        if self.sink_ids and len(spans) > 1:
            # Strongest sink phrase per box; a box only survives if the query
            # out-scores every sink ("none of the above" → dropped).
            sink_score = torch.stack(
                [prob[:, s].max(dim=1).values for s in spans[1:]], dim=0
            ).max(dim=0).values                          # (num_boxes,)
            keep = (qscore >= sink_score) & (qscore >= self.threshold)
        else:
            keep = qscore >= self.threshold

        if not bool(keep.any()):
            return []
        dets = []
        for i in torch.where(keep)[0].tolist():
            cx, cy, bw, bh = boxes[i].tolist()
            x0, y0 = (cx - bw / 2.0) * w, (cy - bh / 2.0) * h
            x1, y1 = (cx + bw / 2.0) * w, (cy + bh / 2.0) * h
            dets.append((float(qscore[i]),
                         (float(x0), float(y0), float(x1), float(y1))))
        dets.sort(key=lambda t: t[0], reverse=True)
        return dets

    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        """Best single detection on `image` for text `query`.

        Returns `(score, box)` where `score` ∈ [0, 1] and `box` is
        (xmin, ymin, xmax, ymax) in pixel coordinates of the input image, or
        `(0.0, None)` if nothing survives the sink gate + `self.threshold`.
        """
        dets = self.detect_all(image, query)
        return dets[0] if dets else (0.0, None)


class LocateAnythingDetector:
    """Open-vocabulary detector using NVIDIA LocateAnything-3B via Hugging Face.

    A generative grounding model (Qwen-VL-style): it is prompted to locate the
    query object and emits `<box><x1><y1><x2><y2></box>` special tokens, which we
    parse into one box with a binary 1.0/0.0 score (1.0 iff it emitted a box).
    Architecturally very different from the DINO-based LLMDet, so its false
    positives are largely uncorrelated — which is what makes it useful as the
    proposer in `CascadeDetector`.
    """

    name = "locate_anything"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "nvidia/LocateAnything-3B",
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.3,
    ):
        import glob
        import os
        # Patch the downloaded remote code for Python 3.9 (it uses PEP 604
        # `X | None` type hints at module import time).
        for f in glob.glob("/root/.cache/huggingface/modules/transformers_modules/"
                           "nvidia/LocateAnything_hyphen_3B/*/*.py"):
            os.system(f"grep -q 'from __future__ import annotations' {f} || "
                      f"sed -i '1s/^/from __future__ import annotations\\n/' {f}")

        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        self.device = device
        self.model_name = model_name
        self.dtype = torch.bfloat16
        # The model's MTP/AR decode loop frequently fails to emit <|im_end|> and
        # instead repeats one box until the token budget runs out, so generate()
        # always runs to max_new_tokens — keep it small to bound per-call latency
        # (~1 s at 128). The meaningful, tight boxes are emitted first; the tail
        # degenerates into oversized/full-image repeats. repetition_penalty only
        # affects already-seen tokens, so the leading box stays pristine.
        self.max_new_tokens = int(max_new_tokens)
        self.repetition_penalty = float(repetition_penalty)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor.tokenizer = self.tokenizer
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=self.dtype,
        ).to(device)
        self.model.eval()

    @classmethod
    def from_config(cls, cfg) -> "LocateAnythingDetector":
        return cls(
            device=cfg.device,
            model_name=cfg.locate_anything_model_name,
            max_new_tokens=cfg.locate_anything_max_new_tokens,
        )

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        arr = to_uint8_rgb(image)
        pil = Image.fromarray(arr).convert("RGB")
        w, h = pil.size

        prompt = ("Locate all the instances that matches the following "
                  f"description: {query}.")
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": pil},
                        {"type": "text", "text": prompt}],
        }]

        # LocateAnything ships its own chat template + vision-info helpers; the
        # generic HF ones don't emit the <image-N> placeholders it needs.
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt",
        ).to(self.device)

        # NOTE: model.generate() runs its own MTP/AR decode loop and returns an
        # already-decoded response *string* (not token ids), so we must NOT call
        # processor.decode() on it. pixel_values is cast to the model dtype
        # internally. image_grid_hws stays a numpy array (generate handles it).
        response = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            use_cache=True,
            generation_mode="hybrid",
        )

        # Boxes are emitted as <box><x1><y1><x2><y2></box>, each coordinate a
        # 0-999 normalized integer special token (<box>None</box> when nothing
        # matches). Take the FIRST box: the model emits its highest-priority,
        # tightest detection first, then degenerates into repeated oversized /
        # full-image boxes — "largest" would pick noise.
        m = re.search(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", response)
        if m is None:
            return 0.0, None
        x1, y1, x2, y2 = (int(g) / 1000.0 for g in m.groups())
        # Corners occasionally come out of order; sort so the box is always
        # (xmin, ymin, xmax, ymax) for every downstream consumer (IoU, CLIP
        # crop, BEV projection, PIL rectangle in viz).
        xs = sorted((x1 * w, x2 * w))
        ys = sorted((y1 * h, y2 * h))
        return 1.0, (xs[0], ys[0], xs[1], ys[1])


class CLIPSegDetector:
    """Open-vocabulary segmentation-based verifier using CLIPSeg (Lueddecke &
    Ecker, 2022).

    Unlike LLMDet/LocateAnything, CLIPSeg is not a box detector: a lightweight
    transformer decoder, trained on top of *frozen* CLIP, predicts a dense
    per-pixel activation map for a text prompt. There is no box regression and
    no vision-language early-fusion — architecturally and in training data it
    is unlike both LLMDet (grounding-DINO) and LocateAnything (generative
    Qwen-VL-style), so its failure modes are plausibly uncorrelated with
    either, which is what would make it useful as a second-stage verifier: a
    proposal box counts only if the region also carries high CLIPSeg
    activation for the same query.

    No boxes of its own, so `detect`/`detect_all` derive one from the
    thresholded activation mask's bounding box (largest connected blob) purely
    for interface parity with the other detectors; the intended use is
    `verify()` against an already-proposed box.
    """

    name = "clipseg"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "CIDAS/clipseg-rd64-refined",
        threshold: float = 0.5,
    ):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        self.device = device
        self.threshold = float(threshold)
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
        self.model.eval()

    @classmethod
    def from_config(cls, cfg) -> "CLIPSegDetector":
        return cls(
            device=cfg.device,
            model_name=cfg.clipseg_model_name,
            threshold=cfg.clipseg_threshold,
        )

    @torch.no_grad()
    def segment(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> torch.Tensor:
        """Dense sigmoid activation map for `query`, upsampled (bilinear) to
        the input image's (H, W). Values in [0, 1]; higher = more relevant.
        CLIPSeg's native output is a low-res (352/16=22^2-ish) logit grid."""
        arr = to_uint8_rgb(image)
        pil = Image.fromarray(arr)
        h, w = arr.shape[:2]

        inputs = self.processor(text=[query], images=[pil], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits          # (H', W') for a single prompt
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        mask = torch.sigmoid(logits[0])  # (H', W')
        mask = F.interpolate(
            mask[None, None], size=(h, w), mode="bilinear", align_corners=False,
        )[0, 0]
        return mask

    @staticmethod
    def score_in_box(
        mask: torch.Tensor,
        box: Tuple[float, float, float, float],
        reduce: str = "mean",
        top_frac: float = 0.2,
    ) -> float:
        """Reduce `mask` (H, W) over `box` (xmin, ymin, xmax, ymax) in pixel
        coords. `reduce`: "mean" (region-wide activation, robust to a box
        that's slightly loose but diluted by background inside a loose box),
        "max" (peak activation anywhere in-box, a single-pixel statistic so
        noisy), or "topk" (mean of the top `top_frac` fraction of in-box
        pixels — a middle ground: robust to a loose box like mean, but not
        swamped by background the way a whole-box mean is)."""
        h, w = mask.shape
        x0 = max(0, int(round(box[0])))
        y0 = max(0, int(round(box[1])))
        x1 = min(w, max(x0 + 1, int(round(box[2]))))
        y1 = min(h, max(y0 + 1, int(round(box[3]))))
        crop = mask[y0:y1, x0:x1]
        if crop.numel() == 0:
            return 0.0
        if reduce == "mean":
            return float(crop.mean())
        if reduce == "max":
            return float(crop.max())
        if reduce == "topk":
            flat = crop.reshape(-1)
            k = max(1, int(round(flat.numel() * top_frac)))
            return float(torch.topk(flat, k).values.mean())
        raise ValueError(f"Unknown reduce {reduce!r} (expected 'mean', 'max', or 'topk')")

    def verify(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
        box: Tuple[float, float, float, float],
        reduce: str = "mean",
        top_frac: float = 0.2,
    ) -> Tuple[bool, float]:
        """Does `query`'s CLIPSeg activation over `box` clear `self.threshold`?
        Returns (accepted, score). Intended as a verifier over a proposal box
        from another detector (mirrors CascadeDetector's role for LLMDet)."""
        mask = self.segment(image, query)
        score = self.score_in_box(mask, box, reduce=reduce, top_frac=top_frac)
        return score >= self.threshold, score

    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        """Bounding box of the largest thresholded activation blob, for
        interface parity with the other detectors. Score is the mean
        activation inside that box. `(0.0, None)` if nothing clears
        `self.threshold`."""
        mask = self.segment(image, query)
        keep = (mask >= self.threshold).cpu().numpy()
        if not keep.any():
            return 0.0, None
        ys, xs = np.where(keep)
        box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        return self.score_in_box(mask, box), box


class CascadeDetector:
    """Two-stage detector: a generative *proposer* (LocateAnything) gates a
    sink-gated *verifier* (LLMDet). A frame counts as a detection only when both
    fire on the same object — the proposer emits a box AND the verifier has a
    sink-gated box, clearing its threshold τ, that spatially overlaps the
    proposal (IoU ≥ `min_iou`). The returned `(score, box)` is the *verifier's*,
    so τ (`llmdet_threshold`), the latch, and the
    goal-projection logic downstream are all unchanged — this only changes
    *which* frames register as detections and drive `w_conf`.

    Rationale: LLMDet alone confidently fires on geometric look-alikes of the
    target (couch↔bed, trashcan↔toilet); its attention sinks absorb attention
    leaking onto featureless background but are structurally blind to a genuine
    look-alike that produces a high query score. LocateAnything is a different
    architecture, so its errors are less correlated — requiring both to agree on
    the same region removes look-alike FPs that survive either detector alone.
    Cost: ~2x detector latency in SEARCH (both models run) and some recall (a
    true positive only one model catches is dropped). The proposer runs first
    and the verifier only when it fires, so cost is bounded by the proposal rate.
    """

    name = "cascade"

    def __init__(self, proposer, verifier, min_iou: float = 0.1):
        self.proposer = proposer      # LocateAnythingDetector-like: detect()
        self.verifier = verifier      # LLMDetDetector: detect_all()
        self.min_iou = float(min_iou)

    @classmethod
    def from_config(cls, cfg) -> "CascadeDetector":
        return cls(
            proposer=LocateAnythingDetector.from_config(cfg),
            verifier=LLMDetDetector.from_config(cfg),
            min_iou=cfg.cascade_min_iou,
        )

    @staticmethod
    def _iou(a, b) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        # Stage 1: proposer. Nothing proposed → no detection (verifier skipped,
        # so the second heavy model only runs when there's a candidate).
        _, p_box = self.proposer.detect(image, query)
        if p_box is None:
            return 0.0, None
        # Stage 2: verifier over the full image (its sink gate is validated on
        # whole images, not tight crops). detect_all is already ≥ τ + sink-gated.
        for score, v_box in self.verifier.detect_all(image, query):
            if self.min_iou <= 0.0 or self._iou(p_box, v_box) >= self.min_iou:
                return score, v_box
        # Proposal not confirmed by any verifier box → treat as a false positive.
        print("  [cascade] proposal rejected by verifier (no sink-gated box "
              "overlaps the LocateAnything proposal)")
        return 0.0, None


def make_detector(cfg):
    """Build the detector from config: the LocateAnything→LLMDet verification
    cascade when `cfg.detector_cascade`, else LLMDet alone (default)."""
    if cfg.detector_cascade:
        return CascadeDetector.from_config(cfg)
    return LLMDetDetector.from_config(cfg)
