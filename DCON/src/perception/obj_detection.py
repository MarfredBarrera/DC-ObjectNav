"""Object detection: `LLMDetDetector.detect(image, query) -> (score, box)`.

MM-Grounding-DINO with training-free attention sinks (Ruis et al., ICLR 2026)
for background false-positive mitigation; box-level acceptance is then further
gated by the pairwise CLIPSeg field (main.py detect_classify_latch). The
YOLO-World / COCO-YOLO / VLFM-hybrid backends, the LocateAnything→LLMDet
verification cascade, and the CLIPSegDetector verifier candidate were all
removed; see git history if one needs to be resurrected.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
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
    pairwise CLIPSeg field instead (`cfg.clipseg_pairwise`, semantics.py).
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


def make_detector(cfg):
    """Build the detector from config."""
    return LLMDetDetector.from_config(cfg)
