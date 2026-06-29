"""Open-vocabulary object detectors.

All backends share the same interface — `detect(image, query) -> (score, box)` —
so the rest of the system can swap between them by config without further
plumbing. Pick one via `make_detector(name, ...)`.

- `ObjectDetector`: YOLO-Worldv2 (fast, ~15 ms/frame, class-name-style queries).
- `CocoYoloDetector`: closed-set COCO YOLOv8 (fast, high precision on the 80
  COCO classes, returns score=0 for queries that don't map to any COCO class).
- `GroundingDinoDetector`: Grounding DINO (slow, ~200 ms/frame, much better
  on natural-language phrases like "the green plant by the window").
- `HybridDetector`: routes COCO-matching queries to `CocoYoloDetector` and
  everything else to `ObjectDetector` (YOLO-World). Mirrors the VLFM paper's
  split, swapping Grounding DINO for YOLO-World on the open-vocab branch
  to stay in the ~15 ms regime.
"""

from typing import Optional, Tuple, Union
import time

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
import re


# Canonical distractor vocabulary (mirrors the default of
# cfg.det_negative_classes — kept here too so the standalone smoke test and
# direct construction get the same behavior without importing config).
DEFAULT_NEGATIVE_CLASSES = [
    "wall", "door", "window", "floor", "ceiling",
    "curtain", "cabinet", "shelf", "picture",
]


def _normalize_query(query: str) -> str:
    """Lower-case, strip leading article and trailing period."""
    q = query.strip().lower().rstrip(".")
    for art in ("a ", "an ", "the "):
        if q.startswith(art):
            q = q[len(art):]
            break
    return q


def _dedupe_negatives(query: str, negatives) -> list:
    """Drop negatives that normalize to the same noun as the query itself."""
    qn = _normalize_query(query)
    return [n for n in (negatives or []) if _normalize_query(n) != qn]


# (id(mask_clip), query, negatives) -> (K, 512) stacked text embeddings.
_TEXT_BANK_CACHE: dict = {}


def encode_query_with_negatives(mask_clip, query: str, negatives) -> torch.Tensor:
    """(K, 512) L2-normalized text bank: row 0 = `query`, rows 1.. = the
    deduped negatives as "a <noun>" prompts (repo's CLIP phrasing convention).
    Cached per (mask_clip, query, negatives) — text encoding is deterministic
    and the query rarely changes within a run."""
    negs = tuple(_dedupe_negatives(query, negatives))
    key = (id(mask_clip), query, negs)
    bank = _TEXT_BANK_CACHE.get(key)
    if bank is None:
        prompts = [query] + [f"a {n}" for n in negs]
        bank = torch.cat([mask_clip.encode_text(p) for p in prompts], dim=0)
        _TEXT_BANK_CACHE[key] = bank
    return bank


def target_prob_from_sims(sims: torch.Tensor, temp: float) -> Tuple[float, float]:
    """(target_prob, raw_target_cos) from a (K,) per-text cosine vector
    whose row 0 is the target query. `temp` is the CLIP-style logit scale.
    With K == 1 (no negatives) the softmax degenerates to prob 1.0 and the
    raw-cosine floor is the only gate, matching the pre-softmax behavior."""
    probs = torch.softmax(temp * sims, dim=0)
    return float(probs[0]), float(sims[0])


# Generic, category-spread vocabulary used to build the "mean" neutral sink:
# the CLIP-output-space analog of the paper's "mean of all word embeddings in
# vocabulary V" init. Deliberately ordinary words spanning many categories so
# their average text embedding lands on no object in particular.
_SINK_MEAN_VOCAB = [
    "thing", "object", "stuff", "area", "place", "surface", "material",
    "color", "shape", "pattern", "texture", "light", "shadow", "space",
    "background", "scene", "image", "picture", "view", "region", "part",
    "side", "edge", "corner", "center", "line", "spot", "mark", "point",
    "person", "animal", "plant", "tool", "food", "vehicle", "building",
    "room", "ground", "sky", "water", "metal", "wood", "plastic", "glass",
    "fabric", "paper", "stone", "dirt", "grass", "cloud",
]

# (id(mask_clip), init, n_sinks, special_str, seed) -> (S, 512) sink bank.
_SINK_BANK_CACHE: dict = {}


def build_sink_bank(
    mask_clip,
    init: str = "mean",
    n_sinks: int = 1,
    special_str: str = "[()]",
    seed: int = 0,
) -> torch.Tensor:
    """(S, 512) L2-normalized neutral *attention-sink* embeddings in CLIP's
    text-output space — the training-free "none of the above" reference from
    Ruis et al., "Fantastic Tractor-Dogs..." (ICLR 2026), adapted to a
    late-interaction CLIP gate.

    Sinks are intentionally NOT semantic negatives ("wall", "door"): the paper
    shows real negative classes do not suppress background false positives. A
    sink is semantically neutral and only serves as a calibrated floor the
    target query must out-score for a detection to survive.

    init strategies (mirroring the paper's three, mapped to CLIP output space):
      - "mean":    mean of CLIP text encodings over a generic, category-spread
                   vocabulary — the output-space analog of "mean of all word
                   embeddings". One sink. Best default for a CLIP comparison.
      - "special": encode a neutral special-character string (default "[()]",
                   the paper's best init for LLM-trained detectors). One sink.
      - "random":  `n_sinks` random unit vectors. Weak in CLIP's
                   near-orthogonal output space (cos ~0 to everything);
                   included for ablation parity with the paper.
    """
    key = (id(mask_clip), str(init).lower(), int(n_sinks), special_str, int(seed))
    cached = _SINK_BANK_CACHE.get(key)
    if cached is not None:
        return cached

    init = str(init).lower()
    if init == "special":
        bank = mask_clip.encode_text(special_str)                       # (1, 512)
    elif init == "mean":
        embs = torch.cat([mask_clip.encode_text(w) for w in _SINK_MEAN_VOCAB], dim=0)
        bank = embs.mean(dim=0, keepdim=True)                           # (1, 512)
    elif init == "random":
        ref = mask_clip.encode_text("object")
        g = torch.Generator(device=ref.device).manual_seed(int(seed))
        bank = torch.randn(int(n_sinks), ref.shape[1], generator=g,
                           device=ref.device, dtype=ref.dtype)
    else:
        raise ValueError(
            f"Unknown sink init {init!r} (expected 'mean', 'special', or 'random')")
    bank = bank / (bank.norm(dim=-1, keepdim=True) + 1e-8)
    _SINK_BANK_CACHE[key] = bank
    return bank


class ObjectDetector:
    """Open-vocabulary detector using YOLO-World (via Ultralytics).

    `negative_classes` are registered as competing classes alongside the
    query (the query is always class id 0); only boxes whose winning class
    is the query are accepted. Without competitors, YOLO-World's contrastive
    head matches salient non-target regions (walls, doors) to the only class
    available — registering distractors lets those regions be claimed by
    their own class instead.
    """

    name = "yolo_world"

    def backend_for(self, query: str) -> str:
        return self.name


    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "yolo/yolov8s-worldv2.pt",
        threshold: float = 0.1,
        imgsz: int = 640,
        half: bool = True,
        negative_classes: Optional[list] = None,
    ):
        self.device = device
        self.threshold = float(threshold)
        self.imgsz = int(imgsz)
        # fp16 only on CUDA; Ultralytics silently no-ops it on CPU.
        self.half = bool(half) and "cuda" in str(device)
        self.negative_classes = (
            list(negative_classes) if negative_classes is not None
            else list(DEFAULT_NEGATIVE_CLASSES)
        )
        self.model = YOLO(model_name)
        self.model.to(device)
        # set_classes() re-embeds the query through CLIP, which is non-trivial.
        # Cache the current query so back-to-back calls with the same text
        # skip the embedding step.
        self._current_query: Optional[str] = None

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        """Run detection on a single image with a single text query.

        Args:
            image: RGB image. Accepts PIL.Image, numpy uint8 (H, W, 3),
                or torch tensor (H, W, 3) in [0, 1] or [0, 255].
            query: Free-form text query, e.g. "a green plant".

        Returns:
            (score, box) where `score` ∈ [0, 1] and `box` is
            (xmin, ymin, xmax, ymax) in pixel coordinates of the input image.
            If no detection clears `self.threshold`, returns (0.0, None).
        """
        if query != self._current_query:
            # Query is always class id 0; negatives compete for the boxes.
            self.model.set_classes([query] + _dedupe_negatives(query, self.negative_classes))
            self._current_query = query

        arr = self._to_uint8(image)

        results = self.model.predict(
            arr,
            imgsz=self.imgsz,
            half=self.half,
            conf=self.threshold,
            verbose=False,
            device=self.device,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return 0.0, None

        # Keep only boxes claimed by the target class — a box that matches
        # "wall" better than the query is a suppressed false positive.
        target_idx = (r.boxes.cls == 0).nonzero(as_tuple=True)[0]
        if target_idx.numel() == 0:
            return 0.0, None
        confs = r.boxes.conf[target_idx]
        best = target_idx[int(torch.argmax(confs))]
        score = float(r.boxes.conf[best])
        xyxy = r.boxes.xyxy[best].tolist()
        box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        return score, box

    @staticmethod
    def _to_uint8(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"))
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image)
        # Float arrays expected in [0, 1]; uint8 in [0, 255].
        if arr.dtype != np.uint8:
            if float(arr.max()) <= 1.0 + 1e-3:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got shape {arr.shape}")
        return arr


class CocoYoloDetector:
    """Closed-set COCO detector using YOLOv8 (via Ultralytics).

    Restricts predictions to a single class derived from the query string.
    `query_to_class_id` normalizes the query (lower-case, strip leading
    article, strip trailing period) and looks it up in `self.model.names`,
    with a small synonym table for common ObjectNav targets that don't
    exactly match a COCO label. Returns (0.0, None) when the query can't
    be mapped to a COCO class — the hybrid wrapper uses that to fall back.
    """

    name = "coco_yolo"

    def backend_for(self, query: str) -> str:
        return self.name


    # Query strings that don't exactly match a COCO label but should still
    # route through the closed-set detector.
    _SYNONYMS = {
        "plant": "potted plant",
        "sofa": "couch",
        "television": "tv",
    }

    _ARTICLES = ("a ", "an ", "the ")

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "yolo/yolov8s.pt",
        threshold: float = 0.25,
        imgsz: int = 640,
        half: bool = True,
    ):
        self.device = device
        self.threshold = float(threshold)
        self.imgsz = int(imgsz)
        self.half = bool(half) and "cuda" in str(device)
        self.model = YOLO(model_name)
        self.model.to(device)
        # Inverse of model.names ({id: name}) so we can convert a class name
        # straight to the integer id Ultralytics wants in `classes=`.
        self.name_to_id = {name: idx for idx, name in self.model.names.items()}
        self._current_query: Optional[str] = None
        self._current_class_id: Optional[int] = None

    def query_to_class_id(self, query: str) -> Optional[int]:
        """Map a free-form query to a COCO class id, or None if no match."""
        q = query.strip().lower().rstrip(".")
        for art in self._ARTICLES:
            if q.startswith(art):
                q = q[len(art):]
                break
        q = self._SYNONYMS.get(q, q)
        return self.name_to_id.get(q)

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        if query != self._current_query:
            self._current_query = query
            self._current_class_id = self.query_to_class_id(query)
        if self._current_class_id is None:
            return 0.0, None

        arr = ObjectDetector._to_uint8(image)
        results = self.model.predict(
            arr,
            imgsz=self.imgsz,
            half=self.half,
            conf=self.threshold,
            classes=[self._current_class_id],
            verbose=False,
            device=self.device,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return 0.0, None
        confs = r.boxes.conf
        best = int(torch.argmax(confs))
        score = float(confs[best])
        xyxy = r.boxes.xyxy[best].tolist()
        box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        return score, box


class HybridDetector:
    """COCO-matching queries → closed-set YOLOv8; everything else → YOLO-World.

    Mirrors the VLFM paper's hybrid scheme. Routing decision is cached per
    query string, so back-to-back calls with the same query skip the
    normalization + lookup.
    """

    name = "hybrid"

    def backend_for(self, query: str) -> str:
        backend = self._route_cache.get(query)
        if backend is None:
            backend = self.coco if self.coco.query_to_class_id(query) is not None else self.world
            self._route_cache[query] = backend
        return backend.name


    def __init__(
        self,
        device: str = "cuda",
        coco_kwargs: Optional[dict] = None,
        world_kwargs: Optional[dict] = None,
        negative_classes: Optional[list] = None,
    ):
        # Negatives only apply to the open-vocab branch — the closed-set
        # COCO branch already has 80-class competition.
        world_kwargs = dict(world_kwargs or {})
        if negative_classes is not None:
            world_kwargs.setdefault("negative_classes", negative_classes)
        self.coco = CocoYoloDetector(device=device, **(coco_kwargs or {}))
        self.world = ObjectDetector(device=device, **world_kwargs)
        self._route_cache: dict = {}

    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        backend = self._route_cache.get(query)
        if backend is None:
            backend = self.coco if self.coco.query_to_class_id(query) is not None else self.world
            self._route_cache[query] = backend
        return backend.detect(image, query)


class GroundingDinoDetector:
    """Open-vocabulary detector using Grounding DINO via HuggingFace transformers.

    Slower than YOLO-World but considerably better at natural-language queries
    ("the small red mug on the table"). Score scale differs from YOLO-World —
      strong detections typically sit at 0.35–0.55 rather than 0.5–0.8, so any
    downstream threshold tuned for YOLO-World likely needs to be retuned.
    """

    name = "gdino"

    def backend_for(self, query: str) -> str:
        return self.name


    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        # Imported here so the YOLO-only path doesn't pay the transformers
        # import cost or require it to be installed.
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.device = device
        self.threshold = float(threshold)
        self.text_threshold = float(text_threshold)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
        self.model.eval()
        # Cache the last query text — re-tokenizing is cheap but the
        # post-processor expects the same string used at inference time.
        self._current_query: Optional[str] = None
        self._query_text: Optional[str] = None

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        # GD expects the prompt to end with a period (separator between phrases).
        if query != self._current_query:
            self._current_query = query
            self._query_text = query.strip().rstrip(".") + "."

        arr = ObjectDetector._to_uint8(image)
        pil = Image.fromarray(arr)

        inputs = self.processor(images=pil, text=self._query_text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # post_process expects (H, W) target sizes for box rescaling.
        h, w = arr.shape[:2]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            # box_threshold=self.threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]

        scores = results.get("scores")
        boxes = results.get("boxes")
        if scores is None or len(scores) == 0:
            return 0.0, None

        best = int(torch.argmax(scores))
        score = float(scores[best])
        xyxy = boxes[best].tolist()
        box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        return score, box


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

    Unlike the cropped-CLIP `SinkGatedDetector`, the sink mechanism here lives
    inside the detector's own fusion layers on the full image, which is exactly
    what the paper validated.
    """

    name = "llmdet"

    def backend_for(self, query: str) -> str:
        return self.name

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "iSEE-Laboratory/llmdet_tiny",
        threshold: float = 0.3,
        use_sinks: bool = True,
        num_sinks: int = 24,
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
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        if query != self._current_query:
            self._current_query = query
            self._prompt = query.strip().rstrip(".") + "." + self._sink_suffix

        arr = ObjectDetector._to_uint8(image)
        pil = Image.fromarray(arr)
        h, w = arr.shape[:2]

        inputs = self.processor(images=pil, text=self._prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        prob = outputs.logits.sigmoid()[0]              # (num_boxes, num_tokens)
        boxes = outputs.pred_boxes[0]                   # (num_boxes, 4) cxcywh in [0, 1]
        token_strs = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        spans = self._phrase_spans(token_strs)
        if not spans:
            return 0.0, None

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
            return 0.0, None
        kept = torch.where(keep)[0]
        best = kept[int(torch.argmax(qscore[kept]))]
        score = float(qscore[best])
        cx, cy, bw, bh = boxes[best].tolist()
        x0, y0 = (cx - bw / 2.0) * w, (cy - bh / 2.0) * h
        x1, y1 = (cx + bw / 2.0) * w, (cy + bh / 2.0) * h
        return score, (float(x0), float(y0), float(x1), float(y1))


class SamRefinedDetector:
    """Detector → MobileSAM (whole-image) → CLIP-scored best mask.

    Pipeline (mirrors what `bev_cells_from_sam` does in main.py, minus the
    BEV projection so it can be tested standalone on a single RGB):

      1. Run a base detector to decide whether the query target is present.
         If no box clears its threshold, return (0.0, None).
      2. Run MobileSAM's automatic mask generator on the WHOLE image
         (no box prompt — at distance the box is often a full box-width
         off, so anchoring SAM on it propagates the error).
      3. Score every mask by softmax over its mean MaskCLIP cosine to
         [``query``] + ``negative_texts`` (raw cosines cluster in a narrow
         ~0.2–0.3 band where distractors score nearly as high as targets;
         the relative "more pillow than wall" probability separates them).
      4. Pick the mask with the highest target probability. If it clears
         ``min_target_prob`` AND its raw target cosine clears
         ``min_clip_sim``, return that mask's bbox + the target probability.
         Otherwise (0.0, None).

    `detect(image, query)` matches the base interface and returns
    (score, box) from the refined mask. `detect_with_mask(image, query)`
    additionally returns the (H, W) bool mask itself — used by the
    __main__ visualization to overlay the contour.
    """

    name = "sam_refined"

    def backend_for(self, query: str) -> str:
        return self.name

    def __init__(
        self,
        base_detector,
        segmenter,
        mask_clip,
        device: str = "cuda",
        min_clip_sim: float = 0.18,
        min_mask_pixels: int = 200,
        negative_texts: Optional[list] = None,
        softmax_temp: float = 100.0,
        min_target_prob: float = 0.5,
    ):
        self.base = base_detector
        self.segmenter = segmenter
        self.mask_clip = mask_clip
        self.device = device
        self.min_clip_sim = float(min_clip_sim)
        self.min_mask_pixels = int(min_mask_pixels)
        self.negative_texts = (
            list(negative_texts) if negative_texts is not None
            else list(DEFAULT_NEGATIVE_CLASSES)
        )
        self.softmax_temp = float(softmax_temp)
        self.min_target_prob = float(min_target_prob)

    @torch.no_grad()
    def detect_with_mask(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]], Optional[np.ndarray]]:
        det_score, det_box = self.base.detect(image, query)
        if det_box is None or det_score <= 0.0:
            return 0.0, None, None

        masks = self.segmenter.segment_all(image)
        if not masks:
            return 0.0, None, None

        rgb_gpu = _to_float_rgb_tensor(image, self.device)
        feats = self.mask_clip.extract_dense_features(rgb_gpu)              # (H, W, 512)
        text_bank = encode_query_with_negatives(
            self.mask_clip, query, self.negative_texts)                     # (K, 512)
        sim_maps = (feats @ text_bank.T).permute(2, 0, 1)                   # (K, H, W) in [-1, 1]
        H, W = sim_maps.shape[1:]

        best_prob = -float("inf")
        best_raw = -float("inf")
        best_seg = None
        for m in masks:
            seg = m["segmentation"]
            if seg.sum() < self.min_mask_pixels:
                continue
            seg_t = torch.from_numpy(seg).to(self.device)
            if seg_t.shape != (H, W):
                seg_t = torch.nn.functional.interpolate(
                    seg_t[None, None].float(), size=(H, W), mode="nearest",
                )[0, 0].bool()
            if not bool(seg_t.any()):
                continue
            sims = sim_maps[:, seg_t].mean(dim=1)                           # (K,)
            prob, raw = target_prob_from_sims(sims, self.softmax_temp)
            if prob > best_prob:
                best_prob = prob
                best_raw = raw
                best_seg = seg

        if best_seg is None or best_prob < self.min_target_prob or best_raw < self.min_clip_sim:
            return 0.0, None, None
        best_score = best_prob

        ys, xs = np.where(best_seg)
        box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        return best_score, box, best_seg

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        score, box, _ = self.detect_with_mask(image, query)
        return score, box


def _to_float_rgb_tensor(image: Union[torch.Tensor, np.ndarray, Image.Image], device: str) -> torch.Tensor:
    """(H, W, 3) float in [0, 1] on `device` — MaskCLIPSemantics input format."""
    arr = ObjectDetector._to_uint8(image)
    t = torch.from_numpy(arr).to(device).float() / 255.0
    return t


class SinkGatedDetector:
    """Wrap any base detector with a neutral *attention-sink* false-positive
    gate (Ruis et al., "Fantastic Tractor-Dogs...", ICLR 2026).

    The paper's training-free fix appends semantically-neutral sink tokens to
    an open-vocabulary detector's prompt and discards any box the model assigns
    to a sink — giving the model a "none of the above" option so it stops
    forcing the target class onto background. That exact mechanism only applies
    to early-fusion OVD heads (Grounding DINO, GLIP, ...). Generative grounding
    models like LocateAnything emit box coordinate *tokens* with no per-box
    class-vs-prompt softmax to redirect, so we realize the same idea post-hoc in
    CLIP space: crop the base detector's box, take its mean CLIP feature, and
    compare it against the target query and the neutral sink(s) via a sharp
    softmax. If a sink out-scores the query (target probability falls below
    `min_target_prob`), the region is "none of the above" → discarded.

    Crucially, and per the paper, the sinks are NOT semantic negatives
    ("wall", "door"); they are neutral references (see `build_sink_bank`). On
    accept, the base detector's score and box pass through unchanged. The gate
    only runs when the base detector actually fires, so it adds one CLIP forward
    per detection — negligible next to a ~1 s/call LocateAnything, more
    noticeable on a ~15 ms YOLO-World.
    """

    name = "sink_gated"

    def __init__(
        self,
        base_detector,
        mask_clip,
        device: str = "cuda",
        sink_init: str = "mean",
        sink_num: int = 1,
        sink_special_str: str = "[()]",
        softmax_temp: float = 100.0,
        min_target_prob: float = 0.5,
        crop_pad: float = 0.0,
        min_crop_px: int = 8,
        pool: str = "mean",
        top_pct: float = 0.15,
        seed: int = 0,
    ):
        self.base = base_detector
        self.mask_clip = mask_clip
        self.device = device
        self.softmax_temp = float(softmax_temp)
        self.min_target_prob = float(min_target_prob)
        self.crop_pad = float(crop_pad)
        self.min_crop_px = int(min_crop_px)
        # Crop pooling for the gate:
        #  - "mean": pools the whole crop into one CLIP feature (high precision —
        #    a wall's mean is wall-like and loses to the neutral sink — but small
        #    objects get diluted by background and missed).
        #  - "top_pct": ranks crop patches by query similarity and pools only the
        #    top `top_pct` fraction of those embeddings (best small-object recall,
        #    but only the query selects patches, so it manufactures a query-
        #    favorable feature from any crop → background/wall false positives).
        #  - "top_pct_pertext": symmetric top-k — query AND each sink average
        #    their own best `top_pct` fraction of patch cosines, so the neutral
        #    sink can defend against walls while the query still isolates small
        #    objects. Middle ground; best when "top_pct" over-fires on background.
        self.pool = str(pool).lower()
        self.top_pct = float(top_pct)
        self.sinks = build_sink_bank(
            mask_clip, init=sink_init, n_sinks=sink_num,
            special_str=sink_special_str, seed=seed)               # (S, 512)
        # Cache the query embedding — the target query rarely changes in a run.
        self._q_text: Optional[str] = None
        self._q_emb: Optional[torch.Tensor] = None
        # Side channels for logging/viz: the target probability of the most
        # recent gated detection and the box it scored. Both are None when the
        # gate didn't run (base detector didn't fire, or the crop was too
        # small). `last_box` is set even when the detection is rejected, so the
        # viz can still draw the false-positive box. They are always set/cleared
        # together: a non-None `last_target_prob` always has a `last_box`.
        self.last_target_prob: Optional[float] = None
        self.last_box: Optional[Tuple[float, float, float, float]] = None

    def backend_for(self, query: str) -> str:
        return self.base.backend_for(query)

    def _query_emb(self, query: str) -> torch.Tensor:
        if query != self._q_text:
            self._q_text = query
            self._q_emb = self.mask_clip.encode_text(query)        # (1, 512)
        return self._q_emb

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        score, box = self.base.detect(image, query)
        self.last_target_prob = None
        self.last_box = None
        if box is None or score <= 0.0:
            return score, box

        arr = ObjectDetector._to_uint8(image)
        H, W = arr.shape[:2]
        x0, y0, x1, y1 = box
        if self.crop_pad > 0.0:
            pw = (x1 - x0) * self.crop_pad
            ph = (y1 - y0) * self.crop_pad
            x0, y0, x1, y1 = x0 - pw, y0 - ph, x1 + pw, y1 + ph
        xi0, yi0 = max(0, int(round(x0))), max(0, int(round(y0)))
        xi1, yi1 = min(W, int(round(x1))), min(H, int(round(y1)))
        # Degenerate / tiny crop: can't verify reliably → pass the box through.
        if xi1 - xi0 < self.min_crop_px or yi1 - yi0 < self.min_crop_px:
            return score, box

        crop = arr[yi0:yi1, xi0:xi1]
        rgb = torch.from_numpy(crop).to(self.device).float() / 255.0   # (h, w, 3)
        feats = self.mask_clip.extract_dense_features(rgb)             # (h, w, 512)

        q_emb = self._query_emb(query)                                 # (1, 512)
        bank = torch.cat([q_emb, self.sinks], dim=0)                   # (1+S, 512)
        P = feats.reshape(-1, feats.shape[-1])                         # (N, 512), L2-normed
        if self.pool == "top_pct":
            # Rank crop patches by query similarity, keep the top-k% most
            # query-like embeddings, pool them into one crop feature, then score
            # that single feature against the whole bank. Only the query selects
            # the patches, so the pooled feature is — by construction — the most
            # query-favorable sub-region of the crop. Best small-object recall,
            # but it manufactures a query-favorable feature from any crop (even a
            # blank wall), so the neutral sink can't defend → background false
            # positives. Use "top_pct_pertext" to restore the sink's defense.
            q_sim = (P @ q_emb.T).squeeze(1)                           # (N,) per-patch query cosine
            k = max(1, int(round(self.top_pct * P.shape[0])))
            idx = q_sim.topk(k, dim=0).indices                         # (k,) most query-like patches
            crop_emb = P[idx].mean(dim=0, keepdim=True)                # (1, 512)
            crop_emb = crop_emb / (crop_emb.norm(dim=-1, keepdim=True) + 1e-8)
            sims = (crop_emb @ bank.T).squeeze(0)                      # (1+S,)
        elif self.pool == "top_pct_pertext":
            # Symmetric top-k: EACH text (query and every sink) averages its OWN
            # best top-k% of patch cosines — different patches per column. The
            # query still cherry-picks the object's patches (small-object recall,
            # like "top_pct"), but the neutral sink also cherry-picks its most
            # neutral/background patches, so a blank wall — which matches the sink
            # well — can win and be rejected. Trades a little of "top_pct"'s recall
            # for much better precision on background regions.
            per_patch = P @ bank.T                                     # (N, 1+S) per-patch cosine
            k = max(1, int(round(self.top_pct * per_patch.shape[0])))
            sims = per_patch.topk(k, dim=0).values.mean(dim=0)        # (1+S,) top-k% mean per text
        else:
            crop_emb = P.mean(dim=0, keepdim=True)
            crop_emb = crop_emb / (crop_emb.norm(dim=-1, keepdim=True) + 1e-8)  # (1, 512)
            sims = (crop_emb @ bank.T).squeeze(0)                      # (1+S,)
        prob, _ = target_prob_from_sims(sims, self.softmax_temp)
        self.last_target_prob = prob
        # Remember the scored box even on rejection so the viz can still draw
        # the false positive (paired with last_target_prob).
        self.last_box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if prob < self.min_target_prob:
            # A neutral sink out-scored the query → background false positive.
            return 0.0, None
        return score, box


class LocateAnythingDetector:
    """Open-vocabulary detector using NVIDIA LocateAnything-3B via Hugging Face.

    Returns binary detection scores: 1.0 if it generates a bounding box, 0.0 otherwise.
    Since it produces point predictions, we parse the output to construct a bounding box.
    """
    name = "locate_anything"

    def backend_for(self, query: str) -> str:
        return self.name

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "nvidia/LocateAnything-3B",
        threshold: float = 0.5, # ignored, binary signal
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.3,
    ):
        import os
        import glob
        # Apply patch for Python 3.9 type hints inside the downloaded remote code
        for f in glob.glob("/root/.cache/huggingface/modules/transformers_modules/nvidia/LocateAnything_hyphen_3B/*/*.py"):
            os.system(f"grep -q 'from __future__ import annotations' {f} || sed -i '1s/^/from __future__ import annotations\\n/' {f}")

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
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        query: str,
    ) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        arr = ObjectDetector._to_uint8(image)
        pil = Image.fromarray(arr).convert("RGB")
        w, h = pil.size

        prompt = (
            "Locate all the instances that matches the following "
            f"description: {query}."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # LocateAnything ships its own chat template + vision-info helpers;
        # the generic HF ones don't emit the <image-N> placeholders it needs.
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
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

        # Output boxes are emitted as <box><x1><y1><x2><y2></box> with each
        # coordinate a 0-999 normalized integer special token (<box>None</box>
        # when nothing matches). Take the FIRST box: the model emits its
        # highest-priority, tightest detection first, then degenerates into
        # repeated oversized / full-image boxes — so "largest" would pick noise.
        m = re.search(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", response)
        if m is not None:
            x1, y1, x2, y2 = (int(g) / 1000.0 for g in m.groups())
            # The model occasionally emits corners out of order; sort so the
            # box is always (xmin, ymin, xmax, ymax) for every downstream
            # consumer (CLIP crop, BEV projection, PIL rectangle in viz).
            xs = sorted((x1 * w, x2 * w))
            ys = sorted((y1 * h, y2 * h))
            box = (xs[0], ys[0], xs[1], ys[1])
            return 1.0, box
        return 0.0, None


def make_detector(name: str = "yolo", device: str = "cuda", **kwargs):
    """Factory: pick a detector backend by short name.

    - "yolo": fast YOLO-Worldv2 (open-vocab).
    - "coco_yolo": closed-set COCO YOLOv8.
    - "hybrid": COCO classes → YOLOv8, everything else → YOLO-Worldv2.
    - "grounding_dino": Grounding DINO Tiny via HuggingFace.
    - "locate_anything": NVIDIA LocateAnything-3B via HuggingFace pipeline.
    - "llmdet": LLMDet via HuggingFace with training-free attention sinks.
    """
    key = name.lower().strip()
    # Only the YOLO-World branch (directly or inside hybrid) uses the
    # negative-class competition; pop it here so closed-set / GDino
    # backends don't choke on an unknown kwarg.
    negative_classes = kwargs.pop("negative_classes", None)
    if key in ("locate_anything", "locateanything"):
        return LocateAnythingDetector(device=device, **kwargs)
    if key in ("yolo", "yolo_world", "yoloworld"):
        if negative_classes is not None:
            kwargs.setdefault("negative_classes", negative_classes)
        return ObjectDetector(device=device, **kwargs)
    if key in ("coco_yolo", "yolov8", "coco"):
        return CocoYoloDetector(device=device, **kwargs)
    if key in ("hybrid",):
        return HybridDetector(device=device, negative_classes=negative_classes, **kwargs)
    if key in ("grounding_dino", "gdino", "groundingdino"):
        return GroundingDinoDetector(device=device, **kwargs)
    if key in ("llmdet", "llm_det"):
        return LLMDetDetector(device=device, **kwargs)
    if key in ("sam_refined", "sam"):
        # Composite: base detector triggers, MobileSAM proposes whole-image
        # masks, MaskCLIP scores them, best mask wins. Heavy deps imported
        # lazily so the other branches don't pay the cost.
        from src.perception.segmentation import MobileSAMSegmenter
        from src.perception.semantics import MaskCLIPSemantics
        base_name = kwargs.pop("base", "yolo")
        sam_checkpoint = kwargs.pop("sam_checkpoint", "SAM_models/mobile_sam.pt")
        min_clip_sim = float(kwargs.pop("min_clip_sim", 0.18))
        min_mask_pixels = int(kwargs.pop("min_mask_pixels", 200))
        maskclip_input_size = int(kwargs.pop("maskclip_input_size", 448))
        maskclip_model_name = kwargs.pop("maskclip_model_name", "ViT-B/16")
        softmax_temp = float(kwargs.pop("softmax_temp", 100.0))
        min_target_prob = float(kwargs.pop("min_target_prob", 0.5))
        base = make_detector(base_name, device=device,
                             negative_classes=negative_classes, **kwargs)
        segmenter = MobileSAMSegmenter(checkpoint=sam_checkpoint, device=device)
        mask_clip = MaskCLIPSemantics(
            device=device, model_name=maskclip_model_name, input_size=maskclip_input_size,
        )
        return SamRefinedDetector(
            base, segmenter, mask_clip, device=device,
            min_clip_sim=min_clip_sim, min_mask_pixels=min_mask_pixels,
            negative_texts=negative_classes, softmax_temp=softmax_temp,
            min_target_prob=min_target_prob,
        )
    raise ValueError(f"Unknown detector backend: {name!r}")