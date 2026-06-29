"""Open-vocabulary object detectors.

All backends share the same interface — `detect(image, query) -> (score, box)` —
so the rest of the system can swap between them by config without further
plumbing. Pick one via `make_detector(name, ...)`.

- `ObjectDetector`: YOLO-Worldv2 (fast, ~15 ms/frame, class-name-style queries).
- `CocoYoloDetector`: closed-set COCO YOLOv8 (fast, high precision on the 80
  COCO classes, returns score=0 for queries that don't map to any COCO class).
- `HybridDetector`: routes COCO-matching queries to `CocoYoloDetector` and
  everything else to `ObjectDetector` (YOLO-World). Mirrors the VLFM paper's
  split, swapping Grounding DINO for YOLO-World on the open-vocab branch
  to stay in the ~15 ms regime.
- `LLMDetDetector`: LLMDet (MM-Grounding-DINO) with training-free attention
  sinks for background false-positive mitigation.
- `LocateAnythingDetector`: NVIDIA LocateAnything-3B generative grounding.
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
    - "locate_anything": NVIDIA LocateAnything-3B via HuggingFace pipeline.
    - "llmdet": LLMDet (MM-Grounding-DINO) via HuggingFace with attention sinks.
    """
    key = name.lower().strip()
    # Only the YOLO-World branch (directly or inside hybrid) uses the
    # negative-class competition; pop it here so the other backends don't
    # choke on an unknown kwarg.
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
    if key in ("llmdet", "llm_det"):
        return LLMDetDetector(device=device, **kwargs)
    raise ValueError(f"Unknown detector backend: {name!r}")