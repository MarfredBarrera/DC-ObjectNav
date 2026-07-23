import re

import torch
import torch.nn.functional as F
import torchvision.transforms as T


def filter_distractors(query: str, distractors) -> list:
    """Drop distractor phrases that share a content word with the query.

    The canonical distractor list (`cfg.distractor_objects`) is fixed while
    the query changes per episode, so the query's own category must be
    removed before it can be used as a competing class ("a couch" must not
    compete with itself). Matching is on article-stripped word overlap, so
    "a potted plant" removes "a potted plant" and "a tv monitor" removes
    "a tv". Synonyms are NOT caught (query "a sofa" keeps "a couch" as a
    distractor) — an accepted limitation for open-vocab queries; the softmax
    temperature is the mitigation, not this filter.
    """
    stop = {"a", "an", "the"}
    qwords = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in stop}
    kept = []
    for d in distractors or []:
        dwords = {w for w in re.findall(r"[a-z0-9]+", str(d).lower()) if w not in stop}
        if dwords & qwords:
            continue
        kept.append(str(d))
    return kept


class CLIPSegSemantics:
    """Dense per-pixel CLIPSeg relevance score for a FIXED text query.

    CLIPSeg is not a query-agnostic embedding (like the MaskCLIP dense-CLIP
    features this replaced — see git history): its dense signal comes from a
    small trained decoder conditioned on one specific prompt, so there is no
    "train once, dot against any later query" API here. The tradeoff accepted
    for this feature field: the query is fixed at construction (matching
    cfg.target_query, which nothing in this codebase changes mid-run anyway),
    and the field regresses CLIPSeg's scalar activation directly instead of a
    512-D embedding. In exchange, the per-pixel signal is much cleaner,
    especially before a target has been well-observed, and the trained field
    aggregates it across every viewpoint into one persistent, multi-view-
    consistent relevance map instead of trusting any single noisy frame.

    GPU-only: no CPU/PIL round-trip per frame. The text-conditional
    embeddings are computed once at construction (via HF's
    `get_conditional_embeddings`, which accepts input_ids directly) and
    cached, so the per-frame path only runs the frozen vision encoder + the
    lightweight decoder.

    **Pairwise mode** (`distractors`): CLIPSeg conditioned on one prompt
    answers the *binary* question "is this couch-like?" — and a bed
    legitimately is, so geometric look-alikes score high and the field
    faithfully aggregates that error across views (no downstream threshold
    can separate them). With a distractor list the output is one sigmoid
    channel per prompt (row 0 = query, rows 1..K-1 the distractors); the
    field regresses the full vector and the contrast happens downstream at
    field-verify time on the multi-view-averaged channels (worst-case
    margin-of-means, see src/perception/detection.py). The K logit maps
    come from one batched forward pass (image repeated K times); the
    distractor conditional embeddings are precomputed like the query's.
    Phrases sharing a content word with the query are dropped
    (`filter_distractors`).

    Usage:
        sem = CLIPSegSemantics(query="a pillow", device="cuda")
        scores = sem.extract_dense_features(rgb_gpu)  # (H, W, 1) in [0, 1]
    """

    # CLIPSeg's HF processor normalizes with plain ImageNet stats (NOT CLIP's
    # own 0.481/0.458/0.408 constants) -- verified against CLIPSegProcessor.
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, query: str, device="cuda", model_name="CIDAS/clipseg-rd64-refined",
                 input_size=352, distractors=None):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        self.device = device
        self.input_size = input_size
        self.query = query
        self.distractors = filter_distractors(query, distractors)

        processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
        self.model.eval()

        self.normalize = T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

        # Fixed prompts -> cache all conditional embeddings once. No per-frame
        # text encoding: every frame reuses these same cached embeddings.
        # Row 0 is always the query; rows 1..K-1 the distractors (if any).
        prompts = [query] + self.distractors
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            self._conditional_embeddings = self.model.get_conditional_embeddings(
                batch_size=len(prompts),
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )
        if self.distractors:
            print(f"[CLIPSeg] pairwise mode: {1 + len(self.distractors)} sigmoid "
                  f"channels ([query] + {len(self.distractors)} distractors) "
                  f"for {query!r}")

    @torch.inference_mode()
    def extract_dense_features(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Extract a dense per-pixel CLIPSeg relevance map for `self.query`.

        Args:
            rgb_tensor: (H, W, 3) float32 tensor in [0, 1] range, on GPU.

        Returns:
            scores: (H, W, K) float32 tensor on GPU in [0, 1] — one sigmoid
            channel per prompt (K=1 plain query sigmoid when no distractors
            are configured; see class docstring).
        """
        H, W, _ = rgb_tensor.shape
        K = self._conditional_embeddings.shape[0]

        img = rgb_tensor.permute(2, 0, 1).unsqueeze(0)
        img = F.interpolate(img, size=(self.input_size, self.input_size),
                            mode="bilinear", align_corners=False)
        img = self.normalize(img.squeeze(0)).unsqueeze(0)

        # One batched pass scores all K prompts against the same frame (image
        # repeated K rows to match the conditional-embedding batch).
        pixel = img.repeat(K, 1, 1, 1) if K > 1 else img
        outputs = self.model(pixel_values=pixel,
                             conditional_embeddings=self._conditional_embeddings)
        logits = outputs.logits          # (K, 352, 352); HF squeezes when K==1
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        # Per-term sigmoid channels, no per-frame contrast: (K, 352, 352).
        # Each channel is independently bounded in [0, 1]; the field
        # regresses the full vector and the margin is taken at read time.
        score = torch.sigmoid(logits)

        score = F.interpolate(score.unsqueeze(1), size=(H, W),
                              mode="bilinear", align_corners=False)  # (C, 1, H, W)
        return score.squeeze(1).permute(1, 2, 0).float()  # (H, W, C)
