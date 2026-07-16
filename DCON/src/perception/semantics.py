import re

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from src.mask_clip.MaskCLIP import MaskCLIP


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


class MaskCLIPSemantics:
    """Dense per-pixel CLIP feature extraction using MaskCLIP value reparameterization.

    Replaces the old SAM+CLIP pipeline. Everything runs on GPU — no CPU transfers
    needed for feature extraction.

    The MaskCLIP approach extracts dense features by reparameterizing the last
    attention layer of the CLIP ViT to output value projections for every patch
    token instead of only the CLS token. These patch features live in the same
    embedding space as CLIP text features, enabling zero-shot dense similarity.

    Usage:
        sem = MaskCLIPSemantics(device="cuda")
        feats = sem.extract_dense_features(rgb_gpu)  # (H, W, 512)
        text  = sem.encode_text("a pillow")           # (1, 512)
    """

    # CLIP normalization constants (ImageNet-tuned for OpenAI CLIP)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, device="cuda", model_name="ViT-B/16", input_size=448):
        self.device = device
        self.input_size = input_size

        # Load the MaskCLIP model (downloads weights on first run)
        self.model = MaskCLIP(model_name=model_name, device=device)

        # GPU-only normalization transform (operates on CHW tensors, no PIL needed)
        self.normalize = T.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)

    @torch.inference_mode()
    def extract_dense_features(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Extract dense per-pixel CLIP features from an RGB image.

        Args:
            rgb_tensor: (H, W, 3) float32 tensor in [0, 1] range, on GPU.

        Returns:
            features: (H, W, 512) L2-normalized CLIP feature map on GPU.
        """
        H, W, _ = rgb_tensor.shape

        # (H, W, 3) → (3, H, W) → (1, 3, H, W)
        img = rgb_tensor.permute(2, 0, 1).unsqueeze(0)

        # Resize to MaskCLIP input size and apply CLIP normalization
        img = F.interpolate(img, size=(self.input_size, self.input_size),
                            mode="bilinear", align_corners=False)
        img = self.normalize(img.squeeze(0)).unsqueeze(0)  # normalize expects (C,H,W)

        # Extract dense patch features via value reparameterization
        # Returns (1, num_patches, 512) where num_patches = (input_size/16)^2
        patch_features = self.model.model.get_patch_encodings(img)

        # Reshape to spatial grid: (1, num_patches, 512) → (1, 512, grid, grid)
        num_patches = patch_features.shape[1]
        grid_size = int(num_patches ** 0.5)
        feature_grid = patch_features.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)

        # Bilinear upsample to original resolution: (1, 512, grid, grid) → (1, 512, H, W)
        feature_grid = F.interpolate(feature_grid, size=(H, W),
                                     mode="bilinear", align_corners=False)

        # (1, 512, H, W) → (H, W, 512)
        features = feature_grid.squeeze(0).permute(1, 2, 0)

        # L2 normalize per-pixel and cast to float32 (MaskCLIP runs in fp16 internally)
        features = features.float()
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)

        return features

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a text prompt into a CLIP embedding.

        Args:
            text: Natural language query string.

        Returns:
            text_features: (1, 512) L2-normalized text embedding on GPU.
        """
        text_tokens = self.model._tokenize(text)
        text_features = self.model.model.encode_text(text_tokens)
        text_features = text_features.float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        return text_features


class CLIPSegSemantics:
    """Dense per-pixel CLIPSeg relevance score for a FIXED text query.

    Unlike MaskCLIP, CLIPSeg is not a query-agnostic embedding: its dense
    signal comes from a small trained decoder conditioned on one specific
    prompt, not a reparameterization that happens to land in the shared
    CLIP text/image space for free. There is no "train once, dot against any
    later query" API here. The tradeoff accepted for this feature field: the
    query is fixed at construction (matching cfg.target_query, which nothing
    in this codebase changes mid-run anyway), and the field regresses
    CLIPSeg's scalar activation directly instead of a 512-D embedding. In
    exchange, CLIPSeg's per-pixel signal is much cleaner than MaskCLIP's,
    especially before a target has been well-observed, and the trained field
    aggregates it across every viewpoint into one persistent, multi-view-
    consistent relevance map instead of trusting any single noisy frame.

    GPU-only, mirroring MaskCLIPSemantics: no CPU/PIL round-trip per frame.
    The text-conditional embeddings are computed once at construction (via
    HF's `get_conditional_embeddings`, which accepts input_ids directly) and
    cached, so the per-frame path only runs the frozen vision encoder + the
    lightweight decoder.

    **Contrastive mode** (`distractors`): CLIPSeg conditioned on one prompt
    answers the *binary* question "is this couch-like?" — and a bed
    legitimately is, so geometric look-alikes score high and the field
    faithfully aggregates that error across views (no downstream threshold
    can separate them). With a distractor list the per-pixel score becomes

        sigmoid(logit_target) * softmax([logit_target; logit_distractors] / T)[target]

    i.e. "is anything here" (the sigmoid, which preserves low scores on
    background/walls — a bare softmax over object classes would force every
    pixel to pick a winner) times "is it more couch than bed/chair/..." (the
    contrastive posterior, which collapses at look-alike pixels because their
    true class out-scores the query). On unambiguous target pixels the
    posterior saturates toward 1 and the score matches the plain sigmoid.
    The K logit maps come from one batched forward pass (image repeated K
    times); the distractor conditional embeddings are precomputed like the
    query's. CLIPSeg logits from separately-conditioned passes are not
    jointly calibrated — `softmax_temp` is the knob (higher = softer
    posterior, more tolerant of near-ties). Phrases sharing a content word
    with the query are dropped (`filter_distractors`).

    Usage:
        sem = CLIPSegSemantics(query="a pillow", device="cuda")
        scores = sem.extract_dense_features(rgb_gpu)  # (H, W, 1) in [0, 1]
    """

    # CLIPSeg's HF processor normalizes with plain ImageNet stats (NOT CLIP's
    # own 0.481/0.458/0.408 constants) -- verified against CLIPSegProcessor.
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, query: str, device="cuda", model_name="CIDAS/clipseg-rd64-refined",
                 input_size=352, distractors=None, softmax_temp: float = 1.0,
                 pairwise: bool = False):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        self.device = device
        self.input_size = input_size
        self.query = query
        self.distractors = filter_distractors(query, distractors)
        self.softmax_temp = float(softmax_temp)
        # Pairwise mode: emit one sigmoid channel per prompt (row 0 = query)
        # instead of collapsing to the scalar sigmoid x softmax-share. The
        # contrast then happens downstream at field-verify time on the
        # multi-view-averaged channels (margin-of-means), not per frame.
        self.pairwise = bool(pairwise)

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
        if self.pairwise:
            print(f"[CLIPSeg] pairwise mode: {1 + len(self.distractors)} sigmoid "
                  f"channels ([query] + {len(self.distractors)} distractors) "
                  f"for {query!r}")
        elif self.distractors:
            print(f"[CLIPSeg] contrastive mode: {len(self.distractors)} distractors "
                  f"for {query!r} (temp={self.softmax_temp})")

    @torch.inference_mode()
    def extract_dense_features(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Extract a dense per-pixel CLIPSeg relevance map for `self.query`.

        Args:
            rgb_tensor: (H, W, 3) float32 tensor in [0, 1] range, on GPU.

        Returns:
            scores: (H, W, 1) float32 tensor on GPU in [0, 1] — the plain
            query sigmoid, or sigmoid x contrastive posterior when
            distractors are configured (see class docstring).
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

        if self.pairwise:
            # Per-term sigmoid channels, no per-frame contrast: (K, 352, 352).
            # Each channel is independently bounded in [0, 1]; the field
            # regresses the full vector and the margin is taken at read time.
            score = torch.sigmoid(logits)
        else:
            score = torch.sigmoid(logits[0])                        # (352, 352)
            if K > 1:
                # Contrastive posterior over [query] + distractors; row 0 = query.
                posterior = torch.softmax(logits / self.softmax_temp, dim=0)[0]
                score = score * posterior
            score = score[None]          # (1, 352, 352)

        score = F.interpolate(score.unsqueeze(1), size=(H, W),
                              mode="bilinear", align_corners=False)  # (C, 1, H, W)
        return score.squeeze(1).permute(1, 2, 0).float()  # (H, W, C)
