"""
CLIP zero-shot disaster damage classification adapter.

Uses OpenAI CLIP (ViT-B/32 by default) to classify input images into one of
seven disaster damage categories using carefully engineered text prompts.

─────────────────────────────────────────────────────────────────────────────
IMPORTANT: Confidence ≠ Severity
─────────────────────────────────────────────────────────────────────────────
CLIP's softmax confidence scores reflect model certainty that the image
matches a text prompt — they do NOT measure physical damage severity.

A high confidence score for "Flood damage" simply means the image resembles
the CLIP model's learned visual-textual representation of flooding. It tells
you NOTHING about:
  - depth of water
  - number of people affected
  - structural compromises
  - accessibility

Severity estimation is handled EXCLUSIVELY by SeverityEstimator (rule-based)
and must NEVER be derived from CLIP classification confidence.
─────────────────────────────────────────────────────────────────────────────

Supported categories (7 + "Other / unclear"):
  - Flood damage
  - Building damage
  - Fire and smoke
  - Landslide
  - Cyclone damage
  - Road and infrastructure obstruction
  - Other / unclear

Fine-tuning notes (see MODEL_LIMITATIONS.md for full dataset requirements):
  For higher accuracy in specific disaster contexts, CLIP can be fine-tuned
  via contrastive learning on labelled disaster image datasets such as:
    - AIDER (Aerial Image Dataset for Emergency Response)
    - xBD (xView2 Building Damage)
    - FloodNet
    - LADI (Large Area Drone Imagery)
  Fine-tuned checkpoint path configurable via CLIP_CHECKPOINT_PATH env var.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.ai.engine.base import (
    AnalysisResult,
    BaseAnalyzerAdapter,
    CandidateClass,
    ABSTENTION_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)
from app.ai.engine.preprocessing import load_and_validate_image, safe_thumbnail


# ── Category taxonomy ─────────────────────────────────────────────────────────

DISASTER_CATEGORIES: Dict[str, List[str]] = {
    "Flood damage": [
        "aerial view of flooded streets and submerged cars",
        "floodwater inside buildings and homes",
        "people stranded on rooftops surrounded by floodwater",
        "muddy brown floodwater covering roads and fields",
        "cars submerged in deep floodwater on a street",
        "river overflowing its banks causing flooding",
    ],
    "Building damage": [
        "collapsed building with rubble and debris on street",
        "partially destroyed house after earthquake",
        "cracked walls and fallen concrete from structural damage",
        "buildings reduced to rubble after a disaster",
        "collapsed roof and exposed interior of damaged building",
        "structural damage with fallen beams and bricks",
    ],
    "Fire and smoke": [
        "building on fire with thick black smoke rising",
        "wildfire burning through a forest or neighbourhood",
        "charred ruins and ash after a fire",
        "flames engulfing a structure with firefighters present",
        "smoke billowing from a burning vehicle or structure",
        "aerial view of wildfire spreading across dry landscape",
    ],
    "Landslide": [
        "landslide with mud and boulders covering a road",
        "hillside collapsed sending debris onto houses below",
        "aerial view of a large mudslide scar on a mountainside",
        "mud and rocks from a landslide blocking a highway",
        "slope failure with earth and trees fallen across infrastructure",
    ],
    "Cyclone damage": [
        "uprooted trees and fallen power lines after a cyclone",
        "destroyed roof blown off a building by strong winds",
        "aerial view of flattened coastal structures after a hurricane",
        "flooded coastal town after storm surge and cyclone",
        "scattered debris and damaged structures from tornado or cyclone",
    ],
    "Road and infrastructure obstruction": [
        "large fallen tree blocking a road or highway",
        "bridge damaged or washed out by flooding",
        "road collapse or sinkhole blocking traffic",
        "damaged power lines and electricity poles blocking a road",
        "debris and rubble covering a road after a disaster",
    ],
    "Other / unclear": [
        "generic outdoor scene with no visible disaster or damage",
        "indoor scene unrelated to disaster response",
        "blurry or unidentifiable photograph with unclear content",
        "vehicle or person with no visible damage or hazard context",
    ],
}

# Multi-prompt ensemble: use mean of all prompts for each category
CATEGORY_LABELS = list(DISASTER_CATEGORIES.keys())

# All prompts in a flat list; remember their category index for aggregation
_ALL_PROMPTS: List[str] = []
_PROMPT_TO_CATEGORY_IDX: List[int] = []
for _cat_idx, (_cat, _prompts) in enumerate(DISASTER_CATEGORIES.items()):
    for _p in _prompts:
        _ALL_PROMPTS.append(_p)
        _PROMPT_TO_CATEGORY_IDX.append(_cat_idx)


# ── Adapter ───────────────────────────────────────────────────────────────────

class CLIPAdapter(BaseAnalyzerAdapter):
    """
    Zero-shot disaster category classification using OpenAI CLIP.

    Uses an ensemble of semantically rich text prompts per category and
    averages their logit scores before computing the final softmax.

    Configuration (via environment variables):
      CLIP_MODEL_NAME      : e.g. "ViT-B/32" (default), "ViT-L/14"
      CLIP_CHECKPOINT_PATH : Path to a fine-tuned state_dict (optional)
    """

    ADAPTER_TYPE = "clip"

    _model = None       # shared across instances (lazy singleton)
    _preprocess = None
    _text_features = None
    _device = None
    _model_name_str = "unloaded"
    _load_error: Optional[str] = None

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        checkpoint_path: Optional[Path] = None,
    ):
        import torch
        self._torch = torch
        self._model_name_key = os.environ.get("CLIP_MODEL_NAME", model_name)
        self._checkpoint_path = checkpoint_path or (
            Path(os.environ["CLIP_CHECKPOINT_PATH"])
            if "CLIP_CHECKPOINT_PATH" in os.environ
            else None
        )
        self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._ensure_loaded()

    # ── Model lifecycle ───────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if CLIPAdapter._model is not None or CLIPAdapter._load_error:
            return
        try:
            import clip  # openai-clip package
            import torch

            CLIPAdapter._device = torch.device(self._device_str)
            CLIPAdapter._model, CLIPAdapter._preprocess = clip.load(
                self._model_name_key, device=CLIPAdapter._device
            )

            if self._checkpoint_path and self._checkpoint_path.exists():
                state = torch.load(self._checkpoint_path, map_location=CLIPAdapter._device)
                CLIPAdapter._model.load_state_dict(state)

            CLIPAdapter._model.eval()
            CLIPAdapter._model_name_str = f"CLIP {self._model_name_key}"

            # Pre-encode all text prompts (done once, cached)
            import clip as clip_module
            tokens = clip_module.tokenize(_ALL_PROMPTS).to(CLIPAdapter._device)
            with torch.no_grad():
                CLIPAdapter._text_features = CLIPAdapter._model.encode_text(tokens)
                CLIPAdapter._text_features = CLIPAdapter._text_features / CLIPAdapter._text_features.norm(
                    dim=-1, keepdim=True
                )

        except Exception as exc:
            CLIPAdapter._load_error = str(exc)

    def is_available(self) -> bool:
        return CLIPAdapter._model is not None and CLIPAdapter._load_error is None

    # ── Inference ─────────────────────────────────────────────────────────

    def _run_inference(self, image_path: Path) -> AnalysisResult:
        if not self.is_available():
            reason = CLIPAdapter._load_error or "CLIP model not loaded"
            return self._fallback(reason)

        import torch
        import numpy as np

        # 1. Load and validate image
        pil_img = load_and_validate_image(image_path)
        pil_img = safe_thumbnail(pil_img)

        # 2. Encode image
        img_tensor = CLIPAdapter._preprocess(pil_img).unsqueeze(0).to(CLIPAdapter._device)
        with torch.no_grad():
            img_features = CLIPAdapter._model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            # Cosine similarity with all text prompts
            logits = (img_features @ CLIPAdapter._text_features.T).squeeze(0)
            logits = logits.cpu().float().numpy()  # shape: (n_prompts,)

        # 3. Aggregate per-category: mean logit across prompts
        n_cats = len(CATEGORY_LABELS)
        cat_logits = np.zeros(n_cats, dtype=np.float32)
        cat_counts = np.zeros(n_cats, dtype=np.int32)
        for i, cat_idx in enumerate(_PROMPT_TO_CATEGORY_IDX):
            cat_logits[cat_idx] += logits[i]
            cat_counts[cat_idx] += 1
        cat_logits /= np.maximum(cat_counts, 1)

        # 4. Softmax over aggregated category logits (temperature from CLIP's logit_scale)
        logit_scale = float(CLIPAdapter._model.logit_scale.exp().cpu())
        scaled = cat_logits * logit_scale
        exp = np.exp(scaled - scaled.max())
        probs = exp / exp.sum()

        # 5. Build result
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        top_cat = CATEGORY_LABELS[top_idx]

        candidates = [
            CandidateClass(label=CATEGORY_LABELS[i], confidence=float(probs[i]))
            for i in np.argsort(probs)[::-1]
        ]

        warnings: list = [
            "CLIP confidence reflects category recognition only. "
            "It is NOT a damage severity score."
        ]

        if top_conf < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(
                "Confidence below review threshold — image may be ambiguous, "
                "blurry, or outside training distribution."
            )

        return AnalysisResult(
            category=top_cat,
            confidence=top_conf,
            top_candidates=candidates,
            model_name=f"{CLIPAdapter._model_name_str} (zero-shot, multi-prompt ensemble)",
            adapter_type=self.ADAPTER_TYPE,
            review_required=top_conf < LOW_CONFIDENCE_THRESHOLD,
            warnings=warnings,
        )
