"""
Abstract base adapter for the disaster damage assessment engine.

All model backends (CLIP, YOLO, future fine-tuned models) must implement
the BaseAnalyzerAdapter interface. This allows the AnalysisEngine to swap
models without touching any API routes.

IMPORTANT DESIGN CONSTRAINTS (read before adding a new adapter):

1. confidence_score is the raw model output – it reflects model certainty
   about a CATEGORY label, NOT damage severity.

2. severity_score MUST be computed by SeverityEstimator (rule-based),
   NOT derived from classification confidence.

3. Never fabricate predictions. If inference fails, set is_fallback=True
   and return a clearly-labeled unavailable result.

4. Adapters should not raise exceptions to the engine; catch and translate
   all inference errors into a FallbackResult.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Shared result types ───────────────────────────────────────────────────────

@dataclass
class CandidateClass:
    """A single labelled candidate from the classification output."""
    label: str
    confidence: float        # raw model probability [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "confidence": round(self.confidence, 4)}


@dataclass
class DetectedObject:
    """A single bounding-box detection from an object-detection model."""
    label: str
    confidence: float
    bbox_xyxy: Optional[List[float]] = None  # [x1, y1, x2, y2] in pixel coords
    # Hazard relevance flag (True if label is mapped to a disaster-related object)
    is_hazard_relevant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "is_hazard_relevant": self.is_hazard_relevant,
        }
        if self.bbox_xyxy:
            d["bbox_xyxy"] = [round(v, 1) for v in self.bbox_xyxy]
        return d


@dataclass
class AnalysisResult:
    """
    Unified result returned by every adapter.

    Callers MUST respect the following invariants:
    - category and confidence come purely from model output.
    - severity is computed by SeverityEstimator, not by this adapter.
    - is_fallback=True means NO real inference was performed.
    """
    # ── Classification output ─────────────────────────────────────────────
    category: str                          # Top predicted disaster category
    confidence: float                      # Raw model probability [0.0, 1.0]
    top_candidates: List[CandidateClass] = field(default_factory=list)

    # ── Detection output (optional, adapter-specific) ─────────────────────
    detected_objects: List[DetectedObject] = field(default_factory=list)

    # ── Model metadata ────────────────────────────────────────────────────
    model_name: str = "unknown"
    adapter_type: str = "unknown"          # "clip", "yolo", "efficientnet", …
    processing_time_ms: float = 0.0

    # ── Reliability flags ─────────────────────────────────────────────────
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    review_required: bool = False          # True when confidence < LOW_CONFIDENCE_THRESHOLD
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "top_candidates": [c.to_dict() for c in self.top_candidates],
            "detected_objects": [o.to_dict() for o in self.detected_objects],
            "model_name": self.model_name,
            "adapter_type": self.adapter_type,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "review_required": self.review_required,
            "warnings": self.warnings,
        }


# ── Confidence tiers ──────────────────────────────────────────────────────────

CONFIDENCE_TIERS = {
    "high":            (0.75, 1.01),
    "medium":          (0.50, 0.75),
    "low":             (0.30, 0.50),
    "review_required": (0.00, 0.30),
}

# Below this threshold the result is marked review_required=True
LOW_CONFIDENCE_THRESHOLD: float = 0.30
# Below this, category is overridden to "Other / unclear"
ABSTENTION_THRESHOLD: float = 0.15


def confidence_tier(score: float) -> str:
    for tier, (lo, hi) in CONFIDENCE_TIERS.items():
        if lo <= score < hi:
            return tier
    return "review_required"


# ── Abstract adapter ──────────────────────────────────────────────────────────

class BaseAnalyzerAdapter(abc.ABC):
    """
    Contract that every model adapter must satisfy.

    Subclasses implement _run_inference(). The base class wraps it with
    timing, exception handling, and fallback generation so adapters never
    crash the engine.
    """

    ADAPTER_TYPE: str = "base"

    def analyze(self, image_path: Path) -> AnalysisResult:
        """Public entry point – times inference and handles all exceptions."""
        t0 = time.perf_counter()
        try:
            result = self._run_inference(image_path)
        except Exception as exc:
            result = self._fallback(str(exc))
        result.processing_time_ms = (time.perf_counter() - t0) * 1000.0

        # Apply review threshold
        if not result.is_fallback and result.confidence < LOW_CONFIDENCE_THRESHOLD:
            result.review_required = True
            result.warnings.append(
                f"Confidence {result.confidence:.2%} is below review threshold "
                f"({LOW_CONFIDENCE_THRESHOLD:.0%}). Human verification required."
            )
        if not result.is_fallback and result.confidence < ABSTENTION_THRESHOLD:
            result.category = "Other / unclear"
            result.warnings.append(
                "Confidence is extremely low. Category overridden to 'Other / unclear'."
            )

        return result

    @abc.abstractmethod
    def _run_inference(self, image_path: Path) -> AnalysisResult:
        """
        Perform model inference. Must NOT catch exceptions – the base class
        handles them. Must NOT fabricate predictions.
        """
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True only if the model is loaded and ready."""
        ...

    def _fallback(self, reason: str) -> AnalysisResult:
        """Return a clearly-marked unavailable result – never a fabricated prediction."""
        return AnalysisResult(
            category="Unavailable",
            confidence=0.0,
            model_name=self.__class__.__name__,
            adapter_type=self.ADAPTER_TYPE,
            is_fallback=True,
            fallback_reason=reason,
            review_required=True,
            warnings=[f"Model unavailable: {reason}"],
        )
