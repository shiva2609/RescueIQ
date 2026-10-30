"""
AnalysisEngine — orchestrates CLIP + YOLO into a unified disaster assessment.

The engine:
  1. Runs CLIP zero-shot category classification
  2. Runs YOLO object detection (if available)
  3. Fuses results using a transparent weighting strategy
  4. Applies the SeverityEstimator (rule-based, never CLIP confidence)
  5. Returns a fully-populated AssessmentResult

Design principles:
  - Adapters are replaceable; the engine API never changes.
  - Fallback is graceful: if CLIP is unavailable, YOLO result is returned.
    If both are unavailable, a clearly-marked unavailable result is returned.
  - No fabrication: if models are not loaded, results say so explicitly.
  - Confidence fusion uses category agreement to weight final confidence,
    NOT to produce a severity score.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ai.engine.base import (
    AnalysisResult,
    BaseAnalyzerAdapter,
    CandidateClass,
    DetectedObject,
    confidence_tier,
    LOW_CONFIDENCE_THRESHOLD,
)
from app.ai.engine.severity import SeverityEstimator, SeverityEstimate
from app.ai.engine.preprocessing import (
    PreprocessingError,
    load_and_validate_image,
    get_image_metadata,
)


# ── Full assessment response ──────────────────────────────────────────────────

@dataclass
class AssessmentResult:
    """
    Complete multi-model disaster damage assessment.

    separation_note: Explains that severity is computed independently from
    classification confidence — required for every response.
    """
    # ── Classification ─────────────────────────────────────────────────────
    category: str
    confidence: float
    confidence_tier: str
    top_candidates: List[CandidateClass] = field(default_factory=list)

    # ── Detection ─────────────────────────────────────────────────────────
    detected_objects: List[DetectedObject] = field(default_factory=list)

    # ── Severity (ALWAYS rule-based, NEVER from CLIP confidence) ──────────
    severity: Optional[SeverityEstimate] = None

    # ── Model metadata ────────────────────────────────────────────────────
    models_used: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    image_metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Reliability ───────────────────────────────────────────────────────
    review_required: bool = False
    warnings: List[str] = field(default_factory=list)
    is_fallback: bool = False
    fallback_reason: Optional[str] = None

    separation_note: str = (
        "Severity is computed by a rule-based estimator using disaster category "
        "and contextual metadata — NOT derived from AI model classification "
        "confidence. Classification confidence measures category recognition only."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "confidence_tier": self.confidence_tier,
            "top_candidates": [c.to_dict() for c in self.top_candidates],
            "detected_objects": [o.to_dict() for o in self.detected_objects],
            "severity": self.severity.to_dict() if self.severity else None,
            "models_used": self.models_used,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "image_metadata": self.image_metadata,
            "review_required": self.review_required,
            "warnings": self.warnings,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "separation_note": self.separation_note,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class AnalysisEngine:
    """
    Orchestrates multiple model adapters into a unified assessment.

    Adapters are plugged in at construction time. The engine supports any
    combination of adapters as long as they implement BaseAnalyzerAdapter.

    Typical setup:
        engine = AnalysisEngine(
            classifiers=[CLIPAdapter()],
            detectors=[YOLOAdapter()],
        )
        result = engine.assess(image_path, urgency_score=70, people_affected=15)
    """

    def __init__(
        self,
        classifiers: Optional[List[BaseAnalyzerAdapter]] = None,
        detectors: Optional[List[BaseAnalyzerAdapter]] = None,
        severity_estimator: Optional[SeverityEstimator] = None,
    ):
        self._classifiers: List[BaseAnalyzerAdapter] = classifiers or []
        self._detectors: List[BaseAnalyzerAdapter] = detectors or []
        self._severity = severity_estimator or SeverityEstimator()

    # ── Public API ────────────────────────────────────────────────────────

    def assess(
        self,
        image_path: Path,
        urgency_score: float = 0.0,
        people_affected: int = 0,
        access_blocked: bool = False,
    ) -> AssessmentResult:
        """
        Run full assessment pipeline on an image.

        Args:
            image_path    : Absolute path to image on disk (already saved).
            urgency_score : Reporter-supplied urgency 0-100.
            people_affected: Estimated number of people affected.
            access_blocked: Whether emergency access is blocked.

        Returns:
            AssessmentResult with category, confidence, detections, and
            rule-based severity estimate.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        models_used: List[str] = []

        # ── 0. Validate image ─────────────────────────────────────────────
        try:
            pil_img = load_and_validate_image(image_path)
            img_meta = get_image_metadata(pil_img)
        except PreprocessingError as exc:
            return self._unavailable(str(exc), t_start)

        # ── 1. Run classifiers ────────────────────────────────────────────
        clf_results: List[AnalysisResult] = []
        for clf in self._classifiers:
            r = clf.analyze(image_path)
            clf_results.append(r)
            if r.is_fallback:
                warnings.append(
                    f"Classifier '{r.model_name}' unavailable: {r.fallback_reason}"
                )
            else:
                models_used.append(r.model_name)
                warnings.extend(r.warnings)

        # ── 2. Run detectors ──────────────────────────────────────────────
        det_results: List[AnalysisResult] = []
        all_detected: List[DetectedObject] = []
        for det in self._detectors:
            r = det.analyze(image_path)
            det_results.append(r)
            if r.is_fallback:
                warnings.append(
                    f"Detector '{r.model_name}' unavailable: {r.fallback_reason}"
                )
            else:
                models_used.append(r.model_name)
                warnings.extend(r.warnings)
                all_detected.extend(r.detected_objects)

        # ── 3. Fuse classification results ────────────────────────────────
        category, final_conf, top_candidates = self._fuse_classifications(
            clf_results, all_detected
        )

        # ── 4. Severity (rule-based — never CLIP confidence) ──────────────
        severity = self._severity.estimate(
            category=category,
            detected_objects=all_detected,
            urgency_score=urgency_score,
            people_affected=people_affected,
            access_blocked=access_blocked,
        )

        # ── 5. Handle complete fallback ───────────────────────────────────
        if not models_used:
            return self._unavailable(
                "All configured models are unavailable", t_start
            )

        # ── 6. Build result ───────────────────────────────────────────────
        is_review = final_conf < LOW_CONFIDENCE_THRESHOLD
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # De-duplicate warnings
        seen_warnings: List[str] = []
        for w in warnings:
            if w not in seen_warnings:
                seen_warnings.append(w)

        return AssessmentResult(
            category=category,
            confidence=final_conf,
            confidence_tier=confidence_tier(final_conf),
            top_candidates=top_candidates,
            detected_objects=all_detected,
            severity=severity,
            models_used=models_used,
            processing_time_ms=elapsed_ms,
            image_metadata=img_meta,
            review_required=is_review,
            warnings=seen_warnings,
            is_fallback=False,
        )

    # ── Fusion logic ──────────────────────────────────────────────────────

    def _fuse_classifications(
        self,
        clf_results: List[AnalysisResult],
        detected_objects: List[DetectedObject],
    ):
        """
        Combine outputs from multiple classifiers.

        Strategy:
          - If only one classifier, pass through its result.
          - If multiple classifiers agree on top category, use the maximum
            confidence (agreement boosts reliability).
          - If they disagree, take the highest-confidence result but reduce
            effective confidence by 20% to reflect uncertainty.
        """
        available = [r for r in clf_results if not r.is_fallback]

        if not available:
            return "Other / unclear", 0.0, []

        if len(available) == 1:
            r = available[0]
            return r.category, r.confidence, r.top_candidates

        # Multiple classifiers
        top_cats = [r.category for r in available]
        agreed = len(set(top_cats)) == 1

        best = max(available, key=lambda r: r.confidence)
        if agreed:
            # Agreement — use best confidence
            return best.category, best.confidence, best.top_candidates
        else:
            # Disagreement — penalise confidence, flag for review
            adjusted_conf = best.confidence * 0.80
            return best.category, adjusted_conf, best.top_candidates

    def _unavailable(self, reason: str, t_start: float) -> AssessmentResult:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return AssessmentResult(
            category="Unavailable",
            confidence=0.0,
            confidence_tier="review_required",
            top_candidates=[],
            detected_objects=[],
            severity=None,
            models_used=[],
            processing_time_ms=elapsed_ms,
            image_metadata={},
            review_required=True,
            warnings=[f"Assessment unavailable: {reason}"],
            is_fallback=True,
            fallback_reason=reason,
        )

    # ── Adapter management ────────────────────────────────────────────────

    def register_classifier(self, adapter: BaseAnalyzerAdapter) -> None:
        """Dynamically register a new classifier adapter."""
        self._classifiers.append(adapter)

    def register_detector(self, adapter: BaseAnalyzerAdapter) -> None:
        """Dynamically register a new detector adapter."""
        self._detectors.append(adapter)

    def model_status(self) -> Dict[str, Any]:
        """Return availability status for all registered adapters."""
        return {
            "classifiers": [
                {
                    "type": a.ADAPTER_TYPE,
                    "class": a.__class__.__name__,
                    "available": a.is_available(),
                }
                for a in self._classifiers
            ],
            "detectors": [
                {
                    "type": a.ADAPTER_TYPE,
                    "class": a.__class__.__name__,
                    "available": a.is_available(),
                }
                for a in self._detectors
            ],
        }
