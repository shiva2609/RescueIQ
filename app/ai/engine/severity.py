"""
Rule-based severity estimator.

─────────────────────────────────────────────────────────────────────────────
CRITICAL DESIGN PRINCIPLE
─────────────────────────────────────────────────────────────────────────────
Severity is NOT derived from AI model classification confidence.

CLIP or any image classification model's confidence score measures how
closely an image resembles a category's text prompt — it is a RECOGNITION
score, not a damage quantification. Treating it as severity would produce
systematic biases (e.g., a very clear, well-lit photo of mild flooding could
score higher "confidence" than a blurry, dark photo of catastrophic damage).

This module implements a transparent, rule-based severity estimation system
that:
  1. Uses detected objects (YOLO) as proxy evidence
  2. Uses category-level heuristics grounded in disaster response doctrine
  3. Uses contextual metadata (reporter-supplied urgency, people affected)
  4. Produces a human-interpretable rationale for every estimate
  5. Always carries a clear "provisional / not validated" disclaimer

This approach is intentionally conservative. In production, severity
should be validated by trained field assessors using domain-specific rubrics
(e.g., FEMA damage assessment protocols, Copernicus EMS grading).
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.ai.engine.base import DetectedObject


# ── Severity levels ───────────────────────────────────────────────────────────

SEVERITY_LEVELS = ["none", "minor", "moderate", "severe", "critical"]

# Base severity lookup by disaster category (domain knowledge heuristic)
# These are conservative starting points, not ground-truth values.
CATEGORY_BASE_SEVERITY: Dict[str, int] = {
    "Flood damage":                       2,   # moderate
    "Building damage":                    2,   # moderate
    "Fire and smoke":                     3,   # severe
    "Landslide":                          3,   # severe
    "Cyclone damage":                     2,   # moderate
    "Road and infrastructure obstruction": 1,  # minor
    "Other / unclear":                    0,   # none
    "Unavailable":                        0,
}

SEVERITY_DISCLAIMER = (
    "PROVISIONAL ESTIMATE ONLY. This severity level is derived from rule-based "
    "heuristics using disaster category, detected objects, and reported metadata. "
    "It has NOT been validated against ground truth, calibrated on historical "
    "events, or reviewed by a trained damage assessor. "
    "Do not use for official emergency resource allocation without field verification."
)


@dataclass
class SeverityEstimate:
    """
    A transparency-first severity estimate.

    level      : one of SEVERITY_LEVELS
    score_0_100: numeric convenience (0=none, 25=minor, 50=moderate, 75=severe, 100=critical)
    rationale  : human-readable explanation of how score was derived
    factors    : list of individual contributing factors and their adjustments
    disclaimer : ALWAYS included warning about provisional nature
    """
    level: str
    score_0_100: float
    rationale: str
    factors: List[Dict[str, Any]] = field(default_factory=list)
    disclaimer: str = SEVERITY_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "score_0_100": round(self.score_0_100, 1),
            "rationale": self.rationale,
            "factors": self.factors,
            "disclaimer": self.disclaimer,
        }


# ── Estimator ─────────────────────────────────────────────────────────────────

class SeverityEstimator:
    """
    Rule-based damage severity estimator.

    Inputs:
      - category: string from CLIP or EfficientNet classification
      - detected_objects: list of DetectedObject from YOLO (may be empty)
      - urgency_score: 0-100 from reporter (optional, from incident metadata)
      - people_affected: int (optional, from incident metadata)
      - access_blocked: bool (optional, from incident metadata)
    """

    def estimate(
        self,
        category: str,
        detected_objects: Optional[List[DetectedObject]] = None,
        urgency_score: float = 0.0,
        people_affected: int = 0,
        access_blocked: bool = False,
    ) -> SeverityEstimate:

        detected_objects = detected_objects or []
        factors: List[Dict[str, Any]] = []

        # ── 1. Base level from category ───────────────────────────────────
        base_idx = CATEGORY_BASE_SEVERITY.get(category, 0)
        factors.append({
            "factor": "Disaster category",
            "value": category,
            "adjustment": base_idx,
            "note": "Conservative base from disaster type taxonomy",
        })
        idx = base_idx

        # ── 2. Person detection (potential casualty indicator) ─────────────
        persons = [d for d in detected_objects if d.label == "person" and d.confidence >= 0.4]
        if persons:
            idx = min(idx + 1, 4)
            factors.append({
                "factor": "Person detected by YOLO",
                "value": f"{len(persons)} individual(s) visible",
                "adjustment": +1,
                "note": "Presence of people indicates potential casualty or rescue need",
            })

        # ── 3. People affected (reporter-supplied) ────────────────────────
        if people_affected > 100:
            idx = min(idx + 1, 4)
            factors.append({
                "factor": "People affected",
                "value": people_affected,
                "adjustment": +1,
                "note": "More than 100 people affected escalates severity",
            })
        elif people_affected > 20:
            factors.append({
                "factor": "People affected",
                "value": people_affected,
                "adjustment": 0,
                "note": "Moderate community impact (no escalation)",
            })

        # ── 4. Access blocked ─────────────────────────────────────────────
        if access_blocked:
            idx = min(idx + 1, 4)
            factors.append({
                "factor": "Access blocked",
                "value": True,
                "adjustment": +1,
                "note": "Blocked access prevents emergency response and escalates severity",
            })

        # ── 5. Urgency from reporter ──────────────────────────────────────
        if urgency_score >= 80:
            idx = min(idx + 1, 4)
            factors.append({
                "factor": "Reporter urgency",
                "value": urgency_score,
                "adjustment": +1,
                "note": "Reporter indicates high urgency (≥80/100)",
            })

        # ── 6. Fire category — always severe or above ─────────────────────
        if category == "Fire and smoke" and idx < 3:
            idx = 3
            factors.append({
                "factor": "Fire / smoke override",
                "value": category,
                "adjustment": "min(3)",
                "note": "Fire events are at minimum severe due to rapid escalation risk",
            })

        # ── Build result ──────────────────────────────────────────────────
        level = SEVERITY_LEVELS[idx]
        score = idx * 25.0

        rationale_parts = [f"Base category '{category}' maps to '{SEVERITY_LEVELS[base_idx]}' severity."]
        adjustments = [f for f in factors if isinstance(f.get("adjustment"), int) and f["adjustment"] != 0]
        if adjustments:
            rationale_parts.append(
                "Escalated by: " + "; ".join(
                    f"{f['factor']} ({'+' if f['adjustment'] > 0 else ''}{f['adjustment']})"
                    for f in adjustments
                ) + "."
            )
        if not adjustments:
            rationale_parts.append("No escalation factors identified.")

        rationale = " ".join(rationale_parts)

        return SeverityEstimate(
            level=level,
            score_0_100=score,
            rationale=rationale,
            factors=factors,
        )
