"""
Module A — Severity Estimation Engine.

Analyses available visual indicators, contextual reporter data, and
situational metadata to produce a preliminary severity classification:
  Low | Moderate | High | Critical

Design principles:
  - Every conclusion is evidence-backed (SeverityEvidence trail)
  - Uncertainty is always quantified and reported
  - A high_risk_flag is raised even at low model confidence when
    contextual signals suggest life-threatening severity
  - Output is explicitly labelled PRELIMINARY DECISION-SUPPORT ASSESSMENT
  - Visual indicators come from CLIP category classification and YOLO detections;
    neither is treated as validated damage quantification

Severity Levels (aligned to FEMA damage categories for interoperability):
  Critical  : Immediate life-threat, infrastructure failure, requires immediate response
  High      : Significant damage, multiple people at risk, response within hours
  Moderate  : Notable damage, limited injuries, response within day
  Low       : Minor incident, no life-threat indicated, standard queue
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from app.triage.models import (
    SeverityAssessment,
    SeverityEvidence,
    PRELIMINARY_LABEL,
)


# ── Severity bands ─────────────────────────────────────────────────────────────

SEVERITY_LEVELS = ["Low", "Moderate", "High", "Critical"]
SEVERITY_SCORES = {"Low": 16.0, "Moderate": 49.0, "High": 74.0, "Critical": 100.0}

# CLIP disaster categories and their baseline severity tendency
CATEGORY_BASELINE: Dict[str, int] = {
    "Flood damage":                        1,   # Moderate
    "Building damage":                     1,   # Moderate
    "Fire and smoke":                      2,   # High
    "Landslide":                           2,   # High
    "Cyclone damage":                      2,   # High
    "Road and infrastructure obstruction": 0,   # Low
    "Other / unclear":                     0,   # Low
    "Unavailable":                         0,
}

# YOLO hazard objects that escalate severity when detected
HAZARD_ESCALATION_OBJECTS = {
    "person": ("Visible person(s) detected — potential casualty or rescue need", 0.4),
    "boat":   ("Boat visible — strong flooding indicator in non-marine context", 0.3),
    "fire":   ("Fire detected by object model", 0.5),
    "smoke":  ("Smoke detected", 0.4),
}

# Uncertainty penalty per data quality issue
UNCERTAINTY_FACTORS = {
    "no_image":           0.35,
    "low_clip_conf":      0.25,
    "no_gps":             0.10,
    "no_people_estimate": 0.10,
    "category_unclear":   0.20,
}


class SeverityEngine:
    """
    Module A: Produces SeverityAssessment from all available incident signals.

    Usage:
        engine = SeverityEngine()
        assessment = engine.assess(
            clip_category="Fire and smoke",
            clip_confidence=0.82,
            yolo_detections=[{"label": "person", "confidence": 0.75}],
            urgency_score=80.0,
            people_affected=50,
            access_blocked=True,
            description="Large structure fire visible from road",
        )
    """

    def assess(
        self,
        clip_category: str = "Other / unclear",
        clip_confidence: float = 0.0,
        yolo_detections: Optional[List[Dict[str, Any]]] = None,
        urgency_score: float = 0.0,
        people_affected: int = 0,
        access_blocked: bool = False,
        description: str = "",
        has_image: bool = True,
        has_gps: bool = True,
    ) -> SeverityAssessment:
        """
        Produce a SeverityAssessment from all available signals.

        Parameters:
            clip_category   : disaster category from CLIP adapter
            clip_confidence : CLIP softmax confidence (NOT used as severity score)
            yolo_detections : list of {"label": str, "confidence": float} dicts
            urgency_score   : reporter-supplied 0–100 urgency
            people_affected : estimated number of people impacted
            access_blocked  : whether emergency vehicle access is blocked
            description     : free-text incident description
            has_image       : whether an image was successfully processed
            has_gps         : whether GPS coordinates were provided
        """
        yolo_detections = yolo_detections or []
        evidence: List[SeverityEvidence] = []
        level_idx = 0        # 0=Low, 1=Moderate, 2=High, 3=Critical
        uncertainty = 0.0
        uncertainty_parts = []

        # ── 1. Category baseline ──────────────────────────────────────────────
        baseline_idx = CATEGORY_BASELINE.get(clip_category, 0)
        level_idx = max(level_idx, baseline_idx)

        evidence.append(SeverityEvidence(
            source="visual_ai",
            signal=f"Disaster category: {clip_category}",
            weight=0.30,
            value=f"Baseline severity index {baseline_idx} → {SEVERITY_LEVELS[baseline_idx]}",
        ))

        # Penalise uncertainty if CLIP confidence is low
        if clip_confidence < 0.30:
            uncertainty += UNCERTAINTY_FACTORS["low_clip_conf"]
            uncertainty_parts.append(
                f"CLIP classification confidence is low ({clip_confidence:.0%}); "
                "category may be incorrect"
            )
        elif clip_category in ("Other / unclear", "Unavailable"):
            uncertainty += UNCERTAINTY_FACTORS["category_unclear"]
            uncertainty_parts.append("Category is 'Other / unclear'; visual evidence ambiguous")

        # ── 2. YOLO object detections ─────────────────────────────────────────
        for det in yolo_detections:
            label = det.get("label", "")
            det_conf = float(det.get("confidence", 0.0))
            if label in HAZARD_ESCALATION_OBJECTS and det_conf >= 0.4:
                signal_text, escalation_weight = HAZARD_ESCALATION_OBJECTS[label]
                level_idx = min(3, level_idx + 1)
                evidence.append(SeverityEvidence(
                    source="visual_ai",
                    signal=signal_text,
                    weight=escalation_weight,
                    value=f"YOLO detected '{label}' with confidence {det_conf:.0%}",
                ))

        # ── 3. Reporter urgency ───────────────────────────────────────────────
        if urgency_score >= 85:
            level_idx = min(3, max(level_idx + 1, 2))  # at least High
            evidence.append(SeverityEvidence(
                source="reporter",
                signal="Reporter indicates very high urgency",
                weight=0.25,
                value=f"Urgency score: {urgency_score:.0f}/100",
            ))
        elif urgency_score >= 60:
            level_idx = min(3, max(level_idx, 1))
            evidence.append(SeverityEvidence(
                source="reporter",
                signal="Reporter indicates elevated urgency",
                weight=0.15,
                value=f"Urgency score: {urgency_score:.0f}/100",
            ))
        else:
            evidence.append(SeverityEvidence(
                source="reporter",
                signal="Reporter urgency at baseline",
                weight=0.05,
                value=f"Urgency score: {urgency_score:.0f}/100",
            ))

        # ── 4. People affected ────────────────────────────────────────────────
        if people_affected > 200:
            level_idx = min(3, max(level_idx + 1, 2))  # at least High
            evidence.append(SeverityEvidence(
                source="reporter",
                signal="Large-scale community impact",
                weight=0.20,
                value=f"{people_affected} people reported affected",
            ))
        elif people_affected > 50:
            level_idx = min(3, max(level_idx, 2))
            evidence.append(SeverityEvidence(
                source="reporter",
                signal="Significant community impact",
                weight=0.15,
                value=f"{people_affected} people reported affected",
            ))
        elif people_affected > 10:
            level_idx = min(3, max(level_idx, 1))
            evidence.append(SeverityEvidence(
                source="reporter",
                signal="Moderate community impact",
                weight=0.08,
                value=f"{people_affected} people reported affected",
            ))
        else:
            uncertainty += UNCERTAINTY_FACTORS["no_people_estimate"]
            uncertainty_parts.append("Number of people affected unknown or zero — may be underestimate")

        # ── 5. Access blocked ─────────────────────────────────────────────────
        if access_blocked:
            level_idx = min(3, level_idx + 1)
            evidence.append(SeverityEvidence(
                source="reporter",
                signal="Emergency vehicle access blocked",
                weight=0.20,
                value="Access blocked = True",
            ))

        # ── 6. Description keyword signals ────────────────────────────────────
        desc_lower = description.lower()
        critical_keywords = ["death", "dead", "dying", "trapped", "buried", "collapse",
                              "explosion", "gas leak", "electrocution", "drowning"]
        high_keywords = ["injured", "hurt", "fire", "flood", "burning", "unstable",
                         "cracking", "sinking", "cut off"]
        matched_critical = [k for k in critical_keywords if k in desc_lower]
        matched_high = [k for k in high_keywords if k in desc_lower]

        if matched_critical:
            level_idx = 3   # Critical override
            evidence.append(SeverityEvidence(
                source="reporter",
                signal="Critical keywords in description",
                weight=0.40,
                value=f"Keywords: {', '.join(matched_critical)}",
            ))
        elif matched_high:
            level_idx = min(3, max(level_idx, 2))
            evidence.append(SeverityEvidence(
                source="reporter",
                signal="High-severity keywords in description",
                weight=0.20,
                value=f"Keywords: {', '.join(matched_high)}",
            ))

        # ── 7. Data-quality uncertainty ───────────────────────────────────────
        if not has_image:
            uncertainty += UNCERTAINTY_FACTORS["no_image"]
            uncertainty_parts.append("No image available — visual indicators absent")
        if not has_gps:
            uncertainty += UNCERTAINTY_FACTORS["no_gps"]
            uncertainty_parts.append("GPS coordinates missing — geo-context unavailable")

        # ── 8. High-risk flag ─────────────────────────────────────────────────
        # Flag reports even when data is sparse if context suggests critical risk
        high_risk_flag = (
            level_idx == 3
            or (clip_confidence < 0.30 and (access_blocked or people_affected > 20))
            or bool(matched_critical)
        )

        # ── 9. Fire category always minimum High ─────────────────────────────
        if clip_category == "Fire and smoke":
            level_idx = max(level_idx, 2)

        # ── Finalise ──────────────────────────────────────────────────────────
        level = SEVERITY_LEVELS[min(3, level_idx)]
        score = SEVERITY_SCORES[level]
        uncertainty = min(1.0, uncertainty)
        uncertainty_reason = (
            "; ".join(uncertainty_parts)
            if uncertainty_parts
            else "Sufficient data available — uncertainty is minimal"
        )

        return SeverityAssessment(
            level=level,
            score_0_100=score,
            evidence=evidence,
            uncertainty=uncertainty,
            uncertainty_reason=uncertainty_reason,
            high_risk_flag=high_risk_flag,
            preliminary_label=PRELIMINARY_LABEL,
        )
