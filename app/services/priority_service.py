"""
Priority scoring service.

Formula: P = 0.40·S + 0.25·U + 0.20·I + 0.15·A (provisional rule-based)
Each factor normalised to 0-100 before weighting.
"""
from __future__ import annotations
import math
from typing import Any

from app.config import settings


def calculate_priority(
    severity_score: float,
    urgency: float,
    people_affected: int,
    access_blocked: bool,
    corroboration_count: int = 0,
) -> dict[str, Any]:
    """Return priority score dict with factor breakdown."""
    W = settings
    S = float(max(0.0, min(100.0, severity_score)))
    U = float(max(0.0, min(100.0, urgency)))

    # Impact: logarithmic scale + corroboration boost (capped)
    corrob = min(15.0, corroboration_count * 5.0)
    I = min(100.0, 15.0 + (math.log1p(max(0, people_affected)) * 13.0) + corrob)

    # Access: high if access is blocked
    A = 100.0 if access_blocked else 20.0

    score = (W.WEIGHT_SEVERITY * S) + (W.WEIGHT_URGENCY * U) + (W.WEIGHT_IMPACT * I) + (W.WEIGHT_ACCESS * A)

    breakdown = {
        "severity": {"raw": round(S, 1), "weighted": round(W.WEIGHT_SEVERITY * S, 1)},
        "urgency": {"raw": round(U, 1), "weighted": round(W.WEIGHT_URGENCY * U, 1)},
        "impact": {"raw": round(I, 1), "weighted": round(W.WEIGHT_IMPACT * I, 1)},
        "access": {"raw": round(A, 1), "weighted": round(W.WEIGHT_ACCESS * A, 1)},
        "provisional_weights_notice": (
            "Provisional rule-based weights. Not scientifically validated."
        ),
    }

    return {
        "priority_score": round(score, 1),
        "severity_score": round(S, 1),
        "urgency_score": round(U, 1),
        "impact_score": round(I, 1),
        "access_score": round(A, 1),
        "breakdown": breakdown,
    }
