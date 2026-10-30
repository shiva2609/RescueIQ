"""
Module C — Response Prioritization Engine.

Calculates a transparent 0–100 priority score from four independent factors:
  S  : Damage Severity (from Module A)
  U  : Reporter Urgency
  I  : Population Impact (logarithmic scale + corroboration boost)
  A  : Accessibility (whether emergency access is blocked)

Formula:
  P = w_s·S + w_u·U + w_i·I + w_a·A

Weights are administrator-configurable, stored in a runtime registry,
and retrieved via API. Changing weights does not alter stored incident
records — scores are re-evaluated on demand.

Design guarantees:
  - All outputs carry PRELIMINARY DECISION-SUPPORT ASSESSMENT label
  - Every factor has a human-readable explanation
  - high_risk_flag is raised when: score ≥ 75, severity=Critical, or
    access is blocked AND urgency is high — even at low model confidence
  - Weights can be updated at runtime; changes are logged
"""
from __future__ import annotations

import math
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.triage.models import (
    PriorityFactor,
    PriorityScore,
    PriorityWeightConfig,
    PRELIMINARY_LABEL,
)


# ── Runtime weight registry (thread-safe singleton) ────────────────────────────

class _WeightRegistry:
    """
    Holds the current weight configuration for the priority engine.
    Updates are atomic and logged with a timestamp and actor.
    Persists for the lifetime of the process (restart resets to defaults).
    """
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._config = PriorityWeightConfig()
        self._history: List[Dict[str, Any]] = []

    def get(self) -> PriorityWeightConfig:
        with self._lock:
            return PriorityWeightConfig(
                weight_severity=self._config.weight_severity,
                weight_urgency=self._config.weight_urgency,
                weight_impact=self._config.weight_impact,
                weight_access=self._config.weight_access,
            )

    def update(
        self,
        weight_severity: float,
        weight_urgency: float,
        weight_impact: float,
        weight_access: float,
        actor: str = "admin",
    ) -> PriorityWeightConfig:
        new_cfg = PriorityWeightConfig(
            weight_severity=weight_severity,
            weight_urgency=weight_urgency,
            weight_impact=weight_impact,
            weight_access=weight_access,
        )
        new_cfg.validate()   # raises ValueError if sum ≠ 1.0
        with self._lock:
            old = self._config
            self._config = new_cfg
            self._history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "actor": actor,
                "old_weights": old.to_dict(),
                "new_weights": new_cfg.to_dict(),
            })
        return new_cfg

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)


# Singleton exported for use by API routes
weight_registry = _WeightRegistry()


# ── Priority bands ─────────────────────────────────────────────────────────────

def _band(score: float) -> str:
    if score >= 75:
        return "Critical"
    elif score >= 50:
        return "High"
    elif score >= 25:
        return "Moderate"
    return "Low"


# ── Normalisation helpers ──────────────────────────────────────────────────────

def _normalise_severity(level: str) -> float:
    """Map severity level string to 0-100 numeric."""
    return {"Low": 16.0, "Moderate": 49.0, "High": 74.0, "Critical": 100.0}.get(level, 0.0)


def _normalise_impact(people_affected: int, corroboration_count: int = 0) -> Tuple:
    """
    Logarithmic people-impact score with corroboration boost.
    Returns (score_0_100, explanation)
    """
    corrob = min(15.0, corroboration_count * 5.0)
    raw = min(100.0, 15.0 + math.log1p(max(0, people_affected)) * 13.0 + corrob)
    expl_parts = [f"People affected: {people_affected}"]
    if corroboration_count:
        expl_parts.append(f"Corroboration: +{corrob:.0f} pts from {corroboration_count} duplicate reports")
    return raw, "; ".join(expl_parts)


# Fix for missing Tuple import
from typing import Tuple  # noqa: E402


# ── Priority Engine ────────────────────────────────────────────────────────────

class PriorityEngine:
    """
    Module C: Configurable response prioritisation engine.

    Usage:
        engine = PriorityEngine()
        score = engine.calculate(
            severity_level="High",
            urgency_score=70.0,
            people_affected=120,
            access_blocked=False,
            corroboration_count=2,
        )
    """

    def calculate(
        self,
        severity_level: str = "Low",
        urgency_score: float = 0.0,
        people_affected: int = 0,
        access_blocked: bool = False,
        corroboration_count: int = 0,
        model_confidence: float = 1.0,
    ) -> PriorityScore:
        """
        Compute a fully explained PriorityScore.

        Parameters:
            severity_level      : "Low" | "Moderate" | "High" | "Critical"
                                  (from Module A — never raw AI confidence)
            urgency_score       : 0–100 reporter-supplied urgency
            people_affected     : estimated count of affected persons
            access_blocked      : True if emergency access is obstructed
            corroboration_count : number of duplicate reports confirming this incident
            model_confidence    : AI model classification confidence (used ONLY for
                                  high_risk_flag heuristic, never as severity proxy)
        """
        weights = weight_registry.get()

        # ── Factor S — Severity ───────────────────────────────────────────────
        S = _normalise_severity(severity_level)
        f_severity = PriorityFactor(
            name="Damage Severity",
            raw_score=S,
            weight=weights.weight_severity,
            weighted_contribution=weights.weight_severity * S,
            explanation=(
                f"Preliminary severity level is '{severity_level}' "
                f"(rule-based assessment, score={S:.0f}/100). "
                "Severity is determined by visual indicators, urgency, people count, "
                "and description keywords — NOT by AI model confidence."
            ),
            evidence=[f"Severity level: {severity_level}", f"Numeric representation: {S:.0f}"],
        )

        # ── Factor U — Urgency ────────────────────────────────────────────────
        U = max(0.0, min(100.0, urgency_score))
        urgency_interpretation = (
            "Critical urgency (≥85)" if U >= 85
            else "High urgency (60–84)" if U >= 60
            else "Moderate urgency (30–59)" if U >= 30
            else "Low urgency (<30)"
        )
        f_urgency = PriorityFactor(
            name="Reporter Urgency",
            raw_score=U,
            weight=weights.weight_urgency,
            weighted_contribution=weights.weight_urgency * U,
            explanation=(
                f"Reporter-supplied urgency: {U:.0f}/100 — {urgency_interpretation}. "
                "This is the reporter's subjective assessment and may differ from "
                "field verification. Use with caution."
            ),
            evidence=[f"Raw urgency input: {urgency_score:.0f}"],
        )

        # ── Factor I — Population Impact ──────────────────────────────────────
        I, impact_expl = _normalise_impact(people_affected, corroboration_count)
        f_impact = PriorityFactor(
            name="Population Impact",
            raw_score=I,
            weight=weights.weight_impact,
            weighted_contribution=weights.weight_impact * I,
            explanation=(
                f"Impact score {I:.0f}/100 computed using logarithmic scale "
                f"(base: {people_affected} people). {impact_expl}."
            ),
            evidence=[
                f"People affected: {people_affected}",
                f"Corroborating reports: {corroboration_count}",
                f"Computed impact score: {I:.1f}",
            ],
        )

        # ── Factor A — Accessibility ──────────────────────────────────────────
        A = 100.0 if access_blocked else 20.0
        f_access = PriorityFactor(
            name="Accessibility",
            raw_score=A,
            weight=weights.weight_access,
            weighted_contribution=weights.weight_access * A,
            explanation=(
                "Emergency vehicle access is BLOCKED — response difficulty is maximum (100). "
                "Blocked access substantially increases response complexity and time-to-aid."
                if access_blocked else
                "Emergency vehicle access appears clear. "
                "Accessibility penalty is minimal (20/100)."
            ),
            evidence=[f"Access blocked: {access_blocked}"],
        )

        # ── Final score ───────────────────────────────────────────────────────
        final_score = (
            f_severity.weighted_contribution
            + f_urgency.weighted_contribution
            + f_impact.weighted_contribution
            + f_access.weighted_contribution
        )
        final_score = min(100.0, max(0.0, final_score))

        # ── High-risk flag ────────────────────────────────────────────────────
        high_risk_flag = False
        flag_reasons = []

        if severity_level == "Critical":
            high_risk_flag = True
            flag_reasons.append("Severity is Critical")
        if final_score >= 75:
            high_risk_flag = True
            flag_reasons.append(f"Priority score {final_score:.0f} ≥ 75")
        if access_blocked and U >= 60:
            high_risk_flag = True
            flag_reasons.append("Access blocked AND high urgency reported")
        if model_confidence < 0.30 and severity_level in ("High", "Critical"):
            high_risk_flag = True
            flag_reasons.append(
                "High/Critical severity at LOW model confidence — may be underestimated; "
                "human review strongly recommended"
            )

        flag_reason = (
            " | ".join(flag_reasons)
            if flag_reasons
            else ""
        )

        return PriorityScore(
            priority_score=round(final_score, 1),
            priority_band=_band(final_score),
            factors=[f_severity, f_urgency, f_impact, f_access],
            weight_config=weights,
            high_risk_flag=high_risk_flag,
            flag_reason=flag_reason,
            preliminary_label=PRELIMINARY_LABEL,
        )
