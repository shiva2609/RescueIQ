"""
app/triage/__init__.py — Triage package public API.

Exports the three primary engine instances (singletons) and their
result model types for use in routes and services.
"""
from app.triage.severity_engine import SeverityEngine
from app.triage.duplicate_engine import DuplicateEngine
from app.triage.priority_engine import PriorityEngine, weight_registry

from app.triage.models import (
    SeverityAssessment,
    SeverityEvidence,
    DuplicateCandidate,
    DuplicateSimilarityBreakdown,
    DuplicateGroup,
    PriorityFactor,
    PriorityScore,
    PriorityWeightConfig,
    PRELIMINARY_LABEL,
)

# Shared singleton engines — import these in routes, not new instances
severity_engine = SeverityEngine()
duplicate_engine = DuplicateEngine()
priority_engine = PriorityEngine()

__all__ = [
    "severity_engine",
    "duplicate_engine",
    "priority_engine",
    "weight_registry",
    "SeverityAssessment",
    "SeverityEvidence",
    "DuplicateCandidate",
    "DuplicateSimilarityBreakdown",
    "DuplicateGroup",
    "PriorityFactor",
    "PriorityScore",
    "PriorityWeightConfig",
    "PRELIMINARY_LABEL",
]
