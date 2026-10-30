"""
AI sub-package: classifier and multimodal deduplication.

Exposes stable interfaces; original module files remain in app/ for
backward compatibility but are soft-aliased here.
"""
from app.ai.classifier import DamageClassifier, SEVERITY_SCORES
from app.ai.duplicates import (
    compute_text_embedding,
    compute_dhash,
    calculate_multimodal_duplicate_score,
)

__all__ = [
    "DamageClassifier",
    "SEVERITY_SCORES",
    "compute_text_embedding",
    "compute_dhash",
    "calculate_multimodal_duplicate_score",
]
