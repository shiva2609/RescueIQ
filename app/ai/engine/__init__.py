"""
app.ai.engine — Disaster damage assessment engine package.

Public API:
    from app.ai.engine import AnalysisEngine, CLIPAdapter, YOLOAdapter
    from app.ai.engine import AssessmentResult, SeverityEstimator
    from app.ai.engine import BaseAnalyzerAdapter, AnalysisResult
    from app.ai.engine import DISASTER_CATEGORIES, confidence_tier
"""
from app.ai.engine.base import (
    BaseAnalyzerAdapter,
    AnalysisResult,
    CandidateClass,
    DetectedObject,
    confidence_tier,
    CONFIDENCE_TIERS,
    LOW_CONFIDENCE_THRESHOLD,
    ABSTENTION_THRESHOLD,
)
from app.ai.engine.engine import AnalysisEngine, AssessmentResult
from app.ai.engine.clip_adapter import CLIPAdapter, DISASTER_CATEGORIES, CATEGORY_LABELS
from app.ai.engine.yolo_adapter import YOLOAdapter, HAZARD_RELEVANT_CLASSES
from app.ai.engine.severity import SeverityEstimator, SeverityEstimate, SEVERITY_LEVELS
from app.ai.engine.preprocessing import (
    PreprocessingError,
    load_and_validate_image,
    validate_raw_bytes,
)

__all__ = [
    "BaseAnalyzerAdapter",
    "AnalysisResult",
    "CandidateClass",
    "DetectedObject",
    "confidence_tier",
    "CONFIDENCE_TIERS",
    "LOW_CONFIDENCE_THRESHOLD",
    "ABSTENTION_THRESHOLD",
    "AnalysisEngine",
    "AssessmentResult",
    "CLIPAdapter",
    "DISASTER_CATEGORIES",
    "CATEGORY_LABELS",
    "YOLOAdapter",
    "HAZARD_RELEVANT_CLASSES",
    "SeverityEstimator",
    "SeverityEstimate",
    "SEVERITY_LEVELS",
    "PreprocessingError",
    "load_and_validate_image",
    "validate_raw_bytes",
]
