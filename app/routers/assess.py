"""
Assessment router — /api/v2/assess/*

New versioned endpoints that do not break the existing /api/analyze/image route.
The legacy route is preserved unchanged; this router adds richer multi-model output.

Endpoints:
  POST /api/v2/assess/image          — full CLIP + YOLO assessment
  GET  /api/v2/assess/status         — adapter availability and model info
  GET  /api/v2/assess/categories     — list supported disaster categories
  POST /api/v2/assess/severity       — rule-based severity estimate from JSON
"""
from __future__ import annotations

import io
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.ai.engine import (
    AnalysisEngine,
    AssessmentResult,
    CLIPAdapter,
    YOLOAdapter,
    CATEGORY_LABELS,
    DISASTER_CATEGORIES,
    SeverityEstimator,
    PreprocessingError,
    validate_raw_bytes,
)
from app.ai.engine.preprocessing import load_and_validate_image, get_image_metadata

router = APIRouter(prefix="/api/v2/assess", tags=["Assessment Engine v2"])


# ── Lazy-initialised singleton engine ─────────────────────────────────────────
# Models are large; we initialise on first request to avoid startup lag.
_engine: Optional[AnalysisEngine] = None


def _get_engine() -> AnalysisEngine:
    global _engine
    if _engine is None:
        _engine = AnalysisEngine(
            classifiers=[CLIPAdapter()],
            detectors=[YOLOAdapter()],
        )
    return _engine


# ── Pydantic request/response schemas ─────────────────────────────────────────

class SeverityRequest(BaseModel):
    category: str = Field(..., description="Disaster category from classification")
    urgency_score: float = Field(default=0.0, ge=0, le=100)
    people_affected: int = Field(default=0, ge=0)
    access_blocked: bool = Field(default=False)
    detected_object_labels: List[str] = Field(
        default_factory=list,
        description="Optional list of YOLO-detected object labels",
    )


class CategoryInfo(BaseModel):
    label: str
    prompts: List[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/image",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Run full CLIP + YOLO disaster damage assessment on an uploaded image",
    description=(
        "Analyses an uploaded image using:\n"
        "1. CLIP zero-shot multi-prompt category classification\n"
        "2. YOLOv8 object detection for visible hazard identification\n"
        "3. Rule-based severity estimation (never derived from AI confidence)\n\n"
        "**Important:** Confidence reflects category recognition, NOT damage severity. "
        "See `severity.disclaimer` in the response for details."
    ),
)
async def assess_image(
    image: UploadFile = File(..., description="JPEG, PNG, or WebP image to assess"),
    urgency_score: float = Form(default=0.0, ge=0.0, le=100.0,
                                description="Reporter-supplied urgency 0-100"),
    people_affected: int = Form(default=0, ge=0,
                                description="Estimated number of people affected"),
    access_blocked: bool = Form(default=False,
                                description="Whether emergency vehicle access is blocked"),
    reporter: str = Form(default="Anonymous"),
):
    # 1. Read and validate raw bytes
    raw = await image.read()
    try:
        validate_raw_bytes(raw)
    except PreprocessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # 2. Save to a temp path (engine works on-disk for YOLO compatibility)
    stem = f"assess_{uuid.uuid4()}"
    ct = (image.content_type or "image/jpeg").split(";")[0].strip()
    ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(ct, ".jpg")
    tmp_path = settings.UPLOAD_DIR / f"{stem}{ext}"
    tmp_path.write_bytes(raw)

    try:
        # 3. Run multi-model assessment
        engine = _get_engine()
        result: AssessmentResult = engine.assess(
            image_path=tmp_path,
            urgency_score=urgency_score,
            people_affected=people_affected,
            access_blocked=access_blocked,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Assessment error: {exc}")
    finally:
        # Clean up temp file (assessment is complete)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return result.to_dict()


@router.get(
    "/status",
    response_model=Dict[str, Any],
    summary="Check model adapter availability and current configuration",
)
def assessment_status():
    engine = _get_engine()
    status = engine.model_status()
    status["engine_version"] = "2.0"
    status["categories_supported"] = CATEGORY_LABELS
    status["notes"] = [
        "CLIP confidence is a category recognition score only — not damage severity.",
        "YOLOv8 uses COCO-80 classes; disaster-specific fine-tuning improves accuracy.",
        "Severity is rule-based; field verification required for resource allocation.",
    ]
    return status


@router.get(
    "/categories",
    response_model=List[CategoryInfo],
    summary="List supported disaster categories and the CLIP text prompts used per category",
)
def list_categories():
    return [
        CategoryInfo(label=label, prompts=prompts)
        for label, prompts in DISASTER_CATEGORIES.items()
    ]


@router.post(
    "/severity",
    response_model=Dict[str, Any],
    summary="Get rule-based severity estimate from category and contextual metadata (no image needed)",
)
def estimate_severity(payload: SeverityRequest):
    from app.ai.engine.base import DetectedObject

    fake_objects = [
        DetectedObject(label=lbl, confidence=0.5, is_hazard_relevant=True)
        for lbl in payload.detected_object_labels
    ]
    estimator = SeverityEstimator()
    estimate = estimator.estimate(
        category=payload.category,
        detected_objects=fake_objects,
        urgency_score=payload.urgency_score,
        people_affected=payload.people_affected,
        access_blocked=payload.access_blocked,
    )
    return estimate.to_dict()
