"""
Incidents router — full CRUD with filtering, search, and pagination.

All endpoints preserve the original API contract; new query params are
additive and default to backward-compatible values.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, List, Dict

from fastapi import APIRouter, File, Form, Query, UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.dependencies import DbSession, Pagination, get_incident_or_404
from app.ai.classifier import SEVERITY_SCORES
from app.ai.duplicates import compute_text_embedding
from app.schemas.common import PaginatedResponse, SuccessResponse
from app.schemas.incident import IncidentRead, IncidentUpdate
from app.services import image_service, incident_service

router = APIRouter(prefix="/api/incidents", tags=["Incidents"])


# ── POST /api/incidents ───────────────────────────────────────────────────────
@router.post(
    "",
    response_model=dict,
    status_code=201,
    summary="Submit a new citizen incident report",
    description=(
        "Accepts a multipart form with an image, description, GPS coordinates "
        "and metadata. Runs EfficientNet-B0 classification, multimodal duplicate "
        "detection, and priority scoring before persisting."
    ),
)
async def create_incident(
    db: DbSession,
    description: str = Form(..., min_length=5, max_length=2000),
    latitude: float = Form(..., ge=-90, le=90),
    longitude: float = Form(..., ge=-180, le=180),
    category: str = Form(default="Flood"),
    urgency: int = Form(default=50, ge=0, le=100),
    people_affected: int = Form(default=0, ge=0),
    access_blocked: bool = Form(default=False),
    reporter: str = Form(default="Anonymous", max_length=200),
    image: UploadFile = File(...),
):
    import uuid as _uuid
    stem = str(_uuid.uuid4())

    # 1. Validate and save image
    img_info = await image_service.validate_and_save_upload(
        image, dest_dir=settings.UPLOAD_DIR, stem=stem
    )

    # 2. Classify image
    from app.ai.classifier import DamageClassifier
    clf = _get_classifier()
    from pathlib import Path
    pred = clf.predict(Path(settings.UPLOAD_DIR / f"{stem}{_ext(img_info['mime_type'])}"))

    # 3. Text embedding for dedup
    emb = compute_text_embedding(description)
    emb_list = emb.tolist() if emb is not None else None

    # 4. Persist via service
    row = incident_service.create_incident(
        db,
        description=description,
        image_path=img_info["image_path"],
        latitude=latitude,
        longitude=longitude,
        category=category,
        urgency=urgency,
        people_affected=people_affected,
        access_blocked=access_blocked,
        reporter=reporter,
        damage_class=pred["damage_class"],
        model_confidence=pred["confidence"],
        is_abstained=pred["is_abstained"],
        probabilities=pred["probabilities"],
        model_name=pred["model_name"],
        severity_score=pred["severity_score"],
        image_hash=img_info.get("dhash"),
        text_embedding=emb_list,
    )

    return incident_service._serialize(row)


def _ext(mime: str) -> str:
    return {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(mime, ".jpg")


_clf_instance = None

def _get_classifier():
    global _clf_instance
    if _clf_instance is None:
        from app.ai.classifier import DamageClassifier
        _clf_instance = DamageClassifier(abstention_threshold=settings.ABSTENTION_THRESHOLD)
    return _clf_instance


# ── GET /api/incidents ────────────────────────────────────────────────────────
@router.get(
    "",
    response_model=List[dict],
    summary="List all incidents (filterable, searchable, paginated)",
)
def list_incidents(
    db: DbSession,
    status: Optional[str] = Query(default=None, description="Filter by verification_status"),
    category: Optional[str] = Query(default=None, description="Filter by hazard category"),
    damage_class: Optional[str] = Query(default=None, description="Filter by AI damage class"),
    incident_status: Optional[str] = Query(default=None, description="Filter by incident lifecycle status"),
    search: Optional[str] = Query(default=None, description="Full-text search across description, reporter, category, ID"),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    paginate: bool = Query(default=False, description="Return PaginatedResponse envelope instead of plain list"),
):
    items, total = incident_service.list_incidents(
        db,
        status=status,
        category=category,
        damage_class=damage_class,
        incident_status=incident_status,
        search=search,
        page=page,
        limit=limit,
    )
    if paginate:
        import math
        return {
            "items": items,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": max(1, math.ceil(total / limit)),
        }
    # Default: flat list (backward-compatible)
    return items


# ── GET /api/incidents/nearby (PostGIS / Geospatial query) ───────────────────
@router.get(
    "/nearby",
    response_model=list[dict],
    summary="Query incidents within a specified radius (PostGIS / Haversine)",
)
def get_nearby_incidents(
    db: DbSession,
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Center point latitude"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Center point longitude"),
    radius_meters: float = Query(default=5000.0, ge=10.0, le=500000.0, description="Search radius in meters"),
    limit: int = Query(default=50, ge=1, le=100),
):
    from app.services.geo_service import find_incidents_nearby
    results = find_incidents_nearby(
        db,
        latitude=latitude,
        longitude=longitude,
        radius_meters=radius_meters,
        limit=limit,
    )
    output = []
    for inc, dist_m in results:
        data = incident_service._serialize(inc)
        data["distance_meters"] = dist_m
        output.append(data)
    return output


# ── GET /api/incidents/{id} ───────────────────────────────────────────────────
@router.get(
    "/{incident_id}",
    response_model=dict,
    summary="Get full incident detail including history and duplicate links",
)
def get_incident(incident_id: str, db: DbSession):
    return incident_service.get_incident(db, incident_id)


# ── PATCH /api/incidents/{id} ─────────────────────────────────────────────────
@router.patch(
    "/{incident_id}",
    response_model=dict,
    summary="Update verification status, assignment, or override AI severity",
)
def update_incident(incident_id: str, payload: dict, db: DbSession):
    return incident_service.update_incident(db, incident_id, payload)


# ── DELETE /api/incidents/{id} ────────────────────────────────────────────────
@router.delete(
    "/{incident_id}",
    status_code=204,
    summary="Soft-delete an incident (sets status to 'Deleted')",
)
def delete_incident(
    incident_id: str,
    db: DbSession,
    actor: str = Query(default="Operator"),
):
    incident_service.delete_incident(db, incident_id, actor=actor)
    return None


# ── GET /api/incidents/{id}/images ────────────────────────────────────────────
@router.get(
    "/{incident_id}/images",
    response_model=list[dict],
    summary="List all images attached to an incident",
)
def list_incident_images(incident_id: str, db: DbSession):
    row = get_incident_or_404(incident_id, db)
    return [
        {
            "id": img.id,
            "incident_id": img.incident_id,
            "image_url": "/" + img.image_path,
            "mime_type": img.mime_type,
            "file_size_bytes": img.file_size_bytes,
            "is_primary": img.is_primary,
            "created_at": img.created_at.isoformat(),
        }
        for img in row.images
    ]
