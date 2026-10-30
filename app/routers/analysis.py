"""Analysis router: stats, image-only analysis."""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from app.dependencies import DbSession
from app.services.stats_service import get_stats
from app.services.image_service import validate_and_save_upload
from app.config import settings
from app.schemas.audit import StatsResponse, AnalysisResponse

router = APIRouter(tags=["Analysis"])


@router.get(
    "/api/stats",
    response_model=StatsResponse,
    summary="Aggregate statistics: counts by severity, status, category",
)
def stats(db: DbSession):
    return get_stats(db)


@router.post(
    "/api/analyze/image",
    response_model=AnalysisResponse,
    summary="Run EfficientNet-B0 on an uploaded image without creating an incident",
)
async def analyze_image(
    db: DbSession,
    image: UploadFile = File(...),
):
    import uuid
    stem = f"temp_{uuid.uuid4()}"
    img_info = await validate_and_save_upload(image, dest_dir=settings.UPLOAD_DIR, stem=stem)

    from pathlib import Path
    from app.routers.incidents import _get_classifier, _ext
    clf = _get_classifier()
    img_path = settings.UPLOAD_DIR / f"{stem}{_ext(img_info['mime_type'])}"
    pred = clf.predict(img_path)

    return AnalysisResponse(
        prediction=pred,
        dhash=img_info.get("dhash"),
        preview_url=f"/uploads/{img_path.name}",
    )
