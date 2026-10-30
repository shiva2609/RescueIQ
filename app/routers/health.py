"""Health-check router."""
from fastapi import APIRouter
from app.config import settings
from app.schemas.audit import HealthResponse

router = APIRouter(tags=["System"])


@router.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Backend health and model information",
)
def health():
    """Returns system status and loaded AI model metadata."""
    return HealthResponse(
        status="ok",
        app=settings.APP_TITLE,
        version=settings.APP_VERSION,
        vision_model="EfficientNet-B0 (3-class triage, staged weights ready)",
        nlp_model="all-MiniLM-L6-v2 (semantic similarity)",
        deduplication="Multimodal (Text + Image + GPS + Time)",
        priority_formula=(
            f"P = {settings.WEIGHT_SEVERITY}·S + {settings.WEIGHT_URGENCY}·U "
            f"+ {settings.WEIGHT_IMPACT}·I + {settings.WEIGHT_ACCESS}·A (provisional rule-based)"
        ),
        db_backend=settings.DATABASE_URL.split("://")[0],
    )
