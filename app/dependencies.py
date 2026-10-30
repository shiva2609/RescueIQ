"""
Shared FastAPI dependencies.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import settings


# ── Database session ──────────────────────────────────────────────────────────
DbSession = Annotated[Session, Depends(get_db)]


# ── Pagination ────────────────────────────────────────────────────────────────
class PaginationParams:
    def __init__(
        self,
        page: int = Query(default=1, ge=1, description="1-indexed page number"),
        limit: int = Query(
            default=settings.DEFAULT_PAGE_SIZE,
            ge=1,
            le=settings.MAX_PAGE_SIZE,
            description="Items per page",
        ),
    ):
        self.page = page
        self.limit = limit
        self.offset = (page - 1) * limit


Pagination = Annotated[PaginationParams, Depends(PaginationParams)]


# ── Common validators ─────────────────────────────────────────────────────────
def get_incident_or_404(incident_id: str, db: Session):
    """Return the incident row or raise 404."""
    from app.models.incident import Incident
    row = db.get(Incident, incident_id)
    if row is None or row.incident_status == "Deleted":
        raise HTTPException(status_code=404, detail="Incident not found")
    return row
