from typing import Optional, List
from fastapi import APIRouter, Query
from sqlalchemy import select
from app.dependencies import DbSession
from app.models.audit import StatusHistory

router = APIRouter(tags=["Audit"])


@router.get(
    "/api/audit-logs",
    response_model=List[dict],
    summary="Retrieve recent immutable audit log entries (IM-4)",
)
def list_audit_logs(
    db: DbSession,
    limit: int = Query(default=100, ge=1, le=1000),
    incident_id: Optional[str] = Query(default=None, description="Filter by incident ID"),
):
    q = select(StatusHistory).order_by(StatusHistory.created_at.desc()).limit(limit)
    if incident_id:
        q = q.where(StatusHistory.incident_id == incident_id)
    rows = db.execute(q).scalars().all()
    return [
        {
            "id": r.id,
            "incident_id": r.incident_id,
            "action": r.action,
            "previous_value": r.previous_value,
            "new_value": r.new_value,
            "actor": r.actor,
            "notes": r.notes,
            "created_at": r.created_at.isoformat() if hasattr(r.created_at, "isoformat") else str(r.created_at),
        }
        for r in rows
    ]
