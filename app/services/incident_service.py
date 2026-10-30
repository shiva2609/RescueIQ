"""
Incident service — all database operations for incidents.

This layer owns the business logic; routers call these functions and
receive plain Python objects / ORM rows that are then serialised by
Pydantic schemas.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from app.ai.classifier import SEVERITY_SCORES
from app.ai.duplicates import (
    calculate_multimodal_duplicate_score,
    compute_text_embedding,
)
from app.models.audit import StatusHistory
from app.models.duplicate import DuplicateLink
from app.models.incident import Incident
from app.services.priority_service import calculate_priority


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_audit(
    db: Session,
    incident_id: str,
    action: str,
    previous: str,
    new: str,
    actor: str = "System",
    notes: str | None = None,
) -> None:
    db.add(StatusHistory(
        id=str(uuid.uuid4()),
        incident_id=incident_id,
        action=action,
        previous_value=str(previous),
        new_value=str(new),
        actor=actor,
        notes=notes,
        created_at=datetime.now(timezone.utc),
    ))


def _serialize(row: Incident) -> dict[str, Any]:
    """Convert ORM row to dict that matches the original API shape."""
    d: dict[str, Any] = {
        c.name: getattr(row, c.name) for c in row.__table__.columns
    }
    d["image_url"] = "/" + (d.get("image_path") or "")
    d["is_abstained"] = bool(d.get("is_abstained", 0))
    if d.get("probabilities_json"):
        try:
            d["probabilities"] = json.loads(d["probabilities_json"])
        except Exception:
            d["probabilities"] = {}
    if d.get("factor_breakdown_json"):
        try:
            d["factor_breakdown"] = json.loads(d["factor_breakdown_json"])
        except Exception:
            d["factor_breakdown"] = None
    # Strip internal-only fields
    for k in ("text_embedding_json", "probabilities_json", "factor_breakdown_json"):
        d.pop(k, None)
    return d


# ── Public API ────────────────────────────────────────────────────────────────

def create_incident(
    db: Session,
    *,
    description: str,
    image_path: str,
    latitude: float,
    longitude: float,
    category: str,
    urgency: int,
    people_affected: int,
    access_blocked: bool,
    reporter: str,
    # From classifier
    damage_class: str,
    model_confidence: float,
    is_abstained: bool,
    probabilities: dict,
    model_name: str,
    severity_score: float,
    # From image service
    image_hash: int | None,
    # From text embedding
    text_embedding: list | None,
) -> Incident:
    """
    Persist a new incident, run duplicate detection, record audit trail.
    Returns the saved Incident ORM row.
    """
    incident_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    # Duplicate detection against last 3 days
    recent = db.execute(
        select(
            Incident.id,
            Incident.latitude,
            Incident.longitude,
            Incident.created_at,
            Incident.image_hash,
            Incident.text_embedding_json,
        ).where(
            Incident.incident_status != "Deleted",
            Incident.created_at >= _three_days_ago(),
        )
    ).fetchall()

    duplicate_of: str | None = None
    for row in recent:
        other_hash = row.image_hash
        other_emb: list | None = None
        if row.text_embedding_json:
            try:
                other_emb = json.loads(row.text_embedding_json)
            except Exception:
                pass

        if other_hash is not None and image_hash is not None:
            dup = calculate_multimodal_duplicate_score(
                text_emb1=text_embedding,
                text_emb2=other_emb,
                hash1=image_hash,
                hash2=other_hash,
                lat1=latitude,
                lon1=longitude,
                lat2=float(row.latitude),
                lon2=float(row.longitude),
                created1=created_at.isoformat(),
                created2=row.created_at.isoformat() if hasattr(row.created_at, "isoformat") else str(row.created_at),
            )
            if dup["is_candidate"]:
                duplicate_of = row.id
                db.add(DuplicateLink(
                    id=str(uuid.uuid4()),
                    incident_a=row.id,
                    incident_b=incident_id,
                    composite_score=dup["composite_score"],
                    text_sim=dup["text_similarity"],
                    image_sim=dup["image_similarity"],
                    geo_dist_m=dup["distance_meters"],
                    time_sim=dup["time_proximity"],
                    status="Candidate",
                ))
                break

    # Priority calculation
    corroboration = 1 if duplicate_of else 0
    pri = calculate_priority(
        severity_score=severity_score,
        urgency=urgency,
        people_affected=people_affected,
        access_blocked=access_blocked,
        corroboration_count=corroboration,
    )

    incident = Incident(
        id=incident_id,
        description=description,
        image_path=image_path,
        latitude=latitude,
        longitude=longitude,
        category=category,
        reporter=reporter,
        damage_class=damage_class,
        model_confidence=model_confidence,
        is_abstained=int(is_abstained),
        probabilities_json=json.dumps(probabilities),
        model_name=model_name,
        severity_score=pri["severity_score"],
        urgency_score=pri["urgency_score"],
        impact_score=pri["impact_score"],
        access_score=pri["access_score"],
        priority_score=pri["priority_score"],
        factor_breakdown_json=json.dumps(pri["breakdown"]),
        duplicate_of=duplicate_of,
        verification_status="Pending",
        incident_status="Reported",
        assigned_to=None,
        created_at=created_at,
        updated_at=created_at,
        image_hash=image_hash,
        text_embedding_json=json.dumps(text_embedding) if text_embedding else None,
    )
    db.add(incident)

    _record_audit(
        db, incident_id,
        "Report Created",
        "None",
        f"Status: Pending, Priority: {pri['priority_score']}",
        actor=reporter or "Anonymous",
    )

    db.commit()
    db.refresh(incident)
    return incident


def _three_days_ago():
    from datetime import timedelta
    return datetime.now(timezone.utc) - timedelta(days=3)


def list_incidents(
    db: Session,
    *,
    status: str | None = None,
    category: str | None = None,
    damage_class: str | None = None,
    incident_status: str | None = None,
    search: str | None = None,
    page: int = 1,
    limit: int = 20,
) -> tuple[list[dict], int]:
    """
    Return (items, total) with full filtering, search, pagination.
    Items are serialised dicts for JSON compatibility.
    """
    q = select(Incident).where(Incident.incident_status != "Deleted")

    if status and status != "all":
        q = q.where(Incident.verification_status == status)
    if category and category != "all":
        q = q.where(Incident.category == category)
    if damage_class and damage_class != "all":
        q = q.where(Incident.damage_class == damage_class)
    if incident_status and incident_status != "all":
        q = q.where(Incident.incident_status == incident_status)
    if search:
        term = f"%{search}%"
        q = q.where(
            or_(
                Incident.description.ilike(term),
                Incident.reporter.ilike(term),
                Incident.category.ilike(term),
                Incident.id.ilike(term),
            )
        )

    total: int = db.execute(select(func.count()).select_from(q.subquery())).scalar_one()

    q = q.order_by(Incident.priority_score.desc()).offset((page - 1) * limit).limit(limit)
    rows = db.execute(q).scalars().all()
    return [_serialize(r) for r in rows], total


def get_incident(db: Session, incident_id: str) -> dict:
    """Get full incident detail including history and duplicate links."""
    row = db.get(Incident, incident_id)
    if row is None or row.incident_status == "Deleted":
        raise HTTPException(status_code=404, detail="Incident not found")

    data = _serialize(row)

    # History (audit trail)
    history_rows = db.execute(
        select(StatusHistory)
        .where(StatusHistory.incident_id == incident_id)
        .order_by(StatusHistory.created_at.desc())
    ).scalars().all()
    data["history"] = [
        {
            "id": h.id,
            "action": h.action,
            "previous_value": h.previous_value,
            "new_value": h.new_value,
            "actor": h.actor,
            "notes": h.notes,
            "created_at": h.created_at.isoformat() if hasattr(h.created_at, "isoformat") else str(h.created_at),
        }
        for h in history_rows
    ]

    # Duplicate links
    dup_rows = db.execute(
        select(DuplicateLink).where(
            or_(DuplicateLink.incident_a == incident_id, DuplicateLink.incident_b == incident_id)
        )
    ).scalars().all()
    data["duplicate_links"] = [
        {c.name: getattr(dl, c.name) for c in dl.__table__.columns}
        for dl in dup_rows
    ]

    return data


def update_incident(db: Session, incident_id: str, payload: dict) -> dict:
    """Apply partial updates; recalculate priority on label override."""
    row = db.get(Incident, incident_id)
    if row is None or row.incident_status == "Deleted":
        raise HTTPException(status_code=404, detail="Incident not found")

    actor = payload.get("actor", "Operator")
    updated = False

    if "verification_status" in payload and payload["verification_status"]:
        new_v = payload["verification_status"]
        _record_audit(db, incident_id, "Verification Updated",
                      row.verification_status, new_v, actor)
        row.verification_status = new_v
        updated = True

    if "incident_status" in payload and payload["incident_status"]:
        new_s = payload["incident_status"]
        _record_audit(db, incident_id, "Status Updated",
                      row.incident_status, new_s, actor)
        row.incident_status = new_s
        updated = True

    if "assigned_to" in payload and payload.get("assigned_to") is not None:
        _record_audit(db, incident_id, "Assignment Updated",
                      row.assigned_to or "None", payload["assigned_to"] or "None", actor)
        row.assigned_to = payload["assigned_to"]
        updated = True

    if "override_damage_class" in payload and payload["override_damage_class"]:
        new_class = payload["override_damage_class"]
        if new_class not in SEVERITY_SCORES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid damage class. Must be one of: {list(SEVERITY_SCORES.keys())}",
            )
        new_sev = SEVERITY_SCORES[new_class]
        pri = calculate_priority(
            severity_score=new_sev,
            urgency=row.urgency_score,
            people_affected=10,
            access_blocked=bool(row.access_score > 50),
        )
        _record_audit(
            db, incident_id, "Human Label Override",
            row.damage_class,
            f"{new_class} (New Priority: {pri['priority_score']})",
            actor,
        )
        row.damage_class = new_class
        row.severity_score = new_sev
        row.priority_score = pri["priority_score"]
        row.factor_breakdown_json = json.dumps(pri["breakdown"])
        row.is_abstained = 0
        updated = True

    if not updated:
        raise HTTPException(status_code=422, detail="No recognised updatable fields provided")

    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return _serialize(row)


def delete_incident(db: Session, incident_id: str, actor: str = "Operator") -> None:
    """Soft-delete: sets incident_status to 'Deleted' and logs audit."""
    row = db.get(Incident, incident_id)
    if row is None or row.incident_status == "Deleted":
        raise HTTPException(status_code=404, detail="Incident not found")

    _record_audit(db, incident_id, "Incident Deleted",
                  row.incident_status, "Deleted", actor)
    row.incident_status = "Deleted"
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
