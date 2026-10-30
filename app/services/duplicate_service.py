"""
Duplicate detection service.
Handles listing enriched duplicates and processing human decisions.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.duplicate import DuplicateLink
from app.models.incident import Incident
from app.models.audit import StatusHistory


def list_duplicates(db: Session) -> list[dict]:
    """Return all duplicate links enriched with incident metadata."""
    rows = db.execute(
        select(DuplicateLink).order_by(DuplicateLink.composite_score.desc())
    ).scalars().all()

    result = []
    for dl in rows:
        d = {c.name: getattr(dl, c.name) for c in dl.__table__.columns}

        inc_a = db.get(Incident, dl.incident_a)
        inc_b = db.get(Incident, dl.incident_b)

        if inc_a:
            d["a_desc"] = inc_a.description
            d["a_cat"] = inc_a.category
            d["a_class"] = inc_a.damage_class
            d["a_img_url"] = "/" + inc_a.image_path
            d["a_pri"] = inc_a.priority_score

        if inc_b:
            d["b_desc"] = inc_b.description
            d["b_cat"] = inc_b.category
            d["b_class"] = inc_b.damage_class
            d["b_img_url"] = "/" + inc_b.image_path
            d["b_pri"] = inc_b.priority_score

        result.append(d)

    return result


def make_decision(db: Session, link_id: str, decision: str, actor: str) -> dict:
    """Confirm or split a duplicate pair and record the decision."""
    if decision not in {"Confirmed", "Split"}:
        raise HTTPException(status_code=422, detail="decision must be 'Confirmed' or 'Split'")

    link = db.get(DuplicateLink, link_id)
    if not link:
        raise HTTPException(status_code=404, detail="Duplicate link not found")

    link.status = decision
    link.decided_by = actor
    link.decided_at = datetime.now(timezone.utc)

    if decision == "Split":
        # Remove the duplicate pointer from incident_b so it stands alone
        inc_b = db.get(Incident, link.incident_b)
        if inc_b and inc_b.duplicate_of == link.incident_a:
            inc_b.duplicate_of = None

    db.add(StatusHistory(
        id=str(uuid.uuid4()),
        incident_id=link.incident_b,
        action=f"Duplicate {decision}",
        previous_value=link.incident_a,
        new_value=decision,
        actor=actor,
        created_at=datetime.now(timezone.utc),
    ))
    db.commit()
    return {"status": "ok", "decision": decision}
