"""Stats aggregation service."""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.duplicate import DuplicateLink
from app.models.incident import Incident


def get_stats(db: Session) -> dict:
    def count(condition=None):
        q = select(func.count()).select_from(Incident)
        if condition is not None:
            q = q.where(condition)
        else:
            q = q.where(Incident.incident_status != "Deleted")
        return db.execute(q).scalar_one()

    total = count(Incident.incident_status != "Deleted")
    pending = count((Incident.verification_status == "Pending") & (Incident.incident_status != "Deleted"))
    verified = count((Incident.verification_status == "Verified") & (Incident.incident_status != "Deleted"))
    rejected = count((Incident.verification_status == "Rejected") & (Incident.incident_status != "Deleted"))
    severe = count((Incident.damage_class == "Severe damage") & (Incident.incident_status != "Deleted"))
    mild = count((Incident.damage_class == "Mild damage") & (Incident.incident_status != "Deleted"))
    little = count((Incident.damage_class == "Little-or-none") & (Incident.incident_status != "Deleted"))
    abstained = count((Incident.is_abstained == 1) & (Incident.incident_status != "Deleted"))
    duplicates = db.execute(
        select(func.count()).select_from(DuplicateLink)
        .where(DuplicateLink.status == "Candidate")
    ).scalar_one()
    assigned = count(
        (Incident.assigned_to.isnot(None)) &
        (Incident.assigned_to != "") &
        (Incident.incident_status != "Deleted")
    )
    resolved = count(
        (Incident.incident_status == "Resolved") & (Incident.incident_status != "Deleted")
    )

    avg_row = db.execute(
        select(func.avg(Incident.priority_score)).where(Incident.incident_status != "Deleted")
    ).scalar_one()
    avg_priority = round(avg_row or 0.0, 1)

    # Category breakdown
    cat_rows = db.execute(
        select(Incident.category, func.count().label("c"))
        .where(Incident.incident_status != "Deleted")
        .group_by(Incident.category)
    ).all()
    by_category = {r.category: r.c for r in cat_rows}

    # Class breakdown
    cls_rows = db.execute(
        select(Incident.damage_class, func.count().label("c"))
        .where(Incident.incident_status != "Deleted")
        .group_by(Incident.damage_class)
    ).all()
    by_class = {r.damage_class: r.c for r in cls_rows}

    return {
        "total": total,
        "pending": pending,
        "verified": verified,
        "rejected": rejected,
        "severe": severe,
        "mild": mild,
        "little": little,
        "abstained": abstained,
        "duplicates": duplicates,
        "assigned": assigned,
        "resolved": resolved,
        "avg_priority": avg_priority,
        "by_category": by_category,
        "by_class": by_class,
    }
