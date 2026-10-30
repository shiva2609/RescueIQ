"""
Triage router — /api/v2/triage/*

Three independent prioritisation modules exposed as REST endpoints:

  Module A — Severity Estimation:
    POST /api/v2/triage/severity           — estimate severity from metadata + AI outputs
    POST /api/v2/triage/severity/incident/{id}  — estimate for a stored incident

  Module B — Duplicate Detection:
    POST /api/v2/triage/duplicates/scan    — pairwise scan of stored incidents
    GET  /api/v2/triage/duplicates         — list existing duplicate links (enriched)
    POST /api/v2/triage/duplicates/{link_id}/decision — human decision (Confirmed/Split)

  Module C — Response Prioritisation:
    POST /api/v2/triage/priority           — score from metadata
    POST /api/v2/triage/priority/incident/{id}  — score a stored incident
    GET  /api/v2/triage/priority/weights   — current weight configuration
    PUT  /api/v2/triage/priority/weights   — update weights (admin)
    GET  /api/v2/triage/priority/weights/history — weight change audit log

  Utility:
    GET  /api/v2/triage/queue             — ranked prioritised incident queue

All outputs carry PRELIMINARY DECISION-SUPPORT ASSESSMENT labels.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.incident import Incident
from app.models.duplicate import DuplicateLink
from app.models.audit import StatusHistory
from app.services.duplicate_service import list_duplicates, make_decision

from app.triage import (
    severity_engine,
    duplicate_engine,
    priority_engine,
    weight_registry,
    PriorityWeightConfig,
    PRELIMINARY_LABEL,
)

router = APIRouter(prefix="/api/v2/triage", tags=["Triage Engine"])


# ════════════════════════════════════════════════════════════════════════════════
# Shared schemas
# ════════════════════════════════════════════════════════════════════════════════

class SeverityRequest(BaseModel):
    clip_category: str = Field(default="Other / unclear", description="Disaster category from CLIP adapter")
    clip_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="CLIP classification confidence (0-1)")
    yolo_detections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description='List of {"label": str, "confidence": float} from YOLO'
    )
    urgency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    people_affected: int = Field(default=0, ge=0)
    access_blocked: bool = Field(default=False)
    description: str = Field(default="")
    has_image: bool = Field(default=True)
    has_gps: bool = Field(default=True)


class PriorityRequest(BaseModel):
    severity_level: str = Field(
        default="Low",
        description="Module A output: Low | Moderate | High | Critical",
    )
    urgency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    people_affected: int = Field(default=0, ge=0)
    access_blocked: bool = Field(default=False)
    corroboration_count: int = Field(default=0, ge=0, description="Number of duplicate reports")
    model_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="AI classification confidence")


class WeightUpdateRequest(BaseModel):
    weight_severity: float = Field(..., ge=0.0, le=1.0)
    weight_urgency: float = Field(..., ge=0.0, le=1.0)
    weight_impact: float = Field(..., ge=0.0, le=1.0)
    weight_access: float = Field(..., ge=0.0, le=1.0)
    actor: str = Field(default="admin", description="Identity of administrator making the change")


class DuplicateDecisionRequest(BaseModel):
    decision: str = Field(..., description="'Confirmed' or 'Split'")
    actor: str = Field(default="reviewer")


# ════════════════════════════════════════════════════════════════════════════════
# Module A — Severity
# ════════════════════════════════════════════════════════════════════════════════

@router.post(
    "/severity",
    response_model=Dict[str, Any],
    summary="[Module A] Estimate preliminary severity from AI outputs and contextual metadata",
    description=(
        "Returns Low / Moderate / High / Critical with full evidence trail and uncertainty score. "
        "**AI confidence is NEVER used as severity proxy.** "
        "All outputs are PRELIMINARY DECISION-SUPPORT ASSESSMENTS requiring human verification."
    ),
)
def estimate_severity(payload: SeverityRequest):
    assessment = severity_engine.assess(
        clip_category=payload.clip_category,
        clip_confidence=payload.clip_confidence,
        yolo_detections=payload.yolo_detections,
        urgency_score=payload.urgency_score,
        people_affected=payload.people_affected,
        access_blocked=payload.access_blocked,
        description=payload.description,
        has_image=payload.has_image,
        has_gps=payload.has_gps,
    )
    return assessment.to_dict()


@router.post(
    "/severity/incident/{incident_id}",
    response_model=Dict[str, Any],
    summary="[Module A] Estimate severity for a stored incident by ID",
)
def estimate_severity_for_incident(incident_id: str, db: Session = Depends(get_db)):
    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id!r} not found")

    import json as _json
    yolo_dets: List[Dict[str, Any]] = []
    # Use stored YOLO detections if available via probabilities_json
    if incident.probabilities_json:
        try:
            probs = _json.loads(incident.probabilities_json)
            if isinstance(probs, list):
                yolo_dets = probs
        except Exception:
            pass

    assessment = severity_engine.assess(
        clip_category=incident.category or "Other / unclear",
        clip_confidence=float(incident.model_confidence or 0.0),
        yolo_detections=yolo_dets,
        urgency_score=float(incident.urgency_score or 0.0),
        people_affected=0,  # not stored on model — use 0
        access_blocked=bool(incident.access_score and incident.access_score >= 80),
        description=incident.description or "",
        has_image=bool(incident.image_path),
        has_gps=bool(incident.latitude and incident.longitude),
    )
    return {
        "incident_id": incident_id,
        **assessment.to_dict(),
    }


# ════════════════════════════════════════════════════════════════════════════════
# Module B — Duplicate Detection
# ════════════════════════════════════════════════════════════════════════════════

@router.post(
    "/duplicates/scan",
    response_model=Dict[str, Any],
    summary="[Module B] Scan stored incidents for duplicate candidates",
    description=(
        "Runs pairwise duplicate detection over the most recent N incidents. "
        "Candidates are grouped into review clusters and persisted as DuplicateLink records. "
        "**Reports are NEVER automatically deleted or merged.**"
    ),
)
def scan_duplicates(
    limit: int = Query(default=200, ge=10, le=1000,
                       description="Number of recent incidents to include in scan"),
    db: Session = Depends(get_db),
):
    # Fetch recent incidents
    rows = db.execute(
        select(Incident).order_by(Incident.created_at.desc()).limit(limit)
    ).scalars().all()

    if len(rows) < 2:
        return {
            "message": "Fewer than 2 incidents found — nothing to compare",
            "candidates_found": 0,
            "groups_found": 0,
            "preliminary_label": PRELIMINARY_LABEL,
        }

    incidents_data = [
        {
            "id": inc.id,
            "image_hash": inc.image_hash,
            "image_embedding_json": None,          # CLIP embeddings stored separately if added later
            "text_embedding_json": inc.text_embedding_json,
            "description": inc.description,
            "latitude": inc.latitude,
            "longitude": inc.longitude,
            "created_at": inc.created_at,
            "priority_score": inc.priority_score,
        }
        for inc in rows
    ]

    candidates, groups = duplicate_engine.detect_all(incidents_data)

    # Persist new candidate pairs to DuplicateLink table (upsert by pair)
    existing_pairs = {
        (dl.incident_a, dl.incident_b)
        for dl in db.execute(select(DuplicateLink)).scalars().all()
    }

    new_links = 0
    for cand in candidates:
        pair = (cand.incident_id_a, cand.incident_id_b)
        rev_pair = (cand.incident_id_b, cand.incident_id_a)
        if pair not in existing_pairs and rev_pair not in existing_pairs:
            link = DuplicateLink(
                id=str(uuid.uuid4()),
                incident_a=cand.incident_id_a,
                incident_b=cand.incident_id_b,
                composite_score=cand.similarity.composite_score,
                hash_sim=cand.similarity.image_hash_similarity,
                image_sim=cand.similarity.image_embedding_similarity,
                text_sim=cand.similarity.text_similarity,
                geo_dist_m=cand.similarity.geo_proximity_score,
                time_sim=cand.similarity.time_proximity_score,
                recommendation=cand.recommendation,
                status="Pending",
                detected_at=datetime.now(timezone.utc),
            )
            db.add(link)
            new_links += 1

    db.commit()

    return {
        "incidents_scanned": len(rows),
        "candidates_found": len(candidates),
        "new_links_created": new_links,
        "groups_found": len(groups),
        "candidates": [c.to_dict() for c in candidates[:50]],   # cap response size
        "groups": [g.to_dict() for g in groups],
        "preliminary_label": PRELIMINARY_LABEL,
        "important_notice": (
            "No incidents have been modified, merged, or deleted. "
            "All groupings are suggestions for human review only."
        ),
    }


@router.get(
    "/duplicates",
    response_model=Dict[str, Any],
    summary="[Module B] List existing duplicate candidate links with incident metadata",
)
def get_duplicate_links(
    status: Optional[str] = Query(
        default=None, description="Filter by status: Pending | Confirmed | Split"
    ),
    db: Session = Depends(get_db),
):
    all_links = list_duplicates(db)
    if status:
        all_links = [lk for lk in all_links if lk.get("status") == status]
    return {
        "count": len(all_links),
        "links": all_links,
        "preliminary_label": PRELIMINARY_LABEL,
    }


@router.post(
    "/duplicates/{link_id}/decision",
    response_model=Dict[str, Any],
    summary="[Module B] Record human decision on a duplicate candidate pair",
)
def decide_duplicate(
    link_id: str,
    payload: DuplicateDecisionRequest,
    db: Session = Depends(get_db),
):
    return make_decision(db, link_id, payload.decision, payload.actor)


# ════════════════════════════════════════════════════════════════════════════════
# Module C — Priority Scoring
# ════════════════════════════════════════════════════════════════════════════════

@router.post(
    "/priority",
    response_model=Dict[str, Any],
    summary="[Module C] Calculate priority score from contextual metadata",
    description=(
        "Returns a 0–100 priority score with per-factor breakdown and weight configuration. "
        "Score is **not** a validated emergency triage score. "
        "All outputs are PRELIMINARY DECISION-SUPPORT ASSESSMENTS."
    ),
)
def calculate_priority(payload: PriorityRequest):
    score = priority_engine.calculate(
        severity_level=payload.severity_level,
        urgency_score=payload.urgency_score,
        people_affected=payload.people_affected,
        access_blocked=payload.access_blocked,
        corroboration_count=payload.corroboration_count,
        model_confidence=payload.model_confidence,
    )
    return score.to_dict()


@router.post(
    "/priority/incident/{incident_id}",
    response_model=Dict[str, Any],
    summary="[Module C] Re-score a stored incident using current weights",
)
def score_incident(incident_id: str, db: Session = Depends(get_db)):
    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id!r} not found")

    # Derive severity level from stored severity_score
    sev_score = float(incident.severity_score or 0.0)
    if sev_score >= 90:
        sev_level = "Critical"
    elif sev_score >= 65:
        sev_level = "High"
    elif sev_score >= 35:
        sev_level = "Moderate"
    else:
        sev_level = "Low"

    # Count confirmed duplicate links for corroboration
    corrob_count = db.execute(
        select(DuplicateLink).where(
            (
                (DuplicateLink.incident_a == incident_id) |
                (DuplicateLink.incident_b == incident_id)
            ) & (DuplicateLink.status == "Confirmed")
        )
    ).scalars().all()

    score = priority_engine.calculate(
        severity_level=sev_level,
        urgency_score=float(incident.urgency_score or 0.0),
        people_affected=0,
        access_blocked=bool(incident.access_score and incident.access_score >= 80),
        corroboration_count=len(corrob_count),
        model_confidence=float(incident.model_confidence or 0.0),
    )

    # Persist updated score
    incident.priority_score = score.priority_score
    incident.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {
        "incident_id": incident_id,
        "stored_priority_updated": True,
        **score.to_dict(),
    }


@router.get(
    "/priority/weights",
    response_model=Dict[str, Any],
    summary="[Module C] Get current priority weight configuration",
)
def get_weights():
    cfg = weight_registry.get()
    return {
        "weights": cfg.to_dict(),
        "description": {
            "weight_severity": "Contribution of Module A severity level to priority score",
            "weight_urgency": "Contribution of reporter-supplied urgency",
            "weight_impact": "Contribution of population exposure (logarithmic scale)",
            "weight_access": "Contribution of emergency vehicle accessibility",
        },
        "formula": "P = w_severity·S + w_urgency·U + w_impact·I + w_access·A  (all normalised 0-100)",
        "notice": "Provisional rule-based weights. Not scientifically validated.",
        "preliminary_label": PRELIMINARY_LABEL,
    }


@router.put(
    "/priority/weights",
    response_model=Dict[str, Any],
    summary="[Module C] Update priority weights (administrator only)",
    description=(
        "Weights must sum to exactly 1.0. Changes take effect immediately for new score calculations. "
        "Previously stored incident scores are NOT retroactively updated. "
        "Use POST /priority/incident/{id} to re-score individual incidents with new weights."
    ),
)
def update_weights(payload: WeightUpdateRequest):
    try:
        new_cfg = weight_registry.update(
            weight_severity=payload.weight_severity,
            weight_urgency=payload.weight_urgency,
            weight_impact=payload.weight_impact,
            weight_access=payload.weight_access,
            actor=payload.actor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "status": "updated",
        "new_weights": new_cfg.to_dict(),
        "updated_by": payload.actor,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "notice": (
            "New weights apply to all future priority calculations. "
            "Existing stored priority_score values are NOT automatically recalculated."
        ),
    }


@router.get(
    "/priority/weights/history",
    response_model=Dict[str, Any],
    summary="[Module C] Audit log of weight configuration changes",
)
def get_weight_history():
    return {
        "history": weight_registry.history(),
        "count": len(weight_registry.history()),
        "note": "History is in-memory and resets on server restart.",
    }


# ════════════════════════════════════════════════════════════════════════════════
# Utility — Prioritised Incident Queue
# ════════════════════════════════════════════════════════════════════════════════

@router.get(
    "/queue",
    response_model=Dict[str, Any],
    summary="Ranked incident queue sorted by stored priority score",
    description=(
        "Returns stored incidents ordered by priority_score descending. "
        "Optionally filtered by minimum priority band or high_risk_flag. "
        "Priority scores are preliminary and must be verified by field personnel."
    ),
)
def get_priority_queue(
    min_score: float = Query(default=0.0, ge=0.0, le=100.0, description="Minimum priority score"),
    band: Optional[str] = Query(
        default=None, description="Filter by band: Critical | High | Moderate | Low"
    ),
    high_risk_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    from sqlalchemy import desc as _desc

    q = select(Incident).where(
        Incident.priority_score >= min_score
    ).order_by(_desc(Incident.priority_score))

    all_rows = db.execute(q).scalars().all()

    # Apply band filter using current engine (band derived from stored score)
    def row_band(inc: Incident) -> str:
        s = inc.priority_score
        if s >= 75:
            return "Critical"
        elif s >= 50:
            return "High"
        elif s >= 25:
            return "Moderate"
        return "Low"

    if band:
        all_rows = [r for r in all_rows if row_band(r) == band]

    total = len(all_rows)
    page = all_rows[offset: offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "incidents": [
            {
                "id": inc.id,
                "description": inc.description[:120],
                "category": inc.category,
                "damage_class": inc.damage_class,
                "priority_score": inc.priority_score,
                "priority_band": row_band(inc),
                "severity_score": inc.severity_score,
                "verification_status": inc.verification_status,
                "incident_status": inc.incident_status,
                "assigned_to": inc.assigned_to,
                "created_at": inc.created_at.isoformat() if inc.created_at else None,
                "latitude": inc.latitude,
                "longitude": inc.longitude,
            }
            for inc in page
        ],
        "preliminary_label": PRELIMINARY_LABEL,
    }
