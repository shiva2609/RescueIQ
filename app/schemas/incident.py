"""Pydantic schemas for Incident CRUD endpoints – Python 3.9 compatible."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional, Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Factor breakdown sub-schemas ──────────────────────────────────────────────
class FactorDetail(BaseModel):
    raw: float
    weighted: float


class FactorBreakdown(BaseModel):
    severity: Optional[FactorDetail] = None
    urgency: Optional[FactorDetail] = None
    impact: Optional[FactorDetail] = None
    access: Optional[FactorDetail] = None
    provisional_weights_notice: Optional[str] = None


# ── Audit entry (embedded in detail responses) ────────────────────────────────
class AuditEntry(BaseModel):
    id: str
    action: str
    previous_value: Optional[str]
    new_value: Optional[str]
    actor: str
    notes: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Image sub-schema ──────────────────────────────────────────────────────────
class IncidentImageRead(BaseModel):
    id: str
    incident_id: str
    image_url: str
    mime_type: str
    file_size_bytes: int
    is_primary: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Request schemas ───────────────────────────────────────────────────────────
class IncidentUpdate(BaseModel):
    """Partial PATCH payload – all fields optional."""
    verification_status: Optional[str] = Field(
        default=None,
        description="Pending | Verified | Rejected",
    )
    incident_status: Optional[str] = Field(
        default=None,
        description="Reported | Assigned | In progress | Resolved",
    )
    assigned_to: Optional[str] = None
    override_damage_class: Optional[str] = Field(
        default=None,
        description="Human-override AI severity class; triggers priority recalculation.",
    )
    actor: str = Field(default="Operator", description="Who performed this action")

    @field_validator("verification_status")
    @classmethod
    def check_verification(cls, v: Optional[str]) -> Optional[str]:
        allowed = {"Pending", "Verified", "Rejected"}
        if v is not None and v not in allowed:
            raise ValueError(f"verification_status must be one of {sorted(allowed)}")
        return v

    @field_validator("incident_status")
    @classmethod
    def check_incident_status(cls, v: Optional[str]) -> Optional[str]:
        allowed = {"Reported", "Assigned", "In progress", "Resolved"}
        if v is not None and v not in allowed:
            raise ValueError(f"incident_status must be one of {sorted(allowed)}")
        return v

    @model_validator(mode="after")
    def at_least_one_field(self) -> "IncidentUpdate":
        updatable = {
            "verification_status", "incident_status",
            "assigned_to", "override_damage_class"
        }
        if not any(getattr(self, f) is not None for f in updatable):
            raise ValueError("At least one updatable field must be provided")
        return self


# ── Response schemas ──────────────────────────────────────────────────────────
class IncidentRead(BaseModel):
    """Flat response matching the original API shape (backward-compatible)."""
    id: str
    description: str
    image_url: str = ""
    image_path: str = ""
    latitude: float
    longitude: float
    category: str
    reporter: Optional[str]
    damage_class: str
    model_confidence: float
    is_abstained: bool
    probabilities: Optional[Dict[str, float]] = None
    model_name: str
    severity_score: float
    urgency_score: float
    impact_score: float
    access_score: float
    priority_score: float
    factor_breakdown: Optional[FactorBreakdown] = None
    duplicate_of: Optional[str]
    verification_status: str
    incident_status: str
    assigned_to: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_row(cls, row: Any) -> "IncidentRead":
        d = {c.name: getattr(row, c.name) for c in row.__table__.columns}
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
        d.pop("probabilities_json", None)
        d.pop("factor_breakdown_json", None)
        d.pop("text_embedding_json", None)
        d.pop("image_hash", None)
        return cls(**d)


class IncidentDetail(IncidentRead):
    """Extended detail view with history and duplicate links."""
    history: List[AuditEntry] = []
    duplicate_links: List[Dict[str, Any]] = []
    images: List[IncidentImageRead] = []
