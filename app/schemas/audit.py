"""Pydantic schemas for audit logs and stats – Python 3.9 compatible."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Dict
from pydantic import BaseModel


class AuditLogRead(BaseModel):
    id: str
    incident_id: str
    action: str
    previous_value: Optional[str]
    new_value: Optional[str]
    actor: str
    notes: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class StatsResponse(BaseModel):
    total: int
    pending: int
    verified: int
    rejected: int
    severe: int
    mild: int
    little: int
    abstained: int
    duplicates: int
    assigned: int
    resolved: int
    avg_priority: float
    by_category: Dict[str, int]
    by_class: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    vision_model: str
    nlp_model: str
    deduplication: str
    priority_formula: str
    db_backend: str


class AnalysisResponse(BaseModel):
    prediction: Dict[str, Any]
    dhash: Optional[int]
    preview_url: str
