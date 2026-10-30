"""Pydantic schemas for duplicate links – Python 3.9 compatible."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class DuplicateLinkRead(BaseModel):
    id: str
    incident_a: str
    incident_b: str
    composite_score: float
    text_sim: float
    image_sim: float
    geo_dist_m: float
    time_sim: float
    status: str
    decided_by: Optional[str]
    decided_at: Optional[datetime]
    created_at: datetime

    # Joined fields from incidents (populated by service layer)
    a_desc: Optional[str] = None
    a_cat: Optional[str] = None
    a_class: Optional[str] = None
    a_img_url: Optional[str] = None
    a_pri: Optional[float] = None
    b_desc: Optional[str] = None
    b_cat: Optional[str] = None
    b_class: Optional[str] = None
    b_img_url: Optional[str] = None
    b_pri: Optional[float] = None

    model_config = {"from_attributes": True}


class DuplicateDecision(BaseModel):
    decision: str = Field(
        ...,
        description="Confirmed – treat as same incident; Split – treat as independent",
    )
    actor: str = Field(default="Operator")

    def validate_decision(self) -> None:
        if self.decision not in {"Confirmed", "Split"}:
            raise ValueError("decision must be 'Confirmed' or 'Split'")
