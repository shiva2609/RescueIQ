"""Demo seeder router – loads realistic disaster scenarios for testing."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.config import settings
from app.dependencies import DbSession
from app.ai.classifier import SEVERITY_SCORES
from app.models.incident import Incident
from app.models.duplicate import DuplicateLink
from app.services.priority_service import calculate_priority

router = APIRouter(tags=["Demo"])

_SAMPLES = [
    {
        "id": "demo-rescue-1",
        "desc": "Severe street inundation; residential ground floors submerged up to 4 feet. Elderly residents trapped inside.",
        "lat": 12.9716, "lon": 77.5946, "cat": "Flood", "cls": "Severe damage",
        "conf": 0.92, "urg": 90, "people": 85, "blocked": True,
    },
    {
        "id": "demo-rescue-2",
        "desc": "Water level rising rapidly near central market crossroad; vehicles unable to move, power lines down.",
        "lat": 12.9725, "lon": 77.5958, "cat": "Flood", "cls": "Severe damage",
        "conf": 0.88, "urg": 85, "people": 40, "blocked": True,
    },
    {
        "id": "demo-rescue-3",
        "desc": "Fallen tree blocking two lanes after storm winds; traffic diverted, no residential flooding.",
        "lat": 12.9650, "lon": 77.5900, "cat": "Road obstruction", "cls": "Mild damage",
        "conf": 0.84, "urg": 45, "people": 10, "blocked": True,
    },
    {
        "id": "demo-rescue-4",
        "desc": "Minor puddle accumulation on pedestrian pavement; drainage is slowly flowing. No structural impact.",
        "lat": 12.9600, "lon": 77.6050, "cat": "Flood", "cls": "Little-or-none",
        "conf": 0.79, "urg": 20, "people": 0, "blocked": False,
    },
]


@router.get(
    "/api/demo-seed",
    response_model=dict,
    summary="Seed realistic flood/hazard demo incidents (idempotent)",
)
def seed_demo(db: DbSession):
    demo_img = settings.UPLOAD_DIR / "demo-scenario.png"
    if not demo_img.exists():
        PILImage.new("RGB", (640, 360), (70, 110, 150)).save(demo_img)

    created = []
    for s in _SAMPLES:
        if db.get(Incident, s["id"]):
            continue
        sev = SEVERITY_SCORES[s["cls"]]
        pri = calculate_priority(
            severity_score=sev,
            urgency=s["urg"],
            people_affected=s["people"],
            access_blocked=s["blocked"],
        )
        now = datetime.now(timezone.utc)
        inc = Incident(
            id=s["id"],
            description=s["desc"],
            image_path="uploads/demo-scenario.png",
            latitude=s["lat"],
            longitude=s["lon"],
            category=s["cat"],
            reporter="Demo Scenario",
            damage_class=s["cls"],
            model_confidence=s["conf"],
            is_abstained=0,
            probabilities_json=json.dumps({s["cls"]: s["conf"]}),
            model_name="EfficientNet-B0 (3-class triage)",
            severity_score=pri["severity_score"],
            urgency_score=pri["urgency_score"],
            impact_score=pri["impact_score"],
            access_score=pri["access_score"],
            priority_score=pri["priority_score"],
            factor_breakdown_json=json.dumps(pri["breakdown"]),
            duplicate_of=None,
            verification_status="Verified",
            incident_status="Reported",
            assigned_to=None,
            created_at=now,
            updated_at=now,
            image_hash=12345,
            text_embedding_json=None,
        )
        db.add(inc)
        created.append(s["id"])

    # Simulate one duplicate link between demo 1 and 2
    if not db.get(DuplicateLink, "demo-link-1-2") and len(created) >= 2:
        db.add(DuplicateLink(
            id="demo-link-1-2",
            incident_a="demo-rescue-1",
            incident_b="demo-rescue-2",
            composite_score=78.5,
            text_sim=74.2,
            image_sim=85.0,
            geo_dist_m=142.0,
            time_sim=95.0,
            status="Candidate",
        ))
        inc2 = db.get(Incident, "demo-rescue-2")
        if inc2:
            inc2.duplicate_of = "demo-rescue-1"

    db.commit()
    return {
        "created": created,
        "message": f"Seeded {len(created)} realistic disaster incidents and 1 multimodal duplicate candidate.",
    }
