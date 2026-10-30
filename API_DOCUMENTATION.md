# Damage Mapper (RescueIQ) API Reference (v2.0)

Damage Mapper provides an AI-powered disaster triage and response platform.

- **Interactive Swagger UI**: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
- **ReDoc Documentation**: [http://localhost:8000/api/redoc](http://localhost:8000/api/redoc)
- **OpenAPI JSON**: [http://localhost:8000/api/openapi.json](http://localhost:8000/api/openapi.json)

---

## Architecture Overview

```
app/
├── config.py              # Environment configuration & paths
├── database.py            # SQLAlchemy engine, sessionmaker & Base
├── dependencies.py        # DbSession dependency, pagination
├── models/                # SQLAlchemy ORM models
│   ├── incident.py        # Incident & IncidentImage
│   ├── duplicate.py       # DuplicateLink & DuplicateGroup
│   ├── audit.py           # StatusHistory (immutable audit trail)
│   ├── assignment.py     # ResponderAssignment
│   └── user.py            # User credentials & role-based access
├── schemas/               # Pydantic validation & response schemas
│   ├── incident.py        # IncidentRead, IncidentUpdate, IncidentDetail
│   ├── duplicate.py       # DuplicateLinkRead, DuplicateDecision
│   ├── audit.py           # AuditLogRead, StatsResponse, HealthResponse
│   ├── user.py            # UserCreate, UserRead, UserUpdate
│   └── common.py          # PaginatedResponse[T], SuccessResponse
├── services/              # Pure business logic layer
│   ├── incident_service.py # Incidents CRUD, deduplication, search
│   ├── duplicate_service.py# Candidate retrieval & confirm/split
│   ├── priority_service.py # Multi-factor priority weighting
│   ├── image_service.py    # Magic-byte check, resizing, SHA-256 & dHash
│   ├── geo_service.py      # PostGIS ST_DWithin + Haversine fallback
│   └── stats_service.py    # Aggregation counts & distributions
├── routers/               # HTTP controllers
│   ├── incidents.py       # Incident endpoints + nearby search
│   ├── duplicates.py      # Deduplication endpoints
│   ├── analysis.py        # AI image inference playground & stats
│   ├── assess.py          # Modular CLIP + YOLO assessment engine (v2)
│   ├── audit.py           # Audit trail logs
│   ├── users.py           # User management
│   ├── demo.py            # Idempotent demo scenario seeder
│   └── health.py          # Health check & model metadata
└── ai/                    # ML model wrappers & Assessment Engine
    ├── classifier.py      # EfficientNet-B0 vision triage model (legacy)
    ├── duplicates.py      # MiniLM semantic similarity + dHash
    └── engine/            # Modular disaster damage assessment engine (v2)
        ├── base.py        # Abstract BaseAnalyzerAdapter & AnalysisResult
        ├── preprocessing.py# MIME validation, EXIF transpose, letterbox/crop
        ├── clip_adapter.py# OpenAI CLIP zero-shot multi-prompt classifier
        ├── yolo_adapter.py# YOLOv8 object detector for hazard objects
        ├── severity.py    # Rule-based severity estimator (never confidence)
        └── engine.py      # Multi-model orchestrator & result fusion
```

---

## Endpoints

### 1. Incidents

#### `POST /api/incidents`
Submit a new citizen incident report with an image and metadata. Runs AI classification, multimodal duplicate analysis, and priority scoring.

- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `description` (str, 5-2000 chars, required)
  - `latitude` (float, -90 to 90, required)
  - `longitude` (float, -180 to 180, required)
  - `category` (str, e.g. "Flood", "Structural damage", "Road obstruction")
  - `urgency` (int, 0-100)
  - `people_affected` (int)
  - `access_blocked` (bool)
  - `reporter` (str)
  - `image` (file, JPEG/PNG/WebP, required)
- **Response**: `201 Created` with full `IncidentRead` JSON.

#### `GET /api/incidents`
List and filter incidents with search and optional pagination.
- **Query Parameters**:
  - `status`: Filter by `verification_status` (`Pending`, `Verified`, `Rejected`)
  - `category`: Filter by hazard category
  - `damage_class`: Filter by AI damage class (`Severe damage`, `Mild damage`, `Little-or-none`, `Abstained / Uncertain`)
  - `incident_status`: Filter by lifecycle status (`Reported`, `Assigned`, `In progress`, `Resolved`)
  - `search`: Full-text substring search across description, reporter, category, and ID
  - `page`: Page index (default: 1)
  - `limit`: Items per page (default: 20, max: 100)
  - `paginate`: Boolean. If true, wraps output in `{items, total, page, limit, pages}`

#### `GET /api/incidents/nearby`
PostGIS-compatible proximity search.
- **Query Parameters**:
  - `latitude`: Center point latitude (required)
  - `longitude`: Center point longitude (required)
  - `radius_meters`: Radius in meters (default: 5000, max: 500000)
  - `limit`: Max results (default: 50)
- **Response**: Array of incidents with `distance_meters` sorted ascending.

#### `GET /api/incidents/{incident_id}`
Retrieve complete incident details, including attached images, audit history trail, and duplicate candidate links.

#### `PATCH /api/incidents/{incident_id}`
Apply operational updates and human overrides.
- **JSON Body**:
  ```json
  {
    "verification_status": "Verified",
    "incident_status": "Assigned",
    "assigned_to": "Rescue Squad Delta",
    "override_damage_class": "Severe damage",
    "actor": "Commander Sarah"
  }
  ```
- **Response**: `200 OK` with updated incident.

#### `DELETE /api/incidents/{incident_id}`
Soft-deletes the incident by marking `incident_status = 'Deleted'` and recording an audit trail entry.
- **Response**: `204 No Content`.

---

### 2. Duplicates

#### `GET /api/duplicates`
Lists all pairwise duplicate candidates scored by multimodal similarity.

#### `POST /api/duplicates/{link_id}/decision`
Submit human decision to confirm or split a duplicate pair.
- **JSON Body**:
  ```json
  {
    "decision": "Confirmed",
    "actor": "Triage Officer"
  }
  ```
  *(Valid decisions: `"Confirmed"` or `"Split"`)*

---

### 3. Users

#### `POST /api/users`
Register a responder, coordinator, or citizen.
- **JSON Body**:
  ```json
  {
    "email": "sarah@rescueiq.org",
    "full_name": "Sarah Connor",
    "password": "securepassword",
    "role": "coordinator",
    "agency_affiliation": "NDRF"
  }
  ```
- **Response**: `201 Created` with `UserRead` (hashed password omitted).

#### `GET /api/users`
List active users with optional `role` filtering.

---

### 4. Operations, Audit & Analytics

#### `GET /api/stats`
Aggregated operational metrics: total counts, pending/verified/rejected, class breakdown, category counts, average priority.

#### `GET /api/audit-logs`
Immutable audit log tracking all actions, field transitions, actors, and timestamps. Supports `?limit=100` and `?incident_id=...`.

#### `POST /api/analyze/image`
Vision playground: runs inference on an uploaded image without creating an incident.

#### `GET /api/demo-seed`
Idempotently seeds 4 realistic disaster scenarios and duplicate links for training and demonstration.

#### `GET /api/health`
System status, AI model versions, and active database dialect.

---

### 5. Modular AI Disaster Damage Assessment Engine (v2)

The `v2/assess` API orchestrates zero-shot vision-language classification (CLIP) and real-time object detection (YOLOv8) behind a unified model adapter contract (`BaseAnalyzerAdapter`).

> **Architectural Guardrails**:
> 1. **Separation of Confidence vs. Severity**: AI classification confidence reflects *visual recognition probability* of a disaster scene; it is **never** used as a damage severity score. Severity is computed via a transparent, rule-based estimator (`SeverityEstimator`) using contextual hazards and report metadata.
> 2. **No Fabrication**: If models are unloaded or fail inference, an explicit `is_fallback: true` payload is returned. Predictions are never fabricated.
> 3. **Review Required Threshold**: If classification confidence falls below `0.30`, the engine automatically tags the output with `review_required: true`.
> 4. **Fine-Tuning Support**: Checkpoints for custom disaster-trained models can be supplied via `CLIP_CHECKPOINT_PATH` and `YOLO_CHECKPOINT_PATH` environment variables.

#### `POST /api/v2/assess/image`
Submit an image for multimodal disaster damage assessment.
- **Content-Type**: `multipart/form-data`
- **Form Parameters**:
  - `image`: Binary image file (JPEG, PNG, WebP)
  - `urgency_score`: Float between `0.0` and `100.0` (optional, default `0.0`)
  - `people_affected`: Integer count of people impacted (optional, default `0`)
  - `access_blocked`: Boolean flag if emergency vehicle access is impassable (optional, default `false`)
- **Response**: `200 OK`
  ```json
  {
    "category": "Flood damage",
    "confidence": 0.8421,
    "confidence_tier": "high",
    "top_candidates": [
      { "label": "Flood damage", "confidence": 0.8421 },
      { "label": "Cyclone damage", "confidence": 0.0912 },
      { "label": "Road and infrastructure obstruction", "confidence": 0.0415 }
    ],
    "detected_objects": [
      {
        "label": "boat",
        "confidence": 0.7812,
        "is_hazard_relevant": true,
        "bbox_xyxy": [120.4, 210.1, 450.0, 520.8]
      }
    ],
    "severity": {
      "level": "moderate",
      "score_0_100": 50.0,
      "rationale": "Base category 'Flood damage' maps to 'moderate' severity. No escalation factors identified.",
      "factors": [
        { "factor": "Disaster category", "value": "Flood damage", "adjustment": 2 }
      ],
      "disclaimer": "PROVISIONAL ESTIMATE ONLY. This severity level is derived from rule-based heuristics..."
    },
    "models_used": [
      "CLIP ViT-B/32 (zero-shot, multi-prompt ensemble)",
      "YOLOv8 (yolov8n.pt, COCO-80)"
    ],
    "processing_time_ms": 142.3,
    "image_metadata": { "width": 1280, "height": 720, "mode": "RGB", "aspect_ratio": 1.778 },
    "review_required": false,
    "warnings": [
      "CLIP confidence reflects category recognition only. It is NOT a damage severity score."
    ],
    "is_fallback": false,
    "separation_note": "Severity is computed by a rule-based estimator using disaster category and contextual metadata — NOT derived from AI model classification confidence."
  }
  ```

#### `GET /api/v2/assess/status`
Retrieve model adapter loading status, device placement, and engine notes.
- **Response**: `200 OK`

#### `GET /api/v2/assess/categories`
List the 7 official supported disaster categories alongside the multi-prompt ensemble dictionary:
1. `Flood damage`
2. `Building damage`
3. `Fire and smoke`
4. `Landslide`
5. `Cyclone damage`
6. `Road and infrastructure obstruction`
7. `Other / unclear`

#### `POST /api/v2/assess/severity`
Calculate transparent rule-based damage severity from category and situational metadata without requiring image upload.
- **JSON Body**:
  ```json
  {
    "category": "Fire and smoke",
    "urgency_score": 85.0,
    "people_affected": 150,
    "access_blocked": true,
    "detected_object_labels": ["person", "car"]
  }
  ```
- **Response**: `200 OK` with full `SeverityEstimate` payload and audit factor trace.

---

### 7. Intelligent Incident Prioritization Engine (`/api/v2/triage/*`)

All outputs are explicitly labeled as **PRELIMINARY DECISION-SUPPORT ASSESSMENTS**. They are designed to support human incident commanders and must never be used as the sole automated basis for life-safety actions.

#### `POST /api/v2/triage/severity`
[Module A] Estimate preliminary damage severity (`Low`, `Moderate`, `High`, `Critical`) with complete evidence trail and quantified uncertainty.
- **Rules & Guarantees**:
  - AI confidence is **never** used as a severity proxy.
  - Life-threatening categories (`Fire and smoke`) automatically baseline at $\ge \text{High}$.
  - Critical keywords (`collapse`, `trapped`, `casualt*`) elevate to `Critical`.
  - Blocked access, high urgency, and large affected population escalate severity by 1–2 levels.
  - Absence of imagery or GPS elevates the uncertainty metric (`0.0`–`1.0`).
- **Request Body**:
  ```json
  {
    "clip_category": "Building damage",
    "clip_confidence": 0.82,
    "yolo_detections": [
      { "label": "person", "confidence": 0.75 }
    ],
    "urgency_score": 75.0,
    "people_affected": 30,
    "access_blocked": true,
    "description": "Partial wall collapse with trapped occupants"
  }
  ```
- **Response**: `200 OK`
  ```json
  {
    "level": "Critical",
    "score_0_100": 95.0,
    "evidence": [
      { "source": "visual_ai", "signal": "Disaster category: Building damage", "weight": 0.3, "value": "Baseline severity index 1 → Moderate" },
      { "source": "keyword", "signal": "Critical keyword 'trapped' detected", "weight": 0.5, "value": "Keyword triggers minimum Critical" },
      { "source": "reporter", "signal": "Emergency vehicle access blocked", "weight": 0.2, "value": "Access blocked = True" }
    ],
    "uncertainty": 0.0,
    "uncertainty_reason": "Sufficient data available — uncertainty is minimal",
    "high_risk_flag": true,
    "preliminary_label": "PRELIMINARY DECISION-SUPPORT ASSESSMENT..."
  }
  ```

#### `POST /api/v2/triage/severity/incident/{id}`
[Module A] Run severity estimation on an existing stored incident by ID, reading persisted metadata and probabilities.

#### `POST /api/v2/triage/duplicates/scan`
[Module B] Pairwise multimodal duplicate detection across stored incidents.
- **Safety Invariant**: Reports are **never automatically deleted or merged**.
- **Composite Similarity Formula**:
  $$S_{\text{dup}} = 0.30 \cdot S_{\text{dHash}} + 0.20 \cdot S_{\text{image}} + 0.20 \cdot S_{\text{text}} + 0.15 \cdot S_{\text{geo}} + 0.15 \cdot S_{\text{time}}$$
- **Query Parameter**: `limit` (int, 10–1000, default: 200).
- **Response**: `200 OK` with candidate pairs, similarity breakdown, review clusters, and persisted `DuplicateLink` records.

#### `GET /api/v2/triage/duplicates`
[Module B] List duplicate candidate links with enriched incident descriptions and similarity breakdowns.
- **Query Parameter**: `status` (`Pending`, `Confirmed`, `Split`).

#### `POST /api/v2/triage/duplicates/{link_id}/decision`
[Module B] Record human operator decision (`Confirmed` or `Split`) on a duplicate candidate pair.

#### `POST /api/v2/triage/priority`
[Module C] Calculate configurable 0–100 priority score with transparent multi-factor breakdown and explainability.
- **Transparent Formula**:
  $$P = w_{\text{severity}} \cdot S + w_{\text{urgency}} \cdot U + w_{\text{impact}} \cdot I + w_{\text{access}} \cdot A$$
  - Default weights: $w_{\text{severity}} = 0.40$, $w_{\text{urgency}} = 0.25$, $w_{\text{impact}} = 0.20$, $w_{\text{access}} = 0.15$ ($\sum w = 1.0$).
  - Population impact ($I$) uses logarithmic scaling: $\min(100, 20 \cdot \ln(1 + \text{people}) + 5 \cdot \text{corroboration})$.
  - High-risk flag is raised if score $\ge 75$, severity is `Critical`, or when model confidence is low ($< 0.40$) but urgency/access-blockage signals danger.
- **Request Body**:
  ```json
  {
    "severity_level": "High",
    "urgency_score": 70.0,
    "people_affected": 25,
    "access_blocked": true,
    "corroboration_count": 1,
    "model_confidence": 0.85
  }
  ```
- **Response**: `200 OK`
  ```json
  {
    "priority_score": 74.6,
    "priority_band": "High",
    "factors": [
      {
        "name": "Damage Severity",
        "raw_score": 74.0,
        "weight": 0.4,
        "weighted_contribution": 29.6,
        "explanation": "Preliminary severity level is 'High' (score=74/100)...",
        "evidence": ["Severity level: High", "Numeric representation: 74"]
      },
      {
        "name": "Reporter Urgency",
        "raw_score": 70.0,
        "weight": 0.25,
        "weighted_contribution": 17.5,
        "explanation": "Reporter-supplied urgency: 70/100 — High urgency...",
        "evidence": ["Raw urgency input: 70"]
      },
      {
        "name": "Population Impact",
        "raw_score": 62.4,
        "weight": 0.2,
        "weighted_contribution": 12.48,
        "explanation": "Impact score 62/100 computed using logarithmic scale...",
        "evidence": ["People affected: 25", "Corroborating reports: 1"]
      },
      {
        "name": "Accessibility",
        "raw_score": 100.0,
        "weight": 0.15,
        "weighted_contribution": 15.0,
        "explanation": "Emergency vehicle access is BLOCKED...",
        "evidence": ["Access blocked: True"]
      }
    ],
    "weight_config": {
      "weight_severity": 0.4,
      "weight_urgency": 0.25,
      "weight_impact": 0.2,
      "weight_access": 0.15
    },
    "high_risk_flag": true,
    "flag_reason": "Access blocked AND high urgency reported",
    "preliminary_label": "PRELIMINARY DECISION-SUPPORT ASSESSMENT..."
  }
  ```

#### `POST /api/v2/triage/priority/incident/{id}`
[Module C] Re-score an existing stored incident and update its persisted `priority_score`.

#### `GET /api/v2/triage/priority/weights`
[Module C] Retrieve current active priority weight configuration, mathematical formula, and descriptions.

#### `PUT /api/v2/triage/priority/weights`
[Module C] Administrator weight update endpoint.
- Validates that weights sum to exactly `1.0` (tolerance `0.001`).
- Records change in in-memory audit history with timestamp and actor.
- **Request Body**:
  ```json
  {
    "weight_severity": 0.35,
    "weight_urgency": 0.30,
    "weight_impact": 0.20,
    "weight_access": 0.15,
    "actor": "emergency_coordinator"
  }
  ```

#### `GET /api/v2/triage/priority/weights/history`
[Module C] Audit trail of all weight adjustments made during the session.

#### `GET /api/v2/triage/queue`
Ranked incident queue sorted by priority score descending.
- **Query Parameters**:
  - `min_score`: Minimum priority score threshold (`0.0`–`100.0`)
  - `band`: Filter by band (`Critical`, `High`, `Moderate`, `Low`)
  - `high_risk_only`: Boolean flag
  - `limit`: Items per page (default: 50)
  - `offset`: Pagination offset (default: 0)

