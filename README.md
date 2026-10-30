# RescueIQ (Damage Mapper)

> **AI-Powered Disaster Damage Assessment, Incident Prioritization & Response Platform**  
> *Full-Stack Platform: Modular FastAPI Backend (v2.0), Pluggable Vision-Language Assessment Engine (CLIP + YOLOv8), and Professional Responsive Command Center*

[![Tests](https://img.shields.io/badge/tests-124%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## 🌟 Overview

RescueIQ converts raw, unstructured field and citizen disaster reports (photographs, descriptions, GPS coordinates, situational metadata, and route access constraints) into an explainable, map-based, prioritized operational pipeline for emergency coordinators, responders, and disaster management agencies.

---

## 🚀 Key Capabilities

### 1. Modular AI Disaster Damage Assessment Engine (v2)
- **CLIP Zero-Shot Multi-Prompt Classification**: Pretrained OpenAI CLIP ViT with curated multi-prompt ensembles for 7 standard disaster categories:
  - *Flood damage*, *Building damage*, *Fire and smoke*, *Landslide*, *Cyclone damage*, *Road & infrastructure obstruction*, and *Other / unclear*.
- **YOLOv8 Visible Hazard Identification**: Real-time object detection recognizing trapped individuals, displaced vehicles, rescue boats, aircraft, and hazardous street debris with bounding box coordinates.
- **Strict Separation of Confidence vs. Damage Severity**:
  - *Confidence* strictly measures visual recognition probability of a category.
  - *Severity* is evaluated independently via a transparent, rule-based estimator (`SeverityEstimator`) using disaster doctrine heuristics, hazard presence, reporter urgency, and route blockage.
- **Low-Confidence Review Gates & Fallback Safety**: Inference scores below 30% flag reports for human triage (`review_required: true`). If models are unavailable or uninitialized, explicit fallback payloads are returned without fabricating predictions.
- **Fine-Tuning Support**: Checkpoints can be plugged in via `CLIP_CHECKPOINT_PATH` and `YOLO_CHECKPOINT_PATH` without changing API routes. Detailed dataset guidance is documented in [`MODEL_LIMITATIONS.md`](file:///Users/vkshivakumar/Downloads/damage-mapper/MODEL_LIMITATIONS.md).

### 2. Intelligent Incident Prioritization & Triage Engine (v2)
- **Module A — Preliminary Severity Estimation**:
  - Combines visual indicators (CLIP + YOLO), reporter urgency, affected population, access status, and critical keyword triggers (`collapse`, `trapped`, `casualt*`).
  - Returns `Low`, `Moderate`, `High`, or `Critical` with comprehensive evidence itemization and quantified uncertainty score (`0.0`–`1.0`).
  - Guarantees AI confidence is **never** used as a severity proxy.
- **Module B — Multimodal Duplicate Detection**:
  - Merges 64-bit perceptual difference hashing (`dHash`), text embeddings (`all-MiniLM-L6-v2`), spatial Haversine proximity, and temporal decay.
  - Pairwise scanning clusters incident reports into candidate groups for coordinator review.
  - **Zero automated deletion**: Reports are strictly grouped for human confirm/split decisions.
- **Module C — Response Prioritization & Dynamic Weights**:
  - Computes explainable 0–100 priority score: $P = w_{\text{severity}} \cdot S + w_{\text{urgency}} \cdot U + w_{\text{impact}} \cdot I + w_{\text{access}} \cdot A$.
  - Provides mathematical contribution breakdown and plain-language explanation for every factor.
  - Live administrator weight reconfiguration API (`PUT /api/v2/triage/priority/weights`) with full in-memory audit trail.
  - Automatic safety flags: raises `high_risk_flag` on high urgency or blocked access even when model confidence is low.
- **Universal Safety Guardrail**: Every triage output carries an explicit, non-removable `PRELIMINARY DECISION-SUPPORT ASSESSMENT` advisory.

### 3. Enterprise Backend Architecture
- Organized into clean layers: `models/`, `schemas/`, `services/`, `routers/`, `ai/engine/`, and `triage/`.
- Dual database support: **SQLite** (with WAL mode & foreign key enforcement for local dev) and **PostgreSQL / PostGIS** (spatial proximity querying using `ST_DWithin`).
- Database migrations managed via **Alembic**.
- Immutable audit log (`StatusHistory`) capturing all state transitions, human damage class overrides, assignments, and actors.

### 4. Professional Responsive Operations Dashboard
- High-contrast, accessibility-tested navy & alert color palette.
- **Landing Page**: Platform capability overview and statistics.
- **Citizen Report Portal**: Drag-and-drop image upload, real-time magic-byte verification, and one-tap GPS locator.
- **Operations Dashboard**: Live KPI statistics, priority-ranked queue, filter controls, and responder dispatching.
- **Geospatial Map**: Leaflet cluster map with coordinate picker, severity radius rendering, and marker clusters.
- **Deduplication Review**: Side-by-side comparison modal with confirm/split actions.
- **AI Analysis Lab**: Multi-model vision inspection playground testing CLIP and YOLO inference on arbitrary images.
- **Admin Management**: User registration, priority weight governance, role assignments, and immutable audit trail viewer.

---

## 🛠️ Project Structure

```
├── API_DOCUMENTATION.md      # Complete REST API reference (v2.0)
├── MODEL_LIMITATIONS.md      # AI safety, failure modes & fine-tuning datasets
├── RescueIQ_PRD.md           # Product Requirements Document
├── alembic.ini               # Database migration configuration
├── requirements.txt          # Python dependencies
├── app/
│   ├── main.py               # FastAPI application factory & router registration
│   ├── config.py             # App settings, thresholds & paths
│   ├── database.py           # Engine, sessionmaker & Base definition
│   ├── dependencies.py       # Dependency injection (sessions, pagination)
│   ├── models/               # SQLAlchemy ORM models (Incident, Duplicate, Audit, User)
│   ├── schemas/              # Pydantic v2 validation and serialization schemas
│   ├── services/             # Pure business logic (Incident, Duplicate, Geo, Priority)
│   ├── routers/              # HTTP API controllers (/incidents, /triage, /assess, etc.)
│   ├── triage/               # Intelligent Prioritization Engine (Modules A, B, C)
│   │   ├── severity_engine.py# Module A: Preliminary Severity & Uncertainty
│   │   ├── duplicate_engine.py# Module B: Multimodal Deduplication & Clustering
│   │   ├── priority_engine.py# Module C: 0-100 Scoring & Weight Registry
│   │   └── models.py         # Triage dataclasses & preliminary constants
│   └── ai/
│       ├── classifier.py     # EfficientNet-B0 triage classifier
│       ├── duplicates.py     # Text embeddings & dHash comparison
│       └── engine/           # Modular AI Damage Assessment Engine (v2)
│           ├── base.py       # Abstract BaseAnalyzerAdapter & AnalysisResult
│           ├── preprocessing.py# MIME validation, EXIF normalization, letterbox
│           ├── clip_adapter.py# OpenAI CLIP zero-shot multi-prompt classifier
│           ├── yolo_adapter.py# YOLOv8 object detection adapter
│           ├── severity.py   # Rule-based severity estimator
│           └── engine.py     # Multi-adapter orchestrator & result fusion
├── migrations/               # Alembic database migrations
├── static/                   # Responsive Single-Page Application (HTML5/CSS3/Vanilla JS)
├── tests/                    # Automated pytest suite (124 tests)
└── uploads/                  # Incident image storage (.gitkeep preserved)
```

---

## ⚡ Quickstart

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/shiva2609/RescueIQ.git
cd RescueIQ

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Database Migrations
```bash
alembic upgrade head
```

### 3. Launch Development Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

- **Interactive API Documentation (Swagger)**: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
- **Alternative ReDoc UI**: [http://localhost:8000/api/redoc](http://localhost:8000/api/redoc)

---

## 🧪 Testing

Run the full automated test suite:
```bash
pytest tests/ -v
```

All 124 automated tests validate:
- **Triage & Incident Prioritization Suite (73 tests)**:
  - Module A: Multi-signal damage severity assessment, keyword elevation, access blockage escalation, population impact scaling, and evidence & uncertainty tracking.
  - Module B: 64-bit dHash Hamming similarity, MiniLM cosine similarity, Haversine spatial proximity, exponential temporal decay, clustering, and no-auto-delete guarantees.
  - Module C: 0–100 priority scoring, factor explanation generation, administrative weight updates with audit logs, and low-confidence high-risk override triggers.
- **AI Assessment Engine Suite (47 tests)**: Preprocessing, magic-byte checks, fallback adapters, CLIP zero-shot classification, YOLO hazard identification, confidence tier gating, and confidence vs. severity decoupling.
- **Backend API & Database Suite (4 tests)**: Incident CRUD lifecycle, verification status, human label overrides, assignment workflows, and duplicate link confirm/split actions.


---

## 📚 Documentation Links
- **[API Reference](file:///Users/vkshivakumar/Downloads/damage-mapper/API_DOCUMENTATION.md)**: Exhaustive endpoint documentation with request/response schemas.
- **[Model Limitations & Training Guide](file:///Users/vkshivakumar/Downloads/damage-mapper/MODEL_LIMITATIONS.md)**: Dataset specifications, fine-tuning protocols, bias considerations, and prohibited uses.

---

## 📄 License
MIT License. Developed for resilient disaster triage and emergency response operations.
