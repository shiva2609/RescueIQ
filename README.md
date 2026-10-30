# RescueIQ (Damage Mapper)

> **AI-Powered Disaster Damage Assessment, Incident Prioritization & Response Platform**  
> *Full-Stack Platform: Modular FastAPI Backend (v2.0), Pluggable Vision-Language Assessment Engine (CLIP + YOLOv8), and Professional Responsive Command Center*

[![Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen.svg)]()
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

### 2. Multi-Factor Priority Scoring ($P = 0.40S + 0.25U + 0.20I + 0.15A$)
- **40% Severity ($S$)**: Rule-based damage severity score.
- **25% Reporter Urgency ($U$)**: Self-reported situation urgency.
- **20% Population Exposure & Corroboration ($I$)**: Community scale and corroborating witness reports.
- **15% Route Access ($A$)**: Penalizes isolated zones requiring immediate access clearance.
- Transparent mathematical factor breakdown rendered on every incident card.

### 3. Multimodal Deduplication Engine
- Combines semantic text similarity (`sentence-transformers` `all-MiniLM-L6-v2`), 64-bit perceptual difference hashing (`dHash`), spatial Haversine proximity, and temporal decay into a single composite duplicate probability score.
- Interactive side-by-side triage modal allowing coordinators to **Confirm** or **Split** duplicates.

### 4. Enterprise Backend Architecture
- Organized into clean layers: `models/`, `schemas/`, `services/`, `routers/`, and `ai/engine/`.
- Dual database support: **SQLite** (with WAL mode & foreign key enforcement for local dev) and **PostgreSQL / PostGIS** (spatial proximity querying using `ST_DWithin`).
- Database migrations managed via **Alembic**.
- Immutable audit log (`StatusHistory`) capturing all state transitions, human damage class overrides, assignments, and actors.

### 5. Professional Responsive Operations Dashboard
- High-contrast, accessibility-tested navy & alert color palette.
- **Landing Page**: Platform capability overview and statistics.
- **Citizen Report Portal**: Drag-and-drop image upload, real-time magic-byte verification, and one-tap GPS locator.
- **Operations Dashboard**: Live KPI statistics, priority-ranked queue, filter controls, and responder dispatching.
- **Geospatial Map**: Leaflet cluster map with coordinate picker, severity radius rendering, and marker clusters.
- **Deduplication Review**: Side-by-side comparison modal with confirm/split actions.
- **AI Analysis Lab**: Multi-model vision inspection playground testing CLIP and YOLO inference on arbitrary images.
- **Admin Management**: User registration, role assignments (`coordinator`, `responder`, `citizen`), and immutable audit trail viewer.

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
│   ├── routers/              # HTTP API controllers (/incidents, /assess, /users, etc.)
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
├── tests/                    # Automated pytest suite (51 tests)
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

All 51 tests validate:
- Preprocessing & MIME magic-byte validation
- Adapter fallback logic & confidence tier calculations
- Independence of damage severity from AI confidence
- Candidate classification ranking & YOLO detection heuristics
- Incident CRUD, human label override, and soft-delete
- Spatial proximity queries and duplicate candidate decisions

---

## 📚 Documentation Links
- **[API Reference](file:///Users/vkshivakumar/Downloads/damage-mapper/API_DOCUMENTATION.md)**: Exhaustive endpoint documentation with request/response schemas.
- **[Model Limitations & Training Guide](file:///Users/vkshivakumar/Downloads/damage-mapper/MODEL_LIMITATIONS.md)**: Dataset specifications, fine-tuning protocols, bias considerations, and prohibited uses.

---

## 📄 License
MIT License. Developed for resilient disaster triage and emergency response operations.
