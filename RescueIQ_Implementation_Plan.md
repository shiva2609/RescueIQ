# RescueIQ: Comprehensive Implementation Plan (Gap-Closure Roadmap)

This implementation plan bridges the prototype ("Damage Mapper") to the complete specification laid out in [RescueIQ_PRD.md](file:///Users/vkshivakumar/Downloads/damage-mapper/RescueIQ_PRD.md). It addresses every critical, major, and minor gap identified in the forensic audit across 4 distinct execution phases.

---

## 1. Overview & Architecture Transition

```
CURRENT STATE (Damage Mapper MVP)                TARGET STATE (RescueIQ v1.0)
┌──────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
│ • Off-the-shelf CLIP Zero-Shot       │        │ • Fine-Tuned EfficientNet-B0 (3-Class)      │
│ • Heuristic hazard severity lookup   │        │ • Visual Damage Severity (Severe/Mild/None)  │
│ • Image dHash + 150m spatial check   │   ──►  │ • Multimodal Deduplication (MiniLM + dHash) │
│ • Plain SQLite (1 unindexed table)   │        │ • Structured Schema + Immutable Audit Trail  │
│ • Zero authentication / open endpoints│       │ • JWT Authentication + RBAC (Citizen/Admin)  │
│ • Monolithic vanilla JS frontend     │        │ • Modular UI with Factor Breakdown & Clusters│
└──────────────────────────────────────┘        └──────────────────────────────────────────────┘
```

---

## 2. Phased Implementation Roadmap

### Phase 1: Security, Auditability & Data Model (Core Foundation)
> **Goal:** Secure open endpoints, establish immutable logging, and prepare relational tables required by PRD Section 10.  
> **Estimated Effort:** Medium | **PRD Requirements:** IM-1, IM-4, IM-5, Non-Functional (Sec 8)

1. **Database Schema Expansion:**
   - Migrate [damage_mapper.db](file:///Users/vkshivakumar/Downloads/damage-mapper/damage_mapper.db) from single-table flat storage to relational entities:
     - `users`: ID, username, hashed_password, role (`citizen`, `coordinator`, `admin`), created_at.
     - `incidents`: Enhanced fields (`damage_severity`, `urgency_score`, `exposure_score`, `access_score`, `priority_score`, `abstain_flag`).
     - `duplicate_links`: Link pair (`incident_a`, `incident_b`), text_sim, image_sim, geo_distance_m, human_status (`pending`, `confirmed`, `rejected`).
     - `status_history`: Immutable log (`id`, `incident_id`, `actor_id`, `action`, `previous_value`, `new_value`, `timestamp`).
2. **Authentication & RBAC:**
   - Implement lightweight authentication (Bearer token / HTTP Basic) in [app/main.py](file:///Users/vkshivakumar/Downloads/damage-mapper/app/main.py).
   - Public endpoint: `POST /api/incidents` (citizen submission).
   - Restricted endpoints: `PATCH /api/incidents/{id}`, `POST /api/duplicates/{id}/decision`, `GET /api/stats` (require coordinator or admin credentials).
3. **Audit Trail Recording:**
   - Any status change, label correction, or assignment writes an immutable record to `status_history`.
4. **Input Sanitization & Storage Access Control:**
   - Restrict direct public directory browsing on `/uploads/` and enforce strict MIME-type & payload size guards.

---

### Phase 2: Explainable Prioritization & Enhanced Duplicate Engine
> **Goal:** Align priority scoring with transparent per-factor explanations and upgrade duplicate detection to multimodal.  
> **Estimated Effort:** Medium | **PRD Requirements:** PR-1, PR-2, PR-3, NL-1, NL-2, DD-1, DD-2, DD-3, DD-5, DD-6

1. **Multimodal Duplicate Detection:**
   - Integrate `sentence-transformers` (`all-MiniLM-L6-v2`) to compute semantic text embeddings for citizen descriptions.
   - Combine signals into weighted duplicate score:
     $$\text{Score}_{\text{dup}} = 0.40 \cdot \text{TextSim} + 0.30 \cdot \text{ImageSim} + 0.20 \cdot \text{SpatialProximity} + 0.10 \cdot \text{TemporalWindow}$$
   - Store candidate pairs in `duplicate_links` table without deleting original records.
2. **Human-in-the-Loop Duplicate Resolution:**
   - Add endpoint `POST /api/duplicates/{id}/decision` to allow coordinators to confirm or split flagged duplicates.
   - Ensure repeat reports from the same coordinate/user do not artificially inflate the incident priority score (DD-6).
3. **Transparent Priority Factor Breakdown:**
   - Formalize formula $P = 0.40 \cdot S + 0.25 \cdot U + 0.20 \cdot I + 0.15 \cdot A$.
   - Return raw and weighted components ($S$, $U$, $I$, $A$) in API incident serialization.
   - Display a visual breakdown bar or tooltip on each UI incident card.

---

### Phase 3: Vision Classifier & Abstention Pipeline
> **Goal:** Replace coarse zero-shot CLIP with a dedicated 3-class disaster damage classifier featuring abstention.  
> **Estimated Effort:** Large | **PRD Requirements:** ML-1, ML-2, ML-3, ML-4, ML-5, ML-6, ML-7

1. **Dataset Preparation & Event-Grouped Splits (Offline):**
   - Clean and standardize disaster dataset (e.g., CrisisMMD / MEDIC damage subset) into 3 classes: `Severe Damage`, `Mild Damage`, `Little-or-none`.
   - Ensure no disaster event leaks across train, validation, and test splits.
2. **Staged Fine-Tuning of EfficientNet-B0:**
   - Stage 1: Freeze backbone, train classification head ($lr = 10^{-3}$).
   - Stage 2: Unfreeze top backbone layers, fine-tune at lower rate ($lr = 10^{-5}$).
   - Export optimized PyTorch weights checkpoint to `app/models/efficientnet_b0_damage.pt`.
3. **Calibrated Abstention Mechanism:**
   - Calculate prediction entropy or confidence cutoff (e.g., if $\max(p) < \tau$, flag as `abstain=True`).
   - Route abstained images to "Needs human review" with $S=0$ contribution until verified.
4. **Baseline Evaluation & Metric Logging:**
   - Execute evaluation script comparing fine-tuned model against frozen ImageNet baseline.
   - Persist metric logs (Macro F1, Severe Recall, confusion matrix) in `evaluation/`.

---

### Phase 4: UI/UX Refinement & Geospatial Enhancements
> **Goal:** Enhance dashboard situational awareness, add clustering, and update demo assets.  
> **Estimated Effort:** Small-Medium | **PRD Requirements:** GEO-1, GEO-2, GEO-3, GEO-5, DB-1, DB-2, DB-3, IM-2

1. **Leaflet Marker Clustering:**
   - Integrate `leaflet.markercluster` to group nearby points and prevent marker occlusion at broad zoom levels.
2. **Interactive Map Pinning:**
   - Allow citizen reporters to click directly on the Leaflet map to set or refine incident coordinates.
3. **AI Label Correction (Human Override):**
   - Provide an "Override AI Severity" dropdown in the coordinator incident card (`Severe`, `Mild`, `None`) which immediately recalculates the priority score.
4. **Realistic Demo Seed Data:**
   - Replace the single grey placeholder image with curated, CC0 disaster scenario photographs demonstrating varied damage severities and duplicate clustering.

---

## 3. Implementation Phasing Matrix

| Phase | Milestone | Primary Deliverable | Exit Proof / Verification Test |
|---|---|---|---|
| **P1** | Security & Data Model | Auth, RBAC, SQLite schema migration, audit table | Unauthenticated PATCH returns 401; status changes logged to `status_history`. |
| **P2** | Deduplication & Priority | Sentence-transformers, multi-factor scoring UI | Duplicate pairs flagged with text+image scores; factor breakdown visible on cards. |
| **P3** | Damage Classifier | EfficientNet-B0 checkpoint, abstention, evaluation logs | Checkpoint outputs 3-class probabilities; logged confusion matrix and recall metrics exist. |
| **P4** | Geospatial & Demo UI | Map clustering, interactive pin-drop, real demo images | Leaflet cluster groups render on zoom; label override updates ranking in real-time. |
