# RescueIQ: Product Requirements Document (PRD)

**Version:** 1.0 (planned-scope baseline) | **Status:** Baseline for gap audit | **Product:** AI-powered disaster damage assessment and emergency response triage

> This PRD records what was *planned*. It does not claim anything is built. Section 15 holds the traceability matrix used to compare plan against the prototype. Every requirement has an ID so the audit can cite it.

---

## 1. Summary

RescueIQ turns citizen disaster reports (photo, description, GPS, category) into a ranked, map-based and explainable list of where responders should act first, how confident the system is, and where reports may be missing. It is decision support: humans verify and decide.

**One-line problem:** Emergency teams do not lack information in a disaster; they lack a fast, reliable way to turn scattered, duplicated, unverified citizen reports into prioritized, explainable intelligence.

## 2. Goals and non-goals

**Goals**
1. Classify visible damage in ground-level citizen photos (Severe / Mild / Little-or-none) with an uncertainty option.
2. Group duplicate reports into incidents and use independent repetition as corroboration.
3. Produce a transparent priority score with per-factor contribution.
4. Show incidents on an interactive map with clustering and severity.
5. Keep humans in control: verify, reject, correct, assign, audit.
6. Surface blind spots (exposed areas with few or no reports).

**Non-goals**
- Detecting casualties, structural safety or actual disaster impact from a photo.
- Replacing emergency professionals or automating dispatch.
- Claiming the priority weights are scientifically validated.
- Training any model from scratch.
- Reporting any metric that was not measured.

## 3. Users

| Persona | Needs | Context |
|---|---|---|
| Citizen reporter | Submit a report in under a minute; get a tracking ID; see status | Mobile, poor connectivity, stress |
| Emergency coordinator | Ranked queue, evidence, confidence, map, duplicate groups | Control room, many reports at once |
| Field responder / team lead | Assigned incidents, status updates | Mobile or tablet |
| Administrator | Users, roles, audit log, analytics | Desktop |

## 4. Scope

**MVP (must work end to end):** report submission, ground-level image upload, fine-tuned EfficientNet-B0 3-class classifier with abstention, description analysis, duplicate detection, priority queue, interactive map, emergency dashboard, incident status management.

**Differentiators (build after the core flow works):** corroboration scoring, silent-zone flag, abstention (coverage vs. accuracy) curve.

**Deferred:** YOLO hazard detection, video processing, multilingual and voice reporting, advanced routing, predictive hotspots, automated situation reports, evidence-request messaging (mock-up only if time allows), learned (XGBoost) priority model.

**Focus hazard for demo and evaluation:** urban flooding and waterlogging (other hazards are extensions).

## 5. Functional requirements

Priority: P0 = MVP, P1 = differentiator, P2 = deferred.

### 5.1 Citizen portal

| ID | Requirement | Pri | Acceptance criterion |
|---|---|---|---|
| CP-1 | Register and log in | P0 | Account created; role assigned (citizen) |
| CP-2 | Submit report with photo, description, category | P0 | Report stored with image, text, category, timestamp |
| CP-3 | Capture GPS or pick a location on a map | P0 | Valid lat/long stored; manual pin works if GPS denied |
| CP-4 | Receive a tracking ID | P0 | Unique ID shown at confirmation |
| CP-5 | View report verification status | P0 | Status updates reflect dashboard actions |
| CP-6 | Mobile-friendly, minimal-step form | P0 | Completes on a phone-width screen without horizontal scroll |
| CP-7 | Short video upload | P2 | Deferred |

### 5.2 Damage classification (ML)

| ID | Requirement | Pri | Acceptance criterion |
|---|---|---|---|
| ML-1 | EfficientNet-B0 (ImageNet weights) with replaced 3-class head | P0 | Checkpoint loads; outputs 3 class probabilities |
| ML-2 | Staged fine-tuning: head only, then partial backbone unfreeze at lower learning rate | P0 | Training log shows both stages; best checkpoint saved by validation metric |
| ML-3 | Abstain / "uncertain or not a disaster image" output | P0 | Low-confidence and non-disaster images are not forced into a class; threshold documented |
| ML-4 | Inference exposed through an API | P0 | Endpoint returns class, confidence and abstain flag |
| ML-5 | Event-grouped train / validation / test split | P0 | No disaster event appears in more than one split |
| ML-6 | Baseline vs. fine-tuned comparison | P0 | Same test set, same metrics for both |
| ML-7 | Abstention curve (coverage vs. accuracy / severe recall) | P1 | Curve produced from real test predictions |
| ML-8 | Evaluation on team-collected ground-level photos | P1 | Separate small set; domain-shift drop reported |
| ML-9 | YOLO object / hazard detection | P2 | Deferred; must not be described as damage detection |

### 5.3 Text analysis and duplicates

| ID | Requirement | Pri | Acceptance criterion |
|---|---|---|---|
| NL-1 | Embed descriptions with all-MiniLM-L6-v2 (no fine-tuning) | P0 | Embedding stored per report |
| NL-2 | Semantic similarity between reports | P0 | Paraphrased reports score higher than unrelated ones |
| DD-1 | Combine text similarity, image embedding similarity, GPS distance, time gap and category | P0 | Pairwise score and its component values stored |
| DD-2 | Flag candidate duplicates; never auto-delete | P0 | Flagged pairs visible; original reports remain |
| DD-3 | Human confirm / split of duplicate groups | P0 | Action recorded in history |
| DD-4 | Ablation: text-only, image-only, GPS-only, combined | P1 | Precision, recall, F1, false-duplicate rate on a labelled set |
| DD-5 | Corroboration score from independent reporters, time spread and location consistency | P1 | Score shown per incident; same-user repeats do not raise it |
| DD-6 | Safeguards against repeated or coordinated submissions | P0 | Duplicate count cannot inflate priority without independence checks |

### 5.4 Prioritization

| ID | Requirement | Pri | Acceptance criterion |
|---|---|---|---|
| PR-1 | Score = 0.40·S + 0.25·U + 0.20·I + 0.15·A, each input normalised 0-100 | P0 | Score reproducible from stored inputs |
| PR-2 | Show each factor's contribution | P0 | Detail view lists S, U, I, A and weighted parts |
| PR-3 | Label weights "provisional, not validated" | P0 | Notice visible in UI and docs |
| PR-4 | Weight sensitivity analysis | P1 | Ranking change shown under altered weights |
| PR-5 | Agreement with independently assigned priorities | P1 | Small labelled set; rank correlation reported with its size |
| PR-6 | Learned (XGBoost) priority model | P2 | Only if real labelled data exists; never fabricate labels |

### 5.5 Geospatial

| ID | Requirement | Pri | Acceptance criterion |
|---|---|---|---|
| GEO-1 | Interactive map with incident markers | P0 | Markers at stored coordinates |
| GEO-2 | Severity-coloured markers and clustering | P0 | Cluster count and colours reflect data |
| GEO-3 | Filter by severity, category, status, area | P0 | Filters change map and list together |
| GEO-4 | Marker click opens incident details | P0 | Detail panel with evidence |
| GEO-5 | Reported road blockage displayed as *reported*, not confirmed | P0 | Label distinguishes reported vs. verified |
| GEO-6 | Silent-zone flag: exposed areas with few or no reports | P1 | Rule-based, labelled as rule-based, with documented inputs |
| GEO-7 | Routing recommendations | P2 | Deferred |

### 5.6 Emergency dashboard and incident management

| ID | Requirement | Pri | Acceptance criterion |
|---|---|---|---|
| DB-1 | Overview counts: total, severe / mild / low, pending verification, potential duplicates | P0 | Counts match the database |
| DB-2 | High-priority queue with reasons | P0 | Sorted by score; "why" visible |
| DB-3 | Incident detail: photo, text, AI label, confidence, abstain flag, duplicates | P0 | All fields shown |
| IM-1 | Verify or reject a report | P0 | Status changes; actor and time logged |
| IM-2 | Correct the AI label | P0 | Override stored; ranking recalculates |
| IM-3 | Assign to a response team and update response status | P0 | Assignment visible to citizen status and dashboard |
| IM-4 | Record actions taken; keep incident history | P0 | Immutable audit trail |
| IM-5 | Role-based access (citizen, responder, admin) | P0 | Citizens cannot open dashboard routes |
| AN-1 | Analytics: distribution, trend, duplicates, priority, model information | P1 | Charts computed from real data |

## 6. Priority scoring specification

- **S** damage severity: from classifier class and confidence; abstained images contribute no severity and are flagged for manual review.
- **U** reported urgency: from citizen input and category; treated as self-reported and unverified.
- **I** estimated impact: from reported affected people, category, and corroboration.
- **A** accessibility / response difficulty: from reported blockages and location.
- Each input is normalised to 0-100 before weighting. Duplicate volume is not a direct factor; corroboration enters only through independence checks (DD-5, DD-6).
- This is a **rule-based** score. It must not be described as a trained or validated model.

## 7. ML evaluation protocol

- **Classifier:** accuracy, per-class precision and recall, macro F1, confusion matrix, **severe-damage recall**, inference time, calibration where feasible.
- **Baseline:** original ImageNet EfficientNet-B0 with an untrained or linear-probe head, same test set.
- **Splits:** grouped by disaster event; duplicates and near-duplicates removed before splitting.
- **Datasets:** CrisisMMD as primary candidate; check suitability, ground-level proportion and licence terms (non-commercial restrictions) before use; supplement with 300-500 labelled ground-level images where available; investigate MEDIC and others.
- **Targets:** set only after the baseline run is recorded. No number is promised in advance.
- **Reporting rule:** every figure in slides or docs must come from a logged run.

## 8. Non-functional requirements

- **Privacy:** photos and locations are personal data; role-based access; minimise stored personal fields; strip or restrict image metadata exposure.
- **Integrity:** detect recycled or old images where feasible; rate-limit submissions.
- **Auditability:** all status changes and overrides logged with actor and time.
- **Resilience:** report submission must not fail if ML inference fails; queue the report and mark "analysis pending".
- **Usability:** clear severity indicators, status labels and responsive layout; clarity over animation.
- **Performance:** inference latency is measured and reported, not assumed.
- **Licensing:** dataset licences and usage limits documented.

## 9. Architecture summary

- **Frontend:** Next.js, React, TypeScript, Tailwind; Leaflet and OpenStreetMap.
- **Backend:** Python, FastAPI, REST.
- **ML:** PyTorch / torchvision (EfficientNet-B0), sentence-transformers, OpenCV, scikit-learn; Ultralytics YOLO deferred.
- **Data:** PostgreSQL with PostGIS; Azure Blob Storage or equivalent for images.
- **Deployment:** Azure or AWS; each ML component independently testable.

**Flow:** submit → validate → preprocess image → classify (with abstain) → embed text → duplicate scoring → geospatial association → priority score → dashboard → human verification → status updates.

## 10. Core data entities

`users`, `reports` (image reference, description, category, location, timestamp, reporter), `analyses` (class, probabilities, abstain flag, model version), `embeddings` (text, image), `incidents` (grouped reports, corroboration, priority and factor parts), `duplicate_links` (pair, score components, human decision), `assignments`, `status_history` (actor, action, time), `teams`.

## 11. Key API surface

`POST /reports`, `GET /reports/{id}`, `GET /reports/{id}/status`, `POST /analyze/image`, `GET /incidents`, `GET /incidents/{id}`, `PATCH /incidents/{id}/status`, `POST /incidents/{id}/assign`, `POST /duplicates/{id}/decision`, `GET /map/incidents`, `GET /analytics/summary`.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Domain shift: social-media training images vs. citizen phone photos | Own ground-level test set (ML-8); report the drop honestly |
| Dataset licence limits | Review licence before training or publishing results |
| False or malicious reports | Human verification, independence checks, rate limits |
| Reporting bias toward connected areas | Silent-zone flag (GEO-6) |
| Overstated claims | Trained vs. rule-based labelling; measured-only metrics |
| Inference failure | Graceful "analysis pending" path |
| Scope creep | Defer P2; lock MVP before differentiators |

## 13. Success metrics

Measured, never assumed: classifier macro F1 and severe recall on unseen events; gain over baseline; duplicate precision / recall / false-duplicate rate; ranking agreement with independent priorities (with sample size stated); simulated-scenario reduction from raw reports to incidents; end-to-end flow working from submission to responder assignment.

## 14. Implementation plan (gated phases)

Phase ordering reflects dependencies. Status of each phase is set after the gap audit.

| Phase | Work | Owner (role) | Gate to exit |
|---|---|---|---|
| 0. Data | Download, inspect, clean, dedupe, normalise labels, event-grouped split, licence notes | AI/ML | Splits reproducible; leakage check passes |
| 1. Classifier | Baseline, staged fine-tuning, abstention threshold, evaluation, checkpoint | AI/ML | ML-1 to ML-6 met; metrics logged |
| 2. Backend core | Schema, auth, report intake, image storage, inference endpoint | Backend | CP-2 to CP-5, ML-4 work end to end |
| 3. Text and duplicates | Embeddings, similarity, duplicate scoring, human decision | NLP | DD-1 to DD-3, DD-6 met |
| 4. Priority | Score, factor breakdown, provisional labelling | NLP | PR-1 to PR-3 met |
| 5. Frontend | Citizen portal, dashboard, queue, detail, map, filters | Frontend | CP-1 to CP-6, DB-1 to DB-3, GEO-1 to GEO-4 met |
| 6. Incident management | Verify, reject, override, assign, history, roles | Backend + Frontend | IM-1 to IM-5 met |
| 7. Integration test | Full flow on seeded realistic scenario | All | Submission to assignment works unattended |
| 8. Differentiators | Corroboration, silent zones, abstention curve, ablations, sensitivity | NLP + AI/ML | DD-4, DD-5, ML-7, PR-4, GEO-6 met |
| 9. Evaluation and docs | Final metrics, baseline comparison, ground-level test, licence notes | All | Every reported number traceable to a log |
| 10. Demo | Flood scenario, one wrong prediction corrected by a human | All | Rehearsed; claims match evidence |

## 15. Traceability matrix (to be completed by the gap audit)

Use this table to compare plan against prototype. "Status" values: Implemented-Verified, Implemented-Unverified, Partial, Mocked/Stubbed, Not implemented, Deviation, Added (not in plan).

| Req ID | Planned (this PRD) | Status | Evidence (file / log / screenshot) | Delta from plan |
|---|---|---|---|---|
| CP-1 to CP-7 | Section 5.1 | | | |
| ML-1 to ML-9 | Section 5.2 | | | |
| NL-1, NL-2, DD-1 to DD-6 | Section 5.3 | | | |
| PR-1 to PR-6 | Section 5.4 | | | |
| GEO-1 to GEO-7 | Section 5.5 | | | |
| DB-1 to DB-3, IM-1 to IM-5, AN-1 | Section 5.6 | | | |
| Non-functional (Section 8) | Privacy, integrity, audit, resilience | | | |
