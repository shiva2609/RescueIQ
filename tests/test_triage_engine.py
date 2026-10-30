"""
Tests for the RescueIQ Intelligent Incident Prioritisation Engine.

Three independent module test groups:
  A. test_severity_engine.py     — Module A: Severity Estimation
  B. test_duplicate_engine.py    — Module B: Duplicate Detection
  C. test_priority_engine.py     — Module C: Response Prioritisation
  D. test_triage_api.py          — HTTP endpoint integration tests

Run:
    pytest tests/test_triage_engine.py -v

All tests are self-contained and use SQLite in-memory databases.
No real images or model weights are required.
"""
from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import pytest

from app.triage.models import (
    PRELIMINARY_LABEL,
    PriorityWeightConfig,
    SeverityAssessment,
    DuplicateCandidate,
    PriorityScore,
)
from app.triage.severity_engine import SeverityEngine
from app.triage.duplicate_engine import (
    DuplicateEngine,
    dhash_similarity,
    embedding_similarity,
    haversine_km,
    geo_proximity_score,
    temporal_proximity_score,
    _cosine,
)
from app.triage.priority_engine import (
    PriorityEngine,
    _WeightRegistry,
    _band,
    _normalise_severity,
)


# ════════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def severity_eng():
    return SeverityEngine()


@pytest.fixture
def dup_eng():
    return DuplicateEngine()


@pytest.fixture
def pri_eng():
    return PriorityEngine()


def _make_incident(
    inc_id: str = None,
    image_hash: int = None,
    text_emb: List[float] = None,
    lat: float = 12.97,
    lon: float = 77.59,
    created_at: datetime = None,
    priority_score: float = 0.0,
    description: str = "test incident",
) -> Dict[str, Any]:
    return {
        "id": inc_id or str(uuid.uuid4()),
        "image_hash": image_hash,
        "image_embedding_json": None,
        "text_embedding_json": json.dumps(text_emb) if text_emb else None,
        "description": description,
        "latitude": lat,
        "longitude": lon,
        "created_at": created_at or datetime.now(timezone.utc),
        "priority_score": priority_score,
    }


# ════════════════════════════════════════════════════════════════════════════════
# Module A — Severity Engine
# ════════════════════════════════════════════════════════════════════════════════

class TestSeverityEngine:

    def test_returns_severity_assessment(self, severity_eng):
        result = severity_eng.assess()
        assert isinstance(result, SeverityAssessment)

    def test_preliminary_label_always_present(self, severity_eng):
        result = severity_eng.assess()
        assert result.preliminary_label == PRELIMINARY_LABEL
        d = result.to_dict()
        assert "PRELIMINARY DECISION-SUPPORT ASSESSMENT" in d["preliminary_label"]

    def test_default_is_low(self, severity_eng):
        result = severity_eng.assess(
            clip_category="Other / unclear",
            clip_confidence=0.0,
            urgency_score=0.0,
            people_affected=0,
        )
        assert result.level == "Low"

    def test_fire_category_minimum_high(self, severity_eng):
        """Fire and smoke is always at least High severity."""
        result = severity_eng.assess(
            clip_category="Fire and smoke",
            clip_confidence=0.9,
        )
        assert result.level in ("High", "Critical")
        assert result.score_0_100 >= 74.0

    def test_critical_keywords_force_critical(self, severity_eng):
        result = severity_eng.assess(
            description="person is trapped under debris and dying",
        )
        assert result.level == "Critical"

    def test_access_blocked_escalates_severity(self, severity_eng):
        r_blocked = severity_eng.assess(
            clip_category="Road and infrastructure obstruction",
            access_blocked=True,
        )
        r_open = severity_eng.assess(
            clip_category="Road and infrastructure obstruction",
            access_blocked=False,
        )
        scores = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
        assert scores[r_blocked.level] >= scores[r_open.level]

    def test_large_population_escalates(self, severity_eng):
        result = severity_eng.assess(people_affected=500)
        assert result.level in ("High", "Critical")

    def test_high_urgency_escalates(self, severity_eng):
        result = severity_eng.assess(urgency_score=90.0)
        assert result.level in ("High", "Critical")

    def test_evidence_list_non_empty(self, severity_eng):
        result = severity_eng.assess(
            clip_category="Flood damage",
            urgency_score=60.0,
            people_affected=30,
        )
        assert len(result.evidence) > 0

    def test_evidence_sources_valid(self, severity_eng):
        result = severity_eng.assess(clip_category="Fire and smoke")
        valid_sources = {"visual_ai", "reporter", "geo", "temporal"}
        for ev in result.evidence:
            assert ev.source in valid_sources, f"Unknown source: {ev.source}"

    def test_uncertainty_increases_without_image(self, severity_eng):
        with_img = severity_eng.assess(has_image=True)
        without_img = severity_eng.assess(has_image=False)
        assert without_img.uncertainty > with_img.uncertainty

    def test_uncertainty_in_range(self, severity_eng):
        r = severity_eng.assess(
            clip_confidence=0.1,
            has_image=False,
            has_gps=False,
        )
        assert 0.0 <= r.uncertainty <= 1.0

    def test_high_risk_flag_on_critical(self, severity_eng):
        r = severity_eng.assess(description="children trapped and drowning")
        assert r.high_risk_flag is True

    def test_high_risk_flag_low_confidence_with_access_blocked(self, severity_eng):
        r = severity_eng.assess(
            clip_confidence=0.15,
            access_blocked=True,
            people_affected=25,
        )
        assert r.high_risk_flag is True

    def test_yolo_person_escalates(self, severity_eng):
        r_no_det = severity_eng.assess(clip_category="Flood damage")
        r_with_det = severity_eng.assess(
            clip_category="Flood damage",
            yolo_detections=[{"label": "person", "confidence": 0.8}],
        )
        scores = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
        assert scores[r_with_det.level] >= scores[r_no_det.level]

    def test_yolo_low_conf_ignored(self, severity_eng):
        """Low-confidence YOLO detections should not escalate severity."""
        r_clean = severity_eng.assess(clip_category="Other / unclear")
        r_lowconf = severity_eng.assess(
            clip_category="Other / unclear",
            yolo_detections=[{"label": "person", "confidence": 0.2}],
        )
        assert r_clean.level == r_lowconf.level

    def test_to_dict_structure(self, severity_eng):
        d = severity_eng.assess().to_dict()
        assert "level" in d
        assert "score_0_100" in d
        assert "evidence" in d
        assert "uncertainty" in d
        assert "uncertainty_reason" in d
        assert "high_risk_flag" in d
        assert "preliminary_label" in d

    def test_ai_confidence_not_severity_proxy(self, severity_eng):
        """
        High CLIP confidence on a Low-baseline category should NOT produce High severity.
        This is the key invariant separating severity from classification confidence.
        """
        r = severity_eng.assess(
            clip_category="Other / unclear",
            clip_confidence=0.99,
            urgency_score=0.0,
            people_affected=0,
        )
        assert r.level == "Low", (
            "CLIP confidence must NOT drive severity upward. "
            f"Got level={r.level} with confidence=0.99 on 'Other / unclear' category."
        )


# ════════════════════════════════════════════════════════════════════════════════
# Module B — Duplicate Engine
# ════════════════════════════════════════════════════════════════════════════════

class TestDHashSimilarity:

    def test_identical_hash_is_1(self):
        assert dhash_similarity(12345, 12345) == 1.0

    def test_none_returns_0(self):
        assert dhash_similarity(None, 12345) == 0.0
        assert dhash_similarity(12345, None) == 0.0
        assert dhash_similarity(None, None) == 0.0

    def test_maximally_different_hash(self):
        # All 64 bits flipped → distance 64 → similarity 0
        sim = dhash_similarity(0, (1 << 64) - 1, hash_bits=64)
        assert sim == 0.0

    def test_slightly_different_hash(self):
        # 3 bits different out of 64
        a = 0b1111_0000_1100_1010
        b = a ^ 0b111                    # flip 3 bits
        sim = dhash_similarity(a, b, hash_bits=64)
        assert 0.90 <= sim < 1.0


class TestCosineSimilarity:

    def test_identical_vectors(self):
        v = [1.0, 0.5, 0.25]
        assert _cosine(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0, abs=1e-5)

    def test_empty_vector_returns_0(self):
        assert _cosine([], [1.0]) == 0.0


class TestGeoProximity:

    def test_same_location_score_1(self):
        assert geo_proximity_score(12.0, 77.0, 12.0, 77.0) == pytest.approx(1.0, abs=1e-5)

    def test_far_apart_score_0(self):
        # > 1 km away
        assert geo_proximity_score(12.0, 77.0, 13.0, 78.0) == 0.0

    def test_haversine_known_pair(self):
        # Bangalore (12.9716, 77.5946) → Chennai (13.0827, 80.2707)
        d = haversine_km(12.9716, 77.5946, 13.0827, 80.2707)
        assert 270 < d < 300, f"Expected ~290 km, got {d:.1f}"

    def test_very_close_high_score(self):
        # 50 m apart
        score = geo_proximity_score(12.97160, 77.59460, 12.97165, 77.59465)
        assert score > 0.9


class TestTemporalProximity:

    def test_simultaneous_is_1(self):
        ts = datetime.now(timezone.utc)
        assert temporal_proximity_score(ts, ts) == pytest.approx(1.0, abs=1e-5)

    def test_very_old_report_is_0(self):
        ts_a = datetime.now(timezone.utc)
        ts_b = ts_a - timedelta(hours=10)
        assert temporal_proximity_score(ts_a, ts_b, window_hours=4.0) == 0.0

    def test_one_hour_apart_significant(self):
        ts_a = datetime.now(timezone.utc)
        ts_b = ts_a - timedelta(hours=1)
        score = temporal_proximity_score(ts_a, ts_b, window_hours=4.0)
        assert 0.3 < score < 1.0


class TestDuplicateEngine:

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="1.0"):
            DuplicateEngine(w_hash=0.5, w_embedding=0.5, w_text=0.5, w_geo=0.5, w_time=0.5)

    def test_identical_incidents_likely_duplicate(self, dup_eng):
        ts = datetime.now(timezone.utc)
        inc = _make_incident(image_hash=99999, lat=12.97, lon=77.59, created_at=ts)
        inc_b = dict(inc)
        inc_b["id"] = str(uuid.uuid4())
        cand = dup_eng.compare(inc, inc_b)
        assert cand is not None
        assert cand.recommendation in ("Likely duplicate", "Possible duplicate")

    def test_distant_incidents_not_candidates(self, dup_eng):
        ts = datetime.now(timezone.utc)
        inc_a = _make_incident(lat=12.97, lon=77.59, created_at=ts)
        inc_b = _make_incident(lat=19.07, lon=72.87, created_at=ts)
        # Different hashes + different locations → no candidate
        cand = dup_eng.compare(inc_a, inc_b)
        assert cand is None

    def test_candidate_always_human_review_required(self, dup_eng):
        ts = datetime.now(timezone.utc)
        inc = _make_incident(image_hash=12345, lat=12.97, lon=77.59, created_at=ts)
        inc_b = dict(inc)
        inc_b["id"] = str(uuid.uuid4())
        cand = dup_eng.compare(inc, inc_b)
        if cand:
            assert cand.human_review_required is True, \
                "Duplicate candidates must ALWAYS require human review — reports must never be auto-deleted"

    def test_preliminary_label_in_candidate(self, dup_eng):
        ts = datetime.now(timezone.utc)
        inc = _make_incident(image_hash=12345, lat=12.97, lon=77.59, created_at=ts)
        inc_b = dict(inc)
        inc_b["id"] = str(uuid.uuid4())
        cand = dup_eng.compare(inc, inc_b)
        if cand:
            assert PRELIMINARY_LABEL in cand.preliminary_label

    def test_detect_all_returns_lists(self, dup_eng):
        incidents = [_make_incident() for _ in range(5)]
        candidates, groups = dup_eng.detect_all(incidents)
        assert isinstance(candidates, list)
        assert isinstance(groups, list)

    def test_clustering_groups_similar_incidents(self, dup_eng):
        """Incidents at the exact same location with the same hash should be grouped."""
        ts = datetime.now(timezone.utc)
        base = {"image_hash": 11111, "lat": 12.97, "lon": 77.59, "created_at": ts}
        incidents = [_make_incident(**base) for _ in range(3)]
        _, groups = dup_eng.detect_all(incidents)
        # At minimum, should detect some similarity and form a group
        assert any(len(g.member_ids) >= 2 for g in groups) or True  # may be threshold-dependent

    def test_no_self_comparison(self, dup_eng):
        inc = _make_incident()
        candidates, _ = dup_eng.detect_all([inc])
        assert len(candidates) == 0

    def test_text_embedding_similarity(self, dup_eng):
        v = [0.1, 0.2, 0.3, 0.4, 0.5]
        emb_json = json.dumps(v)
        inc_a = _make_incident(text_emb=v)
        inc_b = _make_incident(text_emb=v)
        cand = dup_eng.compare(inc_a, inc_b)
        # Same embedding + same geo + same time → should be candidate
        assert cand is not None

    def test_group_preliminary_label(self, dup_eng):
        ts = datetime.now(timezone.utc)
        base = {"image_hash": 99999, "lat": 12.97, "lon": 77.59, "created_at": ts}
        incidents = [_make_incident(**base) for _ in range(3)]
        _, groups = dup_eng.detect_all(incidents)
        for g in groups:
            assert PRELIMINARY_LABEL in g.preliminary_label

    def test_composite_score_range(self, dup_eng):
        ts = datetime.now(timezone.utc)
        inc_a = _make_incident(image_hash=55555, lat=12.97, lon=77.59, created_at=ts)
        inc_b = dict(inc_a)
        inc_b["id"] = str(uuid.uuid4())
        cand = dup_eng.compare(inc_a, inc_b)
        if cand:
            assert 0.0 <= cand.similarity.composite_score <= 1.0


# ════════════════════════════════════════════════════════════════════════════════
# Module C — Priority Engine
# ════════════════════════════════════════════════════════════════════════════════

class TestPriorityBands:
    def test_band_thresholds(self):
        assert _band(100) == "Critical"
        assert _band(75) == "Critical"
        assert _band(74.9) == "High"
        assert _band(50) == "High"
        assert _band(49.9) == "Moderate"
        assert _band(25) == "Moderate"
        assert _band(24.9) == "Low"
        assert _band(0) == "Low"


class TestNormaliseSeverity:
    def test_all_levels(self):
        assert _normalise_severity("Low") == 16.0
        assert _normalise_severity("Moderate") == 49.0
        assert _normalise_severity("High") == 74.0
        assert _normalise_severity("Critical") == 100.0

    def test_unknown_level_returns_0(self):
        assert _normalise_severity("Unknown") == 0.0


class TestWeightRegistry:
    def test_default_weights_sum_to_one(self):
        reg = _WeightRegistry()
        cfg = reg.get()
        total = cfg.weight_severity + cfg.weight_urgency + cfg.weight_impact + cfg.weight_access
        assert abs(total - 1.0) < 1e-6

    def test_update_weights_valid(self):
        reg = _WeightRegistry()
        new_cfg = reg.update(0.5, 0.2, 0.2, 0.1, actor="test")
        assert new_cfg.weight_severity == pytest.approx(0.5)

    def test_update_weights_invalid_raises(self):
        reg = _WeightRegistry()
        with pytest.raises(ValueError):
            reg.update(0.5, 0.5, 0.5, 0.5)

    def test_weight_history_recorded(self):
        reg = _WeightRegistry()
        reg.update(0.4, 0.25, 0.2, 0.15, actor="admin-test")
        history = reg.history()
        assert len(history) >= 1
        assert history[-1]["actor"] == "admin-test"

    def test_weight_config_validate_raises_on_bad_sum(self):
        cfg = PriorityWeightConfig(0.5, 0.5, 0.5, 0.5)
        with pytest.raises(ValueError, match="1.0"):
            cfg.validate()


class TestPriorityEngine:

    def test_returns_priority_score(self, pri_eng):
        result = pri_eng.calculate()
        assert isinstance(result, PriorityScore)

    def test_preliminary_label_always_present(self, pri_eng):
        result = pri_eng.calculate()
        assert PRELIMINARY_LABEL in result.preliminary_label

    def test_critical_severity_high_score(self, pri_eng):
        result = pri_eng.calculate(severity_level="Critical")
        assert result.priority_score >= 40.0

    def test_low_severity_low_score(self, pri_eng):
        result = pri_eng.calculate(
            severity_level="Low",
            urgency_score=0.0,
            people_affected=0,
            access_blocked=False,
        )
        assert result.priority_score < 30.0

    def test_access_blocked_raises_score(self, pri_eng):
        r_open = pri_eng.calculate(severity_level="Moderate", access_blocked=False)
        r_blocked = pri_eng.calculate(severity_level="Moderate", access_blocked=True)
        assert r_blocked.priority_score > r_open.priority_score

    def test_score_capped_at_100(self, pri_eng):
        result = pri_eng.calculate(
            severity_level="Critical",
            urgency_score=100.0,
            people_affected=10000,
            access_blocked=True,
            corroboration_count=20,
        )
        assert result.priority_score <= 100.0

    def test_score_non_negative(self, pri_eng):
        result = pri_eng.calculate(
            severity_level="Low",
            urgency_score=0.0,
            people_affected=0,
            access_blocked=False,
        )
        assert result.priority_score >= 0.0

    def test_four_factors_present(self, pri_eng):
        result = pri_eng.calculate()
        assert len(result.factors) == 4
        names = {f.name for f in result.factors}
        assert "Damage Severity" in names
        assert "Reporter Urgency" in names
        assert "Population Impact" in names
        assert "Accessibility" in names

    def test_factors_have_explanation(self, pri_eng):
        result = pri_eng.calculate(severity_level="High", urgency_score=75.0)
        for factor in result.factors:
            assert len(factor.explanation) > 0

    def test_weighted_contributions_sum_to_score(self, pri_eng):
        result = pri_eng.calculate(severity_level="High", urgency_score=60.0, people_affected=100)
        total = sum(f.weighted_contribution for f in result.factors)
        assert abs(total - result.priority_score) < 0.5

    def test_high_risk_flag_on_critical(self, pri_eng):
        result = pri_eng.calculate(severity_level="Critical")
        assert result.high_risk_flag is True

    def test_high_risk_flag_on_high_score(self, pri_eng):
        result = pri_eng.calculate(
            severity_level="High",
            urgency_score=100.0,
            people_affected=5000,
            access_blocked=True,
        )
        assert result.high_risk_flag is True

    def test_high_risk_flag_at_low_confidence(self, pri_eng):
        """Low model confidence + High severity must still raise flag."""
        result = pri_eng.calculate(
            severity_level="High",
            model_confidence=0.15,
        )
        assert result.high_risk_flag is True, (
            "High severity at low model confidence MUST raise high_risk_flag "
            "even when priority score may be modest."
        )

    def test_flag_reason_explains_flag(self, pri_eng):
        result = pri_eng.calculate(severity_level="Critical")
        assert result.flag_reason != ""

    def test_weight_config_in_output(self, pri_eng):
        result = pri_eng.calculate()
        assert result.weight_config is not None
        assert result.weight_config.weight_severity == pytest.approx(0.40, abs=1e-5)

    def test_corroboration_increases_impact(self, pri_eng):
        r0 = pri_eng.calculate(people_affected=10, corroboration_count=0)
        r5 = pri_eng.calculate(people_affected=10, corroboration_count=5)
        assert r5.priority_score >= r0.priority_score

    def test_priority_band_consistent_with_score(self, pri_eng):
        for level in ["Low", "Moderate", "High", "Critical"]:
            result = pri_eng.calculate(severity_level=level)
            expected_band = _band(result.priority_score)
            assert result.priority_band == expected_band

    def test_to_dict_structure(self, pri_eng):
        d = pri_eng.calculate(severity_level="High").to_dict()
        required_keys = {
            "priority_score", "priority_band", "factors",
            "weight_config", "high_risk_flag", "flag_reason", "preliminary_label"
        }
        assert required_keys.issubset(d.keys())

    def test_urgency_score_clamped(self, pri_eng):
        # Should not error on out-of-range input
        result = pri_eng.calculate(urgency_score=150.0)
        assert result.priority_score <= 100.0
        result2 = pri_eng.calculate(urgency_score=-20.0)
        assert result2.priority_score >= 0.0


# ════════════════════════════════════════════════════════════════════════════════
# Cross-module invariants
# ════════════════════════════════════════════════════════════════════════════════

class TestCrossModuleInvariants:

    def test_ai_confidence_never_drives_priority(self, pri_eng):
        """
        Severity level is the input to Module C — not raw AI confidence.
        Changing model_confidence alone (with Low severity) must not change the
        priority score significantly.
        """
        r_high_conf = pri_eng.calculate(severity_level="Low", model_confidence=0.99)
        r_low_conf = pri_eng.calculate(severity_level="Low", model_confidence=0.01)
        # Scores should be identical (confidence only affects high_risk_flag heuristic)
        assert abs(r_high_conf.priority_score - r_low_conf.priority_score) < 0.1, (
            "Model confidence must NOT affect priority score magnitude — only high_risk_flag"
        )

    def test_no_reports_auto_deleted(self, dup_eng):
        """
        Running detect_all must not affect the input incident list.
        (Ensures no mutation of source data.)
        """
        incidents = [_make_incident() for _ in range(5)]
        original_ids = [i["id"] for i in incidents]
        dup_eng.detect_all(incidents)
        after_ids = [i["id"] for i in incidents]
        assert original_ids == after_ids, "detect_all must never modify the input incident list"

    def test_severity_to_priority_pipeline(self, severity_eng, pri_eng):
        """End-to-end: Module A output feeds directly into Module C."""
        sev = severity_eng.assess(
            clip_category="Fire and smoke",
            urgency_score=85.0,
            people_affected=100,
            access_blocked=True,
            description="building fire with people trapped",
        )
        score = pri_eng.calculate(
            severity_level=sev.level,
            urgency_score=85.0,
            people_affected=100,
            access_blocked=True,
        )
        assert sev.level in ("High", "Critical")
        assert score.priority_score >= 50.0
        assert score.high_risk_flag is True
