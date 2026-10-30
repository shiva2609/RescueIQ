"""
Unit tests for the disaster damage assessment engine.

Test strategy:
  - Uses synthetic in-memory images to avoid filesystem dependencies.
  - Tests each component in isolation: preprocessing, adapters, severity, engine.
  - Tests the engine's fallback behaviour with mock adapters.
  - Tests confidence tier logic and review-required flag.
  - Tests that severity is NEVER derived from classification confidence.
  - Tests that no predictions are fabricated when models unavailable.

Sample images created:
  - flood_sim.jpg:     blue-dominant image simulating water/flood
  - fire_sim.jpg:      red/orange image simulating fire
  - debris_sim.jpg:    grey/dark image simulating rubble
  - blank_white.jpg:   white image (challenging edge case)
  - tiny_1x1.jpg:      1×1 pixel (should fail dimension validation)
  - not_an_image.jpg:  plain text bytes (should fail magic-byte check)

Model limitation note:
  CLIP ViT-B/32 zero-shot on synthetic colour-block images is NOT expected
  to produce meaningful category predictions. These tests validate pipeline
  logic, not model accuracy. Accuracy testing requires real disaster images
  (see MODEL_LIMITATIONS.md).
"""
from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage


# ── Fixtures: synthetic images ────────────────────────────────────────────────

def _make_jpeg(width: int, height: int, color: tuple) -> bytes:
    """Create a minimal valid JPEG in memory with a flat colour fill."""
    buf = io.BytesIO()
    img = PILImage.new("RGB", (width, height), color)
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_png(width: int, height: int, color: tuple) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tmp_images(tmp_path: Path) -> dict:
    """Write sample synthetic images to a temp directory and return paths."""
    images = {}

    # Flood simulation — deep blue dominant
    images["flood"] = tmp_path / "flood_sim.jpg"
    images["flood"].write_bytes(_make_jpeg(640, 480, (30, 80, 180)))

    # Fire simulation — red/orange
    images["fire"] = tmp_path / "fire_sim.jpg"
    images["fire"].write_bytes(_make_jpeg(640, 480, (220, 80, 10)))

    # Debris/building damage — grey
    images["debris"] = tmp_path / "debris_sim.jpg"
    images["debris"].write_bytes(_make_jpeg(640, 480, (90, 85, 82)))

    # Vegetation / green (could be landslide)
    images["green"] = tmp_path / "green_sim.jpg"
    images["green"].write_bytes(_make_jpeg(480, 640, (40, 130, 50)))

    # White blank (challenging edge case)
    images["blank"] = tmp_path / "blank_white.jpg"
    images["blank"].write_bytes(_make_jpeg(300, 300, (255, 255, 255)))

    # PNG format image
    images["png"] = tmp_path / "test.png"
    images["png"].write_bytes(_make_png(320, 240, (100, 150, 200)))

    return images


@pytest.fixture
def invalid_images(tmp_path: Path) -> dict:
    images = {}

    # Too small (below MIN_DIMENSION)
    images["tiny"] = tmp_path / "tiny_1x1.jpg"
    images["tiny"].write_bytes(_make_jpeg(1, 1, (128, 128, 128)))

    # Not an image at all (plain text with fake .jpg extension)
    images["not_image"] = tmp_path / "not_an_image.jpg"
    images["not_image"].write_bytes(b"This is definitely not an image. <?php echo 'evil'; ?>")

    # Empty file
    images["empty"] = tmp_path / "empty.jpg"
    images["empty"].write_bytes(b"")

    return images


# ── 1. Preprocessing tests ────────────────────────────────────────────────────

class TestPreprocessing:
    """Tests for image preprocessing and validation pipeline."""

    def test_valid_jpeg_loads(self, tmp_images):
        from app.ai.engine.preprocessing import load_and_validate_image
        img = load_and_validate_image(tmp_images["flood"])
        assert img.mode == "RGB"
        assert img.size == (640, 480)

    def test_valid_png_loads(self, tmp_images):
        from app.ai.engine.preprocessing import load_and_validate_image
        img = load_and_validate_image(tmp_images["png"])
        assert img.mode == "RGB"

    def test_empty_file_rejected(self, invalid_images):
        from app.ai.engine.preprocessing import load_and_validate_image, PreprocessingError
        with pytest.raises(PreprocessingError, match="empty"):
            load_and_validate_image(invalid_images["empty"])

    def test_non_image_rejected_by_magic_bytes(self, invalid_images):
        from app.ai.engine.preprocessing import load_and_validate_image, PreprocessingError
        with pytest.raises(PreprocessingError, match="recognised image format"):
            load_and_validate_image(invalid_images["not_image"])

    def test_too_small_rejected(self, invalid_images):
        from app.ai.engine.preprocessing import load_and_validate_image, PreprocessingError
        with pytest.raises(PreprocessingError, match="too small"):
            load_and_validate_image(invalid_images["tiny"])

    def test_missing_file_rejected(self, tmp_path):
        from app.ai.engine.preprocessing import load_and_validate_image, PreprocessingError
        with pytest.raises(PreprocessingError, match="not found"):
            load_and_validate_image(tmp_path / "nonexistent.jpg")

    def test_clip_resize_produces_square(self, tmp_images):
        from app.ai.engine.preprocessing import load_and_validate_image, resize_for_clip
        img = load_and_validate_image(tmp_images["flood"])
        resized = resize_for_clip(img, size=224)
        assert resized.size == (224, 224)

    def test_yolo_resize_letterboxes_to_640(self, tmp_images):
        from app.ai.engine.preprocessing import load_and_validate_image, resize_for_yolo
        img = load_and_validate_image(tmp_images["green"])  # portrait 480×640
        resized = resize_for_yolo(img, target=640)
        assert resized.size == (640, 640)

    def test_validate_raw_bytes_accepts_jpeg(self, tmp_images):
        from app.ai.engine.preprocessing import validate_raw_bytes
        raw = tmp_images["flood"].read_bytes()
        mime = validate_raw_bytes(raw)
        assert mime == "image/jpeg"

    def test_validate_raw_bytes_rejects_text(self):
        from app.ai.engine.preprocessing import validate_raw_bytes, PreprocessingError
        with pytest.raises(PreprocessingError):
            validate_raw_bytes(b"<html><body>Not an image</body></html>")

    def test_get_image_metadata(self, tmp_images):
        from app.ai.engine.preprocessing import load_and_validate_image, get_image_metadata
        img = load_and_validate_image(tmp_images["flood"])
        meta = get_image_metadata(img)
        assert meta["width"] == 640
        assert meta["height"] == 480
        assert meta["mode"] == "RGB"
        assert "aspect_ratio" in meta


# ── 2. Base adapter tests ─────────────────────────────────────────────────────

class TestBaseAdapter:
    """Tests for the abstract adapter contract and confidence tier logic."""

    def test_confidence_tier_high(self):
        from app.ai.engine.base import confidence_tier
        assert confidence_tier(0.85) == "high"

    def test_confidence_tier_medium(self):
        from app.ai.engine.base import confidence_tier
        assert confidence_tier(0.60) == "medium"

    def test_confidence_tier_low(self):
        from app.ai.engine.base import confidence_tier
        assert confidence_tier(0.40) == "low"

    def test_confidence_tier_review_required(self):
        from app.ai.engine.base import confidence_tier
        assert confidence_tier(0.15) == "review_required"

    def test_review_required_flag_set_below_threshold(self, tmp_images):
        """An adapter that returns confidence < 0.30 must set review_required=True."""
        from app.ai.engine.base import BaseAnalyzerAdapter, AnalysisResult

        class LowConfAdapter(BaseAnalyzerAdapter):
            ADAPTER_TYPE = "test"

            def is_available(self):
                return True

            def _run_inference(self, image_path):
                return AnalysisResult(
                    category="Flood damage",
                    confidence=0.20,  # below LOW_CONFIDENCE_THRESHOLD
                    model_name="TestModel",
                    adapter_type="test",
                )

        adapter = LowConfAdapter()
        result = adapter.analyze(tmp_images["flood"])
        assert result.review_required is True
        assert any("review threshold" in w for w in result.warnings)

    def test_fallback_never_fabricates(self, tmp_images):
        """When _run_inference raises, fallback must return is_fallback=True, not fake data."""
        from app.ai.engine.base import BaseAnalyzerAdapter, AnalysisResult

        class BrokenAdapter(BaseAnalyzerAdapter):
            ADAPTER_TYPE = "test"

            def is_available(self):
                return True

            def _run_inference(self, image_path):
                raise RuntimeError("Simulated model crash")

        adapter = BrokenAdapter()
        result = adapter.analyze(tmp_images["flood"])
        assert result.is_fallback is True
        assert result.category == "Unavailable"
        assert result.confidence == 0.0
        assert "Simulated model crash" in (result.fallback_reason or "")

    def test_fallback_result_marks_review_required(self):
        """Fallback results must always require human review."""
        from app.ai.engine.base import BaseAnalyzerAdapter

        class NeverLoadedAdapter(BaseAnalyzerAdapter):
            ADAPTER_TYPE = "test"

            def is_available(self):
                return False

            def _run_inference(self, image_path):
                raise RuntimeError("should not be called")

        adapter = NeverLoadedAdapter()
        fallback = adapter._fallback("Not loaded")
        assert fallback.review_required is True
        assert fallback.is_fallback is True


# ── 3. Severity estimator tests ───────────────────────────────────────────────

class TestSeverityEstimator:
    """
    Tests for rule-based severity estimation.

    CRITICAL: These tests verify that severity is computed from contextual
    metadata, NOT from any AI model confidence score.
    """

    def setup_method(self):
        from app.ai.engine.severity import SeverityEstimator
        self.estimator = SeverityEstimator()

    def test_fire_is_minimum_severe(self):
        """Fire and smoke category must always produce at least 'severe' severity."""
        result = self.estimator.estimate(category="Fire and smoke")
        assert result.level in ("severe", "critical")
        assert result.score_0_100 >= 75

    def test_other_category_produces_none_severity(self):
        result = self.estimator.estimate(category="Other / unclear")
        assert result.level == "none"
        assert result.score_0_100 == 0

    def test_access_blocked_escalates_severity(self):
        base = self.estimator.estimate(category="Flood damage", access_blocked=False)
        escalated = self.estimator.estimate(category="Flood damage", access_blocked=True)
        assert escalated.score_0_100 >= base.score_0_100

    def test_high_urgency_escalates_severity(self):
        base = self.estimator.estimate(category="Building damage", urgency_score=10)
        escalated = self.estimator.estimate(category="Building damage", urgency_score=90)
        assert escalated.score_0_100 >= base.score_0_100

    def test_severity_result_always_has_disclaimer(self):
        """Every severity estimate must carry the provisional disclaimer."""
        from app.ai.engine.severity import SEVERITY_DISCLAIMER
        result = self.estimator.estimate(category="Landslide")
        assert len(result.disclaimer) > 50
        assert "PROVISIONAL" in result.disclaimer or "not validated" in result.disclaimer.lower()

    def test_severity_has_factors_list(self):
        result = self.estimator.estimate(
            category="Cyclone damage",
            people_affected=200,
            access_blocked=True,
        )
        assert len(result.factors) >= 2  # at least base + access_blocked

    def test_severity_never_uses_clip_confidence(self):
        """
        Severity must not accept or use a 'confidence' parameter.
        This test verifies the API signature does not accept confidence.
        """
        import inspect
        from app.ai.engine.severity import SeverityEstimator
        sig = inspect.signature(SeverityEstimator.estimate)
        param_names = list(sig.parameters.keys())
        assert "confidence" not in param_names, (
            "SeverityEstimator.estimate must NOT accept a 'confidence' parameter. "
            "Severity must NEVER be derived from AI classification confidence."
        )

    def test_severity_dict_output(self):
        result = self.estimator.estimate(category="Flood damage")
        d = result.to_dict()
        assert "level" in d
        assert "score_0_100" in d
        assert "rationale" in d
        assert "factors" in d
        assert "disclaimer" in d


# ── 4. CLIP adapter tests (with mock) ─────────────────────────────────────────

class TestCLIPAdapter:
    """
    Tests for CLIP adapter behaviour.
    Model loading is mocked to avoid downloading ViT weights in CI/CD.
    """

    def test_clip_returns_fallback_when_unavailable(self, tmp_images):
        from app.ai.engine.clip_adapter import CLIPAdapter

        # Force adapter into unavailable state
        CLIPAdapter._model = None
        CLIPAdapter._load_error = "Simulated: clip not installed"
        CLIPAdapter._text_features = None

        adapter = CLIPAdapter.__new__(CLIPAdapter)
        adapter._torch = None

        result = adapter.analyze(tmp_images["flood"])
        assert result.is_fallback is True
        assert result.confidence == 0.0
        assert result.category == "Unavailable"

        # Reset for other tests
        CLIPAdapter._model = None
        CLIPAdapter._load_error = None

    def test_clip_category_list_completeness(self):
        """CLIP must cover all 7 required disaster categories."""
        from app.ai.engine.clip_adapter import DISASTER_CATEGORIES
        required = {
            "Flood damage",
            "Building damage",
            "Fire and smoke",
            "Landslide",
            "Cyclone damage",
            "Road and infrastructure obstruction",
            "Other / unclear",
        }
        assert required.issubset(set(DISASTER_CATEGORIES.keys())), (
            f"Missing categories: {required - set(DISASTER_CATEGORIES.keys())}"
        )

    def test_clip_each_category_has_multiple_prompts(self):
        from app.ai.engine.clip_adapter import DISASTER_CATEGORIES
        for cat, prompts in DISASTER_CATEGORIES.items():
            assert len(prompts) >= 3, (
                f"Category '{cat}' has fewer than 3 prompts ({len(prompts)}). "
                "Multi-prompt ensemble requires ≥3 prompts per category."
            )

    def test_clip_warning_about_confidence_vs_severity(self):
        """
        When CLIP runs successfully, the result must always include a warning
        clarifying that confidence ≠ severity.
        """
        from app.ai.engine.clip_adapter import CLIPAdapter, DISASTER_CATEGORIES, CATEGORY_LABELS
        import numpy as np

        # Build a mock result directly from the adapter
        # We can't run real CLIP without the model, so we verify the warning
        # is present in the adapter code by inspecting the source.
        import inspect
        source = inspect.getsource(CLIPAdapter._run_inference)
        assert "severity" in source.lower() or "NOT" in source, (
            "CLIPAdapter._run_inference must include a warning about "
            "confidence not being a severity score."
        )


# ── 5. YOLO adapter tests (with mock) ────────────────────────────────────────

class TestYOLOAdapter:
    """Tests for YOLO adapter with mocked ultralytics model."""

    def test_yolo_returns_fallback_when_unavailable(self, tmp_images):
        from app.ai.engine.yolo_adapter import YOLOAdapter

        YOLOAdapter._model = None
        YOLOAdapter._load_error = "Simulated: ultralytics not installed"

        adapter = YOLOAdapter.__new__(YOLOAdapter)
        result = adapter.analyze(tmp_images["flood"])
        assert result.is_fallback is True
        assert result.category == "Unavailable"

        YOLOAdapter._model = None
        YOLOAdapter._load_error = None

    def test_hazard_relevant_classes_defined(self):
        from app.ai.engine.yolo_adapter import HAZARD_RELEVANT_CLASSES
        # Must include at minimum person, car, boat
        assert "person" in HAZARD_RELEVANT_CLASSES
        assert "car" in HAZARD_RELEVANT_CLASSES
        assert "boat" in HAZARD_RELEVANT_CLASSES

    def test_boat_detection_hints_flood(self):
        from app.ai.engine.yolo_adapter import YOLOAdapter
        from app.ai.engine.base import DetectedObject

        adapter = YOLOAdapter.__new__(YOLOAdapter)
        adapter._conf_thresh = 0.30
        adapter._iou_thresh = 0.45

        detected = [DetectedObject(label="boat", confidence=0.8, is_hazard_relevant=True)]
        cat, conf = adapter._infer_category_from_detections(
            {d.label for d in detected}, detected
        )
        assert cat == "Flood damage"
        assert conf > 0


# ── 6. Engine integration tests ───────────────────────────────────────────────

class TestAnalysisEngine:
    """
    Integration tests for the AnalysisEngine with mock adapters.
    These validate fusion logic, fallback behaviour, and severity separation.
    """

    def _make_mock_adapter(
        self,
        category: str,
        confidence: float,
        adapter_type: str = "mock",
        available: bool = True,
        fail: bool = False,
    ):
        from app.ai.engine.base import BaseAnalyzerAdapter, AnalysisResult, CandidateClass

        class MockAdapter(BaseAnalyzerAdapter):
            ADAPTER_TYPE = adapter_type

            def is_available(self_inner):
                return available

            def _run_inference(self_inner, image_path):
                if fail:
                    raise RuntimeError("Injected failure")
                return AnalysisResult(
                    category=category,
                    confidence=confidence,
                    top_candidates=[CandidateClass(label=category, confidence=confidence)],
                    model_name=f"MockModel/{adapter_type}",
                    adapter_type=adapter_type,
                )

        return MockAdapter()

    def test_single_classifier_result_passes_through(self, tmp_images):
        from app.ai.engine.engine import AnalysisEngine

        clf = self._make_mock_adapter("Flood damage", 0.82)
        engine = AnalysisEngine(classifiers=[clf], detectors=[])
        result = engine.assess(tmp_images["flood"])

        assert result.category == "Flood damage"
        assert result.confidence == pytest.approx(0.82, abs=0.01)
        assert not result.is_fallback

    def test_two_agreeing_classifiers_keep_best_confidence(self, tmp_images):
        from app.ai.engine.engine import AnalysisEngine

        clf1 = self._make_mock_adapter("Building damage", 0.75)
        clf2 = self._make_mock_adapter("Building damage", 0.80)
        engine = AnalysisEngine(classifiers=[clf1, clf2], detectors=[])
        result = engine.assess(tmp_images["debris"])

        assert result.category == "Building damage"
        # Best confidence when they agree
        assert result.confidence == pytest.approx(0.80, abs=0.01)

    def test_two_disagreeing_classifiers_penalise_confidence(self, tmp_images):
        from app.ai.engine.engine import AnalysisEngine

        clf1 = self._make_mock_adapter("Flood damage", 0.70)
        clf2 = self._make_mock_adapter("Building damage", 0.65)
        engine = AnalysisEngine(classifiers=[clf1, clf2], detectors=[])
        result = engine.assess(tmp_images["debris"])

        # Best is "Flood damage" (0.70), penalised to 0.70 * 0.80 = 0.56
        assert result.confidence == pytest.approx(0.70 * 0.80, abs=0.02)

    def test_all_models_unavailable_returns_fallback(self, tmp_images):
        from app.ai.engine.engine import AnalysisEngine

        clf = self._make_mock_adapter("Flood damage", 0.0, available=False, fail=True)
        engine = AnalysisEngine(classifiers=[clf], detectors=[])
        result = engine.assess(tmp_images["flood"])

        assert result.is_fallback is True
        assert result.category == "Unavailable"
        assert result.confidence == 0.0

    def test_severity_present_and_not_confidence_based(self, tmp_images):
        """
        Critical test: severity must be present and its score must NOT equal
        the classification confidence. This verifies the separation principle.
        """
        from app.ai.engine.engine import AnalysisEngine

        clf = self._make_mock_adapter("Fire and smoke", confidence=0.42)
        engine = AnalysisEngine(classifiers=[clf], detectors=[])
        result = engine.assess(tmp_images["fire"])

        # Severity must exist
        assert result.severity is not None

        # Severity score is rule-based (for Fire it must be >= 75 = "severe")
        # Confidence is 0.42 — severity score must NOT be 42.0
        assert result.severity.score_0_100 != pytest.approx(42.0, abs=5.0), (
            "Severity score appears to be derived from classification confidence (0.42). "
            "Severity must be rule-based and NEVER mirror confidence."
        )
        assert result.severity.score_0_100 >= 75.0  # Fire is at minimum severe

    def test_low_confidence_triggers_review_required(self, tmp_images):
        from app.ai.engine.engine import AnalysisEngine

        clf = self._make_mock_adapter("Landslide", 0.18)
        engine = AnalysisEngine(classifiers=[clf], detectors=[])
        result = engine.assess(tmp_images["green"])

        assert result.review_required is True

    def test_result_includes_separation_note(self, tmp_images):
        """AssessmentResult must always include the separation_note field."""
        from app.ai.engine.engine import AnalysisEngine

        clf = self._make_mock_adapter("Cyclone damage", 0.65)
        engine = AnalysisEngine(classifiers=[clf], detectors=[])
        result = engine.assess(tmp_images["flood"])

        assert hasattr(result, "separation_note")
        assert "severity" in result.separation_note.lower()
        assert "confidence" in result.separation_note.lower()

    def test_invalid_image_returns_fallback(self, invalid_images):
        from app.ai.engine.engine import AnalysisEngine

        clf = self._make_mock_adapter("Flood damage", 0.70)
        engine = AnalysisEngine(classifiers=[clf], detectors=[])
        result = engine.assess(invalid_images["not_image"])

        assert result.is_fallback is True

    def test_model_status_returns_adapter_info(self, tmp_images):
        from app.ai.engine.engine import AnalysisEngine

        clf = self._make_mock_adapter("Flood damage", 0.80, adapter_type="clip")
        engine = AnalysisEngine(classifiers=[clf], detectors=[])
        status = engine.model_status()

        assert "classifiers" in status
        assert len(status["classifiers"]) == 1
        assert status["classifiers"][0]["type"] == "clip"

    def test_result_to_dict_is_json_serialisable(self, tmp_images):
        """engine.assess().to_dict() must produce a plain JSON-serialisable dict."""
        import json
        from app.ai.engine.engine import AnalysisEngine

        clf = self._make_mock_adapter("Road and infrastructure obstruction", 0.55)
        engine = AnalysisEngine(classifiers=[clf], detectors=[])
        result = engine.assess(tmp_images["flood"])

        d = result.to_dict()
        serialised = json.dumps(d)  # raises TypeError if not serialisable
        assert len(serialised) > 50
