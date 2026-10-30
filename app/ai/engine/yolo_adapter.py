"""
YOLOv8 object detection adapter for disaster scene analysis.

Uses Ultralytics YOLOv8 to identify hazard-relevant objects in disaster images.
Detection results enrich (not replace) category classification — they provide
grounding evidence for identified risks such as:
  - Submerged/damaged vehicles → supports Flood/Building damage
  - People trapped / in danger → urgency signal (NOT severity score)
  - Fire / smoke → supports Fire and smoke category
  - Trees / debris blocking roads → supports Road obstruction
  - Structural elements / rubble → supports Building damage

─────────────────────────────────────────────────────────────────────────────
YOLO Limitations (documented):
- Standard COCO-80 YOLOv8 does NOT have dedicated disaster-specific classes.
  It detects generic objects (person, car, truck, fire hydrant, etc.) which
  can be contextually interpreted for hazard assessment.
- For fire/smoke detection, a specialised fire-detection YOLO model or the
  YOLOv8-fire checkpoint should be preferred.
- Confidence in object detections should NOT be aggregated into a severity
  score without domain expertise and ground-truth validation.
─────────────────────────────────────────────────────────────────────────────

Fine-tuned model support:
  Set YOLO_CHECKPOINT_PATH env var to point to a custom .pt file.
  Recommended training datasets for disaster detection:
    - VisDrone (drone/aerial object detection)
    - HERIDAL (helicopter rescue image dataset)
    - D-Fire (fire & smoke detection dataset)
    - AIDER (multi-class aerial disaster imagery)
  See MODEL_LIMITATIONS.md for complete training specifications.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from app.ai.engine.base import (
    AnalysisResult,
    BaseAnalyzerAdapter,
    CandidateClass,
    DetectedObject,
)
from app.ai.engine.preprocessing import load_and_validate_image, safe_thumbnail


# ── Hazard-relevant object mapping ────────────────────────────────────────────
# COCO class names that are contextually relevant to disaster assessment.
# This mapping is conservative — only include labels with clear disaster relevance.

HAZARD_RELEVANT_CLASSES: Dict[str, str] = {
    # People at risk
    "person":              "Person (potential casualty or trapped individual)",
    # Vehicles
    "car":                 "Vehicle (potentially submerged or displaced)",
    "truck":               "Heavy vehicle (potentially obstructing or displaced)",
    "bus":                 "Bus (potentially displaced or blocking route)",
    "motorcycle":          "Motorcycle (potentially displaced)",
    "boat":                "Boat (indicates flooding or water incident)",
    # Infrastructure
    "traffic light":       "Traffic infrastructure (status indicator)",
    "fire hydrant":        "Fire hydrant (water access or flooding indicator)",
    # Structures
    "bench":               "Street furniture (displacement indicator)",
    "chair":               "Interior furnishing (exterior presence = structural damage)",
    # Animals (may indicate evacuation failure or rural damage)
    "cow":                 "Livestock (potential disaster displacement)",
    "horse":               "Livestock (potential disaster displacement)",
    "dog":                 "Animal (may indicate evacuation area)",
    # Aeronautical context
    "airplane":            "Aircraft (disaster response asset)",
    "helicopter":          "Helicopter (potential rescue operation)",
}

# Classes that map to a likely category
DETECTION_CATEGORY_HINTS: Dict[str, str] = {
    "boat":      "Flood damage",
    "person":    None,  # context-dependent
    "car":       None,  # context-dependent
}


class YOLOAdapter(BaseAnalyzerAdapter):
    """
    YOLOv8 object detection adapter.

    Detection results supplement category classification — they do NOT
    independently assign a category. The engine combines YOLO detections
    with CLIP category scores to produce a grounded final result.

    Configuration (env vars):
      YOLO_MODEL_NAME      : e.g. "yolov8n.pt" (nano, default), "yolov8m.pt"
      YOLO_CHECKPOINT_PATH : Path to a custom/fine-tuned .pt checkpoint
      YOLO_CONFIDENCE      : Detection confidence threshold (default 0.30)
      YOLO_IOU             : NMS IoU threshold (default 0.45)
    """

    ADAPTER_TYPE = "yolo"

    _model = None
    _model_name_str = "unloaded"
    _load_error: Optional[str] = None

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        checkpoint_path: Optional[Path] = None,
        confidence_threshold: float = 0.30,
        iou_threshold: float = 0.45,
    ):
        self._model_name_key = os.environ.get("YOLO_MODEL_NAME", model_name)
        self._checkpoint_path = checkpoint_path or (
            Path(os.environ["YOLO_CHECKPOINT_PATH"])
            if "YOLO_CHECKPOINT_PATH" in os.environ
            else None
        )
        self._conf_thresh = float(os.environ.get("YOLO_CONFIDENCE", str(confidence_threshold)))
        self._iou_thresh = float(os.environ.get("YOLO_IOU", str(iou_threshold)))
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if YOLOAdapter._model is not None or YOLOAdapter._load_error:
            return
        try:
            from ultralytics import YOLO

            checkpoint = self._checkpoint_path or self._model_name_key
            YOLOAdapter._model = YOLO(str(checkpoint))
            YOLOAdapter._model_name_str = (
                f"YOLOv8 ({self._model_name_key}, COCO-80)"
            )
        except Exception as exc:
            YOLOAdapter._load_error = str(exc)

    def is_available(self) -> bool:
        return YOLOAdapter._model is not None and YOLOAdapter._load_error is None

    def _run_inference(self, image_path: Path) -> AnalysisResult:
        if not self.is_available():
            reason = YOLOAdapter._load_error or "YOLOv8 model not loaded"
            return self._fallback(reason)

        # 1. Load and validate image
        pil_img = load_and_validate_image(image_path)
        pil_img = safe_thumbnail(pil_img)

        # 2. Run detection
        results = YOLOAdapter._model.predict(
            source=pil_img,
            conf=self._conf_thresh,
            iou=self._iou_thresh,
            verbose=False,
        )

        # 3. Parse detections
        detected: List[DetectedObject] = []
        class_names = YOLOAdapter._model.names  # dict: int → str

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_idx = int(box.cls[0].item())
                label = class_names.get(cls_idx, f"class_{cls_idx}")
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist() if box.xyxy is not None else None
                is_hazard = label in HAZARD_RELEVANT_CLASSES

                detected.append(DetectedObject(
                    label=label,
                    confidence=conf,
                    bbox_xyxy=xyxy,
                    is_hazard_relevant=is_hazard,
                ))

        # 4. Derive a rough category hint from detection context
        hazard_labels: Set[str] = {d.label for d in detected if d.is_hazard_relevant}
        category, cat_conf = self._infer_category_from_detections(hazard_labels, detected)

        warnings = [
            "YOLO detections are based on COCO-80 generic classes, "
            "not disaster-specific training. Interpret contextually.",
            "Detection confidence is NOT a damage severity score.",
        ]

        if not detected:
            warnings.append(
                "No objects detected above confidence threshold. "
                "Image may be empty, occluded, or require a specialised model."
            )

        return AnalysisResult(
            category=category,
            confidence=cat_conf,
            top_candidates=[CandidateClass(label=category, confidence=cat_conf)],
            detected_objects=detected,
            model_name=YOLOAdapter._model_name_str,
            adapter_type=self.ADAPTER_TYPE,
            review_required=cat_conf < 0.30 or not detected,
            warnings=warnings,
        )

    # ── Category heuristic from detections ───────────────────────────────

    def _infer_category_from_detections(
        self, hazard_labels: Set[str], detections: List[DetectedObject]
    ) -> tuple:
        """
        Lightly infer a probable category from detected object context.

        This is a conservative heuristic — without a disaster-specific
        detector we cannot make confident category claims from detections alone.
        Returns ("Other / unclear", 0.0) when evidence is insufficient.
        """
        if "boat" in hazard_labels:
            return "Flood damage", 0.45

        high_conf = [d for d in detections if d.confidence >= 0.70]
        if len(high_conf) == 0 and not hazard_labels:
            return "Other / unclear", 0.10

        # Many displaced / chaotic objects → possible disaster scene
        if len(detections) >= 5:
            return "Other / unclear", 0.25

        return "Other / unclear", 0.15
