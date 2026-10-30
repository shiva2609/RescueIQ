# AI Model Limitations and Dataset Requirements

> **Version**: 1.0 — Damage Mapper / RescueIQ AI Assessment Engine  
> **Status**: Active

---

## 1. Overview

The Damage Mapper assessment engine uses two AI models in a pipeline:

| Model | Role | Type | Default |
|-------|------|------|---------|
| CLIP ViT-B/32 | Disaster category classification | Zero-shot (no fine-tuning) | OpenAI pretrained |
| YOLOv8-nano | Object detection (hazard identification) | Pretrained COCO-80 | Ultralytics pretrained |

Both models are used **without disaster-specific fine-tuning by default**. This section documents known limitations, failure modes, and the requirements for fine-tuning.

---

## 2. CLIP — Known Limitations

### 2.1 What CLIP can and cannot do

CLIP ViT-B/32 was trained on 400 million image-text pairs from the web. It is strong at matching visual scenes to natural language descriptions. However:

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Confidence ≠ severity** | A clear, well-lit mild flood image can score higher confidence than a blurry catastrophic one | Severity is computed by rule-based estimator only |
| **No spatial reasoning** | Cannot assess depth of water, percentage of a building collapsed, or fire spread | Use YOLO detections as proxy evidence |
| **Distribution shift** | Trained on internet photos; performs worse on UAV/drone imagery, thermal IR, or night-vision | Fine-tune on domain-specific datasets |
| **Prompt sensitivity** | Results vary with prompt wording; our multi-prompt ensemble reduces but does not eliminate this | Expand prompt library; evaluate on held-out sets |
| **Similar visual domains** | Flooded streets vs heavy rain, smoke vs fog, landslide vs construction | Requires disambiguation through context or metadata |
| **Low-resolution images** | Accuracy drops on images smaller than 224×224 or heavily compressed | Preprocessing pipeline enforces minimum 32px; consider upscaling |
| **Text/graphic overlays** | Watermarks and text can bias classification | Preprocessing should strip EXIF data; cropping may help |
| **Single-image limitation** | No temporal context (does not compare before/after satellite imagery) | Multi-temporal analysis requires dedicated model architecture |

### 2.2 Confidence threshold guidance

| Confidence | Tier | Recommendation |
|-----------|------|----------------|
| ≥ 0.75 | High | Category can be used with operator review |
| 0.50–0.75 | Medium | Treat as probable; field confirmation advisable |
| 0.30–0.50 | Low | Treat as a signal only; manual review required |
| < 0.30 | Review Required | Do not use for any decision; flag for expert review |

> **Warning**: These thresholds are heuristic and have not been calibrated against ground-truth disaster assessment data. They must be validated before use in operational emergency response.

---

## 3. YOLOv8 — Known Limitations

### 3.1 What YOLOv8 detects by default

The default YOLOv8 model is trained on COCO-80 (80 generic object categories). It does **not** include:

- ❌ Fire / smoke (use D-Fire specialised model)
- ❌ Structural damage or rubble
- ❌ Floodwater levels
- ❌ Landslide debris
- ❌ Downed power lines
- ❌ Specific infrastructure damage types

It **can** detect:
- ✅ People (potential casualties or trapped individuals)
- ✅ Vehicles (potentially submerged, displaced, or blocking routes)
- ✅ Boats (strong flood indicator)
- ✅ Airplanes / helicopters (disaster response assets)

### 3.2 False positive / negative risk

| Scenario | Risk | Example |
|----------|------|---------|
| Night or low-light imagery | High false negative rate | Missing submerged vehicles |
| Aerial / drone views | Reduced detection at altitude | People appear too small |
| Smoke / dust obscured scene | High false negative | Objects hidden behind smoke |
| Dense debris fields | High false positive | Random shapes detected as objects |

---

## 4. Severity Estimation — Critical Limitations

> **This section is mandatory reading before using severity scores in any decision context.**

The `SeverityEstimator` produces **rule-based provisional estimates**. It is:

- **NOT calibrated** against historical disaster events
- **NOT validated** by trained damage assessors
- **NOT suitable** for official resource allocation without field verification
- **NOT a replacement** for FEMA, Copernicus EMS, or UN OCHA damage grading

### Severity is explicitly NOT derived from:
- CLIP classification confidence
- YOLO detection confidence scores
- Any neural network output probabilities

### Severity IS derived from:
- Disaster category taxonomy (conservative base level)
- Reporter-supplied urgency and people affected
- Emergency access status
- Presence of persons detected by YOLO (escalation factor)
- Category-specific overrides (fire always minimum "severe")

---

## 5. Fine-Tuning Requirements

### 5.1 CLIP Fine-Tuning

To improve accuracy for disaster-specific imagery:

**Recommended Datasets**:

| Dataset | Categories | Size | Access |
|---------|-----------|------|--------|
| [xBD (xView2)](https://xview2.org) | Building damage (4 severity levels) | 850k+ buildings, 22k images | Public (registration) |
| [FloodNet](https://github.com/BinaLab/FloodNet-Supervised_v1.0) | Flood damage, road damage, vehicles | 3200 UAV images | Public |
| [AIDER](https://zenodo.org/record/3888300) | Collapsed building, fire, flood, traffic | 500 images per class | Public |
| [LADI](https://github.com/LADI-Dataset/ladi-dataset) | Multi-class aerial disaster | 6.2k images | Public |
| [D-Fire](https://github.com/gaiasd/DFireDataset) | Fire and smoke only | 21k images | Public |
| [HERIDAL](https://ieeexplore.ieee.org/document/9011036) | Helicopter rescue scenes | 1500 images | IEEE |

**Training Protocol**:
```python
# Contrastive fine-tuning using disaster image-text pairs
# Example with OpenCLIP (compatible with our adapter)
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

# Fine-tune using contrastive loss on (disaster_image, category_text) pairs
# Recommended hyperparameters (starting point, not validated):
#   Learning rate: 1e-6 to 1e-5
#   Batch size: 64–256
#   Epochs: 10–30
#   Augmentation: RandomHorizontalFlip, ColorJitter, RandomCrop
#   Regularisation: Weight decay 0.01, label smoothing 0.1
```

**Checkpoint Registration**:
```bash
# Set env var to use fine-tuned weights
export CLIP_CHECKPOINT_PATH="/path/to/disaster_clip_finetuned.pt"
export CLIP_MODEL_NAME="ViT-B/32"  # or ViT-L/14 for larger model
```

### 5.2 YOLO Fine-Tuning

**Recommended Datasets**:

| Dataset | Categories | Images |
|---------|-----------|--------|
| [VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset) | People, vehicles at altitude | 10k+ |
| [D-Fire](https://github.com/gaiasd/DFireDataset) | Fire, smoke | 21k |
| [RescueNet](https://github.com/BinaLab/RescueNet-Challenge) | Flooding, damage from drone | 4k |

**Training Command (Ultralytics)**:
```bash
yolo detect train \
  data=disaster_dataset.yaml \
  model=yolov8m.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  augment=True \
  name=disaster_yolov8m
```

**dataset.yaml example**:
```yaml
path: /data/disaster_detection
train: images/train
val: images/val
nc: 12
names: ['person', 'vehicle', 'boat', 'fire', 'smoke', 'debris',
        'collapsed_building', 'floodwater', 'downed_powerline',
        'blocked_road', 'rescue_vehicle', 'helicopter']
```

---

## 6. Evaluation Metrics

Before deploying fine-tuned models in operational contexts:

| Metric | Minimum Target | Notes |
|--------|---------------|-------|
| CLIP Top-1 Accuracy | ≥ 70% | Evaluated on held-out disaster images, per category |
| CLIP Category F1 | ≥ 0.65 per class | Especially "Other / unclear" precision |
| YOLO mAP@0.5 | ≥ 0.50 | On disaster-relevant classes |
| False Negative Rate (casualties) | < 10% | Person detection is safety-critical |
| Inference latency | < 2s (GPU) / < 10s (CPU) | Per image at 640px |

---

## 7. Bias and Fairness Considerations

- **Geographic bias**: CLIP training data over-represents North American and European disaster imagery. Performance may degrade on South/Southeast Asian, African, or Pacific Island disaster types.
- **Building type bias**: Models may perform poorly on informal settlements, traditional construction, and high-density urban slums not well-represented in training data.
- **Resolution bias**: Aerial UAV imagery (high altitude, small objects) requires domain-specific training; web-scraped training images are typically ground-level photography.
- **Seasonal/lighting bias**: Nighttime, overcast, and post-storm lighting conditions are underrepresented in most public datasets.

---

## 8. Prohibited Uses

> The following uses of model outputs are explicitly prohibited without additional validation:

1. **Direct resource allocation** based on severity scores alone
2. **Casualty count estimation** from YOLO person detection
3. **Insurance or legal damage assessment** using classification confidence
4. **Automated dispatching** without operator confirmation
5. **Press or public-facing severity claims** based solely on model output

---

## 9. Recommended Validation Process

Before operational deployment:

1. **Collect a validation set** of ≥ 200 labelled disaster images per category from your target geography.
2. **Measure per-category accuracy** and calibration (reliability diagrams).
3. **Identify failure modes** using confusion matrices and grad-CAM visualisation.
4. **Calibrate thresholds** using Platt scaling or temperature scaling.
5. **Perform adversarial testing** with edge cases: night images, smoke-filled, low-resolution, partial images.
6. **Obtain sign-off** from domain experts (disaster response coordinators, structural engineers, hydrologists) before using severity estimates for dispatching.

---

*Document maintained by the RescueIQ Engineering Team. Review quarterly or after any model update.*
