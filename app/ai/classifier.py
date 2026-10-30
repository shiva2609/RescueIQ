"""
Builds and initializes the EfficientNet-B0 3-class damage classifier for RescueIQ.
Outputs:
  Class 0: Little-or-none (score 10)
  Class 1: Mild damage    (score 50)
  Class 2: Severe damage  (score 90)

Includes an uncertainty / abstention check based on max confidence threshold (tau).
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any

CLASSES = ["Little-or-none", "Mild damage", "Severe damage"]
SEVERITY_SCORES = {
    "Little-or-none": 15.0,
    "Mild damage": 55.0,
    "Severe damage": 95.0,
    "Abstained / Uncertain": 0.0
}

# Standard EfficientNet-B0 preprocessing
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_model(checkpoint_path: Path = None, num_classes: int = 3) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    # Replace head for 3 classes
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    if checkpoint_path and checkpoint_path.exists():
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state)
        except Exception as e:
            print(f"Warning loading checkpoint: {e}")
    model.eval()
    return model

class DamageClassifier:
    def __init__(self, checkpoint_path: Path = None, abstention_threshold: float = 0.50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(checkpoint_path).to(self.device)
        self.abstention_threshold = abstention_threshold
        self.classes = CLASSES

    def predict(self, image_path: Path) -> Dict[str, Any]:
        """
        Runs inference on image.
        Returns:
            damage_class: str ("Severe damage", "Mild damage", "Little-or-none", or "Abstained / Uncertain")
            confidence: float (0.0 to 1.0)
            is_abstained: bool
            probabilities: dict of class -> float
            severity_score: float (0 to 100)
            model_info: str
        """
        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                tensor = TRANSFORMS(img_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
            best_idx = int(probs.argmax())
            best_prob = float(probs[best_idx])
            best_class = self.classes[best_idx]

            # Abstention rule: if maximum probability is below threshold, flag as uncertain
            is_abstained = best_prob < self.abstention_threshold
            if is_abstained:
                final_class = "Abstained / Uncertain"
                severity_score = 0.0
            else:
                final_class = best_class
                severity_score = SEVERITY_SCORES[final_class]

            return {
                "damage_class": final_class,
                "confidence": round(best_prob, 3),
                "is_abstained": is_abstained,
                "probabilities": prob_dict,
                "severity_score": severity_score,
                "model_name": f"EfficientNet-B0 (3-class triage, tau={self.abstention_threshold})"
            }
        except Exception as e:
            return {
                "damage_class": "Abstained / Uncertain",
                "confidence": 0.0,
                "is_abstained": True,
                "probabilities": {c: 0.0 for c in self.classes},
                "severity_score": 0.0,
                "model_name": f"Inference fallback: {str(e)[:100]}"
            }
