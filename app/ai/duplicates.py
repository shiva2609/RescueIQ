"""
Multi-modal duplicate detector for RescueIQ:
Combines:
1. Description text semantic similarity (via sentence-transformers all-MiniLM-L6-v2)
2. Image perceptual difference hash (dHash 64-bit normalized distance)
3. Geospatial distance (Haversine formula in meters)
4. Temporal window (hours difference)
5. Category match

Weighted formula:
  Score_dup = 0.40 * text_sim + 0.30 * image_sim + 0.20 * geo_proximity + 0.10 * time_proximity
"""
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

_text_model = None

def get_text_model():
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer
        _text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _text_model

def compute_text_embedding(text: str):
    if not text or not text.strip():
        return None
    try:
        model = get_text_model()
        emb = model.encode(text, convert_to_numpy=True)
        return emb
    except Exception as e:
        print(f"Error computing text embedding: {e}")
        return None

def cosine_similarity(vec1, vec2) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    import numpy as np
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(max(0.0, min(1.0, dot / (norm1 * norm2))))

def compute_dhash(image_path: Path) -> int:
    with Image.open(image_path) as im:
        pixels = list(im.convert("L").resize((9, 8)).getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y*9+x] > pixels[y*9+x+1])
    return bits

def image_hash_similarity(hash1: int, hash2: int) -> float:
    # 64-bit Hamming distance converted to similarity [0, 1]
    # bin().count('1') for full compatibility across Python 3.9+
    dist = bin(hash1 ^ hash2).count('1')
    # 0 distance = 1.0, 16+ distance = 0.0
    sim = max(0.0, 1.0 - (dist / 16.0))
    return round(sim, 3)

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * r * math.asin(math.sqrt(a))

def geo_proximity_score(dist_m: float, max_dist_m: float = 300.0) -> float:
    # Within 50m: 1.0; 300m+: 0.0
    if dist_m <= 50.0:
        return 1.0
    if dist_m >= max_dist_m:
        return 0.0
    return round(1.0 - ((dist_m - 50.0) / (max_dist_m - 50.0)), 3)

def temporal_proximity_score(dt1_iso: str, dt2_iso: str, max_hours: float = 48.0) -> float:
    try:
        t1 = datetime.fromisoformat(dt1_iso)
        t2 = datetime.fromisoformat(dt2_iso)
        diff_hours = abs((t1 - t2).total_seconds()) / 3600.0
        if diff_hours >= max_hours:
            return 0.0
        return round(1.0 - (diff_hours / max_hours), 3)
    except Exception:
        return 0.5

def calculate_multimodal_duplicate_score(
    text_emb1, text_emb2,
    hash1: int, hash2: int,
    lat1: float, lon1: float, lat2: float, lon2: float,
    created1: str, created2: str
) -> Dict[str, Any]:
    text_sim = cosine_similarity(text_emb1, text_emb2)
    image_sim = image_hash_similarity(hash1, hash2)
    dist_m = haversine_m(lat1, lon1, lat2, lon2)
    geo_sim = geo_proximity_score(dist_m)
    time_sim = temporal_proximity_score(created1, created2)

    # Weighted combination
    composite = (0.40 * text_sim) + (0.30 * image_sim) + (0.20 * geo_sim) + (0.10 * time_sim)
    composite_100 = round(composite * 100.0, 1)

    # Flag as duplicate candidate if composite >= 65.0 OR (image_sim >= 0.85 and geo_sim >= 0.8)
    is_candidate = (composite_100 >= 60.0) or (image_sim >= 0.85 and dist_m <= 150.0)

    return {
        "is_candidate": is_candidate,
        "composite_score": composite_100,
        "text_similarity": round(text_sim * 100.0, 1),
        "image_similarity": round(image_sim * 100.0, 1),
        "distance_meters": round(dist_m, 1),
        "geo_proximity": round(geo_sim * 100.0, 1),
        "time_proximity": round(time_sim * 100.0, 1)
    }
