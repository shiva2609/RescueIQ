"""
Module B — Duplicate Detection Engine.

Identifies candidate duplicate incident reports using a multi-signal
composite approach:
  1. Image perceptual hash (dHash) — pixel-level similarity
  2. Image embedding cosine similarity (CLIP ViT features)
  3. Text description embedding cosine similarity (MiniLM)
  4. GPS proximity (Haversine distance)
  5. Temporal proximity (exponential decay window)

Design guarantees:
  - Reports are NEVER automatically deleted or merged
  - Every candidate pair is flagged for human review
  - Grouping is purely for display convenience
  - All outputs carry PRELIMINARY DECISION-SUPPORT ASSESSMENT label
  - Composite scores are explained via breakdown per signal

Thresholds (tunable via DuplicateEngine constructor):
  LIKELY_DUPLICATE   : composite ≥ 0.75
  POSSIBLE_DUPLICATE : composite ≥ 0.50
  UNCERTAIN          : composite < 0.50
"""
from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.triage.models import (
    DuplicateCandidate,
    DuplicateSimilarityBreakdown,
    DuplicateGroup,
    PRELIMINARY_LABEL,
)


# ── Hamming distance helpers ───────────────────────────────────────────────────

def _hamming_bits(a: int, b: int) -> int:
    """Count set bits in XOR (popcount)."""
    return bin(a ^ b).count("1")


def dhash_similarity(hash_a: Optional[int], hash_b: Optional[int],
                     hash_bits: int = 64) -> float:
    """
    Convert dHash Hamming distance to [0,1] similarity.
    Returns 0.0 if either hash is None (no image available).
    """
    if hash_a is None or hash_b is None:
        return 0.0
    distance = _hamming_bits(hash_a, hash_b)
    return max(0.0, 1.0 - distance / hash_bits)


# ── Cosine similarity ──────────────────────────────────────────────────────────

def _cosine(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def embedding_similarity(emb_a: Optional[str], emb_b: Optional[str]) -> float:
    """Parse JSON embedding strings and compute cosine similarity."""
    if not emb_a or not emb_b:
        return 0.0
    try:
        va: List[float] = json.loads(emb_a)
        vb: List[float] = json.loads(emb_b)
        return _cosine(va, vb)
    except (json.JSONDecodeError, TypeError):
        return 0.0


# ── Geo proximity ──────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def geo_proximity_score(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    radius_km: float = 0.5,
) -> float:
    """
    Return [0,1] proximity score. 1.0 at same location, 0.0 at ≥ radius_km.
    Uses exponential decay: score = exp(-d / (radius_km/3))
    """
    dist = haversine_km(lat1, lon1, lat2, lon2)
    if dist >= radius_km:
        return 0.0
    return math.exp(-dist / (radius_km / 3.0))


# ── Temporal proximity ─────────────────────────────────────────────────────────

def temporal_proximity_score(ts_a: datetime, ts_b: datetime,
                              window_hours: float = 4.0) -> float:
    """
    Return [0,1] proximity. 1.0 for simultaneous, decays to 0 at window_hours.
    Uses exponential decay: score = exp(-Δh / (window_hours/3))
    """
    diff_h = abs((ts_b - ts_a).total_seconds()) / 3600.0
    if diff_h >= window_hours:
        return 0.0
    return math.exp(-diff_h / (window_hours / 3.0))


# ── Composite score ────────────────────────────────────────────────────────────

class DuplicateEngine:
    """
    Module B: Multi-signal duplicate detection engine.

    Never automatically removes or merges incidents.
    All candidates are returned for human review.

    Weights for composite score (must sum to 1.0):
        w_hash        : image perceptual hash similarity
        w_embedding   : image embedding cosine similarity
        w_text        : description text embedding similarity
        w_geo         : geo proximity
        w_time        : temporal proximity

    Thresholds:
        likely_threshold   : composite ≥ this → "Likely duplicate"
        possible_threshold : composite ≥ this → "Possible duplicate"
    """

    def __init__(
        self,
        w_hash: float = 0.25,
        w_embedding: float = 0.25,
        w_text: float = 0.20,
        w_geo: float = 0.20,
        w_time: float = 0.10,
        geo_radius_km: float = 0.5,
        time_window_hours: float = 4.0,
        likely_threshold: float = 0.72,
        possible_threshold: float = 0.45,
    ):
        total = w_hash + w_embedding + w_text + w_geo + w_time
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Duplicate engine weights must sum to 1.0, got {total:.4f}")
        self.w_hash = w_hash
        self.w_embedding = w_embedding
        self.w_text = w_text
        self.w_geo = w_geo
        self.w_time = w_time
        self.geo_radius_km = geo_radius_km
        self.time_window_hours = time_window_hours
        self.likely_threshold = likely_threshold
        self.possible_threshold = possible_threshold

    def compare(
        self,
        inc_a: Dict[str, Any],
        inc_b: Dict[str, Any],
    ) -> Optional[DuplicateCandidate]:
        """
        Compare two incident dicts and return a DuplicateCandidate if above
        the possible_threshold, or None if clearly not a duplicate.

        Expected dict keys:
            id, image_hash, text_embedding_json, description,
            latitude, longitude, created_at (datetime)
        """
        # ── Per-signal scores ─────────────────────────────────────────────────
        hash_sim = dhash_similarity(
            inc_a.get("image_hash"), inc_b.get("image_hash")
        )
        img_emb_sim = embedding_similarity(
            inc_a.get("image_embedding_json"),
            inc_b.get("image_embedding_json"),
        )
        txt_sim = embedding_similarity(
            inc_a.get("text_embedding_json"),
            inc_b.get("text_embedding_json"),
        )
        geo_score = geo_proximity_score(
            inc_a.get("latitude", 0.0), inc_a.get("longitude", 0.0),
            inc_b.get("latitude", 0.0), inc_b.get("longitude", 0.0),
            radius_km=self.geo_radius_km,
        )
        ts_a = inc_a.get("created_at") or datetime.now(timezone.utc)
        ts_b = inc_b.get("created_at") or datetime.now(timezone.utc)
        if isinstance(ts_a, str):
            ts_a = datetime.fromisoformat(ts_a)
        if isinstance(ts_b, str):
            ts_b = datetime.fromisoformat(ts_b)
        time_score = temporal_proximity_score(ts_a, ts_b, window_hours=self.time_window_hours)

        composite = (
            self.w_hash * hash_sim
            + self.w_embedding * img_emb_sim
            + self.w_text * txt_sim
            + self.w_geo * geo_score
            + self.w_time * time_score
        )

        if composite < self.possible_threshold:
            return None

        if composite >= self.likely_threshold:
            recommendation = "Likely duplicate"
        elif composite >= self.possible_threshold:
            recommendation = "Possible duplicate"
        else:
            recommendation = "Uncertain"

        breakdown = DuplicateSimilarityBreakdown(
            image_hash_similarity=hash_sim,
            image_embedding_similarity=img_emb_sim,
            text_similarity=txt_sim,
            geo_proximity_score=geo_score,
            time_proximity_score=time_score,
            composite_score=composite,
        )

        return DuplicateCandidate(
            incident_id_a=str(inc_a["id"]),
            incident_id_b=str(inc_b["id"]),
            similarity=breakdown,
            recommendation=recommendation,
            human_review_required=True,   # NEVER auto-deleted
        )

    def detect_all(
        self, incidents: List[Dict[str, Any]]
    ) -> Tuple[List[DuplicateCandidate], List[DuplicateGroup]]:
        """
        Run pairwise duplicate detection over a list of incidents.

        Returns:
            candidates : all pairs above possible_threshold
            groups     : clusters of potentially related incidents
        """
        candidates: List[DuplicateCandidate] = []
        n = len(incidents)

        for i in range(n):
            for j in range(i + 1, n):
                cand = self.compare(incidents[i], incidents[j])
                if cand is not None:
                    candidates.append(cand)

        groups = self._cluster(incidents, candidates)
        return candidates, groups

    def _cluster(
        self,
        incidents: List[Dict[str, Any]],
        candidates: List[DuplicateCandidate],
    ) -> List[DuplicateGroup]:
        """
        Union-Find clustering of incidents linked by duplicate candidates.
        Groups are for display only — no data is deleted or merged.
        """
        parent: Dict[str, str] = {str(inc["id"]): str(inc["id"]) for inc in incidents}
        max_score: Dict[str, float] = {}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str, score: float) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
            root = find(x)
            max_score[root] = max(max_score.get(root, 0.0), score)

        for cand in candidates:
            union(cand.incident_id_a, cand.incident_id_b,
                  cand.similarity.composite_score)

        # Collect groups with ≥2 members
        cluster_members: Dict[str, List[str]] = {}
        for inc_id in parent:
            root = find(inc_id)
            cluster_members.setdefault(root, []).append(inc_id)

        id_to_priority: Dict[str, float] = {
            str(inc["id"]): float(inc.get("priority_score", 0.0))
            for inc in incidents
        }

        groups: List[DuplicateGroup] = []
        for root, members in cluster_members.items():
            if len(members) < 2:
                continue
            primary = max(members, key=lambda x: id_to_priority.get(x, 0.0))
            groups.append(DuplicateGroup(
                group_id=str(uuid.uuid4()),
                member_ids=members,
                primary_id=primary,
                max_composite_score=max_score.get(root, 0.0),
                group_label=(
                    f"{len(members)} reports may describe the same event — human review required"
                ),
            ))

        return groups
