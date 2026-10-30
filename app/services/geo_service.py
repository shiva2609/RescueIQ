"""
Geographic queries service.

Provides unified radius / proximity queries that work seamlessly on both:
1. PostgreSQL + PostGIS (using ST_DWithin / ST_Distance when PostGIS is active)
2. SQLite (using spherical Haversine formula filter in Python / bounding box)
"""
from __future__ import annotations

import math
from typing import List, Tuple
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from app.config import settings
from app.models.incident import Incident


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + (
        math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def get_bounding_box(lat: float, lon: float, radius_meters: float) -> Tuple[float, float, float, float]:
    """Calculate min_lat, max_lat, min_lon, max_lon bounding box for spatial index filtering."""
    # 1 deg latitude ~ 111,000 meters
    delta_lat = radius_meters / 111000.0
    # 1 deg longitude ~ 111,000 * cos(lat) meters
    rad_lat = math.radians(lat)
    cos_lat = math.cos(rad_lat) if abs(math.cos(rad_lat)) > 1e-6 else 1e-6
    delta_lon = radius_meters / (111000.0 * cos_lat)

    return (
        lat - delta_lat,
        lat + delta_lat,
        lon - delta_lon,
        lon + delta_lon,
    )


def find_incidents_nearby(
    db: Session,
    latitude: float,
    longitude: float,
    radius_meters: float = 5000.0,
    limit: int = 50,
) -> List[Tuple[Incident, float]]:
    """
    Find non-deleted incidents within radius_meters of (latitude, longitude).
    Returns list of (Incident, distance_in_meters), sorted by distance ascending.

    PostGIS compatibility:
    If connected to PostgreSQL with PostGIS enabled, this can execute native
    `ST_DWithin(ST_MakePoint(longitude, latitude)::geography, ST_MakePoint(:lon, :lat)::geography, :radius)`.
    For SQLite and generic SQL, it utilizes a bounding box pre-filter followed by exact Haversine calculation.
    """
    # Check if database is PostgreSQL
    if not settings.is_sqlite:
        try:
            from sqlalchemy import text
            # PostGIS query attempt
            sql = text(
                """
                SELECT id,
                       ST_Distance(
                           ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography,
                           ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
                       ) AS distance_meters
                FROM incidents
                WHERE incident_status != 'Deleted'
                  AND ST_DWithin(
                      ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography,
                      ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                      :radius
                  )
                ORDER BY distance_meters ASC
                LIMIT :limit;
                """
            )
            rows = db.execute(
                sql,
                {"lat": latitude, "lon": longitude, "radius": radius_meters, "limit": limit},
            ).fetchall()

            if rows:
                incident_ids = [r[0] for r in rows]
                dist_map = {r[0]: float(r[1]) for r in rows}
                incidents = db.execute(
                    select(Incident).where(Incident.id.in_(incident_ids))
                ).scalars().all()
                # Maintain distance sort order
                inc_map = {inc.id: inc for inc in incidents}
                return [(inc_map[i_id], dist_map[i_id]) for i_id in incident_ids if i_id in inc_map]
        except Exception:
            # Fall back to Haversine bounding-box method if PostGIS extension is not installed
            db.rollback()

    # Generic & SQLite-compatible bounding box search
    min_lat, max_lat, min_lon, max_lon = get_bounding_box(latitude, longitude, radius_meters)

    candidates = db.execute(
        select(Incident).where(
            and_(
                Incident.incident_status != "Deleted",
                Incident.latitude >= min_lat,
                Incident.latitude <= max_lat,
                Incident.longitude >= min_lon,
                Incident.longitude <= max_lon,
            )
        )
    ).scalars().all()

    results: List[Tuple[Incident, float]] = []
    for inc in candidates:
        dist = haversine_distance_m(latitude, longitude, inc.latitude, inc.longitude)
        if dist <= radius_meters:
            results.append((inc, round(dist, 1)))

    results.sort(key=lambda x: x[1])
    return results[:limit]
