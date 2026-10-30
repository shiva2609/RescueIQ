"""Tests for health, model metadata, and stats endpoints."""
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["app"] == "RescueIQ"
    assert "version" in data
    assert "EfficientNet-B0" in data["vision_model"]
    assert "MiniLM" in data["nlp_model"]


def test_empty_stats(client: TestClient):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["pending"] == 0
    assert data["verified"] == 0
    assert data["avg_priority"] == 0.0
