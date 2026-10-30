"""Tests for duplicate links, detection and human decision-making."""
from fastapi.testclient import TestClient


def test_duplicate_decision_flow(client: TestClient, dummy_image_bytes):
    # Seed demo incidents to populate duplicate link
    seed_resp = client.get("/api/demo-seed")
    assert seed_resp.status_code == 200

    # List duplicate candidate links
    dup_resp = client.get("/api/duplicates")
    assert dup_resp.status_code == 200
    links = dup_resp.json()
    assert len(links) >= 1
    link_id = links[0]["id"]
    assert links[0]["status"] == "Candidate"
    assert "composite_score" in links[0]

    # Confirm duplicate
    confirm_resp = client.post(
        f"/api/duplicates/{link_id}/decision",
        json={"decision": "Confirmed", "actor": "Incident Commander"},
    )
    assert confirm_resp.status_code == 200
    assert confirm_resp.json()["decision"] == "Confirmed"

    # Split duplicate
    split_resp = client.post(
        f"/api/duplicates/{link_id}/decision",
        json={"decision": "Split", "actor": "Incident Commander"},
    )
    assert split_resp.status_code == 200
    assert split_resp.json()["decision"] == "Split"
