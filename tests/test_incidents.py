"""Tests for incident CRUD, validation, search, geospatial filtering, and overrides."""
from fastapi.testclient import TestClient


def test_create_and_retrieve_incident(client: TestClient, dummy_image_bytes):
    # 1. Create incident
    payload = {
        "description": "Flash flooding on MG Road with 3ft of standing water.",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "category": "Flood",
        "urgency": 85,
        "people_affected": 25,
        "access_blocked": "true",
        "reporter": "Test Unit Alpha",
    }
    files = {
        "image": ("test_flood.jpg", dummy_image_bytes, "image/jpeg"),
    }
    create_resp = client.post("/api/incidents", data=payload, files=files)
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["id"] is not None
    assert created["category"] == "Flood"
    assert created["verification_status"] == "Pending"
    assert created["incident_status"] == "Reported"
    assert "priority_score" in created
    assert created["priority_score"] > 0
    inc_id = created["id"]

    # 2. Get detail
    get_resp = client.get(f"/api/incidents/{inc_id}")
    assert get_resp.status_code == 200
    detail = get_resp.json()
    assert detail["id"] == inc_id
    assert len(detail["history"]) >= 1
    assert detail["history"][0]["action"] == "Report Created"

    # 3. List incidents
    list_resp = client.get("/api/incidents")
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert len(items) == 1
    assert items[0]["id"] == inc_id

    # 4. Search
    search_resp = client.get("/api/incidents?search=flooding")
    assert len(search_resp.json()) == 1

    search_miss = client.get("/api/incidents?search=earthquake")
    assert len(search_miss.json()) == 0

    # 5. Nearby geospatial search (within 1km should match, within 50m should not match offset point)
    nearby_hit = client.get("/api/incidents/nearby?latitude=12.9716&longitude=77.5946&radius_meters=500")
    assert nearby_hit.status_code == 200
    assert len(nearby_hit.json()) == 1
    assert nearby_hit.json()[0]["distance_meters"] <= 10.0

    nearby_miss = client.get("/api/incidents/nearby?latitude=13.5000&longitude=77.5946&radius_meters=500")
    assert len(nearby_miss.json()) == 0


def test_incident_human_override_and_patch(client: TestClient, dummy_image_bytes):
    # Create incident
    payload = {
        "description": "Roof partially collapsed in residential area.",
        "latitude": 12.9700,
        "longitude": 77.5900,
        "category": "Structural damage",
        "urgency": 70,
        "people_affected": 5,
        "access_blocked": "false",
        "reporter": "Citizen Scout",
    }
    files = {"image": ("test.jpg", dummy_image_bytes, "image/jpeg")}
    create_resp = client.post("/api/incidents", data=payload, files=files)
    inc_id = create_resp.json()["id"]

    # Patch verification and assignment
    patch_resp = client.patch(
        f"/api/incidents/{inc_id}",
        json={"verification_status": "Verified", "assigned_to": "Disaster Response Unit 1", "actor": "Commander"},
    )
    assert patch_resp.status_code == 200
    updated = patch_resp.json()
    assert updated["verification_status"] == "Verified"
    assert updated["assigned_to"] == "Disaster Response Unit 1"

    # Human override damage class to 'Severe damage'
    override_resp = client.patch(
        f"/api/incidents/{inc_id}",
        json={"override_damage_class": "Severe damage", "actor": "Structural Engineer"},
    )
    assert override_resp.status_code == 200
    assert override_resp.json()["damage_class"] == "Severe damage"

    # Verify audit history logged the changes
    detail_resp = client.get(f"/api/incidents/{inc_id}")
    history = detail_resp.json()["history"]
    actions = [h["action"] for h in history]
    assert "Verification Updated" in actions
    assert "Assignment Updated" in actions
    assert "Human Label Override" in actions


def test_soft_delete_incident(client: TestClient, dummy_image_bytes):
    payload = {
        "description": "Erroneous report to be removed.",
        "latitude": 12.9600,
        "longitude": 77.6000,
        "category": "Other",
        "urgency": 10,
        "people_affected": 0,
        "access_blocked": "false",
        "reporter": "Test",
    }
    files = {"image": ("test.jpg", dummy_image_bytes, "image/jpeg")}
    inc_id = client.post("/api/incidents", data=payload, files=files).json()["id"]

    # Delete
    del_resp = client.delete(f"/api/incidents/{inc_id}?actor=Moderator")
    assert del_resp.status_code == 204

    # Detail should now return 404
    assert client.get(f"/api/incidents/{inc_id}").status_code == 404

    # List should not include deleted incident
    assert len(client.get("/api/incidents").json()) == 0


def test_invalid_image_upload_rejected(client: TestClient):
    payload = {
        "description": "Invalid file report.",
        "latitude": 12.97,
        "longitude": 77.59,
        "category": "Flood",
    }
    # Send a plain text file pretending to be an image
    files = {"image": ("malicious.txt", b"plain text script content", "text/plain")}
    resp = client.post("/api/incidents", data=payload, files=files)
    assert resp.status_code in (422, 400)
