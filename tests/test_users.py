"""Tests for user management endpoints."""
from fastapi.testclient import TestClient


def test_user_registration_and_listing(client: TestClient):
    user_data = {
        "email": "field.medic@rescueiq.org",
        "full_name": "Dr. Alex Carter",
        "password": "strongPassword123",
        "role": "responder",
        "phone_number": "+1-555-0199",
        "agency_affiliation": "Red Cross Emergency Medical Team",
    }
    # 1. Register
    reg_resp = client.post("/api/users", json=user_data)
    assert reg_resp.status_code == 201
    user = reg_resp.json()
    assert user["email"] == user_data["email"]
    assert user["full_name"] == user_data["full_name"]
    assert user["role"] == "responder"
    assert "hashed_password" not in user  # Password must never be leaked
    user_id = user["id"]

    # 2. Duplicate email should fail with 400
    dup_resp = client.post("/api/users", json=user_data)
    assert dup_resp.status_code == 400

    # 3. Retrieve user by ID
    get_resp = client.get(f"/api/users/{user_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == user_id

    # 4. List users
    list_resp = client.get("/api/users?role=responder")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1
