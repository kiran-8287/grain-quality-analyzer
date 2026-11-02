"""
API Integration Tests.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data


def test_standards_endpoint(client):
    response = client.get("/standards")
    assert response.status_code == 200
    data = response.json()
    assert "grade_a" in data
    assert "common" in data
    assert data["grade_a"]["broken"]["max_percent"] == 25.0


def test_analyze_empty_file_fails(client):
    response = client.post("/analyze", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_analyze_valid_grain(client):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.ellipse(img, (150, 150), (40, 15), 30, 0, 360, (230, 230, 230), -1)
    _, enc = cv2.imencode(".jpg", img)

    response = client.post(
        "/analyze",
        files={"file": ("grain.jpg", enc.tobytes(), "image/jpeg")},
        data={"grade": "grade_a"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["rice_detected"] is True
    assert len(data["grains"]) == 1
    assert "summary" in data
    assert "standards" in data


def test_analyze_non_rice(client):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", img)

    response = client.post(
        "/analyze",
        files={"file": ("blank.jpg", enc.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["rice_detected"] is False
    assert "No rice grains detected" in data["message"]


def test_export_endpoints(client):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.ellipse(img, (150, 150), (40, 15), 30, 0, 360, (230, 230, 230), -1)
    _, enc = cv2.imencode(".jpg", img)

    res = client.post(
        "/analyze",
        files={"file": ("grain.jpg", enc.tobytes(), "image/jpeg")},
    )
    job_id = res.json()["job_id"]

    # Test JSON export
    res_json = client.get(f"/analysis/{job_id}/export/json")
    assert res_json.status_code == 200
    assert res_json.headers["content-type"].startswith("application/json")

    # Test CSV export
    res_csv = client.get(f"/analysis/{job_id}/export/csv")
    assert res_csv.status_code == 200
    assert "grain_id,confidence,length,breadth,L_over_B" in res_csv.text
