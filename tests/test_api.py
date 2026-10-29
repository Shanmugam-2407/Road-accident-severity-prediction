"""Automated tests for FastAPI endpoints and backend services."""

import io
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.utils.config import TARGET_CLASSES

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint returns application metadata and online status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "RoadGuard AI" in data["application"]
    assert data["status"] == "online"
    assert "version" in data


def test_health_endpoint():
    """Verify health endpoint reports operational readiness."""
    response = client.get("/health")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert "prediction_ready" in data


def test_predict_severity_valid():
    """Verify single severity prediction endpoint returns valid structured response."""
    payload = {
        "number_of_vehicles": 2,
        "speed_limit": 30,
        "accident_hour": 14,
        "day_of_week": "Wednesday",
        "road_type": "Single carriageway",
        "road_surface_conditions": "Dry",
        "weather_conditions": "Fine no high winds",
        "light_conditions": "Daylight",
        "urban_or_rural_area": "Urban",
        "junction_detail": "Crossroads",
    }
    response = client.post("/predict/severity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["predicted_severity"] in TARGET_CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert "class_probabilities" in data
    assert "disclaimer" in data


def test_predict_severity_invalid_input():
    """Verify validation error (422) is raised on invalid payload parameters."""
    bad_payload = {
        "number_of_vehicles": 0,  # Below minimum (ge=1)
        "speed_limit": 30,
    }
    response = client.post("/predict/severity", json=bad_payload)
    assert response.status_code == 422


def test_batch_prediction_csv():
    """Verify batch CSV upload endpoint processes file and returns CSV."""
    csv_content = (
        "number_of_vehicles,speed_limit,accident_hour,day_of_week,road_type,road_surface_conditions,weather_conditions,light_conditions,urban_or_rural_area,junction_detail\n"
        "2,30,12,Monday,Single carriageway,Dry,Fine no high winds,Daylight,Urban,Not at junction\n"
        "1,60,23,Saturday,Dual carriageway,Wet or damp,Raining + high winds,Darkness - no lighting,Rural,Not at junction\n"
    ).encode("utf-8")

    response = client.post(
        "/predict/batch",
        files={"file": ("test_accidents.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("content-type", "")
    content = response.content.decode("utf-8")
    assert "predicted_severity" in content


def test_batch_prediction_non_csv():
    """Verify non-CSV file uploads are rejected with status 400."""
    response = client.post(
        "/predict/batch",
        files={"file": ("test.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only .csv files are supported" in response.json()["detail"]


def test_model_metrics_endpoint():
    """Verify /model/metrics returns actual saved evaluation summary."""
    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "metrics" in data
    assert "model_comparison" in data
    assert "feature_importance" in data

