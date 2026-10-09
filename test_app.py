"""
Pytest unit tests for app.py (Flask routes and API endpoints) in CivicPulse AI.
"""

import json
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Verify that GET / returns the dashboard HTML with injected backend data."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.content_type
    html = response.data.decode("utf-8")
    assert "CIVICPULSE AI" in html
    assert "Complaints Analysed" in html
    assert "Emerging Civic Issues" in html
    assert "fetchDataAndRender" in html
    assert "switchTab" in html


def test_api_data_endpoint(client):
    """Verify that GET /api/data returns the complete CivicPulseBackend JSON payload."""
    response = client.get("/api/data")
    assert response.status_code == 200
    assert "application/json" in response.content_type
    data = response.get_json()

    # Required payload keys
    assert "total_anomalies" in data
    assert "alerts" in data
    assert "spatial_clusters" in data
    assert "ripple_hypotheses" in data

    # Type & content validations
    assert isinstance(data["total_anomalies"], int)
    assert isinstance(data["alerts"], list)
    assert isinstance(data["spatial_clusters"], dict)
    assert isinstance(data["ripple_hypotheses"], list)
    assert "by_geography" in data["spatial_clusters"]
    assert "compounding_crises" in data["spatial_clusters"]


def test_api_overview_endpoint(client):
    """Verify that GET /api/overview returns the unified payload matching /api/data."""
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.get_json()
    assert "total_anomalies" in data
    assert "alerts" in data
    assert "summary" in data


def test_api_alerts_filtered(client):
    """Verify that GET /api/alerts returns alert lists and supports query filtering."""
    response = client.get("/api/alerts?severity=HIGH")
    assert response.status_code == 200
    data = response.get_json()
    assert "alerts" in data
    assert all(a.get("severity") == "HIGH" for a in data["alerts"])


def test_api_spatial_endpoint(client):
    """Verify that GET /api/spatial returns spatial clusters and ripple hypotheses."""
    response = client.get("/api/spatial")
    assert response.status_code == 200
    data = response.get_json()
    assert "spatial_clusters" in data
    assert "ripple_hypotheses" in data


def test_api_health_endpoint(client):
    """Verify that GET /api/health responds with healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"


def test_api_process_endpoint(client):
    """Verify that POST /api/process executes the backend pipeline on posted records."""
    custom_records = [
        {
            "time_window": "2026-10-09 10:00 - 11:00",
            "category": "Traffic Signal Malfunction",
            "area_id": "Downtown-1",
            "observed_count": 35,
            "expected_count": 10.0,
            "anomaly_score": 4.1,
            "is_anomaly": True,
            "statistical_evidence": {},
        }
    ]
    response = client.post(
        "/api/process",
        data=json.dumps(custom_records),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_anomalies"] == 1
    assert data["alerts"][0]["category"] == "Traffic Signal Malfunction"


def test_api_data_real_model_values(client):
    """Verify that GET /api/data returns actual computed model anomalies, spatial clusters, and ripple hypotheses."""
    response = client.get("/api/data")
    assert response.status_code == 200
    data = response.get_json()

    # Verify real computed anomaly count (1271 from real dataset)
    assert data["total_anomalies"] > 0
    assert len(data["alerts"]) == data["total_anomalies"]
    assert data["spatial_clusters"]["total_areas"] > 0
    assert len(data["spatial_clusters"]["compounding_crises"]) > 0
    assert len(data["ripple_hypotheses"]) > 0

    # Summary should indicate live model inference
    summary = data.get("summary", {})
    assert "Live Model Inference" in summary.get("status", "")
    assert summary.get("total_complaints", 0) > 0

    # Trend series should contain actual observed vs expected counts
    trend = data.get("trend_series", {})
    assert "labels" in trend
    assert "actual" in trend
    assert "baseline" in trend
    assert len(trend["labels"]) > 0


def test_api_refresh_endpoint(client):
    """Verify that POST /api/refresh invalidates cache, runs pipeline, and returns updated data."""
    response = client.post("/api/refresh")
    assert response.status_code == 200
    data = response.get_json()
    assert "data" in data
    assert data["message"] == "Pipeline refreshed successfully"
    assert data["data"]["total_anomalies"] > 0

