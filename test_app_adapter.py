"""
Pytest unit tests for app_adapter.py in CivicPulse AI.
"""

import csv
import json
import tempfile
import pytest

from app_adapter import CivicPulseBackend, run_civicpulse_pipeline
from schemas import AlertResponse, ModelOutput


@pytest.fixture
def sample_records():
    return [
        {
            "time_window": "2026-10-08 10:00 - 11:00",
            "category": "Water Main Break",
            "area_id": "Ward-1",
            "observed_count": 50,
            "expected_count": 10.0,
            "anomaly_score": 4.2,
            "is_anomaly": True,
            "statistical_evidence": {"z_score": 4.2, "lat": 28.6139, "lon": 77.2090},
        },
        {
            "time_window": "2026-10-08 10:00 - 11:00",
            "category": "Traffic Congestion",
            "area_id": "Ward-1",
            "observed_count": 30,
            "expected_count": 12.0,
            "anomaly_score": 2.5,
            "is_anomaly": True,
            "statistical_evidence": {"z_score": 2.5, "lat": 28.6142, "lon": 77.2093},
        },
        {
            "time_window": "2026-10-08 10:00 - 11:00",
            "category": "Street Cleaning",
            "area_id": "Ward-1",
            "observed_count": 5,
            "expected_count": 5.0,
            "anomaly_score": 0.0,
            "is_anomaly": False,
            "statistical_evidence": {},
        },
        {
            "time_window": "2026-10-09 12:00 - 13:00",  # +26 hours
            "category": "Sanitation Contamination",
            "area_id": "Ward-1",
            "observed_count": 35,
            "expected_count": 8.0,
            "anomaly_score": 3.9,
            "is_anomaly": True,
            "statistical_evidence": {"z_score": 3.9},
        },
    ]


def test_process_raw_records_structure(sample_records):
    backend = CivicPulseBackend(min_ripple_lag_hours=1.0, max_ripple_lag_hours=48.0)
    response = backend.process(sample_records)

    # Required unified dictionary keys
    assert "total_anomalies" in response
    assert "alerts" in response
    assert "spatial_clusters" in response
    assert "ripple_hypotheses" in response

    # Type verifications
    assert isinstance(response["total_anomalies"], int)
    assert response["total_anomalies"] == 3
    assert isinstance(response["alerts"], list)
    assert len(response["alerts"]) == 3
    assert all(isinstance(a, AlertResponse) for a in response["alerts"])
    assert isinstance(response["spatial_clusters"], dict)
    assert isinstance(response["ripple_hypotheses"], list)


def test_spatial_clustering_integration(sample_records):
    response = run_civicpulse_pipeline(sample_records)
    spatial = response["spatial_clusters"]

    assert "by_geography" in spatial
    assert "compounding_crises" in spatial
    assert "Ward-1" in spatial["by_geography"]
    # Compounding crisis (Water + Traffic) in Ward-1 during same window
    assert len(spatial["compounding_crises"]) >= 1
    crisis = spatial["compounding_crises"][0]
    assert crisis["area_id"] == "Ward-1"
    assert crisis["category_count"] == 2


def test_ripple_engine_integration(sample_records):
    response = run_civicpulse_pipeline(sample_records)
    ripples = response["ripple_hypotheses"]

    assert len(ripples) >= 1
    water_sanitation = next(
        (r for r in ripples if r["antecedent_category"] == "Water Main Break" and r["subsequent_category"] == "Sanitation Contamination"),
        None,
    )
    assert water_sanitation is not None
    assert water_sanitation["is_hypothesis"] is True
    assert water_sanitation["correlation_status"] == "UNVERIFIED_HYPOTHESIS"
    assert 25.0 <= water_sanitation["time_lag_hours"] <= 27.0


def test_json_serialization(sample_records):
    backend = CivicPulseBackend()
    response = backend.process(sample_records)

    # Test .to_json() method
    json_str = response.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["total_anomalies"] == 3
    assert len(parsed["alerts"]) == 3

    # Test json_safe=True mode
    json_safe_response = backend.process(sample_records, json_safe=True)
    direct_json = json.dumps(json_safe_response)
    assert isinstance(direct_json, str)


def test_json_file_loading(sample_records):
    backend = CivicPulseBackend()
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        json.dump(sample_records, f)
        filepath = f.name

    try:
        response = backend.process_file(filepath)
        assert response["total_anomalies"] == 3
        assert len(response["alerts"]) == 3
    finally:
        import os
        if os.path.exists(filepath):
            os.remove(filepath)


def test_csv_file_loading():
    backend = CivicPulseBackend()
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_window", "category", "area_id", "observed_count",
            "expected_count", "anomaly_score", "is_anomaly", "statistical_evidence"
        ])
        writer.writerow(["2026-10-08 10:00 - 11:00", "Pothole Spike", "District-3", "45", "10.0", "4.0", "True", "{}"])
        filepath = f.name

    try:
        response = backend.process_file(filepath)
        assert response["total_anomalies"] == 1
        assert response["alerts"][0].category == "Pothole Spike"
    finally:
        import os
        if os.path.exists(filepath):
            os.remove(filepath)


def test_load_real_model_records_production():
    """Verify load_real_model_records ingests repository dataset and trained model."""
    from app_adapter import load_real_model_records
    records = load_real_model_records()
    assert len(records) > 0
    anomalies = [r for r in records if r.is_anomaly]
    assert len(anomalies) > 0
    # Coordinates / metadata should be enriched
    has_meta = any("latitude" in r.statistical_evidence for r in anomalies)
    assert has_meta is True


def test_graceful_fixture_fallback_when_no_csv():
    """Verify that if no CSV dataset is present, it gracefully falls back to the fixture pipeline."""
    import tempfile
    from app_adapter import load_real_model_records

    with tempfile.TemporaryDirectory() as empty_dir:
        records = load_real_model_records(base_dir=empty_dir)
        assert len(records) > 0
        anomalies = [r for r in records if r.is_anomaly]
        assert len(anomalies) > 0, "Graceful fallback must produce valid anomalies"


def test_backend_process_default_real_inference():
    """Verify CivicPulseBackend.process() with no args runs real inference and produces complete response."""
    backend = CivicPulseBackend()
    response = backend.process()
    assert response["total_anomalies"] > 0
    assert len(response["alerts"]) > 0
    assert response["spatial_clusters"]["total_areas"] > 0
    assert "summary" in response
    assert "trend_series" in response
    assert response["summary"]["status"] == "System Operational — Live Model Inference"

