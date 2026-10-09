"""
tests/test_contract.py - Shared Integration Contract Tests
Verifies that schemas, model output, and canonical inputs meet agreed specifications.
"""

import pytest
from datetime import datetime
from schemas import CanonicalComplaint, ModelOutput, AlertResponse


def test_canonical_complaint_contract():
    complaint = CanonicalComplaint(
        created_at="2022-07-31 10:30:00",
        category="Roads & Infrastructure",
        area_id="ward_22",
        latitude=13.0318,
        longitude=77.6054,
        title="Pothole on main road",
        description="Deep crater creating hazard",
        ward_title="Vishwanath Nagenahalli"
    )
    assert complaint.category == "Roads & Infrastructure"
    assert complaint.area_id == "ward_22"
    assert complaint.latitude == 13.0318
    assert complaint.longitude == 77.6054
    json_data = complaint.model_dump()
    assert "created_at" in json_data


def test_model_output_contract():
    output = ModelOutput(
        time_window="2022-07-31",
        category="Sanitation & Waste",
        area_id="ward_176",
        observed_count=18,
        expected_count=2.5,
        anomaly_score=0.88,
        is_anomaly=True,
        statistical_evidence={
            "rolling_mean_7d": 2.5,
            "rolling_std_7d": 0.8,
            "z_score": 19.38,
            "surge_ratio": 7.2
        }
    )
    assert output.is_anomaly is True
    assert output.observed_count == 18
    assert output.expected_count == 2.5
    assert output.statistical_evidence["z_score"] > 3.0
    dumped = output.model_dump_json()
    assert "statistical_evidence" in dumped


def test_alert_response_contract():
    model_metric = ModelOutput(
        time_window="2022-07-31",
        category="Water Supply",
        area_id="ward_161",
        observed_count=12,
        expected_count=1.2,
        anomaly_score=0.92,
        is_anomaly=True,
        statistical_evidence={
            "rolling_mean_7d": 1.2,
            "z_score": 9.0
        }
    )
    from datetime import timezone
    alert = AlertResponse(
        alert_id="ALT-20220731-001",
        severity="CRITICAL",
        title="Critical Spike in Water Supply Complaints (Ward 161)",
        description="Observed 12 complaints against a 7-day expectation of 1.2 complaints (10x surge).",
        metrics=model_metric,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    assert alert.severity == "CRITICAL"
    assert alert.metrics.observed_count == 12
    assert "ALT-" in alert.alert_id
