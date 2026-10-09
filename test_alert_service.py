"""
Unit tests for alert_service.py and schemas.py in CivicPulse AI.
"""

import pytest
from datetime import datetime
from schemas import ModelOutput, AlertResponse
from alert_service import generate_alerts


def test_anomaly_filtering():
    """Verify that only anomalous outputs (is_anomaly == True) generate alerts."""
    outputs = [
        ModelOutput(
            time_window="2026-10-09 10:00-11:00",
            category="Pothole",
            area_id="Ward-1",
            observed_count=20,
            expected_count=5.0,
            anomaly_score=3.5,
            is_anomaly=True,
            statistical_evidence={"z": 3.5},
        ),
        ModelOutput(
            time_window="2026-10-09 10:00-11:00",
            category="Streetlight",
            area_id="Ward-2",
            observed_count=6,
            expected_count=5.0,
            anomaly_score=0.2,
            is_anomaly=False,
            statistical_evidence={"z": 0.2},
        ),
    ]

    alerts = generate_alerts(outputs)
    assert len(alerts) == 1
    assert alerts[0].title.find("Pothole") != -1
    assert "Streetlight" not in alerts[0].title


def test_severity_calculation_high():
    """Verify HIGH severity when observed > expected * 2."""
    outputs = [
        ModelOutput(
            time_window="2026-10-09 10:00-11:00",
            category="Water Outage",
            area_id="Sector-5",
            observed_count=25,
            expected_count=10.0,  # 25 > 20 -> HIGH
            anomaly_score=4.0,
            is_anomaly=True,
        )
    ]
    alerts = generate_alerts(outputs)
    assert len(alerts) == 1
    assert alerts[0].severity == "HIGH"


def test_severity_calculation_medium():
    """Verify MEDIUM severity when observed <= expected * 2."""
    outputs = [
        ModelOutput(
            time_window="2026-10-09 10:00-11:00",
            category="Traffic Signal",
            area_id="Sector-9",
            observed_count=20,
            expected_count=10.0,  # 20 <= 20 -> MEDIUM
            anomaly_score=2.1,
            is_anomaly=True,
        )
    ]
    alerts = generate_alerts(outputs)
    assert len(alerts) == 1
    assert alerts[0].severity == "MEDIUM"


def test_zero_expected_baseline():
    """Verify handling of zero baseline without division-by-zero errors."""
    outputs = [
        ModelOutput(
            time_window="2026-10-09 10:00-11:00",
            category="Gas Leak",
            area_id="Sector-A",
            observed_count=5,
            expected_count=0.0,
            anomaly_score=5.0,
            is_anomaly=True,
        )
    ]
    alerts = generate_alerts(outputs)
    assert len(alerts) == 1
    assert alerts[0].severity == "HIGH"
    assert alerts[0].metrics["spike_ratio"] is None


def test_alert_response_structure():
    """Verify all fields in AlertResponse conform to schema."""
    output = ModelOutput(
        time_window="2026-10-09 12:00-13:00",
        category="Road Hazard",
        area_id="Downtown",
        observed_count=45,
        expected_count=15.0,
        anomaly_score=3.8,
        is_anomaly=True,
        statistical_evidence={"confidence": 0.99},
    )
    alerts = generate_alerts([output])
    assert len(alerts) == 1
    alert = alerts[0]
    assert isinstance(alert, AlertResponse)
    assert isinstance(alert.alert_id, str)
    assert alert.alert_id.startswith("ALT-")
    assert alert.severity == "HIGH"
    assert "Road Hazard" in alert.title
    assert "Downtown" in alert.description
    assert isinstance(alert.timestamp, datetime)
    assert alert.metrics["observed_count"] == 45
    assert alert.metrics["expected_count"] == 15.0
