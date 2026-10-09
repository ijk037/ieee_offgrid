"""
Pytest unit tests for spatial_analysis.py in CivicPulse AI.
"""

from datetime import datetime, timedelta, timezone
import pytest

from schemas import AlertResponse
from spatial_analysis import (
    GeographyGroupDict,
    civic_ripple_engine,
    cross_category_analysis,
    extract_area_id,
    extract_category,
    extract_coordinates,
    group_by_geography,
    haversine_distance,
)


@pytest.fixture
def base_timestamp():
    return datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


def test_group_by_geography_normal(base_timestamp):
    alerts = [
        AlertResponse(
            alert_id="ALT-1",
            severity="HIGH",
            title="Surge in Water - Area Ward-1",
            description="desc",
            area_id="Ward-1",
            category="Water",
            timestamp=base_timestamp,
        ),
        AlertResponse(
            alert_id="ALT-2",
            severity="MEDIUM",
            title="Surge in Road - Area Ward-2",
            description="desc",
            area_id="Ward-2",
            category="Road",
            timestamp=base_timestamp,
        ),
    ]

    grouped = group_by_geography(alerts)
    assert "Ward-1" in grouped
    assert "Ward-2" in grouped
    assert len(grouped["Ward-1"]) == 1
    assert len(grouped["Ward-2"]) == 1


def test_group_by_geography_handles_missing_and_unknown(base_timestamp):
    alerts = [
        AlertResponse(
            alert_id="ALT-NONE",
            severity="LOW",
            title="No area in title",
            description="desc",
            area_id=None,
            category="Waste",
            timestamp=base_timestamp,
        ),
        AlertResponse(
            alert_id="ALT-EMPTY",
            severity="MEDIUM",
            title="Empty area",
            description="desc",
            area_id="",
            category="Noise",
            timestamp=base_timestamp,
        ),
    ]

    grouped = group_by_geography(alerts)
    assert "UNKNOWN" in grouped
    assert len(grouped["UNKNOWN"]) == 2


def test_group_by_geography_compound_lookup(base_timestamp):
    alerts = [
        AlertResponse(
            alert_id="ALT-W1",
            severity="HIGH",
            title="Surge in Water - Area Ward-1",
            description="desc",
            area_id="Ward-1",
            category="Water Outage",
            timestamp=base_timestamp,
        ),
        AlertResponse(
            alert_id="ALT-W2",
            severity="MEDIUM",
            title="Surge in Power - Area Ward-1",
            description="desc",
            area_id="Ward-1",
            category="Power Grid",
            timestamp=base_timestamp,
        ),
    ]

    grouped = group_by_geography(alerts)
    assert "Ward-1" in grouped
    # Compound lookup via GeographyGroupDict
    water_alerts = grouped["Ward-1:Water Outage"]
    assert len(water_alerts) == 1
    assert water_alerts[0].alert_id == "ALT-W1"

    power_alerts = grouped["Ward-1 - Power Grid"]
    assert len(power_alerts) == 1
    assert power_alerts[0].alert_id == "ALT-W2"


def test_cross_category_analysis_detects_compounding_crisis(base_timestamp):
    alerts = [
        AlertResponse(
            alert_id="ALT-C1",
            severity="HIGH",
            title="Surge in Water - Area Central",
            description="desc",
            area_id="Central",
            category="Water Supply",
            time_window="2026-10-08 12:00-13:00",
            timestamp=base_timestamp,
        ),
        AlertResponse(
            alert_id="ALT-C2",
            severity="MEDIUM",
            title="Surge in Electricity - Area Central",
            description="desc",
            area_id="Central",
            category="Electricity Grid",
            time_window="2026-10-08 12:00-13:00",
            timestamp=base_timestamp,
        ),
        # Alert in different time window -> should not compound with above
        AlertResponse(
            alert_id="ALT-C3",
            severity="LOW",
            title="Surge in Noise - Area Central",
            description="desc",
            area_id="Central",
            category="Noise",
            time_window="2026-10-08 18:00-19:00",
            timestamp=base_timestamp + timedelta(hours=6),
        ),
    ]

    crises = cross_category_analysis(alerts)
    assert len(crises) == 1
    crisis = crises[0]
    assert crisis["area_id"] == "Central"
    assert crisis["time_window"] == "2026-10-08 12:00-13:00"
    assert crisis["category_count"] == 2
    assert "Water Supply" in crisis["categories"]
    assert "Electricity Grid" in crisis["categories"]
    assert crisis["max_severity"] == "HIGH"


def test_cross_category_single_category_ignored(base_timestamp):
    alerts = [
        AlertResponse(
            alert_id="ALT-S1",
            severity="HIGH",
            title="Surge 1",
            description="desc",
            area_id="North",
            category="Traffic",
            time_window="2026-10-08 12:00-13:00",
            timestamp=base_timestamp,
        ),
        AlertResponse(
            alert_id="ALT-S2",
            severity="MEDIUM",
            title="Surge 2",
            description="desc",
            area_id="North",
            category="Traffic",
            time_window="2026-10-08 12:00-13:00",
            timestamp=base_timestamp,
        ),
    ]
    # Same category multiple times is not a multi-category compounding crisis
    crises = cross_category_analysis(alerts)
    assert len(crises) == 0


def test_civic_ripple_engine_water_to_health_cascade(base_timestamp):
    alerts = [
        # Antecedent: Water issue in Zone-4 at T-0
        AlertResponse(
            alert_id="ALT-R1",
            severity="HIGH",
            title="Water Issue in Zone-4",
            description="desc",
            area_id="Zone-4",
            category="Water Infrastructure",
            timestamp=base_timestamp,
            latitude=28.6139,
            longitude=77.2090,
        ),
        # Subsequent: Health issue in Zone-4 at T+30h (within 24-48h window) with MISSING coordinates
        AlertResponse(
            alert_id="ALT-R2",
            severity="HIGH",
            title="Health Issue in Zone-4",
            description="desc",
            area_id="Zone-4",
            category="Health & Sanitation",
            timestamp=base_timestamp + timedelta(hours=30),
            latitude=None,   # Missing coordinates
            longitude=None,
        ),
    ]

    hypotheses = civic_ripple_engine(alerts, min_lag_hours=1.0, max_lag_hours=48.0)
    assert len(hypotheses) == 1
    hyp = hypotheses[0]
    # Explicit hypothesis requirements
    assert hyp["is_hypothesis"] is True
    assert hyp["correlation_status"] == "UNVERIFIED_HYPOTHESIS"
    assert "HYPOTHESIS" in hyp["disclaimer"].upper()
    assert hyp["time_lag_hours"] == 30.0
    assert hyp["antecedent_category"] == "Water Infrastructure"
    assert hyp["subsequent_category"] == "Health & Sanitation"
    assert "WARD_CO_LOCATION" in hyp["proximity_basis"]


def test_civic_ripple_engine_coordinate_distance(base_timestamp):
    # Two alerts 10 hours apart with coordinates ~0.35 km apart
    alerts = [
        AlertResponse(
            alert_id="ALT-GEO1",
            severity="MEDIUM",
            title="Flooding",
            description="desc",
            area_id="Ward-9",
            category="Flooding",
            timestamp=base_timestamp,
            latitude=28.5355,
            longitude=77.3910,
        ),
        AlertResponse(
            alert_id="ALT-GEO2",
            severity="MEDIUM",
            title="Pothole",
            description="desc",
            area_id="Ward-9",
            category="Pothole",
            timestamp=base_timestamp + timedelta(hours=10),
            latitude=28.5380,
            longitude=77.3930,
        ),
    ]

    hypotheses = civic_ripple_engine(alerts, min_lag_hours=1.0, max_lag_hours=48.0)
    assert len(hypotheses) == 1
    hyp = hypotheses[0]
    assert hyp["spatial_distance_km"] is not None
    assert hyp["spatial_distance_km"] < 1.0
    assert "COORDINATE_PROXIMITY" in hyp["proximity_basis"]


def test_haversine_distance_calculation():
    # Distance between New Delhi (28.6139, 77.2090) and Noida Sector 18 (28.5700, 77.3250) ~12 km
    dist = haversine_distance(28.6139, 77.2090, 28.5700, 77.3250)
    assert 10.0 < dist < 15.0
