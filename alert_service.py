"""
Alert Service for CivicPulse AI.
Transforms detected anomalies from machine learning / statistical models
into actionable operational alerts for municipal authorities and operators.
"""

import uuid
from datetime import datetime, timezone
from typing import List

from schemas import AlertResponse, ModelOutput


def generate_alerts(model_outputs: List[ModelOutput]) -> List[AlertResponse]:
    """
    Generate actionable alerts from model anomaly detection outputs.

    Filtering & Severity Rules:
    - Only processes items where `is_anomaly == True`.
    - Computes severity:
        - "HIGH": if observed_count > expected_count * 2
        - "MEDIUM": otherwise (observed_count <= expected_count * 2)
    - Generates human-readable titles, descriptions, and metrics explaining the civic spike.

    Args:
        model_outputs: List of ModelOutput instances from anomaly detection models.

    Returns:
        List of AlertResponse objects for all detected civic anomalies.
    """
    alerts: List[AlertResponse] = []

    for output in model_outputs:
        # 1. Filter for anomalies only
        if not output.is_anomaly:
            continue

        # 2. Calculate severity ("HIGH" if observed > expected * 2 else "MEDIUM")
        if output.observed_count > (output.expected_count * 2):
            severity = "HIGH"
        else:
            severity = "MEDIUM"

        # 3. Generate clear human-readable title
        title = f"[{severity} ALERT] Surge in {output.category} - Area {output.area_id}"

        # 4. Generate descriptive explanation of the civic spike
        if output.expected_count > 0:
            ratio = output.observed_count / output.expected_count
            spike_percentage = ((output.observed_count - output.expected_count) / output.expected_count) * 100
            ratio_detail = f"{ratio:.1f}x expected baseline (+{spike_percentage:.0f}% surge)"
        else:
            ratio_detail = "unprecedented spike over zero baseline"

        description = (
            f"Civic spike detected: An anomalous surge of '{output.category}' incidents was recorded "
            f"in Area '{output.area_id}' during time window '{output.time_window}'. "
            f"Observed count is {output.observed_count} compared to an expected baseline of {output.expected_count:.1f} "
            f"({ratio_detail}) with anomaly score {output.anomaly_score:.2f}. "
            f"Immediate municipal review and dispatch recommended."
        )

        # 5. Extract structured metrics
        metrics = {
            "observed_count": output.observed_count,
            "expected_count": output.expected_count,
            "anomaly_score": output.anomaly_score,
            "spike_ratio": (
                round(output.observed_count / output.expected_count, 2)
                if output.expected_count > 0
                else None
            ),
            "statistical_evidence": output.statistical_evidence,
        }

        # 6. Construct AlertResponse
        clean_area = str(output.area_id).replace(" ", "-").upper()
        alert_id = f"ALT-{clean_area}-{uuid.uuid4().hex[:8].upper()}"

        evidence = output.statistical_evidence or {}
        lat = evidence.get("latitude") or evidence.get("lat")
        lon = evidence.get("longitude") or evidence.get("lon") or evidence.get("lng")

        # Infer timestamp from evidence or time_window if present
        alert_ts = None
        if "timestamp" in evidence:
            try:
                alert_ts = datetime.fromisoformat(str(evidence["timestamp"]).replace("Z", "+00:00"))
            except Exception:
                pass

        if alert_ts is None and output.time_window:
            import re
            tw_match = re.search(r"(\d{4}-\d{2}-\d{2})[T\s]+(\d{1,2}:\d{2})", output.time_window)
            if tw_match:
                try:
                    alert_ts = datetime.strptime(
                        f"{tw_match.group(1)} {tw_match.group(2)}", "%Y-%m-%d %H:%M"
                    ).replace(tzinfo=timezone.utc)
                except Exception:
                    pass

        if alert_ts is None:
            alert_ts = datetime.now(timezone.utc)

        alert = AlertResponse(
            alert_id=alert_id,
            severity=severity,
            title=title,
            description=description,
            metrics=metrics,
            timestamp=alert_ts,
            area_id=output.area_id,
            category=output.category,
            time_window=output.time_window,
            latitude=float(lat) if lat is not None else None,
            longitude=float(lon) if lon is not None else None,
        )
        alerts.append(alert)

    return alerts


if __name__ == "__main__":
    print("=" * 70)
    print("Running CivicPulse AI Alert Service - Mock Test Suite")
    print("=" * 70)

    # Mock ModelOutput data representing various civic scenarios
    mock_model_outputs = [
        # Case 1: High severity anomaly (52 > 12.0 * 2 = 24.0)
        ModelOutput(
            time_window="2026-10-09 20:00 - 21:00",
            category="Water Supply Outage",
            area_id="Zone-4",
            observed_count=52,
            expected_count=12.0,
            anomaly_score=4.85,
            is_anomaly=True,
            statistical_evidence={"z_score": 4.85, "p_value": 0.0001, "historical_std": 2.4},
        ),
        # Case 2: Medium severity anomaly (26 <= 15.0 * 2 = 30.0)
        ModelOutput(
            time_window="2026-10-09 20:00 - 21:00",
            category="Traffic Signal Malfunction",
            area_id="Sector-12",
            observed_count=26,
            expected_count=15.0,
            anomaly_score=2.31,
            is_anomaly=True,
            statistical_evidence={"z_score": 2.31, "p_value": 0.012, "historical_std": 3.1},
        ),
        # Case 3: Non-anomaly (is_anomaly == False) -> Should be filtered out
        ModelOutput(
            time_window="2026-10-09 20:00 - 21:00",
            category="Street Cleaning Request",
            area_id="Ward-3",
            observed_count=11,
            expected_count=10.0,
            anomaly_score=0.45,
            is_anomaly=False,
            statistical_evidence={"z_score": 0.45, "p_value": 0.65, "historical_std": 2.2},
        ),
        # Case 4: High severity anomaly with zero expected count (8 > 0.0 * 2 = 0)
        ModelOutput(
            time_window="2026-10-09 20:00 - 21:00",
            category="Hazardous Chemical Leak",
            area_id="Industrial-Park-A",
            observed_count=8,
            expected_count=0.0,
            anomaly_score=6.20,
            is_anomaly=True,
            statistical_evidence={"z_score": 6.20, "p_value": 0.0, "historical_std": 0.0},
        ),
    ]

    # Generate alerts
    generated_alerts = generate_alerts(mock_model_outputs)

    # Validations
    assert len(generated_alerts) == 3, f"Expected 3 alerts, got {len(generated_alerts)}"
    assert generated_alerts[0].severity == "HIGH", f"Expected HIGH, got {generated_alerts[0].severity}"
    assert generated_alerts[1].severity == "MEDIUM", f"Expected MEDIUM, got {generated_alerts[1].severity}"
    assert generated_alerts[2].severity == "HIGH", f"Expected HIGH, got {generated_alerts[2].severity}"

    print(f"\n[PASS] Successfully generated {len(generated_alerts)} alerts from {len(mock_model_outputs)} model outputs.")
    print("[PASS] Non-anomalous events correctly filtered out.\n")

    for idx, alert in enumerate(generated_alerts, 1):
        print(f"--- Alert #{idx} ---")
        print(f"ID:          {alert.alert_id}")
        print(f"Severity:    {alert.severity}")
        print(f"Title:       {alert.title}")
        print(f"Timestamp:   {alert.timestamp.isoformat()}")
        print(f"Metrics:     {alert.metrics}")
        print(f"Description: {alert.description}\n")

    print("=" * 70)
    print("Mock test block passed all assertions successfully!")
    print("=" * 70)
