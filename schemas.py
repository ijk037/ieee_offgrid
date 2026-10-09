"""
Pydantic schemas for CivicPulse AI.
Defines data contracts for model anomaly detection outputs and operational alert responses.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Union
from pydantic import BaseModel, ConfigDict, Field


class ModelOutput(BaseModel):
    """
    Schema representing the output of an anomaly detection model for civic events.
    """
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    time_window: str = Field(
        ...,
        description="Time window for the aggregated metrics (e.g., '2026-10-09 10:00-11:00' or ISO timestamp)",
    )
    category: str = Field(
        ...,
        description="Civic category or issue type (e.g., 'Water Outage', 'Traffic Signal', 'Pothole')",
    )
    area_id: str = Field(
        ...,
        description="Geographic region, ward, or zone identifier",
    )
    observed_count: Union[int, float] = Field(
        ...,
        description="Observed number of civic reports/incidents in the given time window",
    )
    expected_count: float = Field(
        ...,
        description="Baseline expected number of incidents predicted by the statistical model",
    )
    anomaly_score: float = Field(
        ...,
        description="Anomaly score or magnitude metric (e.g., Z-score, deviation score)",
    )
    is_anomaly: bool = Field(
        ...,
        description="Boolean flag indicating whether the event is classified as an anomaly",
    )
    statistical_evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary containing supporting statistical evidence (e.g., z-score, p-value, historical mean)",
    )


class AlertResponse(BaseModel):
    """
    Schema representing an actionable alert generated from anomalous civic events.
    """
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    alert_id: str = Field(
        ...,
        description="Unique identifier for the generated alert",
    )
    severity: str = Field(
        ...,
        description="Alert severity level: 'HIGH' if observed > expected * 2, else 'MEDIUM'",
    )
    title: str = Field(
        ...,
        description="Human-readable title describing the civic spike",
    )
    description: str = Field(
        ...,
        description="Detailed contextual description explaining the civic incident surge",
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key metrics associated with the alert (observed, expected, ratio, etc.)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the alert was generated",
    )
    area_id: Union[str, None] = Field(
        default=None,
        description="Geographic region, ward, or zone identifier",
    )
    category: Union[str, None] = Field(
        default=None,
        description="Civic category or issue type",
    )
    time_window: Union[str, None] = Field(
        default=None,
        description="Time window of the aggregated event",
    )
    latitude: Union[float, None] = Field(
        default=None,
        description="Geographic latitude coordinate",
    )
    longitude: Union[float, None] = Field(
        default=None,
        description="Geographic longitude coordinate",
    )
