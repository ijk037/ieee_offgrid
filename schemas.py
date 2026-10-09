"""
schemas.py - Data Contracts & Schemas for CivicPulse AI
Jointly agreed and merged across Person A (ML/Data) and Person B (Backend/UI).
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------
# Canonical Raw Complaint Input Schema (Person A)
# ---------------------------------------------------------
class CanonicalComplaint(BaseModel):
    """
    Standardized complaint structure produced by data_pipeline.py.
    """
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    created_at: str = Field(description="ISO 8601 formatted datetime string (YYYY-MM-DD HH:MM:SS)")
    category: str = Field(description="Normalized civic issue category (e.g. Roads & Infrastructure)")
    area_id: str = Field(description="Canonical area/ward identifier (e.g. ward_22)")
    latitude: Optional[float] = Field(default=None, description="GPS Latitude")
    longitude: Optional[float] = Field(default=None, description="GPS Longitude")
    title: Optional[str] = Field(default=None, description="Complaint summary title")
    description: Optional[str] = Field(default=None, description="Raw complaint description")
    ward_title: Optional[str] = Field(default=None, description="Human readable ward name")


# ---------------------------------------------------------
# Model Output Anomaly Detection Schema (Joint Contract)
# ---------------------------------------------------------
class ModelOutput(BaseModel):
    """
    Standardized ML anomaly detection output produced by anomaly detection models.
    Consumed by alert_service.py, app_adapter.py, and downstream UI.
    """
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    time_window: str = Field(
        ...,
        description="Time window for the aggregated metrics (e.g. 'YYYY-MM-DD' or ISO timestamp)"
    )
    category: str = Field(
        ...,
        description="Civic category or issue type (e.g. 'Roads & Infrastructure', 'Water Outage')"
    )
    area_id: str = Field(
        ...,
        description="Geographic region, ward, or zone identifier (e.g. 'ward_22', 'Ward-1')"
    )
    observed_count: Union[int, float] = Field(
        ...,
        description="Observed number of civic reports/incidents in the given time window"
    )
    expected_count: float = Field(
        ...,
        description="Baseline expected number of incidents predicted by the statistical model"
    )
    anomaly_score: float = Field(
        ...,
        description="Anomaly score or magnitude metric"
    )
    is_anomaly: bool = Field(
        ...,
        description="Boolean flag indicating whether the event is classified as an anomaly"
    )
    statistical_evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary containing supporting statistical evidence (e.g. z_score, surge_ratio)"
    )


# ---------------------------------------------------------
# Actionable Alert Response Schema (Joint Contract)
# ---------------------------------------------------------
class AlertResponse(BaseModel):
    """
    Actionable alert formatted for dashboards, Stitch UI, and municipal operators.
    """
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    alert_id: str = Field(
        ...,
        description="Unique identifier for the generated alert"
    )
    severity: str = Field(
        ...,
        description="Alert severity level: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'"
    )
    title: str = Field(
        ...,
        description="Human-readable title describing the civic spike"
    )
    description: str = Field(
        ...,
        description="Detailed contextual description explaining the civic incident surge"
    )
    metrics: Union[ModelOutput, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Key metrics associated with the alert (observed, expected, ratio, etc.)"
    )
    timestamp: Union[datetime, str] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the alert was generated"
    )
    area_id: Optional[str] = Field(
        default=None,
        description="Geographic region, ward, or zone identifier"
    )
    category: Optional[str] = Field(
        default=None,
        description="Civic category or issue type"
    )
    time_window: Optional[str] = Field(
        default=None,
        description="Time window of the aggregated event"
    )
    latitude: Optional[float] = Field(
        default=None,
        description="Geographic latitude coordinate"
    )
    longitude: Optional[float] = Field(
        default=None,
        description="Geographic longitude coordinate"
    )
