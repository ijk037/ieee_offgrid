"""
schemas.py - Data Contracts & Schemas for CivicPulse AI
Jointly agreed and frozen in Phase 0.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Phase 0 Person A: Canonical Input Requirements
# ---------------------------------------------------------
class CanonicalComplaint(BaseModel):
    """
    Standardized complaint structure produced by Person A's data_pipeline.py.
    """
    created_at: str = Field(description="ISO 8601 formatted datetime string (YYYY-MM-DD HH:MM:SS)")
    category: str = Field(description="Normalized civic issue category (e.g. Roads & Infrastructure)")
    area_id: str = Field(description="Canonical area/ward identifier (e.g. ward_22)")
    latitude: Optional[float] = Field(default=None, description="GPS Latitude")
    longitude: Optional[float] = Field(default=None, description="GPS Longitude")
    title: Optional[str] = Field(default=None, description="Complaint summary title")
    description: Optional[str] = Field(default=None, description="Raw complaint description")
    ward_title: Optional[str] = Field(default=None, description="Human readable ward name")


# ---------------------------------------------------------
# Phase 0 Joint Contract: Model Output & Alert Schemas
# ---------------------------------------------------------
class ModelOutput(BaseModel):
    """
    Standardized ML anomaly detection output produced by Person A's model.
    Consumed by Person B's alert_service.py and app_adapter.py.
    """
    time_window: str = Field(description="Time window identifier, e.g. YYYY-MM-DD")
    category: str = Field(description="Civic issue category")
    area_id: str = Field(description="Ward or area identifier")
    observed_count: int = Field(description="Actual number of complaints observed")
    expected_count: float = Field(description="Baseline / expected complaints from historical trends")
    anomaly_score: float = Field(description="Normalized anomaly score (0.0 to 1.0, higher is more anomalous)")
    is_anomaly: bool = Field(description="Flag indicating if the observation is an anomaly")
    statistical_evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting statistics (e.g. rolling_mean_7d, rolling_std_7d, z_score, surge_ratio)"
    )


class AlertResponse(BaseModel):
    """
    Standardized alert object formatted for Stitch AI / Frontend UI.
    Produced by Person B's alert_service.py.
    """
    alert_id: str = Field(description="Unique identifier for the alert")
    severity: str = Field(description="Severity level: CRITICAL, HIGH, MEDIUM, LOW")
    title: str = Field(description="Human-readable alert title")
    description: str = Field(description="Clear explanation of the surge and statistical context")
    metrics: ModelOutput = Field(description="Associated model inference metrics")
    timestamp: str = Field(description="Alert creation timestamp (ISO 8601)")
