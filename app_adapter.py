"""
Application Adapter for CivicPulse AI.
Person B Implementation:
Serves as the clean bridge between upstream ML model outputs (Person A)
and downstream operational backend services:
1. Ingests raw model output records (dicts, ModelOutput instances) or files (.json, .jsonl, .csv).
2. Generates alerts via alert_service.generate_alerts.
3. Conducts spatial intelligence via spatial_analysis (grouping, compounding crises, ripple engine).
4. Returns a unified, JSON-serializable dictionary.
"""

import csv
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from alert_service import generate_alerts
from pydantic import BaseModel
from schemas import AlertResponse, ModelOutput
from spatial_analysis import (
    GeographyGroupDict,
    civic_ripple_engine,
    cross_category_analysis,
    group_by_geography,
)

# Optional pandas support for DataFrames
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# JSON Serialization Helpers
# ---------------------------------------------------------------------------

def make_json_serializable(obj: Any) -> Any:
    """
    Recursively converts Pydantic models, datetime objects, GeographyGroupDicts,
    and sets into JSON-serializable primitives (dicts, lists, strings, numbers, booleans).
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, GeographyGroupDict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(v) for v in obj]
    return obj


class CivicPulseBackendResponse(dict):
    """
    Unified dictionary response produced by CivicPulseBackend.
    Acts as a standard Python dictionary containing:
    - total_anomalies: int
    - alerts: List[AlertResponse]
    - spatial_clusters: dict
    - ripple_hypotheses: list

    Also provides:
    - `.to_dict()`: Recursively serializes all elements into JSON primitives.
    - `.to_json()`: Returns a JSON-formatted string.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert all nested models and datetimes into pure Python primitives."""
        return make_json_serializable(dict(self))

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize the unified response to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Raw Model Output Ingestion & Normalization
# ---------------------------------------------------------------------------

def _normalize_record_keys(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map common field aliases from Person A's model output to ModelOutput schema.
    """
    d = dict(raw_dict)

    # Alias mappings
    alias_map = {
        "observed": "observed_count",
        "observed_value": "observed_count",
        "actual_count": "observed_count",
        "expected": "expected_count",
        "expected_value": "expected_count",
        "baseline_count": "expected_count",
        "score": "anomaly_score",
        "z_score": "anomaly_score",
        "anomaly": "is_anomaly",
        "area": "area_id",
        "ward": "area_id",
        "zone": "area_id",
        "window": "time_window",
        "time": "time_window",
    }

    for alias, standard in alias_map.items():
        if alias in d and standard not in d:
            d[standard] = d[alias]

    # Type coercion for CSV / string imports
    if "is_anomaly" in d and isinstance(d["is_anomaly"], str):
        d["is_anomaly"] = d["is_anomaly"].strip().lower() in ("true", "1", "yes", "t")

    if "observed_count" in d and isinstance(d["observed_count"], str):
        try:
            d["observed_count"] = int(float(d["observed_count"]))
        except ValueError:
            pass

    if "expected_count" in d and isinstance(d["expected_count"], str):
        try:
            d["expected_count"] = float(d["expected_count"])
        except ValueError:
            pass

    if "anomaly_score" in d and isinstance(d["anomaly_score"], str):
        try:
            d["anomaly_score"] = float(d["anomaly_score"])
        except ValueError:
            pass

    # Parse JSON string in statistical_evidence if present
    if "statistical_evidence" in d and isinstance(d["statistical_evidence"], str):
        try:
            d["statistical_evidence"] = json.loads(d["statistical_evidence"])
        except Exception:
            d["statistical_evidence"] = {"raw": d["statistical_evidence"]}
    elif "statistical_evidence" not in d or d["statistical_evidence"] is None:
        d["statistical_evidence"] = {}

    # Extract coordinates into statistical_evidence if provided as flat columns
    evidence = d.get("statistical_evidence", {})
    if isinstance(evidence, dict):
        for coord_key in ("latitude", "lat", "longitude", "lon", "lng"):
            if coord_key in d and coord_key not in evidence:
                evidence[coord_key] = d[coord_key]
        d["statistical_evidence"] = evidence

    return d


def load_model_outputs(
    data: Union[str, Path, List[Union[ModelOutput, Dict[str, Any]]], Any]
) -> List[ModelOutput]:
    """
    Ingests and validates raw model output records or files into List[ModelOutput].

    Supported Input Formats:
    - List of ModelOutput instances
    - List of raw dictionaries (e.g. inference outputs)
    - Path to .json file (list of objects or {"records": [...]})
    - Path to .jsonl file (one JSON object per line)
    - Path to .csv file (table with incident/anomaly rows)
    - Raw JSON string
    - pandas.DataFrame (if pandas is installed)

    Args:
        data: Raw model outputs in any supported format.

    Returns:
        List of validated ModelOutput instances.
    """
    # 1. Handle pandas DataFrame
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        records = data.to_dict(orient="records")
        return [ModelOutput(**_normalize_record_keys(r)) for r in records]

    # 2. Handle file path or raw string
    if isinstance(data, (str, Path)):
        str_path = str(data).strip()

        # Check if it's an existing file on disk
        if os.path.isfile(str_path):
            path_obj = Path(str_path)
            suffix = path_obj.suffix.lower()

            if suffix == ".csv":
                records = []
                with open(str_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        records.append(row)
                return [ModelOutput(**_normalize_record_keys(r)) for r in records]

            elif suffix == ".jsonl":
                records = []
                with open(str_path, mode="r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
                return [ModelOutput(**_normalize_record_keys(r)) for r in records]

            else:  # Assume standard .json
                with open(str_path, mode="r", encoding="utf-8") as f:
                    content = json.load(f)
                if isinstance(content, dict):
                    content = content.get("records") or content.get("data") or content.get("model_outputs") or [content]
                return [ModelOutput(**_normalize_record_keys(r)) for r in content]

        else:
            # Not a file path, try parsing as inline JSON string
            try:
                parsed = json.loads(str_path)
                if isinstance(parsed, dict):
                    parsed = parsed.get("records") or parsed.get("data") or [parsed]
                return [ModelOutput(**_normalize_record_keys(r)) for r in parsed]
            except Exception as e:
                raise ValueError(f"Input string is neither an existing file nor valid JSON: {e}")

    # 3. Handle single dictionary
    if isinstance(data, dict):
        extracted = data.get("records") or data.get("data") or data.get("model_outputs")
        if extracted is not None and isinstance(extracted, list):
            return [
                item if isinstance(item, ModelOutput) else ModelOutput(**_normalize_record_keys(item))
                for item in extracted
            ]
        return [ModelOutput(**_normalize_record_keys(data))]

    # 4. Handle list/tuple of records
    if isinstance(data, (list, tuple)):
        outputs: List[ModelOutput] = []
        for item in data:
            if isinstance(item, ModelOutput):
                outputs.append(item)
            elif isinstance(item, dict):
                outputs.append(ModelOutput(**_normalize_record_keys(item)))
            else:
                raise TypeError(f"Unsupported record type inside list: {type(item)}")
        return outputs

    raise TypeError(f"Unsupported input type for model outputs: {type(data)}")


# ---------------------------------------------------------------------------
# CivicPulseBackend Class
# ---------------------------------------------------------------------------

class CivicPulseBackend:
    """
    Core backend adapter for CivicPulse AI.
    Connects upstream Machine Learning predictions to downstream municipal services.

    Pipeline Workflow:
    1. Ingests raw model outputs or files.
    2. Invokes alert_service.generate_alerts (filters anomalies, assigns severity, builds descriptions).
    3. Runs spatial intelligence (group_by_geography, cross_category_analysis, civic_ripple_engine).
    4. Delivers unified, clean JSON-serializable responses.
    """

    def __init__(
        self,
        min_ripple_lag_hours: float = 0.5,
        max_ripple_lag_hours: float = 48.0,
        max_ripple_distance_km: float = 5.0,
    ):
        """
        Initialize the backend adapter.

        Args:
            min_ripple_lag_hours: Minimum time lag in hours for cascading ripple hypotheses (default 0.5h).
            max_ripple_lag_hours: Maximum horizon in hours for ripple detection (default 48.0h).
            max_ripple_distance_km: Maximum physical distance in km for coordinate-based ripples (default 5.0 km).
        """
        self.min_ripple_lag_hours = min_ripple_lag_hours
        self.max_ripple_lag_hours = max_ripple_lag_hours
        self.max_ripple_distance_km = max_ripple_distance_km

    def process(
        self,
        raw_input: Union[str, Path, List[Union[ModelOutput, Dict[str, Any]]], Any],
        json_safe: bool = False,
    ) -> CivicPulseBackendResponse:
        """
        Execute the end-to-end backend processing pipeline.

        Args:
            raw_input: Raw model output records (dicts, ModelOutput instances) or file paths (.json, .csv).
            json_safe: If True, all nested Pydantic models are pre-converted to pure JSON dicts.

        Returns:
            CivicPulseBackendResponse dictionary containing:
            - total_anomalies: int
            - alerts: List[AlertResponse]
            - spatial_clusters: dict
            - ripple_hypotheses: list
        """
        # Step 1: Ingest raw model output records or files
        model_outputs = load_model_outputs(raw_input)

        # Step 2: Pass through alert_service to generate operational alerts
        alerts: List[AlertResponse] = generate_alerts(model_outputs)

        # Step 3: Pass resulting alerts through spatial analysis modules
        geo_groups = group_by_geography(alerts)
        compounding_crises = cross_category_analysis(alerts)
        ripple_hypotheses = civic_ripple_engine(
            alerts,
            min_lag_hours=self.min_ripple_lag_hours,
            max_lag_hours=self.max_ripple_lag_hours,
            max_distance_km=self.max_ripple_distance_km,
        )

        # Assemble structured spatial clusters
        spatial_clusters: Dict[str, Any] = {
            "by_geography": geo_groups,
            "compounding_crises": compounding_crises,
            "active_areas": list(geo_groups.keys()),
            "total_areas": len(geo_groups),
            "total_compounding_crises": len(compounding_crises),
        }

        # Step 4: Construct unified response
        response = CivicPulseBackendResponse({
            "total_anomalies": len(alerts),
            "alerts": alerts,
            "spatial_clusters": spatial_clusters,
            "ripple_hypotheses": ripple_hypotheses,
        })

        if json_safe:
            return CivicPulseBackendResponse(response.to_dict())

        return response

    def __call__(
        self,
        raw_input: Union[str, Path, List[Union[ModelOutput, Dict[str, Any]]], Any],
        json_safe: bool = False,
    ) -> CivicPulseBackendResponse:
        """Convenience callable interface."""
        return self.process(raw_input, json_safe=json_safe)

    def process_records(
        self,
        records: List[Union[ModelOutput, Dict[str, Any]]],
        json_safe: bool = False,
    ) -> CivicPulseBackendResponse:
        """Convenience method to process in-memory records."""
        return self.process(records, json_safe=json_safe)

    def process_file(
        self,
        file_path: Union[str, Path],
        json_safe: bool = False,
    ) -> CivicPulseBackendResponse:
        """Convenience method to process records directly from a file."""
        return self.process(file_path, json_safe=json_safe)


# Top-level functional API
def run_civicpulse_pipeline(
    raw_input: Union[str, Path, List[Union[ModelOutput, Dict[str, Any]]], Any],
    json_safe: bool = False,
) -> CivicPulseBackendResponse:
    """Functional wrapper executing the CivicPulseBackend pipeline."""
    backend = CivicPulseBackend()
    return backend.process(raw_input, json_safe=json_safe)


# ---------------------------------------------------------------------------
# Standalone Mock Test Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 75)
    print("Running CivicPulse AI App Adapter - End-to-End Pipeline Verification")
    print("=" * 75)

    backend = CivicPulseBackend(min_ripple_lag_hours=1.0, max_ripple_lag_hours=48.0)

    # -----------------------------------------------------------------------
    # Test 1: Ingestion from Raw Python Dictionaries (Person A Simulation)
    # -----------------------------------------------------------------------
    print("\n[TEST 1] Processing Raw Model Output Records (In-Memory Dictionaries)...")

    mock_raw_records = [
        # Anomaly 1: Water Infrastructure in Ward-4 (High Severity, T-0)
        {
            "time_window": "2026-10-08 10:00 - 11:00",
            "category": "Water Infrastructure",
            "area_id": "Ward-4",
            "observed_count": 60,
            "expected_count": 15.0,  # 60 > 15*2=30 -> HIGH
            "anomaly_score": 4.5,
            "is_anomaly": True,
            "statistical_evidence": {"z_score": 4.5, "latitude": 28.6139, "longitude": 77.2090},
        },
        # Anomaly 2: Traffic Signal in Ward-4 (Medium Severity, same time window -> Compounding Crisis)
        {
            "time_window": "2026-10-08 10:00 - 11:00",
            "category": "Traffic Signal Malfunction",
            "area_id": "Ward-4",
            "observed_count": 25,
            "expected_count": 15.0,  # 25 <= 30 -> MEDIUM
            "anomaly_score": 2.1,
            "is_anomaly": True,
            "statistical_evidence": {"z_score": 2.1, "latitude": 28.6145, "longitude": 77.2095},
        },
        # Record 3: Normal ticket volume (Non-anomaly -> Must be filtered out)
        {
            "time_window": "2026-10-08 10:00 - 11:00",
            "category": "Street Cleaning",
            "area_id": "Ward-1",
            "observed_count": 10,
            "expected_count": 10.0,
            "anomaly_score": 0.1,
            "is_anomaly": False,
            "statistical_evidence": {"z_score": 0.1},
        },
        # Anomaly 4: Health & Sanitation in Ward-4 (T + 28 hours -> Cascading Ripple Hypothesis)
        {
            "time_window": "2026-10-09 14:00 - 15:00",
            "category": "Health & Sanitation",
            "area_id": "Ward-4",
            "observed_count": 40,
            "expected_count": 10.0,
            "anomaly_score": 3.8,
            "is_anomaly": True,
            "statistical_evidence": {"z_score": 3.8},  # Sparse/missing coordinates
        },
    ]

    response = backend.process(mock_raw_records)

    # Verifications
    assert response["total_anomalies"] == 3, f"Expected 3 anomalies, got {response['total_anomalies']}"
    assert len(response["alerts"]) == 3, f"Expected 3 alerts, got {len(response['alerts'])}"
    assert isinstance(response["alerts"][0], AlertResponse), "Alerts must be AlertResponse instances"
    assert isinstance(response["spatial_clusters"], dict), "spatial_clusters must be a dict"
    assert isinstance(response["ripple_hypotheses"], list), "ripple_hypotheses must be a list"

    print(f"[PASS] Ingested {len(mock_raw_records)} records -> Detected {response['total_anomalies']} anomalies.")
    print(f"       Filtered out {len(mock_raw_records) - response['total_anomalies']} non-anomalous record(s).")
    print(f"       Generated {len(response['alerts'])} operational AlertResponse objects.")

    # Spatial cluster verification
    spatial = response["spatial_clusters"]
    assert "Ward-4" in spatial["by_geography"], "Ward-4 must be in geographic clusters"
    assert len(spatial["compounding_crises"]) >= 1, "Must detect compounding crisis in Ward-4"
    crisis = spatial["compounding_crises"][0]
    print(f"[PASS] Spatial Clusters: {spatial['total_areas']} active area(s), "
          f"{spatial['total_compounding_crises']} compounding crisis cluster(s):")
    print(f"       * {crisis['title']} ({', '.join(crisis['categories'])})")

    # Ripple hypotheses verification
    assert len(response["ripple_hypotheses"]) >= 1, "Must detect cascading ripple hypotheses"
    ripple = response["ripple_hypotheses"][0]
    assert ripple["is_hypothesis"] is True, "Must be explicitly labeled as hypothesis"
    print(f"[PASS] Ripple Engine: {len(response['ripple_hypotheses'])} hypothesis/hypotheses generated:")
    print(f"       * {ripple['hypothesis_title']}")
    print(f"         Status: {ripple['correlation_status']} | Proximity: {ripple['proximity_basis']}")

    # -----------------------------------------------------------------------
    # Test 2: JSON File Ingestion & JSON-Serializability Test
    # -----------------------------------------------------------------------
    print("\n[TEST 2] Testing File Ingestion (.json) and JSON Serializability...")

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp_json:
        json.dump(mock_raw_records, tmp_json)
        tmp_json_path = tmp_json.name

    try:
        file_response = backend.process_file(tmp_json_path)
        assert file_response["total_anomalies"] == 3

        # Test full JSON serialization
        json_output = file_response.to_json()
        assert isinstance(json_output, str)
        reparsed = json.loads(json_output)
        assert reparsed["total_anomalies"] == 3
        assert len(reparsed["alerts"]) == 3
        assert "spatial_clusters" in reparsed
        assert "ripple_hypotheses" in reparsed

        print(f"[PASS] Successfully ingested from JSON file: {tmp_json_path}")
        print(f"[PASS] Successfully serialized unified backend response to valid JSON ({len(json_output)} bytes)")
    finally:
        if os.path.exists(tmp_json_path):
            os.remove(tmp_json_path)

    # -----------------------------------------------------------------------
    # Test 3: CSV File Ingestion Test (Standard Library Fallback)
    # -----------------------------------------------------------------------
    print("\n[TEST 3] Testing File Ingestion (.csv)...")

    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False, newline="") as tmp_csv:
        writer = csv.writer(tmp_csv)
        writer.writerow([
            "time_window", "category", "area_id", "observed_count",
            "expected_count", "anomaly_score", "is_anomaly", "statistical_evidence"
        ])
        writer.writerow(["2026-10-08 10:00 - 11:00", "Gas Leak", "Sector-8", "15", "2.0", "5.1", "True", '{"z": 5.1}'])
        writer.writerow(["2026-10-08 10:00 - 11:00", "Noise", "Sector-8", "5", "5.0", "0.2", "False", '{}'])
        tmp_csv_path = tmp_csv.name

    try:
        csv_response = backend.process_file(tmp_csv_path)
        assert csv_response["total_anomalies"] == 1
        assert csv_response["alerts"][0].severity == "HIGH"
        print(f"[PASS] Successfully ingested from CSV file: {tmp_csv_path}")
        print(f"       Extracted alert: {csv_response['alerts'][0].title} (Severity: {csv_response['alerts'][0].severity})")
    finally:
        if os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)

    print("\n" + "=" * 75)
    print("All end-to-end adapter tests passed cleanly without Person A's model!")
    print("=" * 75)
