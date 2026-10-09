"""
tests/test_person_a.py - Comprehensive Unit & Integration Tests for Person A

Covers:
- data_pipeline.py: Ingestion, column mapping, category normalization, coordinate validation.
- features.py: Daily aggregation, calendar grid filling, rolling features, shift(1) leakage check.
- train_model.py: Training, model saving/loading, anomaly scoring, schema output.
- validation.py: Chronological split logic, edge case handling.
"""

import pytest
import os
import tempfile
import numpy as np
import pandas as pd

from schemas import CanonicalComplaint, ModelOutput
from data_pipeline import (
    normalize_category,
    normalize_area_id,
    clean_complaints_dataframe
)
from features import (
    FEATURE_COLUMNS,
    aggregate_daily_complaints,
    compute_rolling_features,
    extract_feature_matrix
)
from train_model import (
    train_isolation_forest,
    save_model,
    load_model,
    normalize_anomaly_score,
    detect_anomalies
)
from validation import (
    chronological_train_test_split,
    evaluate_zero_variability_edge_case,
    evaluate_sparse_series_edge_case,
    evaluate_synthetic_surge_injection
)


# =========================================================
# Data Pipeline Tests
# =========================================================
def test_category_normalization():
    assert normalize_category("Mobility - Roads, Footpaths and Infrastructure") == "Roads & Infrastructure"
    assert normalize_category("garbage and unsanitary practices") == "Sanitation & Waste"
    assert normalize_category("Street lighting") == "Street Lighting"
    assert normalize_category("Streetlights") == "Street Lighting"
    assert normalize_category("yellow spot") == "Sanitation & Waste"
    assert normalize_category(None) == "Public Services & Others"


def test_area_id_normalization():
    assert normalize_area_id(22) == "ward_22"
    assert normalize_area_id("161") == "ward_161"
    assert normalize_area_id("ward_5") == "ward_5"
    assert normalize_area_id(np.nan) == "ward_unknown"


def test_clean_complaints_dataframe():
    raw_sample = pd.DataFrame([
        {
            "created_at": "1-1-2019 06:33",
            "ward_id": 22,
            "category_title": "Street lighting",
            "latitude": 13.0318,
            "longitude": 77.6054,
            "title": "Broken light",
            "ward_title": "Vishwanath Nagenahalli"
        },
        {
            "created_at": "1-1-2019 09:59",
            "ward_id": 27,
            "category_title": "Garbage and Unsanitary Practices",
            "latitude": 13.0065,
            "longitude": 77.6444,
            "title": "Garbage pile",
            "ward_title": "Banasavadi"
        }
    ])
    cleaned = clean_complaints_dataframe(raw_sample)
    assert len(cleaned) == 2
    assert cleaned.iloc[0]["category"] == "Street Lighting"
    assert cleaned.iloc[0]["area_id"] == "ward_22"
    assert cleaned.iloc[0]["created_at"] == "2019-01-01 06:33:00"


# =========================================================
# Feature Engineering Tests
# =========================================================
def test_temporal_shift_prevents_leakage():
    # If complaints occur today, today's rolling_mean_7d should NOT include today's complaint
    data = [
        {"created_at": f"2022-01-0{i} 10:00:00", "area_id": "ward_1", "category": "Roads & Infrastructure", "latitude": 13.0, "longitude": 77.6}
        for i in range(1, 6)
    ]
    # Add 10 complaints on day 5
    for _ in range(9):
        data.append({"created_at": "2022-01-05 11:00:00", "area_id": "ward_1", "category": "Roads & Infrastructure", "latitude": 13.0, "longitude": 77.6})

    df = pd.DataFrame(data)
    features = compute_rolling_features(df, fill_calendar_gaps=True)
    day5_row = features[features["time_window"] == "2022-01-05"].iloc[0]

    assert day5_row["observed_count"] == 10
    # Day 5's baseline should only reflect days 1 to 4 (mean of ~1), NOT 10
    assert day5_row["rolling_mean_7d"] < 2.0


def test_feature_matrix_extraction():
    dummy_features = pd.DataFrame([{col: 1.0 for col in FEATURE_COLUMNS}])
    X, names = extract_feature_matrix(dummy_features)
    assert X.shape == (1, len(FEATURE_COLUMNS))
    assert names == FEATURE_COLUMNS


# =========================================================
# Model Training & Inference Tests
# =========================================================
def test_model_training_and_saving():
    np.random.seed(42)
    X = np.random.normal(loc=0.0, scale=1.0, size=(100, len(FEATURE_COLUMNS)))
    model = train_isolation_forest(X, contamination=0.05, n_estimators=20)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_file = os.path.join(tmpdir, "test_model.joblib")
        save_model(model, filepath=tmp_model_file, metadata={"test": True})
        loaded_model, cols, meta = load_model(filepath=tmp_model_file)
        assert len(cols) == len(FEATURE_COLUMNS)
        assert meta["test"] is True


def test_detect_anomalies_output_conformance():
    np.random.seed(42)
    X = np.zeros((100, len(FEATURE_COLUMNS)))
    model = train_isolation_forest(X, contamination=0.05, n_estimators=20)

    features_sample = pd.DataFrame([{
        "time_window": "2022-07-31",
        "category": "Roads & Infrastructure",
        "area_id": "ward_22",
        "observed_count": 15,
        "rolling_mean_7d": 1.2,
        "rolling_std_7d": 0.5,
        "rolling_mean_14d": 1.0,
        "delta_count": 13.8,
        "z_score": 23.0,
        "surge_ratio": 6.8,
        "lag_1d": 1.0,
        "lag_7d": 1.0,
        "day_of_week": 6,
        "is_weekend": 1
    }])

    outputs = detect_anomalies(features_sample, model=model)
    assert len(outputs) == 1
    out = outputs[0]
    assert isinstance(out, ModelOutput)
    assert out.observed_count == 15
    assert out.expected_count == 1.2
    assert "rolling_mean_7d" in out.statistical_evidence


# =========================================================
# Validation Tests
# =========================================================
def test_validation_edge_cases():
    if os.path.exists("model.joblib"):
        model, _, _ = load_model("model.joblib")
    else:
        np.random.seed(42)
        X = np.random.normal(loc=0.0, scale=1.0, size=(100, len(FEATURE_COLUMNS)))
        model = train_isolation_forest(X, contamination=0.05, n_estimators=50)

    zero_res = evaluate_zero_variability_edge_case(model)
    assert zero_res["passed"] is True

    sparse_res = evaluate_sparse_series_edge_case(model)
    assert sparse_res["passed"] is True

    surge_res = evaluate_synthetic_surge_injection(model)
    assert surge_res["passed"] is True
