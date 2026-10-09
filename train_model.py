"""
train_model.py - Person A: Isolation Forest Anomaly Detection Model

Responsibilities:
- Train scikit-learn IsolationForest on historical features.
- Persist model artifact (model.joblib) containing the trained estimator, feature list, and metadata.
- Implement live inference engine that maps raw predictions to canonical ModelOutput instances (from schemas.py).
- Filter out non-actionable negative deviations (quiet days), ensuring alerts reflect true civic surges.
- Format statistical evidence explaining why an event is flagged as anomalous.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

from schemas import ModelOutput
from features import FEATURE_COLUMNS, compute_rolling_features, extract_feature_matrix

logger = logging.getLogger("civicpulse.train_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

DEFAULT_MODEL_PATH = "model.joblib"


def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.03,
    n_estimators: int = 150,
    random_state: int = 42
) -> IsolationForest:
    """
    Trains an Isolation Forest anomaly detector on the feature matrix.
    """
    logger.info(
        f"Training IsolationForest with {n_estimators} trees, "
        f"contamination={contamination}, samples={len(X)}..."
    )
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X)
    logger.info("Model training successfully completed.")
    return model


def save_model(
    model: IsolationForest,
    filepath: str = DEFAULT_MODEL_PATH,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Persists trained model and configuration metadata using joblib.
    """
    payload = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {}
    }
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(payload, filepath)
    logger.info(f"Model artifact saved to: {filepath}")

    # Also save copy to models/ directory if saving to root
    if filepath == DEFAULT_MODEL_PATH:
        models_dir_path = os.path.join("models", DEFAULT_MODEL_PATH)
        os.makedirs("models", exist_ok=True)
        joblib.dump(payload, models_dir_path)


def load_model(filepath: str = DEFAULT_MODEL_PATH) -> Tuple[IsolationForest, List[str], Dict[str, Any]]:
    """
    Loads saved model artifact and metadata.
    """
    if not os.path.exists(filepath):
        alt_path = os.path.join("models", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at {filepath} or {alt_path}")

    logger.info(f"Loading model artifact from {filepath}...")
    payload = joblib.load(filepath)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("feature_columns", FEATURE_COLUMNS), payload.get("metadata", {})
    else:
        # Fallback if raw estimator was saved
        return payload, FEATURE_COLUMNS, {}


def normalize_anomaly_score(
    raw_scores: np.ndarray,
    min_dec: float = -0.20,
    max_dec: float = 0.28
) -> np.ndarray:
    """
    Transforms IsolationForest raw decision function scores into an intuitive [0.0, 1.0] scale.
    - 0.0 - 0.30: Completely normal background activity
    - 0.30 - 0.55: Elevated activity / minor variance
    - 0.55 - 0.75: Anomaly threshold (outliers flagged by Isolation Forest)
    - 0.75 - 1.00: Severe / critical civic surges
    """
    # Inverse linear calibration with clipping
    norm = (max_dec - raw_scores) / (max_dec - min_dec + 1e-6)
    return np.clip(norm, 0.0, 1.0)


def detect_anomalies(
    features_df: pd.DataFrame,
    model: Optional[IsolationForest] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    min_observed_count: int = 2,
    min_z_score: float = 1.5
) -> List[ModelOutput]:
    """
    Executes inference over a features DataFrame and outputs validated ModelOutput objects.
    Enforces civic business logic: only flags surges where observed count exceeds baseline.
    """
    if model is None:
        model, feat_cols, _ = load_model(model_path)
    else:
        feat_cols = FEATURE_COLUMNS

    X, _ = extract_feature_matrix(features_df)
    
    # Raw decision scores and predictions
    raw_decisions = model.decision_function(X)
    predictions = model.predict(X)  # -1 for anomaly, 1 for inlier
    norm_scores = normalize_anomaly_score(raw_decisions)

    outputs: List[ModelOutput] = []

    for idx, row in features_df.reset_index(drop=True).iterrows():
        is_iso_anomaly = bool(predictions[idx] == -1)
        score = float(norm_scores[idx])
        observed = int(row["observed_count"])
        expected = float(row["rolling_mean_7d"])
        z = float(row.get("z_score", 0.0))
        surge_ratio = float(row.get("surge_ratio", 1.0))
        delta = float(row.get("delta_count", 0.0))

        # Actionable Civic Surge Logic:
        # 1. Model flagged as anomaly OR significant statistical surge (z_score >= 3.0)
        # 2. Observed complaints must exceed expected baseline
        # 3. Minimum absolute count threshold to avoid alerting on trivial counts
        is_surge = (observed > expected) and (observed >= min_observed_count)
        is_civic_anomaly = bool((is_iso_anomaly or (z >= 3.0)) and is_surge and (z >= min_z_score))

        evidence = {
            "rolling_mean_7d": round(expected, 2),
            "rolling_std_7d": round(float(row.get("rolling_std_7d", 0.0)), 2),
            "rolling_mean_14d": round(float(row.get("rolling_mean_14d", 0.0)), 2),
            "delta_count": round(delta, 2),
            "z_score": round(z, 2),
            "surge_ratio": round(surge_ratio, 2),
            "day_of_week": int(row.get("day_of_week", 0)),
            "is_weekend": bool(row.get("is_weekend", 0))
        }

        output = ModelOutput(
            time_window=str(row["time_window"]),
            category=str(row["category"]),
            area_id=str(row["area_id"]),
            observed_count=observed,
            expected_count=round(expected, 2),
            anomaly_score=round(score, 4),
            is_anomaly=is_civic_anomaly,
            statistical_evidence=evidence
        )
        outputs.append(output)

    return outputs


def run_training_pipeline(
    features_csv: str = "data/features.csv",
    model_save_path: str = DEFAULT_MODEL_PATH,
    contamination: float = 0.03
) -> IsolationForest:
    """
    Loads features, trains Isolation Forest, and persists model artifact.
    """
    logger.info(f"Loading feature records from {features_csv}...")
    df_feat = pd.read_csv(features_csv)
    X, feat_names = extract_feature_matrix(df_feat)
    
    metadata = {
        "training_samples": len(X),
        "contamination": contamination,
        "date_min": str(df_feat["time_window"].min()),
        "date_max": str(df_feat["time_window"].max()),
        "feature_names": feat_names
    }
    model = train_isolation_forest(X, contamination=contamination)
    save_model(model, model_save_path, metadata=metadata)
    return model


def main():
    parser = argparse.ArgumentParser(description="CivicPulse AI - Person A Model Training & Inference")
    parser.add_argument("--features", default="data/features.csv", help="Path to features CSV")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Destination for saved model")
    parser.add_argument("--contamination", type=float, default=0.03, help="Contamination rate")
    parser.add_argument("--inference", action="store_true", help="Run inference and output anomalies")
    parser.add_argument("--output-json", default="data/model_outputs.json", help="Path to save ModelOutput JSON")
    args = parser.parse_args()

    if not os.path.exists(args.features):
        print(f"Error: {args.features} not found. Run features.py first.")
        sys.exit(1)

    model = run_training_pipeline(args.features, args.model_path, args.contamination)

    if args.inference:
        df_feat = pd.read_csv(args.features)
        outputs = detect_anomalies(df_feat, model=model)
        anomalies = [o for o in outputs if o.is_anomaly]
        print(f"\nInference complete: {len(outputs)} total records evaluated.")
        print(f"Total Anomalies Flagged: {len(anomalies)} ({len(anomalies)/len(outputs):.2%})")

        # Save outputs to JSON
        import json
        with open(args.output_json, "w") as f:
            json.dump([o.model_dump() for o in outputs], f, indent=2)
        print(f"Model outputs saved to {args.output_json}")

        if anomalies:
            print("\nTop 5 Flagged Anomalies by Delta Count:")
            sorted_anomalies = sorted(
                anomalies,
                key=lambda x: x.statistical_evidence.get("delta_count", 0),
                reverse=True
            )[:5]
            for a in sorted_anomalies:
                print(
                    f" - [{a.time_window}] {a.area_id} | {a.category}: "
                    f"Observed={a.observed_count}, Expected={a.expected_count:.1f}, "
                    f"z={a.statistical_evidence['z_score']}, score={a.anomaly_score}"
                )


if __name__ == "__main__":
    main()
