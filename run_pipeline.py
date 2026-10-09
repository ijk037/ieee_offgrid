"""
run_pipeline.py - Person A: End-to-End CivicPulse AI ML Pipeline Runner

Executes the complete Person A pipeline in sequence:
1. Ingests and cleans raw complaints (data_pipeline.py) -> data/cleaned_complaints.csv
2. Computes rolling features and gap-filled grid (features.py) -> data/features.csv
3. Trains Isolation Forest and saves model.joblib (train_model.py)
4. Executes full-scale anomaly detection inference -> data/model_outputs.json
5. Prepares sample anomaly & visualization payload for Person B / UI demonstration.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone
import pandas as pd

from data_pipeline import load_and_clean_data
from features import compute_rolling_features
from train_model import run_training_pipeline, detect_anomalies
from schemas import AlertResponse

logger = logging.getLogger("civicpulse.runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def run_full_pipeline(
    raw_path: str = "data/raw_complaints.csv",
    cleaned_path: str = "data/cleaned_complaints.csv",
    features_path: str = "data/features.csv",
    model_path: str = "model.joblib",
    outputs_path: str = "data/model_outputs.json"
):
    print("="*60)
    print("=== CIVICPULSE AI - FULL END-TO-END PIPELINE EXECUTION ===")
    print("="*60)

    # Step 1: Ingestion & Cleaning
    logger.info("[Step 1/4] Ingesting and cleaning raw dataset...")
    cleaned_df = load_and_clean_data(raw_path, cleaned_path)
    print(f"  [OK] Cleaned {len(cleaned_df):,} records across {cleaned_df['area_id'].nunique()} wards.")

    # Step 2: Feature Engineering
    logger.info("[Step 2/4] Computing temporal rolling features...")
    features_df = compute_rolling_features(cleaned_df)
    features_df.to_csv(features_path, index=False)
    print(f"  [OK] Computed features for {len(features_df):,} observations.")

    # Step 3: Model Training
    logger.info("[Step 3/4] Training Isolation Forest model...")
    model = run_training_pipeline(features_path, model_path, contamination=0.03)
    print(f"  [OK] Model trained and saved to {model_path}.")

    # Step 4: Inference & Alert Export
    logger.info("[Step 4/4] Executing batch anomaly detection inference...")
    outputs = detect_anomalies(features_df, model=model)
    anomalies = [o for o in outputs if o.is_anomaly]

    # Save full outputs to JSON
    os.makedirs(os.path.dirname(os.path.abspath(outputs_path)), exist_ok=True)
    with open(outputs_path, "w") as f:
        json.dump([o.model_dump() for o in outputs], f, indent=2)
    
    # Save compact anomalies-only payload for fast UI & alert consumption
    anomalies_path = os.path.join(os.path.dirname(outputs_path), "anomalies_only.json")
    with open(anomalies_path, "w") as f:
        json.dump([a.model_dump() for a in anomalies], f, indent=2)

    print(f"  [OK] Inference completed: {len(outputs):,} records evaluated.")
    print(f"  [OK] Total Actionable Anomalies Flagged: {len(anomalies):,} ({len(anomalies)/len(outputs):.2%})")
    print(f"  [OK] Exported full model outputs to: {outputs_path}")
    print(f"  [OK] Exported compact anomalies to: {anomalies_path}")

    # Step 5: Highlight Demo Showcase Anomaly (Phase 6 requirement)
    if anomalies:
        top_anomaly = max(
            anomalies,
            key=lambda x: x.statistical_evidence.get("delta_count", 0)
        )
        # Find geographic coordinates for this ward from cleaned data
        ward_coords = cleaned_df[cleaned_df["area_id"] == top_anomaly.area_id]
        lat = float(ward_coords["latitude"].mean()) if not ward_coords.empty else 12.9716
        lon = float(ward_coords["longitude"].mean()) if not ward_coords.empty else 77.5946
        ward_title = ward_coords.iloc[0]["ward_title"] if not ward_coords.empty else top_anomaly.area_id

        demo_alert = {
            "alert_id": f"ALERT-DEMO-{top_anomaly.time_window}-{top_anomaly.area_id}",
            "severity": "CRITICAL" if top_anomaly.anomaly_score > 0.70 else "HIGH",
            "title": f"Critical Surge: {top_anomaly.category} in {ward_title} ({top_anomaly.area_id})",
            "description": (
                f"Observed {top_anomaly.observed_count} complaints on {top_anomaly.time_window} "
                f"against a 7-day expected baseline of {top_anomaly.expected_count:.1f} "
                f"(Surge Delta: +{top_anomaly.statistical_evidence['delta_count']:.1f}, "
                f"z-score: {top_anomaly.statistical_evidence['z_score']:.1f} sigma)."
            ),
            "coordinates": {"latitude": lat, "longitude": lon},
            "metrics": top_anomaly.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        demo_file = "data/demo_showcase_alert.json"
        with open(demo_file, "w") as f:
            json.dump(demo_alert, f, indent=2)

        print("\n" + "="*60)
        print("PHASE 6 DEMO SHOWCASE ARTIFACT CREATED")
        print("="*60)
        print(f"Ward:        {ward_title} ({top_anomaly.area_id})")
        print(f"Category:    {top_anomaly.category}")
        print(f"Date:        {top_anomaly.time_window}")
        print(f"Observed:    {top_anomaly.observed_count} complaints (Expected: {top_anomaly.expected_count:.1f})")
        print(f"Z-Score:     +{top_anomaly.statistical_evidence['z_score']} sigma")
        print(f"Score:       {top_anomaly.anomaly_score}")
        print(f"Coordinates: {lat:.4f}, {lon:.4f}")
        print(f"Saved to:    {demo_file}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="CivicPulse AI End-to-End Runner")
    parser.add_argument("--raw", default="data/raw_complaints.csv")
    parser.add_argument("--cleaned", default="data/cleaned_complaints.csv")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--outputs", default="data/model_outputs.json")
    args = parser.parse_args()

    run_full_pipeline(args.raw, args.cleaned, args.features, args.model, args.outputs)


if __name__ == "__main__":
    main()
