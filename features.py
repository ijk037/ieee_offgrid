"""
features.py - Person A: Feature Engineering & Temporal Aggregation

Responsibilities:
- Aggregate cleaned complaint records into canonical (area_id, category, time_window) observations.
- Construct continuous daily time grid ensuring missing days are represented as 0 counts (preventing sparse time-series distortion).
- Compute robust rolling window statistics (7-day, 14-day rolling mean and std with shift(1) to prevent lookahead bias).
- Generate lag features (lag_1d, lag_7d), calendar signals, and statistical surge metrics (z-score, surge ratio).
- Export standardized feature matrices ready for IsolationForest training and online inference.
"""

import os
import logging
import argparse
from typing import List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("civicpulse.features")

# Standard ML Feature Set
FEATURE_COLUMNS: List[str] = [
    "observed_count",
    "rolling_mean_7d",
    "rolling_std_7d",
    "rolling_mean_14d",
    "delta_count",
    "z_score",
    "surge_ratio",
    "lag_1d",
    "lag_7d",
    "day_of_week",
    "is_weekend"
]


def aggregate_daily_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups cleaned complaints by area_id, category, and date into daily complaint counts.
    """
    df = df.copy()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["created_at"]).dt.date

    agg = (
        df.groupby(["area_id", "category", "date"])
        .size()
        .reset_index(name="observed_count")
    )
    agg["date"] = pd.to_datetime(agg["date"])
    return agg


def compute_rolling_features(
    df_cleaned: pd.DataFrame,
    fill_calendar_gaps: bool = True
) -> pd.DataFrame:
    """
    Computes rolling baseline features and surge metrics for each area-category pair.
    Uses shift(1) before rolling aggregations to strictly enforce zero-lookahead bias.
    """
    logger.info("Aggregating complaints into daily counts...")
    agg = aggregate_daily_complaints(df_cleaned)

    if agg.empty:
        raise ValueError("Cannot compute features on empty aggregated dataset.")

    if fill_calendar_gaps:
        logger.info("Constructing continuous temporal calendar grid...")
        # Pivot into matrix (date x [area_id, category])
        piv = agg.pivot_table(
            index="date",
            columns=["area_id", "category"],
            values="observed_count",
            fill_value=0
        )
        # Reindex across continuous daily date range
        all_dates = pd.date_range(piv.index.min(), piv.index.max(), freq="D")
        piv = piv.reindex(all_dates, fill_value=0)
        piv.index.name = "date"

        # Shift(1) to avoid lookahead leakage
        shifted = piv.shift(1)

        logger.info("Computing vectorized rolling and lag metrics...")
        r_mean_7 = shifted.rolling(7, min_periods=1).mean()
        r_std_7 = shifted.rolling(7, min_periods=1).std().fillna(0.0)
        r_mean_14 = shifted.rolling(14, min_periods=1).mean()
        lag_1 = piv.shift(1).fillna(0.0)
        lag_7 = piv.shift(7).fillna(0.0)

        # Reshape back to long format
        count_long = piv.stack(level=["area_id", "category"], future_stack=True).rename("observed_count")
        mean7_long = r_mean_7.stack(level=["area_id", "category"], future_stack=True).rename("rolling_mean_7d")
        std7_long = r_std_7.stack(level=["area_id", "category"], future_stack=True).rename("rolling_std_7d")
        mean14_long = r_mean_14.stack(level=["area_id", "category"], future_stack=True).rename("rolling_mean_14d")
        lag1_long = lag_1.stack(level=["area_id", "category"], future_stack=True).rename("lag_1d")
        lag7_long = lag_7.stack(level=["area_id", "category"], future_stack=True).rename("lag_7d")

        features_df = pd.concat(
            [count_long, mean7_long, std7_long, mean14_long, lag1_long, lag7_long],
            axis=1
        ).reset_index()

    else:
        # Non-gap-filled mode (operates on active days only)
        features_df = agg.sort_values(by=["area_id", "category", "date"]).copy()
        grouped = features_df.groupby(["area_id", "category"])["observed_count"]
        shifted = grouped.shift(1)
        features_df["rolling_mean_7d"] = shifted.rolling(7, min_periods=1).mean().fillna(0.0)
        features_df["rolling_std_7d"] = shifted.rolling(7, min_periods=1).std().fillna(0.0)
        features_df["rolling_mean_14d"] = shifted.rolling(14, min_periods=1).mean().fillna(0.0)
        features_df["lag_1d"] = grouped.shift(1).fillna(0.0)
        features_df["lag_7d"] = grouped.shift(7).fillna(0.0)

    # Statistical derivations
    features_df["observed_count"] = features_df["observed_count"].fillna(0).astype(int)
    features_df["rolling_mean_7d"] = features_df["rolling_mean_7d"].fillna(0.0).round(3)
    features_df["rolling_std_7d"] = features_df["rolling_std_7d"].fillna(0.0).round(3)
    features_df["rolling_mean_14d"] = features_df["rolling_mean_14d"].fillna(0.0).round(3)
    features_df["lag_1d"] = features_df["lag_1d"].fillna(0.0).astype(float)
    features_df["lag_7d"] = features_df["lag_7d"].fillna(0.0).astype(float)

    # Surge delta & z-score
    features_df["delta_count"] = features_df["observed_count"] - features_df["rolling_mean_7d"]
    features_df["z_score"] = (
        (features_df["observed_count"] - features_df["rolling_mean_7d"])
        / (features_df["rolling_std_7d"] + 0.1)
    ).round(3)
    features_df["surge_ratio"] = (
        features_df["observed_count"] / (features_df["rolling_mean_7d"] + 1.0)
    ).round(3)

    # Temporal context
    features_df["time_window"] = pd.to_datetime(features_df["date"]).dt.strftime("%Y-%m-%d")
    features_df["day_of_week"] = pd.to_datetime(features_df["date"]).dt.dayofweek
    features_df["is_weekend"] = (features_df["day_of_week"] >= 5).astype(int)

    # Filter out empty warm-up records where both observed_count and rolling_mean_7d are 0
    # to focus training/inference on non-trivial activity, while retaining days with complaints or non-zero baselines
    active_mask = (features_df["observed_count"] > 0) | (features_df["rolling_mean_7d"] > 0.1)
    filtered_df = features_df[active_mask].copy().reset_index(drop=True)

    logger.info(
        f"Features generated: {len(filtered_df)} observations across "
        f"{filtered_df['area_id'].nunique()} areas and {filtered_df['category'].nunique()} categories."
    )
    return filtered_df


def extract_feature_matrix(df_features: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts the feature matrix X and verified feature column names.
    """
    for col in FEATURE_COLUMNS:
        if col not in df_features.columns:
            raise KeyError(f"Missing required feature column: {col}")

    X = df_features[FEATURE_COLUMNS].values.astype(np.float32)
    # Replace any accidental NaN or Inf
    X = np.nan_to_num(X, nan=0.0, posinf=999.0, neginf=-999.0)
    return X, FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description="CivicPulse AI - Person A Feature Engineering")
    parser.add_argument("--input", default="data/cleaned_complaints.csv", help="Path to cleaned complaints CSV")
    parser.add_argument("--output", default="data/features.csv", help="Path for output features CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} does not exist. Run data_pipeline.py first.")
        return

    cleaned_df = pd.read_csv(args.input)
    features_df = compute_rolling_features(cleaned_df)
    features_df.to_csv(args.output, index=False)
    print(f"Features saved to {args.output} (Shape: {features_df.shape})")
    print("\nFeature Summary:")
    print(features_df[FEATURE_COLUMNS].describe().round(2))


if __name__ == "__main__":
    main()
