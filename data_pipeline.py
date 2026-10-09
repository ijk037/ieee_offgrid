"""
data_pipeline.py - Person A: Civic Complaint Ingestion & Cleaning Pipeline

Responsibilities:
- Ingest raw civic complaint records (CSV).
- Normalize column names to canonical schema (created_at, category, area_id, latitude, longitude).
- Parse and validate datetime formats across mixed string patterns.
- Normalize civic complaint categories into consistent canonical domain taxonomy.
- Clean and sanitize geographic coordinates.
- Filter invalid / incomplete records.
- Export cleaned canonical dataset for downstream feature engineering and ML training.
"""

import os
import sys
import logging
import argparse
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("civicpulse.data_pipeline")

# Standard bounding box for validation (Bengaluru metropolitan region)
DEFAULT_LAT_MIN, DEFAULT_LAT_MAX = 12.0, 14.0
DEFAULT_LON_MIN, DEFAULT_LON_MAX = 76.0, 79.0

# Canonical Category Taxonomy Mapping
CATEGORY_NORMALIZATION_MAP: Dict[str, str] = {
    # Roads & Infrastructure
    "mobility - roads, footpaths and infrastructure": "Roads & Infrastructure",
    "roads and footpaths": "Roads & Infrastructure",
    "mobility - roads, public transport": "Roads & Infrastructure",
    
    # Sanitation & Waste
    "garbage and unsanitary practices": "Sanitation & Waste",
    "yellow spot": "Sanitation & Waste",
    "sanitation": "Sanitation & Waste",
    "solid waste management": "Sanitation & Waste",
    "waste management": "Sanitation & Waste",
    
    # Street Lighting
    "street lighting": "Street Lighting",
    "streetlights": "Street Lighting",
    
    # Traffic & Mobility
    "traffic and road safety": "Traffic & Safety",
    "public transport - bmtc": "Traffic & Safety",
    "public transport (bmtc and metro)": "Traffic & Safety",
    "public transport - ksrtc": "Traffic & Safety",
    
    # Water & Drainage
    "water supply and services": "Water Supply & Drainage",
    "water supply": "Water Supply & Drainage",
    "sewerage systems": "Water Supply & Drainage",
    "storm water drains": "Water Supply & Drainage",
    "lakes": "Water Supply & Drainage",
    
    # Electricity & Power
    "electricity and power supply": "Electricity & Power",
    "electricity & power": "Electricity & Power",
    "power supply": "Electricity & Power",
    
    # Animal Control
    "animal husbandry": "Animal Control",
    "animal catcher": "Animal Control",
    
    # Public Safety
    "crime and safety": "Public Safety",
    "safety and crime": "Public Safety",
    "fire safety": "Public Safety",
    
    # Parks & Environment
    "parks & recreation": "Parks & Greenery",
    "trees and saplings": "Parks & Greenery",
    "parks & garden": "Parks & Greenery",
    "playgrounds": "Parks & Greenery",
    
    # Pollution
    "pollution": "Pollution",
    "fire pollution": "Pollution",
    
    # Civic Amenities & Public Services
    "public toilets": "Civic Amenities",
    "community infrastructure and services": "Civic Amenities",
    "certificates": "Public Services & Others",
    "pwd": "Public Services & Others",
    "ration card": "Public Services & Others",
    "pension": "Public Services & Others",
    "emp grievance redressal": "Public Services & Others",
    "covid 19": "Public Services & Others",
    "proibition & sale of tobacco products, plastic carry bags": "Public Services & Others",
    "others": "Public Services & Others"
}


def normalize_category(raw_category: Optional[str]) -> str:
    """
    Maps varied category labels into clean, canonical category strings.
    """
    if not raw_category or pd.isna(raw_category):
        return "Public Services & Others"
    clean_cat = str(raw_category).strip().lower()
    return CATEGORY_NORMALIZATION_MAP.get(clean_cat, str(raw_category).strip())


def normalize_area_id(raw_area: Any) -> str:
    """
    Standardizes ward / area identifiers into canonical format (e.g. 'ward_22').
    """
    if pd.isna(raw_area):
        return "ward_unknown"
    try:
        val = int(float(raw_area))
        return f"ward_{val}"
    except (ValueError, TypeError):
        clean_str = str(raw_area).strip().lower().replace(" ", "_")
        return f"ward_{clean_str}" if not clean_str.startswith("ward_") else clean_str


def clean_complaints_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw DataFrame and conforms to CanonicalComplaint schema.
    """
    logger.info(f"Initial raw rows: {len(df)}")
    df = df.copy()

    # Determine created_at column
    time_col = None
    for candidate in ["created_at", "created_date", "timestamp", "date", "complaint_date"]:
        if candidate in df.columns:
            time_col = candidate
            break
    if not time_col:
        raise ValueError("Could not find datetime column in raw dataset.")

    # Parse created_at
    df["created_at_parsed"] = pd.to_datetime(df[time_col], format="mixed", errors="coerce")
    initial_valid_dates = df["created_at_parsed"].notna().sum()
    df = df.dropna(subset=["created_at_parsed"])
    logger.info(f"Rows after datetime parsing ({time_col}): {len(df)} (dropped {initial_valid_dates - len(df)} invalid)")

    # Normalize category
    cat_col = None
    for candidate in ["category_title", "category", "category_name", "sub_category_title"]:
        if candidate in df.columns:
            cat_col = candidate
            break
    if not cat_col:
        raise ValueError("Could not find category column in raw dataset.")
    
    df["category"] = df[cat_col].apply(normalize_category)

    # Normalize area_id
    area_col = None
    for candidate in ["ward_id", "area_id", "ward_number", "ward_no", "ward_title"]:
        if candidate in df.columns:
            area_col = candidate
            break
    if not area_col:
        raise ValueError("Could not find area/ward column in raw dataset.")
    
    df["area_id"] = df[area_col].apply(normalize_area_id)

    # Ward title if available
    if "ward_title" in df.columns:
        df["ward_title"] = df["ward_title"].fillna("Unknown Ward").astype(str).str.strip()
    else:
        df["ward_title"] = df["area_id"]

    # Parse and validate coordinates
    lat_col = "latitude" if "latitude" in df.columns else None
    lon_col = "longitude" if "longitude" in df.columns else None

    if lat_col and lon_col:
        df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
        # Validate coordinate range
        valid_coords = (
            df["latitude"].isna() |
            ((df["latitude"] >= DEFAULT_LAT_MIN) & (df["latitude"] <= DEFAULT_LAT_MAX))
        ) & (
            df["longitude"].isna() |
            ((df["longitude"] >= DEFAULT_LON_MIN) & (df["longitude"] <= DEFAULT_LON_MAX))
        )
        dropped_coords = (~valid_coords).sum()
        if dropped_coords > 0:
            logger.warning(f"Dropping {dropped_coords} rows with out-of-bounds coordinates.")
            df = df[valid_coords]

    # Optional descriptive columns
    if "title" in df.columns:
        df["title"] = df["title"].fillna("").astype(str)
    else:
        df["title"] = ""

    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str)
    else:
        df["description"] = ""

    # Format canonical created_at string
    df["created_at"] = df["created_at_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Select canonical columns
    canonical_cols = [
        "created_at",
        "category",
        "area_id",
        "latitude",
        "longitude",
        "ward_title",
        "title",
        "description"
    ]
    canonical_cols = [c for c in canonical_cols if c in df.columns]
    cleaned_df = df[canonical_cols].copy()

    # Sort chronologically
    cleaned_df = cleaned_df.sort_values(by="created_at").reset_index(drop=True)
    logger.info(f"Cleaning complete. Output clean rows: {len(cleaned_df)}")
    return cleaned_df


def load_and_clean_data(
    input_path: str,
    output_path: Optional[str] = None,
    encoding: Optional[str] = None
) -> pd.DataFrame:
    """
    Ingests raw file, runs cleaning pipeline, and optionally writes output CSV.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Robust encoding detection
    encodings_to_try = [encoding] if encoding else ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    df = None
    for enc in encodings_to_try:
        if not enc:
            continue
        try:
            logger.info(f"Attempting to load {input_path} with encoding '{enc}'...")
            df = pd.read_csv(input_path, encoding=enc)
            logger.info(f"Successfully loaded {len(df)} rows using '{enc}'.")
            break
        except (UnicodeDecodeError, Exception) as e:
            logger.warning(f"Failed with encoding '{enc}': {e}")

    if df is None:
        raise ValueError(f"Could not read {input_path} with any supported encoding.")

    cleaned_df = clean_complaints_dataframe(df)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset saved to: {output_path}")

    return cleaned_df


def main():
    parser = argparse.ArgumentParser(description="CivicPulse AI - Person A Ingestion Pipeline")
    parser.add_argument("--input", default="data/raw_complaints.csv", help="Path to raw complaints CSV")
    parser.add_argument("--output", default="data/cleaned_complaints.csv", help="Path for cleaned output CSV")
    parser.add_argument("--summary", action="store_true", help="Print summary statistics")
    args = parser.parse_args()

    cleaned_df = load_and_clean_data(args.input, args.output)

    if args.summary:
        print("\n=== Cleaned Dataset Summary ===")
        print(f"Total Records: {len(cleaned_df):,}")
        print(f"Date Range: {cleaned_df['created_at'].min()} to {cleaned_df['created_at'].max()}")
        print(f"Unique Areas: {cleaned_df['area_id'].nunique()}")
        print(f"Unique Categories: {cleaned_df['category'].nunique()}")
        print("\nCategory Distribution:")
        print(cleaned_df["category"].value_counts())


if __name__ == "__main__":
    main()
