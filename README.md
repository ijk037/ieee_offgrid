# CivicPulse AI 🏙️⚡

An intelligent, real-time civic complaint anomaly detection and spatial intelligence platform. Built for municipal monitoring and predictive governance.

---

## 👥 Repository & File Ownership Matrix

This repository enforces non-intersecting developer workstreams to ensure zero merge conflicts:

| File | Owner | Description | Status |
| :--- | :--- | :--- | :--- |
| [`schemas.py`](file:///D:/vsc/ieee_offgrid/schemas.py) | **Joint (Frozen)** | Canonical data contracts & Pydantic models (`ModelOutput`, `AlertResponse`, `CanonicalComplaint`) | ✅ **Frozen (Phase 0)** |
| [`requirements.txt`](file:///D:/vsc/ieee_offgrid/requirements.txt) | **Joint** | Shared project dependencies | ✅ **Complete** |
| [`tests/test_contract.py`](file:///D:/vsc/ieee_offgrid/tests/test_contract.py) | **Joint** | Shared contract verification test suite | ✅ **100% Passed** |
| [`data_pipeline.py`](file:///D:/vsc/ieee_offgrid/data_pipeline.py) | **Person A** | Raw ingestion, encoding fallback, datetime parsing, canonical categorization | ✅ **Complete (Phase 1)** |
| [`features.py`](file:///D:/vsc/ieee_offgrid/features.py) | **Person A** | Daily temporal aggregation, zero-filled continuous calendar grid, rolling statistics | ✅ **Complete (Phase 2)** |
| [`train_model.py`](file:///D:/vsc/ieee_offgrid/train_model.py) | **Person A** | Isolation Forest training, model persistence (`model.joblib`), calibrated scoring | ✅ **Complete (Phase 2)** |
| [`validation.py`](file:///D:/vsc/ieee_offgrid/validation.py) | **Person A** | Chronological temporal split validation, edge case stress testing | ✅ **Complete (Phase 3)** |
| [`run_pipeline.py`](file:///D:/vsc/ieee_offgrid/run_pipeline.py) | **Person A** | End-to-end automated ML pipeline execution runner | ✅ **Complete (Phase 4)** |
| [`benchmark.py`](file:///D:/vsc/ieee_offgrid/benchmark.py) | **Person A** | Throughput, latency, memory profiling, and SLA compliance | ✅ **Complete (Phase 5)** |
| [`BENCHMARK_REPORT.md`](file:///D:/vsc/ieee_offgrid/BENCHMARK_REPORT.md) | **Person A** | Comprehensive system performance report | ✅ **Complete** |
| [`spatial_analysis.py`](file:///D:/vsc/ieee_offgrid/spatial_analysis.py) | **Person B** | Geographic grouping, cross-category overlap & Civic Ripple Engine | ⏳ *Person B* |
| [`alert_service.py`](file:///D:/vsc/ieee_offgrid/alert_service.py) | **Person B** | Anomaly-to-alert formatting & explanation generator | ⏳ *Person B* |
| [`app_adapter.py`](file:///D:/vsc/ieee_offgrid/app_adapter.py) | **Person B** | Model-to-UI data binding and priority ranking | ⏳ *Person B* |
| [`app.py`](file:///D:/vsc/ieee_offgrid/app.py) | **Person B** | Backend FastAPI / Stitch UI integration | ⏳ *Person B* |

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Tests
```bash
python -m pytest tests/ -v
```

### 3. Run the Complete End-to-End ML Pipeline
```bash
python run_pipeline.py
```
This executes:
1. Ingestion (`data/raw_complaints.csv` ➡️ `data/cleaned_complaints.csv`)
2. Feature engineering (`data/features.csv`)
3. Isolation Forest training (`model.joblib`)
4. Full inference output (`data/model_outputs.json` & `data/anomalies_only.json`)
5. Demo showcase extraction (`data/demo_showcase_alert.json`)

---

## 📊 Dataset & Pipeline Overview (Person A)

- **Input Dataset:** Real Bengaluru BBMP Civic Complaints (16,071 records across 198 wards, 2019-01-01 to 2022-07-31).
- **Taxonomy Normalization:** Mapped 43 noisy raw categories into 12 clean canonical domains:
  - `Roads & Infrastructure` (5,204)
  - `Sanitation & Waste` (4,939)
  - `Street Lighting` (1,500)
  - `Traffic & Safety` (1,013)
  - `Animal Control` (860)
  - `Water Supply & Drainage` (834)
  - `Public Services & Others` (451)
  - `Pollution` (438)
  - `Electricity & Power` (241)
  - `Parks & Greenery` (233)
  - `Civic Amenities` (194)
  - `Public Safety` (164)
- **Temporal Reindexing:** Full daily calendar grid across 1,308 days; zero-complaint days are represented with `0` counts to eliminate sparse time-series distortion.
- **Lookahead Bias Prevention:** Enforced via `shift(1)` rolling statistics (`rolling_mean_7d`, `rolling_std_7d`, `rolling_mean_14d`, `lag_1d`, `lag_7d`).

---

## 🎯 Model & Anomaly Scoring

- **Algorithm:** Scikit-learn `IsolationForest(n_estimators=150, contamination=0.03)`
- **Artifact:** [`model.joblib`](file:///D:/vsc/ieee_offgrid/model.joblib)
- **Calibrated Scoring:**
  - `0.0 - 0.30`: Normal background activity
  - `0.30 - 0.55`: Elevated activity / minor variance
  - `0.55 - 0.75`: Anomaly threshold
  - `0.75 - 1.00`: Critical civic surge
- **Actionable Filter:** Suppresses quiet/dormant periods; only flags surges where `observed_count > expected_count` and `z_score >= 1.5`.

---

## 🧪 Validation & Edge Case Results

Run validation via:
```bash
python validation.py
```
- **Train Window (2019-01-01 to 2021-06-30):** 76,033 observations, 1.42% anomaly rate.
- **Held-Out Test Window (2021-07-01 to 2022-07-31):** 17,784 observations, 1.06% anomaly rate.
- **Rate Stability Ratio:** `0.746` (no catastrophic drift).
- **Edge Cases Tested & Passed:**
  - ✅ **Zero-Variability Edge Case:** Flatlines / zero complaints never crash or trigger false alerts.
  - ✅ **Sparse Series Single-Event Edge Case:** Isolated noise complaints do not cause alert fatigue.
  - ✅ **Controlled Surge Sensitivity:** Synthetic emergency surge detected at `0.84` score and `+48.75` z-score.

---

## ⚡ Performance & Benchmarks

| Metric | Measured Value | Standard SLA |
| :--- | :--- | :--- |
| **Ingestion Throughput** | **13,881 rows/sec** | > 2,000 rows/sec |
| **Feature Engineering Latency** | **6.59s** (93,817 observations) | < 30s |
| **Model Training Duration** | **0.86s** (150 trees) | < 15s |
| **Inference Throughput** | **12,601 obs/sec** | > 5,000 obs/sec |
| **Per-Observation Latency** | **0.079 ms** | < 1.0 ms |
| **Model Size** | **1.15 MB** | < 50 MB |
| **Memory Footprint** | **463 MB** | < 1,024 MB |

---

## 🤝 Hand-off for Person B

Person B can directly import the frozen contracts and consume the generated outputs:

```python
from schemas import ModelOutput, AlertResponse
import json

# Pre-computed anomaly alerts (1,271 records):
with open("data/anomalies_only.json") as f:
    anomalies = json.load(f)

# Or load model directly for online scoring:
from train_model import load_model, detect_anomalies
model, feature_cols, meta = load_model("model.joblib")
```

### 🌟 Phase 6 Demo Showcase Artifact
Ready at [`data/demo_showcase_alert.json`](file:///D:/vsc/ieee_offgrid/data/demo_showcase_alert.json):
- **Ward:** Uttarahalli (`ward_184`)
- **Category:** `Sanitation & Waste`
- **Date:** `2021-01-11`
- **Observed:** 18 complaints (Expected: 0.6)
- **Surge Magnitude:** `+16.2 sigma` z-score, `0.88` anomaly score
- **Coordinates:** `12.8968, 77.5399`
