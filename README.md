# CivicPulse AI 🏙️⚡

An intelligent, real-time municipal grievance anomaly detection, spatial intelligence, and predictive governance platform. Built for the **IEEE OFFGRID Hackathon (Tech Ideate, Manipal University Jaipur)**.

---

## 👥 Repository Architecture & Complete Implementation

CivicPulse AI divides responsibilities into clean, non-intersecting engineering workstreams that merge into a single production pipeline:

| Module / File | Component | Description | Status |
| :--- | :--- | :--- | :--- |
| [`schemas.py`](file:///D:/vsc/ieee_offgrid/schemas.py) | **Joint Contract** | Pydantic v2 data models (`CanonicalComplaint`, `ModelOutput`, `AlertResponse`) | ✅ **100% Complete** |
| [`requirements.txt`](file:///D:/vsc/ieee_offgrid/requirements.txt) | **Joint Config** | Dependencies (`scikit-learn`, `pydantic`, `fastapi`, `uvicorn`, `streamlit`, etc.) | ✅ **100% Complete** |
| [`Procfile`](file:///D:/vsc/ieee_offgrid/Procfile) | **Deployment** | Production startup command for cloud deployment (Render / Railway) | ✅ **100% Complete** |
| [`data_pipeline.py`](file:///D:/vsc/ieee_offgrid/data_pipeline.py) | **Person A (Data)** | Ingests raw BBMP complaints, Latin-1 fallback, datetime parsing, 43 ➡️ 12 taxonomy | ✅ **100% Complete** |
| [`features.py`](file:///D:/vsc/ieee_offgrid/features.py) | **Person A (Data)** | 1,308-day continuous calendar grid, zero-filling, `shift(1)` anti-leakage rolling features | ✅ **100% Complete** |
| [`train_model.py`](file:///D:/vsc/ieee_offgrid/train_model.py) | **Person A (ML)** | Scikit-learn `IsolationForest(n_estimators=150)`, calibrated 0.0–1.0 scoring | ✅ **100% Complete** |
| [`validation.py`](file:///D:/vsc/ieee_offgrid/validation.py) | **Person A (ML)** | Chronological 81/19 temporal split, zero-variability and synthetic surge stress tests | ✅ **100% Complete** |
| [`run_pipeline.py`](file:///D:/vsc/ieee_offgrid/run_pipeline.py) | **Person A (ML)** | End-to-end automated ML pipeline execution & demo artifact generator | ✅ **100% Complete** |
| [`benchmark.py`](file:///D:/vsc/ieee_offgrid/benchmark.py) | **Person A (ML)** | Throughput profiling, latency measurement (0.079ms per observation) | ✅ **100% Complete** |
| [`BENCHMARK_REPORT.md`](file:///D:/vsc/ieee_offgrid/BENCHMARK_REPORT.md) | **Person A (ML)** | Production SLA report and operational boundaries | ✅ **100% Complete** |
| [`alert_service.py`](file:///D:/vsc/ieee_offgrid/alert_service.py) | **Person B (Backend)** | Anomaly-to-alert formatting, severity calculation, plain-English explanations | ✅ **100% Complete** |
| [`spatial_analysis.py`](file:///D:/vsc/ieee_offgrid/spatial_analysis.py) | **Person B (Spatial)** | Geographic clustering, Haversine distance, and **Civic Ripple Engine** cascades | ✅ **100% Complete** |
| [`app_adapter.py`](file:///D:/vsc/ieee_offgrid/app_adapter.py) | **Person B (Backend)** | Clean bridge connecting ML predictions to downstream services (`CivicPulseBackend`) | ✅ **100% Complete** |
| [`app.py`](file:///D:/vsc/ieee_offgrid/app.py) | **Person B (UI/API)** | FastAPI backend server serving REST endpoints & multi-page Stitch Dashboard | ✅ **100% Complete** |
| [`stitch_civicpulse_ai_dashboard/`](file:///D:/vsc/ieee_offgrid/stitch_civicpulse_ai_dashboard) | **UI Interface** | 5 distinct web dashboard pages styled in Orange, Green, Black, and White | ✅ **100% Complete** |
| `tests/` & root test suites | **QA / Tests** | 30 unit, contract, and integration tests covering Person A and Person B | ✅ **30/30 Passed** |

---

## 🚀 Quickstart: Running Locally

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run the Full Test Suite
Verify that all 30 tests for both ML models and backend services pass:
```powershell
python -m pytest tests/ test_alert_service.py test_app_adapter.py test_spatial_analysis.py -v
```

### 3. Launch the Integrated Web Dashboard & API
Start the FastAPI server:
```powershell
python app.py
```
*(Or with Uvicorn directly: `uvicorn app:app --port 8000`)*

Open your browser to **[http://localhost:8000/](http://localhost:8000/)** to access the live dashboard:
- 🏠 **Command Center:** `http://localhost:8000/`
- 🗺️ **Spatial Ward Map:** `http://localhost:8000/spatial-ward-map`
- 🌊 **Civic Ripple Engine:** `http://localhost:8000/civic-ripple-engine`
- 📈 **Ward Analytics:** `http://localhost:8000/ward-analytics`
- 📋 **Triage & Incident Dispatch:** `http://localhost:8000/triage-and-dispatch`
- 📖 **Interactive Swagger API Docs:** `http://localhost:8000/docs`

---

## ⚡ Option: Run the End-to-End ML Pipeline Individually

If you wish to re-train the model, re-generate features, or run validation from scratch:

```powershell
# 1. Ingest & clean raw data
python data_pipeline.py --input dataset/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv --output data/cleaned_complaints.csv --summary

# 2. Extract 11 rolling features on continuous calendar grid
python features.py --input data/cleaned_complaints.csv --output data/features.csv

# 3. Fit Isolation Forest & export model.joblib (1.15 MB)
python train_model.py --features data/features.csv --inference --output-json data/model_outputs.json

# 4. Run temporal validation & edge-case stress tests
python validation.py

# 5. Measure latency and throughput SLAs
python benchmark.py

# OR run everything with a single command:
python run_pipeline.py
```

---

## 📊 Dataset Profile & Authenticity

CivicPulse AI trains on authentic municipal grievance data from Bengaluru, Karnataka (sourced from the IChangeMyCity / Janaagraha platform for the BBMP):

| Metric | Dataset Details |
| :--- | :--- |
| **Total Ingested Records** | **16,071 real citizen complaints** |
| **Time Horizon** | **January 1, 2019 – July 31, 2022** (3.5 continuous years) |
| **Administrative Coverage** | **198 Municipal Wards** across Bengaluru |
| **GPS Bounds** | Latitude `12.71° N` to `13.18° N`, Longitude `77.43° E` to `77.81° E` (100% valid coordinates) |
| **Consolidated Categories** | 43 raw labels normalized into **12 canonical municipal domains**: Roads & Infrastructure (5,204), Sanitation & Waste (4,939), Street Lighting (1,500), Traffic & Safety (1,013), Animal Control (860), Water Supply & Drainage (834), Public Services & Others (451), Pollution (438), Electricity & Power (241), Parks & Greenery (233), Civic Amenities (194), Public Safety (164). |

---

## 🧠 Machine Learning & Anomaly Detection

- **Algorithm:** Scikit-learn `IsolationForest(n_estimators=150, contamination=0.03)`
- **Feature Vector (11 dimensions):** `observed_count`, `rolling_mean_7d`, `rolling_std_7d`, `rolling_mean_14d`, `delta_count`, `z_score`, `surge_ratio`, `lag_1d`, `lag_7d`, `day_of_week`, `is_weekend`.
- **Sparse Time-Series Architecture:** Generates a continuous daily grid across 1,308 calendar days; zero-fills inactive days with `0` so quiet periods do not distort rolling statistics.
- **Strict Anti-Leakage Shift:** Enforces `shift(1)` so today's complaint count never leaks into today's baseline average.
- **Calibrated Scoring:** Maps raw decision scores into a normalized `0.0 – 1.0` scale (Baseline: `<0.30`, Elevated: `0.30–0.55`, Outlier: `0.55–0.75`, Critical Surge: `0.75–1.00`).
- **Actionable Civic Filtering:** Suppresses quiet/negative anomalies (0 complaints is not an emergency); flags only when `observed > expected`, $z \ge 1.5\sigma$, and `count >= 2`.
- **Result:** Detected **1,271 actionable emergency spikes** out of 93,817 observations (1.35% actionable rate).

---

## 🌊 The Civic Ripple Engine (Spatial Intelligence)

Municipal failures trigger multi-ward domino effects. The **Civic Ripple Engine** (`spatial_analysis.py`):
1. Computes pairwise geographic distance matrices across all 198 wards using the **Haversine formula**.
2. Scans for time-lagged spillovers within a **0.5 to 48-hour horizon** and **5.0 km radius**.
3. Detects cross-category cascades *(e.g., Water Pipe burst on Day 1 in Ward 161 ➡️ Road cave-in and sanitation overflow in adjacent Ward 184 on Day 2)*.

---

## 🌟 Flagship Demo Showcase Incident

Ready for screen-share demonstration at [`data/demo_showcase_alert.json`](file:///D:/vsc/ieee_offgrid/data/demo_showcase_alert.json):
- **Location:** Uttarahalli (`ward_184`), Bengaluru (GPS: `12.8968° N, 77.5399° E`)
- **Category:** Sanitation & Waste
- **Date:** January 11, 2021
- **Observed Volume:** **18 complaints**
- **7-Day Historical Expected Baseline:** **0.6 complaints**
- **Surge Magnitude:** **+17.4 delta volume**, **+16.2σ z-score**, **0.8814 anomaly score**

---

## 📈 Performance & Latency Benchmarks

| Metric | Measured Result | Benchmark SLA | Status |
| :--- | :--- | :--- | :--- |
| **Ingestion Throughput** | **13,881 rows/sec** (1.15s total) | > 2,000 rows/sec | **OPTIMAL** |
| **Feature Engineering Latency** | **6.59s** (93,817 observations) | < 30s | **OPTIMAL** |
| **Model Training Speed** | **0.86 seconds** (150 trees) | < 15s | **OPTIMAL** |
| **Inference Throughput** | **12,601 observations/sec** | > 5,000 obs/sec | **OPTIMAL** |
| **Per-Observation Latency** | **0.079 milliseconds** | < 1.0 ms | **REAL-TIME** |
| **Peak Memory Footprint** | **463 MB RSS** | < 1,024 MB | **LEAN** |
| **Model Artifact Footprint** | **1.15 MB** (`model.joblib`) | < 50 MB | **COMPACT** |
| **Hardware Used** | **11th Gen Intel Core i7-1185G7 (CPU only, 0 GPU required)** | Any CPU | **PORTABLE** |

---

## 🌐 Cloud Deployment Guide

### Option 1: Free Cloud Deployment on Render.com
1. Push this repository to GitHub:
   ```powershell
   git push origin main
   ```
2. Log in to [render.com](https://render.com) and click **New + ➡️ Web Service**.
3. Connect repository `ijk037/ieee_offgrid`.
4. Configure:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free ($0/mo)
5. Click **Deploy Web Service** — your live public HTTPS link will be ready in 2 minutes.

### Option 2: Instant Public Tunnel (60 Seconds)
```powershell
# Terminal 1: Run backend
python app.py

# Terminal 2: Generate public link
npx localtunnel --port 8000
```

---

## ⚖️ Project Verification Summary

- **Total Automated Tests:** **30 passed in 1.85s** (`pytest`)
- **Codebase Integrity:** Zero merge conflicts, zero overlapping files between Person A and Person B.
- **Hackathon Rulebook Compliance:** 100% compliant with IEEE OFFGRID guidelines (Real city dataset, working multi-page interactive UI, sub-second performance, B2G SaaS business model).
