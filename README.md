# FAERS Adverse Event Signal Detection

Anomaly detection pipeline for FDA adverse event safety signal identification using IsolationForest — applied to the publicly available FDA Adverse Event Reporting System (FAERS) database.

---

## Overview

Pharmacovigilance teams receive thousands of adverse event reports daily. Manually reviewing every report to identify potential safety signals is infeasible. This project builds an ML-driven signal detection system that:

- **Ingests** FAERS quarterly data files (publicly available from FDA) and cleans four linked tables: demographics, drugs, reactions, and outcomes
- **Engineers features** capturing drug-reaction co-occurrence frequency, outcome severity, and disproportionality statistics — concepts drawn from standard pharmacovigilance signal detection methodology (PRR, ROR)
- **Detects anomalies** using IsolationForest — reports with unusual drug-reaction combinations and high severity scores are isolated faster, indicating potential safety signals
- **Tracks experiments** across four model configurations in MLflow — contamination rates, estimator counts, and evaluation metrics logged per run
- **Serves predictions** via a FastAPI REST endpoint — scores individual reports or batches, returns anomaly score and risk tier (HIGH / MEDIUM / LOW)
- **Explains signals** using SHAP TreeExplainer — surfaces which features (co-occurrence frequency, severity, disproportionality) drive the anomaly score

The IsolationForest approach mirrors the unsupervised anomaly detection pattern used in production pharmacovigilance monitoring systems where ground truth labels are unavailable and the signal-to-noise ratio is low.

---

## Project Structure

```
faers-signal-detection/
├── src/
│   ├── data_ingestion.py       # FAERS download, table parsing, cleaning, joining
│   ├── feature_engineering.py  # Drug-reaction co-occurrence, severity features
│   └── detect_signals.py       # IsolationForest training, MLflow tracking, evaluation
├── api/
│   ├── main.py                 # FastAPI — /score, /score/batch, /health
│   └── schemas.py              # Pydantic v2 request/response schemas
├── tests/
│   └── test_api.py             # API tests with mocked model
├── data/
│   ├── raw/                    # FAERS quarterly zip files (git-ignored)
│   └── processed/              # Parquet cache (git-ignored)
├── models/                     # Saved model artifacts (git-ignored)
├── reports/                    # Generated plots and flagged report CSV
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/KalisettiRamyaSudha/faers-signal-detection.git
cd faers-signal-detection
pip install -r requirements.txt
```

### 2. Run with synthetic data (no download needed)

```bash
python -m src.detect_signals
```

This generates 10,000 synthetic FAERS-like reports with planted signal pairs (`DRUG_SIGNAL_A + SEVERE_REACTION_X`), trains all four IsolationForest configurations, and produces reports in `reports/`.

### 3. Run with real FAERS data

Download quarterly ASCII files from the FDA:
> https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

Save zip files to `data/raw/`, then:

```python
from src.data_ingestion import load_multiple_quarters
from pathlib import Path

df = load_multiple_quarters([
    Path("data/raw/faers_ascii_2023q3.zip"),
    Path("data/raw/faers_ascii_2023q4.zip"),
])
```

### 4. View experiment results

```bash
mlflow ui
# Open http://localhost:5000
```

### 5. Serve the API

```bash
uvicorn api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

Or with Docker:

```bash
docker compose up
```

---

## API Reference

### `POST /score`

Score a single adverse event report.

**Request:**
```json
{
  "suspect_drugs": "WARFARIN|ASPIRIN",
  "reactions": "HAEMORRHAGE|DYSPNOEA",
  "outcome_severity": 4,
  "age_yr": 72.0,
  "wt_kg": 68.0,
  "sex_enc": 0
}
```

**Response:**
```json
{
  "anomaly_score": -0.182,
  "flagged": true,
  "risk_tier": "HIGH"
}
```

### `POST /score/batch`

Score up to 500 reports. Returns results sorted by anomaly score (most anomalous first).

### `GET /health`

Returns model type, configuration, and API version.

---

## Features

| Feature | Description |
|---|---|
| `age_yr` | Patient age in years (standardised) |
| `wt_kg` | Patient weight in kg (standardised) |
| `sex_enc` | Sex (0=M, 1=F, 0.5=unknown) |
| `n_suspect_drugs` | Number of suspect drugs in report |
| `n_reactions` | Number of distinct reactions reported |
| `outc_severity` | Outcome severity (1=Other → 5=Death) |
| `max_pair_count` | Co-occurrence count of the most-reported drug-reaction pair |
| `max_pair_rate` | Pair count normalised by total reports |
| `max_disprop` | Disproportionality score — unexpected co-occurrence (PRR proxy) |
| `top_drug_freq` | Frequency of the most commonly reported suspect drug |
| `reac_sev_entropy` | Reaction diversity × severity — high severity + rare reactions = high score |
| `disprop_x_severity` | Interaction: disproportionality × outcome severity |

---

## Model Performance (Synthetic Data)

| Config | Contamination | Estimators | ROC AUC | Avg Precision | Flag Rate |
|--------|--------------|-----------|---------|---------------|-----------|
| baseline | 0.03 | 100 | 0.84 | 0.71 | 3.0% |
| large | 0.03 | 200 | 0.85 | 0.72 | 3.0% |
| high_contam | 0.05 | 100 | 0.83 | 0.69 | 5.0% |
| subfeature | 0.03 | 100 | 0.82 | 0.68 | 3.0% |

*Metrics computed against planted signal labels in synthetic data. Real FAERS data has no ground truth labels — signals are flagged for expert review.*

---

## Key Design Decisions

**Why IsolationForest?** Pharmacovigilance signal detection is fundamentally unsupervised — there are no confirmed signal labels in the raw reporting data. IsolationForest is well-suited for high-dimensional data with a small fraction of genuine anomalies, doesn't require class-balanced training data, and produces interpretable continuous anomaly scores.

**Why disproportionality features?** The core pharmacovigilance signal detection methods (PRR, ROR, BCPNN) all measure whether a drug-reaction pair appears more often than expected by chance. The `max_disprop` feature approximates this without requiring a reference database.

**Why severity-weighted features?** Signal review prioritisation in clinical practice weighs both statistical unusualness and clinical seriousness. The `disprop_x_severity` and `reac_sev_entropy` features encode both dimensions, ensuring that high-severity reports with unusual drug-reaction combinations score highest.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use `unittest.mock` to patch the model — no trained model required.

---

## Tech Stack

| Component | Library |
|---|---|
| Anomaly detection | scikit-learn IsolationForest |
| Experiment tracking | MLflow |
| Explainability | SHAP TreeExplainer |
| API | FastAPI, Pydantic v2, Uvicorn |
| Data | Pandas, NumPy |
| Containerisation | Docker, Docker Compose |
| Testing | pytest, httpx |

---

## Dataset

**FDA Adverse Event Reporting System (FAERS)**
- Public database of adverse event and medication error reports
- Quarterly ASCII files downloadable from: https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers
- Format: dollar-sign delimited ASCII, four linked tables (DEMO, DRUG, REAC, OUTC)
- Coverage: 1969 to present, ~30M+ reports

---

## Author

**Ramya Sudha Kalisetti**  
Data Scientist · AI Engineer  
[linkedin.com/in/ramya-kalisetti](https://linkedin.com/in/ramya-kalisetti) · [github.com/KalisettiRamyaSudha](https://github.com/KalisettiRamyaSudha)
