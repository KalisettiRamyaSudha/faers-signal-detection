"""
FastAPI endpoint for FAERS adverse event signal scoring.

Accepts a single adverse event report and returns an anomaly score
plus a risk tier — suitable for integration with pharmacovigilance
review workflows.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    ReportRequest, SignalResponse,
    BatchRequest, BatchResponse, HealthResponse,
)

MODELS_DIR = Path("models")

app = FastAPI(
    title="FAERS Adverse Event Signal Detection API",
    description=(
        "Scores adverse event reports for anomalous drug-reaction patterns "
        "using IsolationForest anomaly detection. Designed for integration "
        "with pharmacovigilance monitoring workflows."
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_model  = None
_scaler = None
_meta   = None


def get_model():
    global _model, _scaler, _meta
    if _model is None:
        model_path  = MODELS_DIR / "isolation_forest.joblib"
        scaler_path = MODELS_DIR / "scaler.joblib"
        meta_path   = MODELS_DIR / "model_meta.json"

        if not model_path.exists():
            raise RuntimeError("Model not found. Run `python -m src.detect_signals` first.")

        _model  = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        with open(meta_path) as f:
            _meta = json.load(f)

    return _model, _scaler, _meta


def report_to_features(req: ReportRequest, meta: dict) -> pd.DataFrame:
    """Convert a ReportRequest into a scaled feature vector."""
    feature_names = meta["feature_names"]

    # Simple feature extraction from request
    drugs = [d.strip() for d in req.suspect_drugs.split("|") if d.strip()]
    reacs = [r.strip() for r in req.reactions.split("|") if r.strip()]

    # Basic frequency proxies (real deployment would use the full dataset counts)
    max_pair_count  = 1
    max_pair_rate   = 0.001
    max_disprop     = 1.0
    top_drug_freq   = 0.01
    reac_sev_entropy = float(len(reacs)) * float(req.outcome_severity)

    raw = {
        "age_yr":            req.age_yr if req.age_yr is not None else 50.0,
        "wt_kg":             req.wt_kg if req.wt_kg is not None else 75.0,
        "sex_enc":           req.sex_enc if req.sex_enc is not None else 0.5,
        "n_suspect_drugs":   float(len(drugs)),
        "n_reactions":       float(len(reacs)),
        "outc_severity":     float(req.outcome_severity),
        "max_pair_count":    float(max_pair_count),
        "max_pair_rate":     float(max_pair_rate),
        "max_disprop":       float(max_disprop),
        "top_drug_freq":     float(top_drug_freq),
        "reac_sev_entropy":  float(reac_sev_entropy),
        "disprop_x_severity": float(max_disprop * req.outcome_severity),
    }

    # Align to training feature order
    df = pd.DataFrame([{k: raw.get(k, 0.0) for k in feature_names}])
    return df


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    try:
        _, _, meta = get_model()
        return {"status": "healthy", "model_type": meta["model_type"],
                "best_config": meta["best_config"], "version": "1.0.0"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/score", response_model=SignalResponse, tags=["Signal Detection"])
def score_report(req: ReportRequest):
    """
    Score a single adverse event report for anomalous signal strength.

    Returns:
    - **anomaly_score**: continuous score (more negative = more anomalous)
    - **risk_tier**: HIGH / MEDIUM / LOW
    - **flagged**: boolean at the model's contamination threshold
    """
    try:
        model, scaler, meta = get_model()
        X = report_to_features(req, meta)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

        score = float(model.decision_function(X_scaled)[0])
        pred  = int(model.predict(X_scaled)[0])  # -1 = anomaly

        flagged   = pred == -1
        risk_tier = "HIGH" if score < -0.1 else ("MEDIUM" if score < 0 else "LOW")

        return {
            "anomaly_score": round(score, 4),
            "flagged":       flagged,
            "risk_tier":     risk_tier,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/batch", response_model=BatchResponse, tags=["Signal Detection"])
def score_batch(req: BatchRequest):
    """Score up to 500 reports. Returns results sorted by anomaly score (most anomalous first)."""
    if len(req.reports) > 500:
        raise HTTPException(status_code=400, detail="Max batch size: 500")
    try:
        model, scaler, meta = get_model()
        results = []
        for report in req.reports:
            X = report_to_features(report, meta)
            X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
            score = float(model.decision_function(X_scaled)[0])
            pred  = int(model.predict(X_scaled)[0])
            flagged   = pred == -1
            risk_tier = "HIGH" if score < -0.1 else ("MEDIUM" if score < 0 else "LOW")
            results.append({"anomaly_score": round(score, 4),
                            "flagged": flagged, "risk_tier": risk_tier})

        results.sort(key=lambda x: x["anomaly_score"])
        return {
            "total_scored": len(results),
            "total_flagged": sum(1 for r in results if r["flagged"]),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["System"])
def model_info():
    try:
        _, _, meta = get_model()
        return meta
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
