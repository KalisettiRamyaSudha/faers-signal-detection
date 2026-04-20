"""Tests for the FAERS signal detection API."""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app

client = TestClient(app)

SAMPLE_REPORT = {
    "suspect_drugs": "WARFARIN|ASPIRIN",
    "reactions":     "HAEMORRHAGE|DYSPNOEA",
    "outcome_severity": 4,
    "age_yr": 72.0,
    "wt_kg":  68.0,
    "sex_enc": 0,
}

MOCK_META = {
    "model_type":    "IsolationForest",
    "best_config":   "baseline",
    "contamination": 0.03,
    "feature_names": [
        "age_yr", "wt_kg", "sex_enc", "n_suspect_drugs", "n_reactions",
        "outc_severity", "max_pair_count", "max_pair_rate", "max_disprop",
        "top_drug_freq", "reac_sev_entropy", "disprop_x_severity",
    ],
    "metrics": {"flag_rate": 0.03, "roc_auc": 0.82},
}


def make_mock_model(score=-0.15):
    m = MagicMock()
    m.decision_function.return_value = np.array([score])
    m.predict.return_value = np.array([-1 if score < 0 else 1])
    return m


def make_mock_scaler():
    s = MagicMock()
    import pandas as pd
    s.transform = lambda X: X.values
    return s


def test_health():
    with patch("api.main.get_model", return_value=(
        make_mock_model(), make_mock_scaler(), MOCK_META
    )):
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_score_high_risk():
    """High severity report with rare drug combination → HIGH risk tier."""
    with patch("api.main.get_model", return_value=(
        make_mock_model(-0.2), make_mock_scaler(), MOCK_META
    )):
        r = client.post("/score", json=SAMPLE_REPORT)
    assert r.status_code == 200
    data = r.json()
    assert data["flagged"] is True
    assert data["risk_tier"] == "HIGH"
    assert data["anomaly_score"] < 0


def test_score_low_risk():
    """Common drug with mild reaction → LOW risk."""
    with patch("api.main.get_model", return_value=(
        make_mock_model(0.15), make_mock_scaler(), MOCK_META
    )):
        r = client.post("/score", json={
            "suspect_drugs": "PARACETAMOL",
            "reactions": "NAUSEA",
            "outcome_severity": 1,
        })
    assert r.status_code == 200
    assert r.json()["risk_tier"] == "LOW"
    assert r.json()["flagged"] is False


def test_batch_score():
    with patch("api.main.get_model", return_value=(
        make_mock_model(-0.15), make_mock_scaler(), MOCK_META
    )):
        r = client.post("/score/batch", json={"reports": [SAMPLE_REPORT, SAMPLE_REPORT]})
    assert r.status_code == 200
    data = r.json()
    assert data["total_scored"] == 2
    assert len(data["results"]) == 2


def test_batch_limit():
    big_batch = {"reports": [SAMPLE_REPORT] * 501}
    r = client.post("/score/batch", json=big_batch)
    assert r.status_code == 422   # Pydantic max_length


def test_missing_required_field():
    r = client.post("/score", json={"reactions": "NAUSEA", "outcome_severity": 1})
    assert r.status_code == 422


def test_model_info():
    with patch("api.main.get_model", return_value=(
        make_mock_model(), make_mock_scaler(), MOCK_META
    )):
        r = client.get("/model/info")
    assert r.status_code == 200
    assert "feature_names" in r.json()
