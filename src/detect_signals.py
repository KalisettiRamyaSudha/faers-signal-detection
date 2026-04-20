"""
Adverse event signal detection using IsolationForest anomaly detection.

IsolationForest isolates observations by randomly selecting a feature and
a split value — anomalous reports (rare drug-reaction combinations with
high severity) require fewer splits to isolate and receive lower scores.

This mirrors the approach used in production pharmacovigilance systems to
surface potential safety signals before full statistical analysis.

MLflow tracks all experiment configurations, contamination parameters,
and evaluation metrics across runs.
"""

import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)

from src.data_ingestion import generate_demo_data
from src.feature_engineering import build_features, scale_features


# ── Paths ──────────────────────────────────────────────────────────────────────

MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")
EXPERIMENT  = "faers_signal_detection"

# ── Model configs to compare ───────────────────────────────────────────────────

CONFIGS = [
    {"contamination": 0.03, "n_estimators": 100, "max_features": 1.0, "label": "baseline"},
    {"contamination": 0.03, "n_estimators": 200, "max_features": 0.8, "label": "large"},
    {"contamination": 0.05, "n_estimators": 100, "max_features": 1.0, "label": "high_contam"},
    {"contamination": 0.03, "n_estimators": 100, "max_features": 0.6, "label": "subfeature"},
]


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    model: IsolationForest,
    X: pd.DataFrame,
    y_true: np.ndarray = None,
    threshold: float = None,
) -> dict:
    """
    Evaluate an IsolationForest model.

    - anomaly_score: continuous score (lower = more anomalous)
    - If y_true labels are available (from synthetic planted signals),
      compute classification metrics at an optimal threshold.
    """
    scores = model.decision_function(X)   # higher = more normal
    preds  = model.predict(X)             # -1 = anomaly, 1 = normal
    binary = (preds == -1).astype(int)    # 1 = flagged as anomaly

    metrics = {
        "n_flagged":    int(binary.sum()),
        "flag_rate":    float(binary.mean()),
        "score_mean":   float(scores.mean()),
        "score_std":    float(scores.std()),
        "score_min":    float(scores.min()),
    }

    if y_true is not None:
        # Anomaly score direction: invert so higher = more anomalous for AUC
        inv_scores = -scores
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, inv_scores)), 4)
            metrics["avg_precision"] = round(float(average_precision_score(y_true, inv_scores)), 4)
        except Exception:
            pass

        metrics["precision"] = round(float(precision_score(y_true, binary, zero_division=0)), 4)
        metrics["recall"]    = round(float(recall_score(y_true, binary, zero_division=0)), 4)
        metrics["f1"]        = round(float(f1_score(y_true, binary, zero_division=0)), 4)

    return metrics, scores, binary


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_score_distribution(
    scores: np.ndarray,
    binary: np.ndarray,
    y_true: np.ndarray = None,
    save_path: str = None,
):
    """Histogram of anomaly scores split by flagged/normal."""
    fig, ax = plt.subplots(figsize=(9, 5))
    normal_scores  = scores[binary == 0]
    anomaly_scores = scores[binary == 1]

    ax.hist(normal_scores,  bins=50, alpha=0.6, color="#378ADD", label="Normal reports")
    ax.hist(anomaly_scores, bins=50, alpha=0.8, color="#D85A30", label="Flagged as anomaly")

    if y_true is not None:
        signal_scores = scores[y_true == 1]
        ax.hist(signal_scores, bins=20, alpha=0.9, color="#1D9E75",
                label="True signal (planted)", histtype="step", linewidth=2)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xlabel("IsolationForest Decision Score (lower = more anomalous)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Anomaly Score Distribution — FAERS Signal Detection", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Score distribution saved → {save_path}")
    plt.close()


def plot_top_flagged(
    df_raw: pd.DataFrame,
    scores: np.ndarray,
    top_n: int = 20,
    save_path: str = None,
):
    """Bar chart of the top flagged drug-reaction combinations."""
    df = df_raw.copy()
    df["anomaly_score"] = scores

    # Explode multi-drug reports into individual drug rows
    rows = []
    for _, row in df.nsmallest(top_n * 5, "anomaly_score").iterrows():
        drugs = str(row.get("suspect_drugs", "")).split("|")
        reacs = str(row.get("reactions", "")).split("|")[:2]  # top 2 reactions
        for d in drugs:
            d = d.strip()
            if d:
                rows.append({
                    "drug":   d,
                    "reac":   " / ".join([r.strip() for r in reacs if r.strip()]),
                    "score":  row["anomaly_score"],
                    "severity": row.get("outc_severity", 1),
                })

    flagged = (
        pd.DataFrame(rows)
          .groupby(["drug", "reac"])
          .agg(mean_score=("score", "mean"), count=("score", "count"))
          .sort_values("mean_score")
          .head(top_n)
          .reset_index()
    )
    flagged["label"] = flagged["drug"] + "\n" + flagged["reac"]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#D85A30" if s < -0.1 else "#EF9F27" for s in flagged["mean_score"]]
    ax.barh(flagged["label"], -flagged["mean_score"], color=colors)
    ax.set_xlabel("Anomaly Strength (higher = more anomalous)", fontsize=11)
    ax.set_title(f"Top {top_n} Flagged Drug-Reaction Combinations", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Top flagged chart saved → {save_path}")
    plt.close()


def plot_feature_importance_shap(
    model: IsolationForest,
    X: pd.DataFrame,
    save_path: str = None,
):
    """
    Mean |SHAP| feature importance using shap.TreeExplainer.
    Requires shap package — skipped gracefully if not installed.
    """
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        sample      = X.sample(min(300, len(X)), random_state=42)
        shap_values = explainer.shap_values(sample)

        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = pd.DataFrame({
            "feature":   X.columns,
            "mean_shap": mean_abs,
        }).sort_values("mean_shap", ascending=False)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(importance["feature"][::-1], importance["mean_shap"][::-1], color="#185FA5")
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("Feature Importance — IsolationForest SHAP", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"SHAP importance saved → {save_path}")
        plt.close()
        return importance
    except ImportError:
        print("shap not installed — skipping SHAP plot")
        return None


# ── Main training pipeline ─────────────────────────────────────────────────────

def train(df_raw: pd.DataFrame = None, use_demo: bool = True):
    """
    Full signal detection training pipeline:
    1. Load/generate data
    2. Build features
    3. Train and compare IsolationForest configurations with MLflow
    4. Select best model by ROC AUC (or flag rate if no labels)
    5. Save model and metadata
    6. Generate reports
    """
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    mlflow.set_experiment(EXPERIMENT)

    # ── Data ──────────────────────────────────────────────────────────────────
    if df_raw is None:
        if use_demo:
            print("No data provided — using synthetic demo data.")
            df_raw = generate_demo_data(n=10_000, seed=42)
        else:
            raise ValueError("Provide df_raw or set use_demo=True")

    # Ground truth signal labels (only available in synthetic data)
    if "suspect_drugs" in df_raw.columns:
        y_true = df_raw["suspect_drugs"].str.contains("SIGNAL", na=False).astype(int).values
        print(f"Ground truth signals available: {y_true.sum():,} ({y_true.mean():.1%})")
    else:
        y_true = None
        print("No ground truth labels — evaluating by flag rate only")

    # ── Feature engineering ──────────────────────────────────────────────────
    print("\nBuilding features ...")
    X, feature_names = build_features(df_raw)
    X_scaled, _, scaler = scale_features(X)

    # ── Train all configs ────────────────────────────────────────────────────
    print(f"\nTraining {len(CONFIGS)} IsolationForest configurations ...")
    results = []

    for cfg in CONFIGS:
        label = cfg.pop("label")
        with mlflow.start_run(run_name=f"isolation_forest_{label}"):
            mlflow.log_params({**cfg, "label": label, "n_features": len(feature_names)})

            model = IsolationForest(random_state=42, n_jobs=-1, **cfg)
            model.fit(X_scaled)

            metrics, scores, binary = evaluate(model, X_scaled, y_true)
            mlflow.log_metrics(metrics)

            print(f"\n── {label} ──────────────────────────")
            for k, v in metrics.items():
                print(f"  {k}: {v}")

            results.append({
                "label":   label,
                "model":   model,
                "scores":  scores,
                "binary":  binary,
                "metrics": metrics,
                **cfg,
            })
        cfg["label"] = label  # restore

    # ── Select best model ────────────────────────────────────────────────────
    if y_true is not None and any("roc_auc" in r["metrics"] for r in results):
        best = max(results, key=lambda r: r["metrics"].get("roc_auc", 0))
    else:
        # Without labels: prefer lowest flag rate (fewest false positives)
        best = min(results, key=lambda r: r["metrics"]["flag_rate"])

    print(f"\nBest configuration: {best['label']}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_score_distribution(
        best["scores"], best["binary"], y_true,
        save_path=str(REPORTS_DIR / "score_distribution.png"),
    )
    plot_top_flagged(
        df_raw, best["scores"],
        save_path=str(REPORTS_DIR / "top_flagged_signals.png"),
    )
    plot_feature_importance_shap(
        best["model"], X_scaled,
        save_path=str(REPORTS_DIR / "shap_feature_importance.png"),
    )

    # ── Save artifacts ───────────────────────────────────────────────────────
    joblib.dump(best["model"],  MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(scaler,         MODELS_DIR / "scaler.joblib")

    meta = {
        "model_type":     "IsolationForest",
        "best_config":    best["label"],
        "contamination":  best.get("contamination", 0.03),
        "feature_names":  feature_names,
        "metrics":        best["metrics"],
    }
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Signal report ────────────────────────────────────────────────────────
    flagged_df = df_raw.copy()
    flagged_df["anomaly_score"] = best["scores"]
    flagged_df["is_flagged"]    = best["binary"]
    flagged_df = flagged_df[flagged_df["is_flagged"] == 1].sort_values("anomaly_score")
    flagged_df.to_csv(REPORTS_DIR / "flagged_reports.csv", index=False)

    print(f"\nTotal flagged reports: {best['binary'].sum():,}")
    print(f"Flag rate: {best['metrics']['flag_rate']:.1%}")
    print("\nRun `mlflow ui` to view experiment runs.")
    print("=" * 55)

    return best["model"], feature_names, best["metrics"]


if __name__ == "__main__":
    train(use_demo=True)
