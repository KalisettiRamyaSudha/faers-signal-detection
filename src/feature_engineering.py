"""
Feature engineering for FAERS adverse event signal detection.

Transforms raw report-level data into a structured feature matrix
suitable for IsolationForest anomaly detection.

Feature groups:
  1. Patient demographics       — age, weight, sex
  2. Drug exposure              — drug count, drug encoding
  3. Reaction profile           — reaction count, severity
  4. Signal co-occurrence       — drug-reaction pair frequency (PRR-inspired)
  5. Report-level aggregates    — derived statistical features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── Drug-reaction co-occurrence features ───────────────────────────────────────

def compute_drug_reaction_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-drug and per-reaction occurrence frequencies across
    all reports in the dataset.

    These proxy the Proportional Reporting Ratio (PRR), a standard
    pharmacovigilance signal detection statistic, without requiring
    a reference database.
    """
    drug_counts = {}
    reac_counts = {}
    pair_counts = {}

    for _, row in df.iterrows():
        drugs = str(row.get("suspect_drugs", "")).split("|")
        reacs = str(row.get("reactions", "")).split("|")
        drugs = [d.strip() for d in drugs if d.strip()]
        reacs = [r.strip() for r in reacs if r.strip()]

        for d in drugs:
            drug_counts[d] = drug_counts.get(d, 0) + 1
        for r in reacs:
            reac_counts[r] = reac_counts.get(r, 0) + 1
        for d in drugs:
            for r in reacs:
                key = f"{d}|{r}"
                pair_counts[key] = pair_counts.get(key, 0) + 1

    return drug_counts, reac_counts, pair_counts


def report_max_pair_frequency(
    row: pd.Series,
    drug_counts: dict,
    reac_counts: dict,
    pair_counts: dict,
    n_reports: int,
) -> tuple[float, float, float]:
    """
    For a single report, compute the maximum drug-reaction pair statistics:
    - max_pair_count: raw count of the most-reported drug-reaction pair
    - max_pair_rate:  pair count / total reports (proxy PRR numerator)
    - max_disprop:    pair_count / (drug_count * reac_count / n_reports)
                      (disproportionality measure — high = unexpected co-occurrence)
    """
    drugs = [d.strip() for d in str(row.get("suspect_drugs", "")).split("|") if d.strip()]
    reacs = [r.strip() for r in str(row.get("reactions", "")).split("|") if r.strip()]

    max_pair  = 0
    max_disp  = 0.0

    for d in drugs:
        for r in reacs:
            key   = f"{d}|{r}"
            pc    = pair_counts.get(key, 1)
            dc    = max(drug_counts.get(d, 1), 1)
            rc    = max(reac_counts.get(r, 1), 1)
            disp  = (pc * n_reports) / (dc * rc)

            if pc > max_pair:
                max_pair = pc
            if disp > max_disp:
                max_disp = disp

    return (
        max_pair,
        max_pair / max(n_reports, 1),
        max_disp,
    )


# ── Main feature builder ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the feature matrix from a cleaned FAERS DataFrame.

    Returns:
        feature_df:    DataFrame with one row per report, feature columns only
        feature_names: ordered list of feature column names
    """
    print("Computing drug-reaction co-occurrence frequencies ...")
    drug_counts, reac_counts, pair_counts = compute_drug_reaction_frequencies(df)
    n_reports = len(df)

    # Co-occurrence features
    print("Computing per-report signal features ...")
    co_features = df.apply(
        lambda row: report_max_pair_frequency(
            row, drug_counts, reac_counts, pair_counts, n_reports
        ),
        axis=1,
        result_type="expand",
    )
    co_features.columns = ["max_pair_count", "max_pair_rate", "max_disprop"]

    # Drug name frequency encoding
    # Use count of how many times the most-common suspect drug appears
    def top_drug_freq(suspect_drugs: str) -> float:
        drugs = [d.strip() for d in str(suspect_drugs).split("|") if d.strip()]
        if not drugs:
            return 0.0
        return max(drug_counts.get(d, 0) for d in drugs) / n_reports

    # Reaction severity entropy — more distinct severe reactions = higher entropy
    def reaction_entropy(reactions: str, severity: float) -> float:
        reacs = [r.strip() for r in str(reactions).split("|") if r.strip()]
        if not reacs:
            return 0.0
        freqs = np.array([reac_counts.get(r, 1) / n_reports for r in reacs])
        freqs = freqs / freqs.sum()
        entropy = -np.sum(freqs * np.log(freqs + 1e-10))
        return entropy * severity

    # Assemble feature DataFrame
    feat = pd.DataFrame({
        # Demographics
        "age_yr":          df["age_yr"].fillna(df["age_yr"].median()),
        "wt_kg":           df["wt_kg"].fillna(df["wt_kg"].median()),
        "sex_enc":         df["sex_enc"].fillna(0.5),

        # Drug / reaction counts
        "n_suspect_drugs": df["n_suspect_drugs"].fillna(0),
        "n_reactions":     df["n_reactions"].fillna(0),

        # Outcome severity
        "outc_severity":   df["outc_severity"].fillna(1),

        # Co-occurrence signal features
        "max_pair_count":  co_features["max_pair_count"],
        "max_pair_rate":   co_features["max_pair_rate"],
        "max_disprop":     co_features["max_disprop"],

        # Drug frequency
        "top_drug_freq":   df["suspect_drugs"].apply(top_drug_freq),

        # Reaction severity entropy
        "reac_sev_entropy": df.apply(
            lambda r: reaction_entropy(r.get("reactions", ""), r.get("outc_severity", 1)),
            axis=1,
        ),

        # Interaction: high disprop + high severity
        "disprop_x_severity": co_features["max_disprop"] * df["outc_severity"].fillna(1),
    })

    feature_names = feat.columns.tolist()
    print(f"Feature matrix shape: {feat.shape}")
    return feat, feature_names


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, StandardScaler]:
    """Fit StandardScaler on training data, apply to train and optionally test."""
    scaler  = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
    return X_train_scaled, X_test_scaled, scaler
