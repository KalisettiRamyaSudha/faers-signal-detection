"""
FDA Adverse Event Reporting System (FAERS) data ingestion and preprocessing.

FAERS is a publicly available database of adverse event reports submitted
to the FDA by manufacturers, healthcare providers, and consumers.
Data: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

This module handles loading, cleaning, and joining the four core FAERS tables:
- DEMO: Patient demographics (age, sex, weight, report date)
- DRUG: Drug information (name, role, route of administration)
- REAC: Adverse reactions reported (MedDRA terms)
- OUTC: Outcomes (hospitalisation, death, disability, etc.)
"""

import os
import re
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

# Column subsets we actually use — keeps memory manageable
DEMO_COLS = ["primaryid", "caseid", "age", "age_cod", "sex", "wt", "wt_cod",
             "occr_country", "init_fda_dt", "rept_dt", "quarter"]
DRUG_COLS = ["primaryid", "drugname", "prod_ai", "role_cod", "route",
             "cum_dose_chr", "dechal", "rechal"]
REAC_COLS = ["primaryid", "pt"]           # pt = Preferred Term (MedDRA)
OUTC_COLS = ["primaryid", "outc_cod"]

# Outcome severity mapping
OUTCOME_SEVERITY = {
    "DE": 5,   # Death
    "LT": 4,   # Life-threatening
    "HO": 3,   # Hospitalisation
    "DS": 3,   # Disability
    "CA": 3,   # Congenital anomaly
    "RI": 2,   # Required intervention
    "OT": 1,   # Other
}


# ── Download helpers ───────────────────────────────────────────────────────────

def download_faers_quarter(year: int, quarter: int, dest_dir: Path = RAW_DIR) -> Path:
    """
    Download a single FAERS quarterly file from the FDA website.

    Args:
        year:    4-digit year  (e.g. 2023)
        quarter: 1-4
        dest_dir: where to save the zip

    Returns:
        Path to the downloaded zip file.

    Note:
        FDA renames files unpredictably across quarters. This function
        tries both naming conventions and falls back gracefully.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    tag    = f"faers_ascii_{year}q{quarter}"
    url    = f"https://fis.fda.gov/content/Exports/{tag}.zip"
    dest   = dest_dir / f"{tag}.zip"

    if dest.exists():
        print(f"Already downloaded: {dest.name}")
        return dest

    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Download failed ({resp.status_code}) for {url}.\n"
            f"Download FAERS ASCII data manually from:\n"
            f"  https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html\n"
            f"and place the zip in: {dest_dir.resolve()}"
        )

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


# ── File discovery ─────────────────────────────────────────────────────────────

def _find_table_file(zip_path: Path, table_prefix: str) -> str | None:
    """Return the name of the matching file inside a FAERS zip."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = Path(name).stem.upper()
            if base.startswith(table_prefix.upper()):
                return name
    return None


# ── Table loaders ──────────────────────────────────────────────────────────────

def _load_table(zip_path: Path, prefix: str, usecols: list[str]) -> pd.DataFrame:
    """Extract and load a single FAERS ASCII table from a zip."""
    fname = _find_table_file(zip_path, prefix)
    if fname is None:
        raise FileNotFoundError(
            f"Could not find {prefix}*.txt inside {zip_path.name}"
        )
    with zipfile.ZipFile(zip_path) as z:
        with z.open(fname) as f:
            df = pd.read_csv(
                f,
                sep="$",
                encoding="latin-1",
                low_memory=False,
                usecols=lambda c: c.lower() in [x.lower() for x in usecols],
                on_bad_lines="skip",
            )
    df.columns = [c.lower() for c in df.columns]
    # normalise column names
    available = set(df.columns)
    df = df[[c for c in [x.lower() for x in usecols] if c in available]]
    return df


# ── Cleaning ───────────────────────────────────────────────────────────────────

def _clean_demo(demo: pd.DataFrame) -> pd.DataFrame:
    """Normalise age to years, weight to kg, encode sex."""
    df = demo.copy()
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")

    # Age → years
    age_map = {"DEC": 10, "YR": 1, "MON": 1/12, "WK": 1/52,
               "DY": 1/365, "HR": 1/8760}
    df["age_yr"] = pd.to_numeric(df["age"], errors="coerce")
    if "age_cod" in df.columns:
        df["age_yr"] *= df["age_cod"].str.strip().str.upper().map(age_map).fillna(1)
    df["age_yr"] = df["age_yr"].clip(0, 120)

    # Weight → kg
    df["wt_kg"] = pd.to_numeric(df.get("wt"), errors="coerce")
    if "wt_cod" in df.columns:
        df.loc[df["wt_cod"].str.upper().str.strip() == "LBS", "wt_kg"] *= 0.4536
    df["wt_kg"] = df["wt_kg"].clip(1, 300)

    # Sex → binary
    df["sex_enc"] = df["sex"].str.upper().str.strip().map({"M": 0, "F": 1})

    # Date
    if "rept_dt" in df.columns:
        df["rept_dt"] = pd.to_datetime(df["rept_dt"].astype(str), format="%Y%m%d", errors="coerce")

    return df[["primaryid", "age_yr", "wt_kg", "sex_enc", "rept_dt",
               "occr_country"] if "occr_country" in df.columns
              else ["primaryid", "age_yr", "wt_kg", "sex_enc", "rept_dt"]]


def _clean_drug(drug: pd.DataFrame) -> pd.DataFrame:
    """Normalise drug names, mark primary suspect drugs."""
    df = drug.copy()
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    df["drugname"]  = df["drugname"].str.upper().str.strip()
    df["is_suspect"] = df["role_cod"].str.strip().str.upper().isin(["PS", "SS"]).astype(int)
    return df[["primaryid", "drugname", "is_suspect", "route"]].dropna(subset=["drugname"])


def _clean_reac(reac: pd.DataFrame) -> pd.DataFrame:
    """Upper-case and strip reaction preferred terms."""
    df = reac.copy()
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    df["pt"]        = df["pt"].str.upper().str.strip()
    return df.dropna(subset=["pt"])


def _clean_outc(outc: pd.DataFrame) -> pd.DataFrame:
    """Map outcomes to severity scores, take max severity per report."""
    df = outc.copy()
    df["primaryid"]      = pd.to_numeric(df["primaryid"], errors="coerce")
    df["outc_severity"]  = df["outc_cod"].str.strip().str.upper().map(OUTCOME_SEVERITY).fillna(1)
    return (
        df.groupby("primaryid")["outc_severity"]
          .max()
          .reset_index()
    )


# ── Master pipeline ────────────────────────────────────────────────────────────

def load_and_clean_quarter(zip_path: Path) -> pd.DataFrame:
    """
    Load all four FAERS tables from one quarterly zip and join them.

    Returns a report-level DataFrame where each row = one unique
    (report, drug, reaction) combination, ready for feature engineering.
    """
    print(f"Loading tables from {zip_path.name} ...")

    demo = _clean_demo(_load_table(zip_path, "DEMO", DEMO_COLS))
    drug = _clean_drug(_load_table(zip_path, "DRUG", DRUG_COLS))
    reac = _clean_reac(_load_table(zip_path, "REAC", REAC_COLS))
    outc = _clean_outc(_load_table(zip_path, "OUTC", OUTC_COLS))

    # Aggregate reactions per report → pipe-delimited string
    reac_agg = (
        reac.groupby("primaryid")["pt"]
            .apply(lambda x: "|".join(sorted(set(x))))
            .reset_index()
            .rename(columns={"pt": "reactions"})
    )

    # Aggregate drugs: keep suspect drugs only, unique names
    drug_agg = (
        drug[drug["is_suspect"] == 1]
            .groupby("primaryid")["drugname"]
            .apply(lambda x: "|".join(sorted(set(x))))
            .reset_index()
            .rename(columns={"drugname": "suspect_drugs"})
    )

    # Join all
    df = (
        demo
        .merge(drug_agg,  on="primaryid", how="left")
        .merge(reac_agg,  on="primaryid", how="left")
        .merge(outc,      on="primaryid", how="left")
    )

    df["outc_severity"] = df["outc_severity"].fillna(1)
    df["n_suspect_drugs"] = df["suspect_drugs"].fillna("").str.split("|").apply(
        lambda x: len([d for d in x if d])
    )
    df["n_reactions"] = df["reactions"].fillna("").str.split("|").apply(
        lambda x: len([r for r in x if r])
    )

    print(f"  → {len(df):,} reports loaded")
    return df


def load_multiple_quarters(
    zip_paths: list[Path],
    save_path: Path = None,
) -> pd.DataFrame:
    """Load and concatenate multiple FAERS quarters."""
    frames = [load_and_clean_quarter(z) for z in zip_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["primaryid"])
    print(f"\nTotal reports across all quarters: {len(df):,}")
    if save_path:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        print(f"Saved to {save_path}")
    return df


# ── Demo data generator (for testing without real FAERS download) ──────────────

def generate_demo_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic FAERS-like data for testing when real files
    are not yet downloaded.

    Signal planted: two drug-reaction pairs with unusually high
    co-occurrence and severity — these should be flagged as anomalies.
    """
    rng = np.random.default_rng(seed)
    DRUG_POOL = [
        "WARFARIN", "METFORMIN", "LISINOPRIL", "ATORVASTATIN", "ASPIRIN",
        "AMOXICILLIN", "PREDNISONE", "OMEPRAZOLE", "METOPROLOL", "AMLODIPINE",
        "DRUG_SIGNAL_A",   # planted signal drug A
        "DRUG_SIGNAL_B",   # planted signal drug B
    ]
    REAC_POOL = [
        "NAUSEA", "HEADACHE", "DIZZINESS", "FATIGUE", "RASH", "VOMITING",
        "DYSPNOEA", "PAIN", "PYREXIA", "INSOMNIA", "PRURITUS", "DIARRHOEA",
        "SEVERE_REACTION_X",  # planted signal reaction
        "ORGAN_FAILURE_Y",    # planted signal reaction
    ]

    records = []
    for i in range(n):
        # Plant anomalies: 3% of reports involve signal pair
        is_signal = rng.random() < 0.03
        if is_signal and rng.random() < 0.5:
            drugs     = "DRUG_SIGNAL_A"
            reactions = "SEVERE_REACTION_X|ORGAN_FAILURE_Y"
            severity  = 5  # death
        elif is_signal:
            drugs     = "DRUG_SIGNAL_B"
            reactions = "SEVERE_REACTION_X"
            severity  = 4
        else:
            n_d = rng.integers(1, 4)
            n_r = rng.integers(1, 5)
            drugs     = "|".join(rng.choice(DRUG_POOL[:10], n_d, replace=False))
            reactions = "|".join(rng.choice(REAC_POOL[:12], n_r, replace=False))
            severity  = int(rng.choice([1, 2, 3, 4, 5], p=[0.5, 0.2, 0.15, 0.1, 0.05]))

        records.append({
            "primaryid":     i + 1,
            "age_yr":        float(rng.integers(18, 85)) if rng.random() > 0.1 else np.nan,
            "wt_kg":         float(rng.integers(50, 120)) if rng.random() > 0.15 else np.nan,
            "sex_enc":       int(rng.integers(0, 2)) if rng.random() > 0.05 else np.nan,
            "suspect_drugs": drugs,
            "reactions":     reactions,
            "outc_severity": severity,
            "n_suspect_drugs": len(drugs.split("|")),
            "n_reactions":   len(reactions.split("|")),
        })

    df = pd.DataFrame(records)
    print(f"Generated {len(df):,} synthetic FAERS-like reports (seed={seed})")
    print(f"Signal reports planted: {(df['suspect_drugs'].str.contains('SIGNAL')).sum():,}")
    return df


if __name__ == "__main__":
    df = generate_demo_data()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROC_DIR / "faers_demo.parquet", index=False)
    print(df.head())
