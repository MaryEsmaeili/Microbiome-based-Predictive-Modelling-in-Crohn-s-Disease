#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metatablepool.py

Build a pooled, long-format meta table for modeling (no Snakemake needed).

Key points you asked for:
- site is numeric: oral=0, fecal=1  (no text labels anywhere)
- Preserve leading zeros and original case of Sample_ID (no uppercasing, no lstrip)
- Remove replicate suffix like ".1" from Sample_ID (configurable)
- Strong NA cleaning; drop any row missing required covariates
- Outputs:
    data/meta/model_table_pooled.csv
    data/meta/model_table_pooled_missing.csv
- Helper functions (normalize_sample_id, clean_na_like, etc.) are reusable across scripts.

Run:
    python metatablepool.py
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import pandas as pd

# ============================= CONFIG =============================

# Inputs (hard-coded as requested)
CROHN_CORE_CSV     = Path("data/meta/crohn_metadata_core.csv")
CROHN_EXTENDED_CSV = Path("data/meta/crohn_metadata_extended.csv")   # read to keep parity; not used
HEALTHY_CSV        = Path("data/meta/healthy_metadata.csv")

# Outputs
OUT_POOLED_CSV   = Path("data/meta/model_table_pooled.csv")
OUT_MISSING_CSV  = Path("data/meta/model_table_pooled_missing.csv")

# Keep ".1" replicate suffix?  (Set True if your abundance headers use them)
KEEP_DOT_SUFFIX  = False

# Numeric coding for site
ORAL_CODE  = 0
FECAL_CODE = 1

# Required covariates (any missing → row dropped)
REQ_COVARS: List[str] = [
    "Age", "Sex", "BMI", "Smoking",
    "Antibiotics_3m", "PPI_use",
    "Steroids_ongoing", "Immuno_ongoing"
]

# Final column order (site is numeric)
TARGET_COLS: List[str] = [
    "Sample_ID", "STUDY_ID", "site", "disease",
    "Age", "Sex", "BMI", "Smoking",
    "Antibiotics_3m", "PPI_use", "Steroids_ongoing", "Immuno_ongoing"
]

# Integer-coded columns (nullable Int64). BMI is float.
CAT_INT_COLS: List[str] = [
    "disease", "Sex", "Smoking",
    "Antibiotics_3m", "PPI_use",
    "Steroids_ongoing", "Immuno_ongoing", "site"
]

# NA-like strings to coerce
NA_STRINGS = {"", "NA", "NaN", "NAN", "NULL", "Null", "null", "None", "none"}

# ====================== REUSABLE HELPERS (copy-paste friendly) ======================

def read_csv_obj(path: Path) -> pd.DataFrame:
    """Read CSV as object dtype (robust) and fail if not found."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, dtype="object")

def clean_na_like(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert common NA-like strings to real <NA>."""
    upper_na = {x.upper() for x in NA_STRINGS}
    for c in cols:
        if c in df.columns:
            s = df[c].astype("string").str.strip()
            df[c] = s.where(~s.str.upper().isin(upper_na), pd.NA)
    return df

def normalize_sample_id_series(s: pd.Series, keep_dot_suffix: bool = False) -> pd.Series:
    """
    Normalize Sample_ID safely for merging with abundance matrices:
    - strip spaces
    - optionally drop replicate suffix '.<digits>' (default: drop)
    - DO NOT uppercase; DO NOT strip leading zeros
    """
    s = s.astype("string").str.strip()
    if not keep_dot_suffix:
        s = s.str.replace(r"\.\d+$", "", regex=True)
    return s

def as_int64(s: pd.Series) -> pd.Series:
    """Coerce to pandas nullable Int64 (keeps <NA>)."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def as_float64(s: pd.Series) -> pd.Series:
    """Coerce to float64 (keeps NaN)."""
    return pd.to_numeric(s, errors="coerce").astype("float64")

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure unique column names (keep the right-most duplicate)."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df

def retype_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce final dtypes and column order."""
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    if "Age" in df: df["Age"] = as_int64(df["Age"])
    if "BMI" in df: df["BMI"] = as_float64(df["BMI"])
    for c in CAT_INT_COLS:
        if c in df: df[c] = as_int64(df[c])
    return df[TARGET_COLS].copy()

def require_and_drop(df: pd.DataFrame,
                     required_cols: List[str],
                     whoami: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows missing ANY required column; return (kept, dropped_report).
    Report contains which columns were missing.
    """
    miss_mask = df[required_cols].isna().any(axis=1)
    kept = df.loc[~miss_mask].copy()
    dropped = df.loc[miss_mask].copy()
    if len(dropped):
        # Provide detailed missing list per-row
        dropped["__missing_cols__"] = df[required_cols].apply(
            lambda r: ",".join([c for c, v in r.items() if pd.isna(v)]), axis=1
        )[miss_mask].values
        dropped["__source__"] = whoami
        dropped = dropped[["Sample_ID","STUDY_ID","site","disease","__source__","__missing_cols__"]]
    return kept, dropped

# ============================== BUILDERS ==============================

def build_crohn_long(core_df: pd.DataFrame) -> pd.DataFrame:
    need = {"STUDY_ID","Oral_sample_ID","Fecal_sample_ID", *REQ_COVARS}
    missing = sorted(list(need - set(core_df.columns)))
    if missing:
        raise KeyError(f"[Crohn] Missing required columns: {missing}")

    df = core_df.copy()
    df = clean_na_like(df, list(need))

    # Normalize ID-like columns (no case change, no leading-zero change)
    for c in ["STUDY_ID","Oral_sample_ID","Fecal_sample_ID"]:
        df[c] = normalize_sample_id_series(df[c], keep_dot_suffix=KEEP_DOT_SUFFIX)

    cov = df[["STUDY_ID", *REQ_COVARS]].copy()

    # ORAL rows
    oral = (
        df[["STUDY_ID","Oral_sample_ID"]]
        .rename(columns={"Oral_sample_ID":"Sample_ID"})
        .assign(site=ORAL_CODE)
        .merge(cov, on="STUDY_ID", how="left")
    )
    oral = oral[oral["Sample_ID"].notna()]

    # FECAL rows
    fecal = (
        df[["STUDY_ID","Fecal_sample_ID"]]
        .rename(columns={"Fecal_sample_ID":"Sample_ID"})
        .assign(site=FECAL_CODE)
        .merge(cov, on="STUDY_ID", how="left")
    )
    fecal = fecal[fecal["Sample_ID"].notna()]

    out = pd.concat([oral, fecal], ignore_index=True)
    out["disease"] = 1  # Crohn
    out["Sample_ID"] = normalize_sample_id_series(out["Sample_ID"], keep_dot_suffix=KEEP_DOT_SUFFIX)
    out = ensure_unique_columns(out).drop_duplicates(subset=["Sample_ID","site"])
    out = retype_and_order(out)
    return out


def build_healthy_long(healthy_df: pd.DataFrame) -> pd.DataFrame:
    need_min = {"STUDY_ID","Oral_sample_ID","Fecal_sample_ID",
                "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing"}
    missing = sorted(list(need_min - set(healthy_df.columns)))
    if missing:
        raise KeyError(f"[Healthy] Missing required columns: {missing}")

    df = healthy_df.copy()
    df = clean_na_like(df, list(need_min))

    for c in ["STUDY_ID","Oral_sample_ID","Fecal_sample_ID"]:
        df[c] = normalize_sample_id_series(df[c], keep_dot_suffix=KEEP_DOT_SUFFIX)

    cov = df[["STUDY_ID","Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing"]].copy()
    cov["Immuno_ongoing"] = 0  # not collected for healthy

    oral = (
        df[["STUDY_ID","Oral_sample_ID"]]
        .rename(columns={"Oral_sample_ID":"Sample_ID"})
        .assign(site=ORAL_CODE)
        .merge(cov, on="STUDY_ID", how="left")
    )
    oral = oral[oral["Sample_ID"].notna()]

    fecal = (
        df[["STUDY_ID","Fecal_sample_ID"]]
        .rename(columns={"Fecal_sample_ID":"Sample_ID"})
        .assign(site=FECAL_CODE)
        .merge(cov, on="STUDY_ID", how="left")
    )
    fecal = fecal[fecal["Sample_ID"].notna()]

    out = pd.concat([oral, fecal], ignore_index=True)
    out["disease"] = 0  # Healthy
    out["Sample_ID"] = normalize_sample_id_series(out["Sample_ID"], keep_dot_suffix=KEEP_DOT_SUFFIX)
    out = ensure_unique_columns(out).drop_duplicates(subset=["Sample_ID","site"])
    out = retype_and_order(out)
    return out

# ============================== MAIN ==============================

def main() -> None:
    crohn_core = read_csv_obj(CROHN_CORE_CSV)
    # keep parity; not used for building
    if CROHN_EXTENDED_CSV.exists():
        _ = read_csv_obj(CROHN_EXTENDED_CSV)
    healthy = read_csv_obj(HEALTHY_CSV)

    crohn_long   = build_crohn_long(crohn_core)
    healthy_long = build_healthy_long(healthy)

    pooled = pd.concat([crohn_long, healthy_long], ignore_index=True)
    pooled = retype_and_order(ensure_unique_columns(pooled))

    # Strict requirement: Sample_ID, site, disease, and all covariates must exist
    required_all = ["Sample_ID","site","disease", *REQ_COVARS]
    kept, dropped = require_and_drop(pooled, required_all, whoami="pooled")

    OUT_POOLED_CSV.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(OUT_POOLED_CSV, index=False)
    dropped.to_csv(OUT_MISSING_CSV, index=False)

    # Compact console summary
    counts = (
        kept.assign(disease=kept["disease"].map({0:"Healthy",1:"Crohn"}))
            .pivot_table(index="site", columns="disease", values="Sample_ID",
                         aggfunc="count", fill_value=0)
            .rename(index={ORAL_CODE:"oral(0)", FECAL_CODE:"fecal(1)"})
            .reset_index()
    )
    print(f"[OK] pooled rows kept: {len(kept)} (dropped: {len(dropped)})")
    print("\nCounts by site(0/1) × disease")
    print(counts.to_string(index=False))
    print(f"\nWritten:\n  {OUT_POOLED_CSV}\n  {OUT_MISSING_CSV}")

if __name__ == "__main__":
    main()
