#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a pooled, long-format model table:
- Crohn core/extended (site split into oral/fecal using *_sample_ID)
- Healthy metadata (site split similarly)
- Enforces integer types (no floats) for categorical/numeric codes
- Writes a "missing IDs" report

Inputs:
  --crohn           : crohn_metadata_core.csv
  --crohn-extended  : crohn_metadata_extended.csv (not heavily used here, but kept for future)
  --healthy         : healthy_metadata.csv (wide; has Oral_sample_ID/Fecal_sample_ID + covariates)
  --out-pooled      : output pooled model table (CSV)
  --out-missing     : output missing report (CSV)
"""

import argparse
import numpy as np
import pandas as pd

# --------------------------- helpers ---------------------------

REQ_COLS_CORE = [
    "STUDY_ID","Oral_sample_ID","Fecal_sample_ID",
    "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use",
    "Steroids_ongoing","Immuno_ongoing"
]
REQ_COLS_HEALTHY = [
    "STUDY_ID","Oral_sample_ID","Fecal_sample_ID",
    "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use",
    "Steroids_ongoing"  # Healthy has no Immuno_ongoing (all NA/0 per your dataset)
]

TARGET_COLS = [
    "Sample_ID","STUDY_ID","site","disease",
    "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing"
]

INT_COLS = ["disease","Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing"]

def _as_int64(series, fill=0):
    """Coerce to pandas nullable Int64 (no decimals in CSV)."""
    s = pd.to_numeric(series, errors="coerce").astype("Int64")
    # If you prefer leaving NAs as NA, comment next line. For strict ints, fill NAs:
    s = s.fillna(fill).astype("Int64")
    return s

MISSING_STRS = {"", "NA", "NaN", "NAN", "NULL", "Null", "null", "None", "none"}

def _clean_ids(df, cols):
    """Normalize ID-like columns and coerce common 'NA' strings to true NA."""
    for c in cols:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            # turn common NA-like strings into actual NA
            s = s.where(~s.str.upper().isin({m.upper() for m in MISSING_STRS}), pd.NA)
            # normalize sample-like columns to UPPER (optional, handy)
            if c.endswith("_sample_ID") or c == "Sample_ID":
                s = s.str.upper()
            df[c] = s
    return df

def _unique_columns(df):
    """Ensure the DataFrame has unique column names (drop duplicated right-most)."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df

# --------------------------- builders ---------------------------

def crohn_to_long(core_df: pd.DataFrame) -> pd.DataFrame:
    """Split Crohn core wide table into long rows for oral/fecal; disease=1."""
    # Check required columns exist
    missing = [c for c in REQ_COLS_CORE if c not in core_df.columns]
    if missing:
        raise KeyError(f"[Crohn] Missing required columns: {missing}")

    df = core_df.copy()
    df = _clean_ids(df, ["STUDY_ID","Oral_sample_ID","Fecal_sample_ID"])
    # Keep only START rows (robust to accidental junk rows)
    df = df[df["STUDY_ID"].astype(str).str.startswith("START", na=False)].copy()

    # Prepare shared covariates (will be attached to both sites per subject if IDs exist)
    cov = df[["STUDY_ID","Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing"]].copy()

    # ORAL rows
    oral = df[["STUDY_ID","Oral_sample_ID"]].rename(columns={"Oral_sample_ID":"Sample_ID"})
    oral = oral[oral["Sample_ID"].notna() & (oral["Sample_ID"].str.upper()!="NA")].copy()
    oral["site"] = "oral"
    oral = oral.merge(cov, on="STUDY_ID", how="left")

    # FECAL rows
    fecal = df[["STUDY_ID","Fecal_sample_ID"]].rename(columns={"Fecal_sample_ID":"Sample_ID"})
    fecal = fecal[fecal["Sample_ID"].notna() & (fecal["Sample_ID"].str.upper()!="NA")].copy()
    fecal["site"] = "fecal"
    fecal = fecal.merge(cov, on="STUDY_ID", how="left")

    # after building 'oral' and 'fecal' DataFrames for healthy
    oral  = oral[ oral["Sample_ID"].notna()].copy()
    fecal = fecal[fecal["Sample_ID"].notna()].copy()

    out = pd.concat([oral, fecal], ignore_index=True)
    out["disease"] = 1  # Crohn patients
    out = _unique_columns(out)

    # Reorder/complete columns
    for col in TARGET_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[TARGET_COLS].copy()

    # Types: integers (nullable)
    for c in INT_COLS:
        out[c] = _as_int64(out[c], fill=0)

    # Clean IDs again
    out = _clean_ids(out, ["Sample_ID","STUDY_ID"])
    # Drop any accidental duplicates
    out = out.drop_duplicates(subset=["Sample_ID","site"])

    return out

def healthy_to_long(healthy_df: pd.DataFrame) -> pd.DataFrame:
    """Split Healthy wide table into long rows for oral/fecal; disease=0."""
    # Minimal required columns for healthy
    missing = [c for c in REQ_COLS_HEALTHY if c not in healthy_df.columns]
    if missing:
        raise KeyError(f"[Healthy] Missing required columns: {missing}")

    df = healthy_df.copy()
    df = _clean_ids(df, ["STUDY_ID","Oral_sample_ID","Fecal_sample_ID"])

    cov_cols = ["STUDY_ID","Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing"]
    cov = df[cov_cols].copy()
    # By your note, Immuno_ongoing is not used for Healthy; set to 0
    cov["Immuno_ongoing"] = 0

    # ORAL rows
    oral = df[["STUDY_ID","Oral_sample_ID"]].rename(columns={"Oral_sample_ID":"Sample_ID"})
    oral = oral[oral["Sample_ID"].notna() & (oral["Sample_ID"].str.upper()!="NA")].copy()
    oral["site"] = "oral"
    oral = oral.merge(cov, on="STUDY_ID", how="left")

    # FECAL rows
    fecal = df[["STUDY_ID","Fecal_sample_ID"]].rename(columns={"Fecal_sample_ID":"Sample_ID"})
    fecal = fecal[fecal["Sample_ID"].notna() & (fecal["Sample_ID"].str.upper()!="NA")].copy()
    fecal["site"] = "fecal"
    fecal = fecal.merge(cov, on="STUDY_ID", how="left")

    out = pd.concat([oral, fecal], ignore_index=True)
    out["disease"] = 0  # Healthy
    out = _unique_columns(out)

    # Reorder/complete columns
    for col in TARGET_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[TARGET_COLS].copy()

    # Types: integers (nullable)
    for c in INT_COLS:
        out[c] = _as_int64(out[c], fill=0)

    out = _clean_ids(out, ["Sample_ID","STUDY_ID"])
    out = out.drop_duplicates(subset=["Sample_ID","site"])
    return out

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crohn", required=True)
    ap.add_argument("--crohn-extended", required=True)  # kept for compatibility
    ap.add_argument("--healthy", required=True)
    ap.add_argument("--out-pooled", required=True)
    ap.add_argument("--out-missing", required=True)
    args = ap.parse_args()

    crohn_core = pd.read_csv(args.crohn)
    crohn_ext  = pd.read_csv(args.crohn_extended)  # not used right now
    healthy    = pd.read_csv(args.healthy)

    # Build long tables
    crohn_long   = crohn_to_long(crohn_core)
    healthy_long = healthy_to_long(healthy)

    # Pooled
    pooled = pd.concat([crohn_long, healthy_long], ignore_index=True)
    pooled = _unique_columns(pooled)

    # Final type enforcement (in case concat introduced object dtypes)
    for c in INT_COLS:
        pooled[c] = _as_int64(pooled[c], fill=0)

    # Save pooled
    pooled.to_csv(args.out_pooled, index=False)

    # Missing report: rows with missing key fields or any NA in INT_COLS
    miss = []
    if pooled["Sample_ID"].isna().any():
        miss.append({
            "issue":"missing_sample_id",
            "n": int(pooled["Sample_ID"].isna().sum())
        })
    # Any rows where all covariates are NA (optional)
    all_na_cov = pooled[INT_COLS].isna().all(axis=1)
    if all_na_cov.any():
        miss.append({
            "issue":"all_covariates_na",
            "n": int(all_na_cov.sum())
        })
    miss_df = pd.DataFrame(miss)
    miss_df.to_csv(args.out_missing, index=False)

    print(f"[OK] pooled rows={len(pooled)}  (Crohn={len(crohn_long)}  Healthy={len(healthy_long)})")

if __name__ == "__main__":
    main()
