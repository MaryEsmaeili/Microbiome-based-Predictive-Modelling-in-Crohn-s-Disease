#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a pooled, long-format model table:
- Crohn core (site split into oral/fecal using *_sample_ID)
- Healthy metadata (site split similarly)
- Robust ID normalization (uppercase, drop replicate suffix, trim leading zeros)
- Clean NA-like strings, enforce sensible dtypes
- Writes a small "missing" report

CLI:
  --crohn           data/meta/crohn_metadata_core.csv
  --crohn-extended  data/meta/crohn_metadata_extended.csv   (kept for compatibility)
  --healthy         data/meta/healthy_metadata.csv
  --out-pooled      data/meta/model_table_pooled.csv
  --out-missing     data/meta/model_table_pooled_missing.csv
"""
import argparse
import re
import numpy as np
import pandas as pd

# --------------------------- required columns ---------------------------
REQ_COLS_CORE = [
    "STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID",
    "Age", "Sex", "BMI", "Smoking", "Antibiotics_3m", "PPI_use",
    "Steroids_ongoing", "Immuno_ongoing"
]
REQ_COLS_HEALTHY = [
    "STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID",
    "Age", "Sex", "BMI", "Smoking", "Antibiotics_3m", "PPI_use",
    "Steroids_ongoing"  # Healthy has no Immuno_ongoing in your data
]

TARGET_COLS = [
    "Sample_ID", "STUDY_ID", "site", "disease",
    "Age", "Sex", "BMI", "Smoking", "Antibiotics_3m", "PPI_use", "Steroids_ongoing", "Immuno_ongoing"
]

# categorical / coded ints (leave BMI as float, Age as integer)
CAT_INT_COLS = ["disease","Sex","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing"]
MISSING_STRS = {"", "NA", "NaN", "NAN", "NULL", "Null", "null", "None", "none"}

# --------------------------- helpers ---------------------------
def _as_int64(series, fill=0):
    """Coerce to pandas nullable Int64 (keeps NA); optionally fill with a code (default 0)."""
    s = pd.to_numeric(series, errors="coerce").astype("Int64")
    s = s.fillna(fill).astype("Int64")
    return s

def _as_float(series):
    """Coerce to float (BMI etc.)."""
    return pd.to_numeric(series, errors="coerce").astype("float64")

def normalize_id(x):
    """Remove replicate suffix '.<digits>', drop leading zeros if purely numeric, uppercase."""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    s = re.sub(r"\.\d+$", "", s)           # kill replicate suffixes like '.1'
    if re.fullmatch(r"\d+", s):            # pure numeric? drop leading zeros
        s = s.lstrip("0") or "0"
    return s.upper()

def _clean_ids(df, cols):
    """Normalize ID-like columns and coerce common 'NA' strings to real NA."""
    for c in cols:
        if c in df.columns:
            s = df[c].astype("string").str.strip()
            # cast NA-like strings to NA
            s = s.where(~s.str.upper().isin({m.upper() for m in MISSING_STRS}), pd.NA)
            # normalize sample-like columns
            if c.endswith("_sample_ID") or c == "Sample_ID":
                s = s.apply(normalize_id)
            df[c] = s
    return df

def _unique_columns(df):
    """Ensure unique column names (drop right-most duplicates)."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df

# --------------------------- builders ---------------------------
def crohn_to_long(core_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQ_COLS_CORE if c not in core_df.columns]
    if missing:
        raise KeyError(f"[Crohn] Missing required columns: {missing}")

    df = core_df.copy()
    df = _clean_ids(df, ["STUDY_ID","Oral_sample_ID","Fecal_sample_ID"])

    # Keep only real participants (START...) if present
    if "STUDY_ID" in df.columns:
        df = df[df["STUDY_ID"].astype(str).str.startswith("START", na=False)].copy()

    cov = df[["STUDY_ID","Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing"]].copy()

    # ORAL rows
    oral = df[["STUDY_ID","Oral_sample_ID"]].rename(columns={"Oral_sample_ID":"Sample_ID"})
    oral = oral[oral["Sample_ID"].notna()].copy()
    oral["site"] = "oral"
    oral = oral.merge(cov, on="STUDY_ID", how="left")

    # FECAL rows
    fecal = df[["STUDY_ID","Fecal_sample_ID"]].rename(columns={"Fecal_sample_ID":"Sample_ID"})
    fecal = fecal[fecal["Sample_ID"].notna()].copy()
    fecal["site"] = "fecal"
    fecal = fecal.merge(cov, on="STUDY_ID", how="left")

    out = pd.concat([oral, fecal], ignore_index=True)
    out["disease"] = 1  # Crohn
    out = _unique_columns(out)

    # Reorder / complete
    for col in TARGET_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[TARGET_COLS].copy()

    # Types
    out["Age"]  = _as_int64(out["Age"],  fill=0)
    out["BMI"]  = _as_float(out["BMI"])
    for c in CAT_INT_COLS:
        out[c] = _as_int64(out[c], fill=0)

    out = _clean_ids(out, ["Sample_ID","STUDY_ID"])
    out = out.drop_duplicates(subset=["Sample_ID","site"])
    return out

def healthy_to_long(healthy_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQ_COLS_HEALTHY if c not in healthy_df.columns]
    if missing:
        raise KeyError(f"[Healthy] Missing required columns: {missing}")

    df = healthy_df.copy()
    df = _clean_ids(df, ["STUDY_ID","Oral_sample_ID","Fecal_sample_ID"])

    cov = df[["STUDY_ID","Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing"]].copy()
    cov["Immuno_ongoing"] = 0  # not present for healthy in your data

    # ORAL rows
    oral = df[["STUDY_ID","Oral_sample_ID"]].rename(columns={"Oral_sample_ID":"Sample_ID"})
    oral = oral[oral["Sample_ID"].notna()].copy()
    oral["site"] = "oral"
    oral = oral.merge(cov, on="STUDY_ID", how="left")

    # FECAL rows
    fecal = df[["STUDY_ID","Fecal_sample_ID"]].rename(columns={"Fecal_sample_ID":"Sample_ID"})
    fecal = fecal[fecal["Sample_ID"].notna()].copy()
    fecal["site"] = "fecal"
    fecal = fecal.merge(cov, on="STUDY_ID", how="left")

    out = pd.concat([oral, fecal], ignore_index=True)
    out["disease"] = 0  # Healthy
    out = _unique_columns(out)

    # Reorder / complete
    for col in TARGET_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[TARGET_COLS].copy()

    # Types
    out["Age"]  = _as_int64(out["Age"],  fill=0)
    out["BMI"]  = _as_float(out["BMI"])
    for c in CAT_INT_COLS:
        out[c] = _as_int64(out[c], fill=0)

    out = _clean_ids(out, ["Sample_ID","STUDY_ID"])
    out = out.drop_duplicates(subset=["Sample_ID","site"])
    return out

# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crohn",           required=True)
    ap.add_argument("--crohn-extended",  required=True)  # kept for compatibility
    ap.add_argument("--healthy",         required=True)
    ap.add_argument("--out-pooled",      required=True)
    ap.add_argument("--out-missing",     required=True)
    args = ap.parse_args()

    crohn_core = pd.read_csv(args.crohn)
    _ = pd.read_csv(args.crohn_extended)  # not used now, but keep CLI stable
    healthy    = pd.read_csv(args.healthy)

    crohn_long   = crohn_to_long(crohn_core)
    healthy_long = healthy_to_long(healthy)

    pooled = pd.concat([crohn_long, healthy_long], ignore_index=True)
    pooled = _unique_columns(pooled)

    # Final type enforcement (in case concat messed types)
    pooled["Age"] = _as_int64(pooled["Age"], fill=0)
    pooled["BMI"] = _as_float(pooled["BMI"])
    for c in CAT_INT_COLS:
        pooled[c] = _as_int64(pooled[c], fill=0)

    pooled.to_csv(args.out_pooled, index=False)

    # Missing report
    cov_total = ["Age","BMI"] + CAT_INT_COLS
    miss_rows = pooled[cov_total].isna().all(axis=1)
    report = []
    if pooled["Sample_ID"].isna().any():
        report.append({"issue": "missing_sample_id", "n": int(pooled["Sample_ID"].isna().sum())})
    if miss_rows.any():
        report.append({"issue": "all_covariates_na_rowcount", "n": int(miss_rows.sum())})
    pd.DataFrame(report).to_csv(args.out_missing, index=False)

    print(f"[OK] pooled rows={len(pooled)}  (Crohn={len(crohn_long)}  Healthy={len(healthy_long)})")

if __name__ == "__main__":
    main()
