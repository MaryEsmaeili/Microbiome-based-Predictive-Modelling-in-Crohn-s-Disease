#!/usr/bin/env python3
"""
MakeCleanMetadata.py
--------------------
Reads a row-oriented Excel metadata file, normalizes it, and outputs
a clean, analysis-ready CSV.

Usage with Snakemake:
    python scripts/MakeCleanMetadata.py \
        --input data/Metadata_metagenomics.xlsx \
        --output data/processed/clinical_clean.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from difflib import get_close_matches

# ===== Mapping: Raw column name -> Clean column name =====
RAW_TO_CLEAN = {
    "SampleID":              "Oral_sample_ID",
    "Fecal_sample_ID":       "Fecal_sample_ID",
    "STUDY_ID":              "STUDY_ID",
    "V1_AgeatFecalSampling": "Age",
    "V1_Sex":                "Sex",
    "V1_BMI":                "BMI",
    "V1_PPI_yes_or_no":      "PPI_use",
    "V1_HarveyBradshawScore":"HBI",
    "Responder_study":       "Responder",
}

# ===== Functions =====
def read_excel_row_oriented(meta_path: str) -> pd.DataFrame:
    """Reads a row-oriented Excel (var names in first col, samples in rows after transpose)."""
    df = pd.read_excel(meta_path, header=None)
    df_t = df.set_index(0).T
    df_t.columns = df_t.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    if "SampleID" not in df_t.columns:
        if "Oral_sample_ID" in df_t.columns:
            df_t = df_t.rename(columns={"Oral_sample_ID": "SampleID"})
        else:
            close = get_close_matches("SampleID", list(map(str, df_t.columns)), n=5)
            raise KeyError(f"Couldn't find 'SampleID'. Close matches: {close}")

    df_t["SampleID"] = df_t["SampleID"].astype(str).str.strip().str.upper()

    return df_t.reset_index(drop=True)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def extract_clean(df_row_fixed: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df_row_fixed)
    missing = [c for c in RAW_TO_CLEAN if c not in df.columns]
    if missing:
        print("[ERROR] Missing required columns:")
        for m in missing:
            close = get_close_matches(m, df.columns, n=3)
            print(f"  - {m}   (close?: {close})")
        raise KeyError("Some required columns are missing.")

    out = df[list(RAW_TO_CLEAN.keys())].rename(columns=RAW_TO_CLEAN)

    for idcol in ["Oral_sample_ID", "Fecal_sample_ID"]:
        if idcol in out.columns:
            out[idcol] = out[idcol].astype(str).str.strip().str.upper()

    for numcol in ["Age", "BMI", "HBI"]:
        if numcol in out.columns:
            out[numcol] = pd.to_numeric(out[numcol], errors="ignore")

    return out

def normalize_clean_df_intbool(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()

    # Standardize NA values
    na_like = {"", " ", "NA", "N A", "N/A", "n/a", "NaN", "-", "—", "None", "null"}
    def _to_nan(x):
        if isinstance(x, str) and x.strip() in na_like:
            return np.nan
        return x
    df = df.applymap(_to_nan)

    # Trim strings
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            df.loc[df[c].isin(["", " "]), c] = np.nan

    # Round Age
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").round(2)

    # Sex: 0/1 int
    if "Sex" in df.columns:
        sex_map = {"F": 1, "Female": 1, "FEMALE": 1, "V": 1, "Woman": 1, "W": 1,
                   "M": 0, "Male": 0, "MALE": 0, "Man": 0, "H": 0}
        df["Sex"] = df["Sex"].map(lambda x: sex_map.get(str(x).strip(), np.nan)).astype("Int64")

    # PPI_use: 0/1 int
    if "PPI_use" in df.columns:
        yn_map = {"Yes": 1, "YES": 1, "Y": 1, "y": 1, True: 1, "1": 1, 1: 1,
                  "No": 0, "NO": 0, "N": 0, "n": 0, False: 0, "0": 0, 0: 0}
        df["PPI_use"] = df["PPI_use"].map(lambda x: yn_map.get(str(x).strip(), np.nan)).astype("Int64")

    # Responder: 0/1 int
    if "Responder" in df.columns:
        yn_map = {"Yes": 1, "YES": 1, "Y": 1, "y": 1, True: 1, "1": 1, 1: 1,
                  "No": 0, "NO": 0, "N": 0, "n": 0, False: 0, "0": 0, 0: 0}
        df["Responder"] = df["Responder"].map(lambda x: yn_map.get(str(x).strip(), np.nan)).astype("Int64")

    return df

import numpy as np
import pandas as pd

def drop_trailing_empty_rows(df: pd.DataFrame, na_tokens=None) -> pd.DataFrame:
    """
    Remove trailing rows at the *end* of the dataframe that are completely empty.
    'Empty' = all cells are NA after coercing common 'NA-like' strings to NaN.
    Keeps internal empty rows; only trims the tail.
    """
    if na_tokens is None:
        na_tokens = {"", " ","nan","NA", "N A", "N/A", "n/a", "NaN", "NAN", "-", "—", "None", "null"}

    # Work on a copy for detection
    df2 = df.copy()

    # Coerce NA-like strings -> NaN (only where values are strings)
    for col in df2.columns:
        if pd.api.types.is_string_dtype(df2[col]) or df2[col].dtype == object:
            s = df2[col].astype(str).str.strip()
            # empty-after-trim -> NaN
            s = s.mask(s.eq(""), np.nan)
            # tokens (case-insensitive) -> NaN
            s_upper = s.str.upper()
            df2[col] = s.mask(s_upper.isin({t.upper() for t in na_tokens}), np.nan)

    # Boolean mask: row has at least one non-NaN?
    non_empty = ~df2.isna().all(axis=1)

    if non_empty.any():
        # last index (positional) that is non-empty
        last_pos = np.where(non_empty.values)[0][-1]
        return df.iloc[: last_pos + 1].reset_index(drop=True)
    else:
        # everything empty -> return empty df
        return df.iloc[0:0].reset_index(drop=True)

# ===== Main =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and normalize row-oriented metadata Excel.")
    parser.add_argument("--input", required=True, help="Path to row-oriented Excel file")
    parser.add_argument("--output", required=True, help="Path to save cleaned CSV")
    args = parser.parse_args()

    df_raw = read_excel_row_oriented(args.input)
    clean_df = extract_clean(df_raw)
    norm_df = normalize_clean_df_intbool(clean_df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm_df = drop_trailing_empty_rows(norm_df)
    norm_df.to_csv(out_path, index=False)
   

    print(f"[OK] Clean metadata saved to: {out_path}")
