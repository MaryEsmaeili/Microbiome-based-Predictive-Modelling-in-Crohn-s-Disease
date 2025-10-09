#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import pandas as pd
import re

# --------------------------
# Expected pooled meta schema (you provided):
# Sample_ID, STUDY_ID, site, disease, Age, Sex, BMI, Smoking,
# Antibiotics_3m, PPI_use, Steroids_ongoing, Immuno_ongoing
# --------------------------

DEFAULT_META_COLS = [
    "Sample_ID","STUDY_ID","site","disease",
    "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use",
    "Steroids_ongoing","Immuno_ongoing"
]

def normalize_id(x: str) -> str:
    """
    Robust sample-id normalization:
    - strip/uppercase
    - drop leading 'S'
    - keep digits only
    - strip leading zeros
    return "0" if ends up empty (won't match anything anyway)
    """
    s = str(x).strip().upper()
    if s.startswith('S'):
        s = s[1:]
    s = re.sub(r'[^0-9]', '', s)
    s = s.lstrip('0')
    return s if s != "" else "0"

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_csv(p)

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # drop duplicated column *names* after trimming
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", required=True, help="QC'd wide(-CLR) matrix (rows = samples)")
    ap.add_argument("--meta", required=True, help="Pooled metadata table (csv/xlsx)")
    ap.add_argument("--out",  required=True, help="Output CSV: wide + selected meta")
    ap.add_argument("--id-col-wide", default="Sample",
                    help="Sample ID column in wide matrix (default: Sample)")
    ap.add_argument("--id-col-meta", default="Sample_ID",
                    help="Sample ID column in metadata (default: Sample_ID)")
    ap.add_argument("--meta-cols", default="",
                    help="Comma-separated meta columns to keep; if empty, use sensible defaults.")
    ap.add_argument("--fill-na-site", default=None,
                    help="Optional fallback value to fill NA in 'site' (e.g., NA)")
    ap.add_argument("--fill-na-group", default=None,
                    help="Optional fallback value to fill NA in 'disease' (e.g., 0/1)")
    args = ap.parse_args()

    # ---- Load & clean tables ----
    wide = clean_headers(read_table(args.wide))
    meta = clean_headers(read_table(args.meta))

    idw = args.id_col_wide
    idm = args.id_col_meta

    if idw not in wide.columns:
        raise SystemExit(f"[attach_meta] wide missing id column: {idw}")
    if idm not in meta.columns:
        raise SystemExit(f"[attach_meta] meta missing id column: {idm}")

    # ---- Which meta columns to keep ----
    if args.meta_cols.strip():
        keep_meta_cols = [c.strip() for c in args.meta_cols.split(",")]
        keep_meta_cols = [idm] + [c for c in keep_meta_cols if c != idm and c in meta.columns]
    else:
        keep_meta_cols = [c for c in DEFAULT_META_COLS if c in meta.columns]
        if idm not in keep_meta_cols:
            keep_meta_cols = [idm] + keep_meta_cols

    meta_sub = meta[keep_meta_cols].copy()
    # safety: drop rows with missing id
    meta_sub = meta_sub.dropna(subset=[idm])

    # ---- Build normalized join key (do NOT overwrite original IDs) ----
    wide["_id_norm"] = wide[idw].map(normalize_id)
    meta_sub["_id_norm"] = meta_sub[idm].map(normalize_id)

    before_n = len(wide)

    # ---- Inner-merge on normalized id (only intersecting samples) ----
    merged = wide.merge(
        meta_sub,
        on="_id_norm",
        how="inner",
        suffixes=("", "_meta")
    )

    after_n = len(merged)

    # ---- Ensure we carry canonical Sample_ID in output ----
    # Prefer metadata's Sample_ID if present; otherwise, reconstruct from wide id
    if "Sample_ID" not in merged.columns and idm in merged.columns:
        merged["Sample_ID"] = merged[idm]
    elif "Sample_ID" not in merged.columns:
        merged["Sample_ID"] = merged[idw]

    # ---- Optional NA fills ----
    if args.fill_na_site and "site" in merged.columns:
        merged["site"] = merged["site"].fillna(args.fill_na_site)
    if args.fill_na_group and "disease" in merged.columns:
        merged["disease"] = merged["disease"].fillna(args.fill_na_group)

    # ---- Feature vs meta columns ----
    meta_order = [c for c in [
        "Sample_ID","STUDY_ID","site","disease",
        "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use",
        "Steroids_ongoing","Immuno_ongoing"
    ] if c in merged.columns]

    non_feature = set([idw, idm, "_id_norm"] + meta_order)
    feature_cols = [c for c in merged.columns if c not in non_feature]

    # ---- Fill NaN features with 0 (safe for CLR / numeric features) ----
    if feature_cols:
        merged[feature_cols] = merged[feature_cols].fillna(0)

    # ---- Prefer keeping Sample_ID; drop the wide id column if redundant ----
    if idw in merged.columns and idm in merged.columns and idw != idm:
        # Keep Sample_ID and meta id; drop wide id
        if "Sample_ID" in merged.columns:
            merged = merged.drop(columns=[idw])

    # ---- Dropped IDs (for log) ----
    dropped_norm = sorted(set(wide["_id_norm"]) - set(merged["_id_norm"]))

    # ---- Column order: meta first, then features (exclude helper key) ----
    front = ["Sample_ID"] if "Sample_ID" in merged.columns else [idw]
    front += [c for c in meta_order if c not in front]
    others = [c for c in merged.columns if c not in front and c != "_id_norm"]
    merged = merged[front + others]

    # ---- Write ----
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    # ---- Log ----
    report = {
        "wide_in": args.wide,
        "meta_in": args.meta,
        "out_csv": args.out,
        "n_wide_samples": before_n,
        "n_after_merge": after_n,
        "dropped_samples_norm": dropped_norm[:200]
    }
    with open(str(Path(args.out).with_suffix(".log.json")), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
