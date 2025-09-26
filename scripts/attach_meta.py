#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd
import re

# ---- Default metadata columns (use any that exist) ----
DEFAULT_META_COLS = [
    "Sample_ID","STUDY_ID","site","disease",
    "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use",
    "Steroids_ongoing","Immuno_ongoing"
]

def normalize_id(x: str) -> str:
    """
    Normalize sample IDs for robust merges:
    - strip/uppercase
    - drop leading 'S'
    - keep digits only
    - strip leading zeros
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", required=True, help="QC'd wide-CLR matrix (samples x features)")
    ap.add_argument("--meta", required=True, help="Pooled metadata table (csv/xlsx)")
    ap.add_argument("--out",  required=True, help="Output: wide + metadata")
    ap.add_argument("--id-col-wide", default="Sample", help="Sample ID column in wide matrix")
    ap.add_argument("--id-col-meta", default="Sample_ID", help="Sample ID column in metadata")
    ap.add_argument("--meta-cols", default="", help="Comma-separated metadata columns to keep")
    ap.add_argument("--fill-na-site", default=None, help="Optional site fallback value")
    ap.add_argument("--fill-na-group", default=None, help="Optional disease fallback value")
    args = ap.parse_args()

    # ---- Load tables ----
    wide = pd.read_csv(args.wide)
    meta = read_table(args.meta)

    # ---- Trim column names ----
    wide.columns = [c.strip() for c in wide.columns]
    meta.columns = [c.strip() for c in meta.columns]

    idw = args.id_col_wide
    idm = args.id_col_meta

    # ---- Basic presence checks BEFORE using ----
    if idw not in wide.columns:
        raise SystemExit(f"[attach_meta] wide missing id column: {idw}")
    if idm not in meta.columns:
        raise SystemExit(f"[attach_meta] meta missing id column: {idm}")

    # ---- Select metadata columns ----
    if args.meta_cols.strip():
        keep_meta_cols = [c.strip() for c in args.meta_cols.split(",")]
        # ensure ID column present
        keep_meta_cols = [idm] + [c for c in keep_meta_cols if c != idm and c in meta.columns]
    else:
        # use defaults that exist
        keep_meta_cols = [c for c in DEFAULT_META_COLS if c in meta.columns]
        if idm not in keep_meta_cols:
            keep_meta_cols = [idm] + keep_meta_cols

    meta_sub = meta[keep_meta_cols].copy()
    meta_sub = meta_sub.dropna(subset=[idm])

    # ---- Normalize IDs on both sides (do NOT overwrite original IDs) ----
    wide["_id_norm"]    = wide[idw].map(normalize_id)
    meta_sub["_id_norm"] = meta_sub[idm].map(normalize_id)

    before_n = len(wide)

    # ---- Single robust inner-merge ON normalized id ----
    merged = wide.merge(
        meta_sub,
        on="_id_norm",
        how="inner",
        suffixes=("", "_meta")
    )
    after_n = len(merged)

    # ---- Ensure we have Sample_ID column ----
    if "Sample_ID" not in merged.columns:
        # prefer metadata ID if available
        if idm in merged.columns:
            merged["Sample_ID"] = merged[idm]
        else:
            merged["Sample_ID"] = merged[idw]

    # ---- Optional fallbacks ----
    if args.fill_na_site and "site" in merged.columns:
        merged["site"] = merged["site"].fillna(args.fill_na_site)
    if args.fill_na_group and "disease" in merged.columns:
        merged["disease"] = merged["disease"].fillna(args.fill_na_group)

    # ---- Work out feature columns (exclude ID/meta helpers) ----
    meta_order = [c for c in [
        "Sample_ID","STUDY_ID","site","disease",
        "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use",
        "Steroids_ongoing","Immuno_ongoing"
    ] if c in merged.columns]

    non_feature = set([idw, idm, "_id_norm"] + meta_order)
    feature_cols = [c for c in merged.columns if c not in non_feature]

    # ---- Fill missing feature values with zero (safe for CLR matrices) ----
    if feature_cols:
        merged[feature_cols] = merged[feature_cols].fillna(0)

    # ---- Prefer keeping Sample_ID; drop duplicate original id columns if both exist ----
    if idw in merged.columns and idm in merged.columns and idw != idm:
        # keep Sample_ID; drop the original wide ID if redundant
        if "Sample_ID" in merged.columns:
            # keep idm only if you really need it; otherwise we can drop idw
            merged = merged.drop(columns=[idw])

    # ---- Compute dropped ids BEFORE removing helper column ----
    dropped_norm = sorted(set(wide["_id_norm"]) - set(merged["_id_norm"]))

    # ---- Column order: metadata up front, then features ----
    front = ["Sample_ID"] if "Sample_ID" in merged.columns else [idw]
    front += [c for c in meta_order if c not in front]
    others = [c for c in merged.columns if c not in front and c not in ["_id_norm"]]
    merged = merged[front + others]

    # ---- Now it's safe to drop helper column from the written CSV ----
    # (_id_norm) is already excluded by column selection above

    # ---- Write output ----
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    # ---- Lightweight log next to CSV ----
    report = {
        "wide_in": args.wide,
        "meta_in": args.meta,
        "out_csv": args.out,
        "n_wide_samples": before_n,
        "n_after_merge": after_n,
        "dropped_samples_norm": dropped_norm[:200]  # preview up to 200
    }
    with open(str(Path(args.out).with_suffix(".log.json")), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
