#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- columns ----------
CORE = [
    "Age","Sex","BMI","Smoking","Antibiotics_3m",
    "PPI_use","Steroids_ongoing","Immuno_ongoing"
]
# Crohn-only: فقط RAW و فیلدهای بالینی که گفتی
CROHN_ONLY_BASE = [
    "Calprotectin_baseline_raw", "HBI_baseline_raw",
    "Disease_duration_years","Perianal_disease","Any_resection",
    "Anti_TNF_current","5ASA_current",
]
MONTREAL_COLS = ["Montreal_behavior","Montreal_L4"]

NA_SYMS = {"", "na", "n/a", "none", "null", "nan", "NaN", "NAN"}

# ---------- helpers ----------
def norm_sample_id(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"\.\d+$", "", s)  # drop .rep suffix
    if s.isdigit():
        try: s = str(int(s))      # strip leading zeros on numeric-only
        except: pass
    return s.upper()

def to_int01(x):
    """map yes/no, y/n, true/false, 1/0 -> {0,1,NA} (nullable Int64)"""
    if x is None or (isinstance(x, float) and np.isnan(x)): return pd.NA
    s = str(x).strip().lower()
    if s in NA_SYMS: return pd.NA
    if s in {"1","true","t","yes","y"}: return 1
    if s in {"0","false","f","no","n"}: return 0
    try:
        v = float(s)
        if np.isnan(v): return pd.NA
        return int(v)
    except:
        return pd.NA

def sex_to_int01(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return pd.NA
    s = str(x).strip().lower()
    if s in {"f","female","0"}: return 0
    if s in {"m","male","1"}:   return 1
    try:
        v = float(s)
        if np.isnan(v): return pd.NA
        return int(v)
    except:
        return pd.NA

def map_montreal_behavior(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return pd.NA
    s = str(x).strip().upper()
    if s in {"B1","1"}: return 1
    if s in {"B2","2"}: return 2
    if s in {"B3","3"}: return 3
    return pd.NA

def expand_subject_to_samples(df_subject: pd.DataFrame, disease_flag: int,
                              crohn_only_cols: list[str]) -> pd.DataFrame:
    """از ردیفِ فردی، سطر-نمونه بساز (oral/fecal اگر ID دارد)."""
    rows = []
    for _, r in df_subject.iterrows():
        base = {}
        # core
        for c in CORE:
            base[c] = r.get(c, pd.NA)
        # crohn-only فقط برای disease=1
        for c in crohn_only_cols:
            base[c] = r.get(c, pd.NA) if disease_flag == 1 else pd.NA

        # ORAL
        o = str(r.get("Oral_sample_ID","")).strip()
        if o and o.lower() not in NA_SYMS:
            rows.append({
                "Sample_ID": norm_sample_id(o),
                "site": "oral",
                "site_bin": 0,                # oral -> 0
                "disease": int(disease_flag), # 0/1 int
                **base
            })
        # FECAL
        f = str(r.get("Fecal_sample_ID","")).strip()
        if f and f.lower() not in NA_SYMS:
            rows.append({
                "Sample_ID": norm_sample_id(f),
                "site": "fecal",
                "site_bin": 1,                # fecal -> 1
                "disease": int(disease_flag),
                **base
            })
    return pd.DataFrame(rows)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--healthy", required=True, help="data/meta/healthy_metadata.csv")
    ap.add_argument("--crohn",   required=True, help="data/meta/crohn_metadata.csv")
    ap.add_argument("--out",     required=True, help="data/meta/model_table_pooled.csv")
    ap.add_argument("--report",  required=False, help="results/meta_table_pooled.json")
    ap.add_argument("--include-montreal", action="store_true",
                    help="Add Montreal_behavior (1/2/3) and Montreal_L4 (0/1) for Crohn-only.")
    args = ap.parse_args()

    # read (as string, then clean)
    H = pd.read_csv(args.healthy, dtype=str).replace(NA_SYMS, np.nan)
    C = pd.read_csv(args.crohn,   dtype=str).replace(NA_SYMS, np.nan)

    # ---- typing: core ----
    for df in (H, C):
        if "Age" in df: df["Age"] = pd.to_numeric(df["Age"], errors="coerce").round(1)
        if "BMI" in df:
            df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce").round(1) 
        if "Sex" in df: df["Sex"] = df["Sex"].map(sex_to_int01).astype("Int64")
        for c in ["Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing"]:
            if c in df: df[c] = df[c].map(to_int01).astype("Int64")

    # ---- typing: Crohn-only ----
    for c in ["Calprotectin_baseline_raw","HBI_baseline_raw","Disease_duration_years"]:
        if c in C: C[c] = pd.to_numeric(C[c], errors="coerce")
    for c in ["Perianal_disease","Any_resection","Anti_TNF_current","5ASA_current"]:
        if c in C: C[c] = C[c].map(to_int01).astype("Int64")

    # Montreal (optional)
    crohn_only_cols = CROHN_ONLY_BASE.copy()
    if args.include_montreal:
        if "Montreal_behavior" in C: C["Montreal_behavior"] = C["Montreal_behavior"].map(map_montreal_behavior).astype("Int64")
        if "Montreal_L4" in C:        C["Montreal_L4"]       = C["Montreal_L4"].map(to_int01).astype("Int64")
        crohn_only_cols += [c for c in MONTREAL_COLS if c in C.columns]

    # ---- expand to sample-level ----
    h_samples = expand_subject_to_samples(H, disease_flag=0, crohn_only_cols=crohn_only_cols)
    c_samples = expand_subject_to_samples(C, disease_flag=1, crohn_only_cols=crohn_only_cols)

    pooled = pd.concat([h_samples, c_samples], ignore_index=True)

    # ---- final column order ----
    final_cols = ["Sample_ID","site","site_bin","disease"] \
                 + [c for c in CORE if c in pooled.columns] \
                 + [c for c in crohn_only_cols if c in pooled.columns]

    pooled = (pooled[final_cols]
              .drop_duplicates(subset=["Sample_ID","site"])
              .sort_values(["site","disease","Sample_ID"])
              .reset_index(drop=True))

    # enforce dtypes on outputs
    pooled["disease"] = pooled["disease"].astype("Int64")
    if "site_bin" in pooled: pooled["site_bin"] = pooled["site_bin"].astype("Int64")
    for c in CORE:
        if c in pooled and pooled[c].dtype.name == "object" and c != "Sex":
            # leave non-binary core as is (e.g., Age float), binaries already cast
            pass

    # ---- write ----
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(outp, index=False)

    # ---- report (optional) ----
    if args.report:
        rep = {
            "rows_out": int(len(pooled)),
            "unique_samples": int(pooled["Sample_ID"].nunique()),
            "by_site": pooled["site"].value_counts(dropna=False).to_dict(),
            "missing_core_share": {c: float(pooled[c].isna().mean()) for c in CORE if c in pooled.columns},
            "crohn_only_kept": [c for c in crohn_only_cols if c in pooled.columns],
            "dtypes": {c: str(pooled[c].dtype) for c in pooled.columns},
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f: json.dump(rep, f, indent=2)

    print(f"[OK] wrote {args.out} (rows={len(pooled)})")

if __name__ == "__main__":
    main()

# python metatablepool.py \
#   --crohn   data/meta/crohn_metadata.csv \
#   --healthy data/meta/healthy_metadata.csv \
#   --out     data/meta/model_table_pooled.csv \
#   --report  results/meta_table_pooled.json \
#   --include-montreal      # اگر خواستی مونترآل را هم اضافه کن؛ در غیر این صورت حذفش کن
