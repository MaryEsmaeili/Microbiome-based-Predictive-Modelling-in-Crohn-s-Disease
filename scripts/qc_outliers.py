#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def _clean_cols(df): df.columns = [c.strip() for c in df.columns]; return df
def _find_sample_col(df):
    low = {c.lower(): c for c in df.columns}
    return low.get("sample") or low.get("sample_id")

def _ensure_sample(df):
    df = _clean_cols(df).copy()
    # drop duplicated column names if any
    df = df.loc[:, ~df.columns.duplicated()]
    sc = _find_sample_col(df)
    if sc is None:
        df = df.reset_index().rename(columns={"index":"Sample"})
        sc = "Sample"
    # Build a single 'Sample' column (normalized)
    df["Sample"] = df[sc].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    # If sc was something else (Sample_ID), rename to Sample and drop duplicates of name
    if sc != "Sample":
        df = df.rename(columns={sc:"Sample"})
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _numeric_block(df):
    return df.drop(columns=[c for c in df.columns if c.lower()=="sample"], errors="ignore").select_dtypes(include=[np.number])

def l2_norms(df_wide):
    X = _numeric_block(df_wide)
    if X.shape[1]==0: raise ValueError("No numeric features to compute norms.")
    return pd.Series(np.linalg.norm(X.to_numpy(dtype=float), axis=1), index=df_wide.index, name="l2")

def iqr_bounds(s, mult=1.5):
    q1, q3 = np.nanpercentile(s, 25), np.nanpercentile(s, 75)
    iqr = q3-q1; return q1,q3,iqr,(q1-mult*iqr),(q3+mult*iqr)

def detect(df, label, mult=1.5):
    n = l2_norms(df); q1,q3,iqr,lo,hi = iqr_bounds(n, mult)
    out = pd.DataFrame({"Sample":df["Sample"], f"norm_{label}":n, f"outlier_{label}":(n<lo)|(n>hi),
                        f"{label}_q1":q1, f"{label}_q3":q3, f"{label}_iqr":iqr,
                        f"{label}_lower_thr":lo, f"{label}_upper_thr":hi})
    return out, dict(q1=q1,q3=q3,iqr=iqr,lower=lo,upper=hi)

def read_meta(path):
    """
    Read pooled metadata and return a table with a SINGLE 'Sample' column.
    Pooled schema we expect:
      Sample_ID, STUDY_ID, site, disease, Age, Sex, BMI, Smoking,
      Antibiotics_3m, PPI_use, Steroids_ongoing, Immuno_ongoing
    - Trim header whitespace
    - Drop duplicated column names after trimming
    - Build 'Sample' from 'Sample_ID' if available; otherwise use existing 'Sample'
    """
    if not path:
        return None
    # Read robustly
    try:
        m = pd.read_csv(path)
    except Exception:
        try:
            m = pd.read_excel(path)
        except Exception:
            return None

    # Clean headers and drop duplicate column NAMES
    m = m.copy()
    m.columns = [str(c).strip() for c in m.columns]
    m = m.loc[:, ~m.columns.duplicated()]

    # If both Sample and Sample_ID exist, prefer Sample_ID to define a clean 'Sample'
    if "Sample_ID" in m.columns:
        # Build a clean 'Sample' from Sample_ID
        m["Sample"] = m["Sample_ID"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
        # If another 'Sample' also exists, keep the new one by dropping the old-before we set it
        # (Since we just overwrote/created 'Sample', ensure there is only one column named 'Sample')
        m = m.loc[:, ~m.columns.duplicated()]
    else:
        # Fall back to any existing 'Sample' (normalize)
        if "Sample" in m.columns:
            m["Sample"] = m["Sample"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
        else:
            # No key to merge: cannot attach meta safely
            return None

    return m


def breakdown(title, m):
    if m is None or "Sample" not in m.columns: return ""
    lines=[title]
    for col in ("disease","Group","site","Site"):
        if col in m.columns: lines += [f"  {col}:", str(m[col].value_counts(dropna=False))]
    return "\n".join(lines)+"\n"

def main():
    ap = argparse.ArgumentParser(description="QC outliers on genus/species wide matrices using L2+IQR.")
    ap.add_argument("--in-genus", required=True)
    ap.add_argument("--in-species", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--out-genus", required=True)
    ap.add_argument("--out-species", required=True)
    ap.add_argument("--out-exclusion", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-norms", default="results/qc/qc_norms.csv")
    ap.add_argument("--iqr-mult", type=float, default=1.5)
    args = ap.parse_args()

    outdir = Path(args.out_report).parent; outdir.mkdir(parents=True, exist_ok=True)

    g = _ensure_sample(pd.read_csv(args.in_genus))
    s = _ensure_sample(pd.read_csv(args.in_species))

    meta = read_meta(args.meta)
    meta_g_before = g[["Sample"]].merge(meta, on="Sample", how="left") if meta is not None else None
    meta_s_before = s[["Sample"]].merge(meta, on="Sample", how="left") if meta is not None else None

    g_out, g_thr = detect(g, "genus", args.iqr_mult)
    s_out, s_thr = detect(s, "species", args.iqr_mult)
    merged = g_out.merge(s_out[["Sample","outlier_species","norm_species","species_q1","species_q3","species_iqr","species_lower_thr","species_upper_thr"]], on="Sample", how="outer")
    merged["outlier_any"] = merged["outlier_genus"].fillna(False) | merged["outlier_species"].fillna(False)

    excl = merged.loc[merged["outlier_any"], "Sample"].dropna().astype(str).tolist()
    Path(args.out_exclusion).write_text("\n".join(excl), encoding="utf-8")

    g_qc = g[~g["Sample"].isin(excl)].copy()
    s_qc = s[~s["Sample"].isin(excl)].copy()
    g_qc.to_csv(args.out_genus, index=False); s_qc.to_csv(args.out_species, index=False)

    meta_g_after = g_qc[["Sample"]].merge(meta, on="Sample", how="left") if meta is not None else None
    meta_s_after = s_qc[["Sample"]].merge(meta, on="Sample", how="left") if meta is not None else None

    rep = []
    rep.append("=== QC Outlier Report (L2 + IQR) ===\n")
    rep.append(f"GENUS : Q1={g_thr['q1']:.6f}, Q3={g_thr['q3']:.6f}, IQR={g_thr['iqr']:.6f}, Lower={g_thr['lower']:.6f}, Upper={g_thr['upper']:.6f}")
    rep.append(f"SPECIES: Q1={s_thr['q1']:.6f}, Q3={s_thr['q3']:.6f}, IQR={s_thr['iqr']:.6f}, Lower={s_thr['lower']:.6f}, Upper={s_thr['upper']:.6f}\n")
    rep.append(f"Original samples: genus={len(g)}, species={len(s)}")
    rep.append(f"Excluded (union): {len(excl)}")
    rep.append("List: " + (", ".join(excl) if excl else "(none)") + "\n")
    rep.append(breakdown("Before QC (GENUS):", meta_g_before))
    rep.append(breakdown("After  QC (GENUS):", meta_g_after))
    rep.append(breakdown("Before QC (SPECIES):", meta_s_before))
    rep.append(breakdown("After  QC (SPECIES):", meta_s_after))
    Path(args.out_report).write_text("\n".join(rep), encoding="utf-8")

    cols = ["Sample",
            "norm_genus","outlier_genus","genus_q1","genus_q3","genus_iqr","genus_lower_thr","genus_upper_thr",
            "norm_species","outlier_species","species_q1","species_q3","species_iqr","species_lower_thr","species_upper_thr",
            "outlier_any"]
    merged[cols].to_csv(args.out_norms, index=False)

    print(json.dumps({
        "in_genus": args.in_genus, "in_species": args.in_species,
        "excluded_n": len(excl), "excluded_samples": excl,
        "genus_thresholds": g_thr, "species_thresholds": s_thr,
        "out_genus": args.out_genus, "out_species": args.out_species,
        "out_exclusion": args.out_exclusion, "out_report": args.out_report, "out_norms": args.out_norms
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
