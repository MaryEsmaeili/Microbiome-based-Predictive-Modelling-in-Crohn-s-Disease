#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml/data.py
----------
Data loading, normalization, and featurization for ML tasks.

- Reads taxa×samples percentage table (0..100) and pooled metadata.
- Builds a samples×features matrix using CLR transform, then Z-score.
- Robust to metadata column naming; gracefully filters to selected site(s).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional

# -------------------- Metadata normalization --------------------

def _first_existing_column(df: pd.DataFrame, candidates) -> Optional[str]:
    lower_to_true = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_true:
            return lower_to_true[c.lower()]
    return None

def norm_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Return columns: sample_id(str), site in {Oral,Fecal,NaN}, disease in {0,1,NaN}."""
    m = meta.copy()
    m.columns = [str(c).strip() for c in m.columns]

    def get_col(*cands):
        name = _first_existing_column(m, cands)
        if name is None:
            return pd.Series([np.nan]*len(m), index=m.index)
        return m[name]

    out = pd.DataFrame({
        "sample_id": get_col("sample_id","id","sample","sampleid").astype(str),
        "site":      get_col("site","body_site","location"),
        "disease":   get_col("disease","status","group"),
    })

    # site normalization
    out["site"] = out["site"].astype(str).str.strip().str.capitalize().replace({"Faecal":"Fecal"})
    mask = ~out["site"].isin(["Oral","Fecal"])
    out.loc[mask, "site"] = np.nan

    def to01(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().lower()
        if s in {"1","yes","true","y","crohn","cd","case","ibd"}: return 1
        if s in {"0","no","false","n","healthy","control","hc","non-ibd","nonibd"}: return 0
        try:
            v = int(float(s))
            if v in (0,1): return v
        except Exception:
            pass
        return np.nan

    out["disease"] = out["disease"].map(to01)
    return out

# -------------------- CLR transform --------------------

def clr_transform(pct_df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """
    Input: taxa×samples percentages (0..100). Output: taxa×samples CLR.
    """
    X = (pct_df.astype(float) / 100.0) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)
    return logX.sub(gm, axis=1)

# -------------------- Feature matrix builder --------------------

def build_matrix(
    pct_all_csv: str,
    meta_csv: str,
    site: Optional[str] = None,
    rank: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Returns
    -------
    Xz : pd.DataFrame  (samples×features, z-scored CLR)
    y  : pd.Series     (binary disease label {0,1})
    meta_sel : pd.DataFrame  (aligned metadata for selected samples)

    Notes
    -----
    - site: "oral" or "fecal" (case-insensitive). If None, keep all and ignore site filter.
    - rank is not used here but kept for CLI parity/logging.
    """
    pct = pd.read_csv(pct_all_csv, index_col=0)
    pct.columns = pct.columns.astype(str)

    clr = clr_transform(pct, 1e-6).T  # samples×taxa
    clr.index.name = "Sample_ID"

    meta_raw = pd.read_csv(meta_csv)
    meta_n = norm_meta(meta_raw)
    meta_n["sample_id"] = meta_n["sample_id"].astype(str)

    # Keep only samples present in the matrix
    meta_n = meta_n[meta_n["sample_id"].isin(clr.index.astype(str))].copy()

    # Site filter
    site_norm = None
    if site is not None:
        s = str(site).strip().lower()
        if s not in {"oral","fecal"}:
            warnings.warn(f"[data] Unknown site '{site}', proceeding without site filter.")
        else:
            site_norm = "Oral" if s == "oral" else "Fecal"
            meta_n = meta_n[meta_n["site"] == site_norm].copy()

    # Label
    lab = meta_n["disease"]
    sel_ids = meta_n.loc[lab.isin([0,1]), "sample_id"].astype(str)

    X = clr.loc[clr.index.astype(str).isin(sel_ids)].copy()
    # Align in the same order as meta
    X = X.reindex(sel_ids)

    y = meta_n.set_index("sample_id").loc[sel_ids, "disease"].astype(int)

    # Z-score features (per column)
    # Add small epsilon to std to avoid division by zero
    std = X.std(axis=0, ddof=0).replace(0, np.nan)
    Xz = (X - X.mean(axis=0)) / std
    Xz = Xz.fillna(0.0)

    meta_sel = meta_n.set_index("sample_id").loc[sel_ids]
    return Xz, y, meta_sel
