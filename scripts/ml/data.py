# scripts/ml/data.py

"""
Data loading and label construction utilities for microbiome ML tasks.

This module builds analysis-ready feature matrices (X), binary labels (y),
aligned metadata, and optional group labels from taxa tables and pooled
metadata, with special handling for:
  - disease vs healthy classification,
  - PPI-use classification,
  - treatment responder vs non-responder tasks in Crohn’s disease.

High-level behaviour
--------------------
The main entry point is load_X_y(), which:

  1. Reads a taxa-by-sample table (taxa_csv)
     - Columns = taxa/features, rows = samples (auto-flips if needed).
     - Index must be sample IDs that can be matched to metadata.

  2. Builds metadata depending on the chosen target:
     - target == "disease" or "ppi":
         * Uses pooled metadata (model_table_pooled.csv) via _build_pooled_meta().
         * Optionally merges in a source-map (da_pct_all_sources.csv) to obtain
           site_fallback, disease_fallback, cohort labels and harmonised IDs.
     - target == "responder":
         * Uses crohn_metadata.csv via _build_crohn_responder_meta().
         * Creates one row per sample (oral and/or fecal) per Crohn subject,
           with shared subject-level responder status.

  3. Aligns taxa and metadata:
     - Automatically detects whether samples are in rows or columns by comparing
       row/column labels to metadata index.
     - Restricts both tables to the intersection of sample IDs.
     - Raises a detailed error if there is no overlap.

  4. Optional site filtering:
     - site parameter can be "oral", "fecal" or None.
     - For responder tasks, uses the explicit "site" column from the
       crohn_metadata-derived table.
     - For pooled tasks, prefers "site_fallback" from the source-map, otherwise
       auto-detects a site column (e.g. "site", "body_site").
     - Maps common synonyms to canonical "oral"/"fecal" labels and filters rows.

  5. Target construction:
     - target == "disease":
         * Prefer disease_fallback if present; otherwise auto-detect a disease
           column (e.g. "disease", "disease_status", "diagnosis").
         * Requires binary 0/1 values, dropping any samples with invalid labels.
     - target == "ppi":
         * Auto-detects the PPI column (e.g. "PPI_use", "ppi_status").
         * Keeps only samples with interpretable 0/1 usage indicators.
     - target == "responder":
         * Uses "Responder" from crohn_metadata; coerce to 0/1 and drop invalids.

  6. Feature construction:
     - Compositional taxa features are transformed with _safe_log_clr():
         * If values look like percentages, they are converted to proportions.
         * A small pseudocount is added.
         * Centered log-ratio (CLR) is applied per sample.
     - Optional covariates (add_covars):
         * _select_covars() collects requested covariate columns from metadata.
         * Numeric columns are used as-is.
         * Non-numeric columns are one-hot encoded via _one_hot().
         * All covariate columns are coerced to numeric and NaNs filled with 0.
     - Final feature matrix X is an early-fusion concatenation:
         [CLR taxa features | covariates]

  7. Group labels for CV:
     - If subject_id_col exists in metadata (default "subject_id"), its values
       are returned as a "groups" Series aligned to X, suitable for group-aware
       cross-validation (e.g. StratifiedGroupKFold).
     - If missing, groups is returned as None.

  8. Sanity checks:
     - If filtering leaves zero samples, raises a detailed error message
       summarising shapes and, if available, site value counts.

Key helpers
-----------
- _detect_sample_id_col(): heuristically find the sample ID column in metadata.
- _set_index_from_sample_id(): set metadata index to the detected sample ID.
- _as_str_lower(): robustly normalise Series to lowercased strings.
- _binary_from_disease_col(): generic mapping from disease labels to 0/1.
- _safe_log_clr(): CLR transform with pseudocount, assuming rows = samples.
- _one_hot(): safe one-hot encoding for selected metadata columns.
- _select_covars(): build numeric covariate matrix (continuous + one-hot).
- _detect_col(): generic fuzzy matching for key metadata columns.
- _build_crohn_responder_meta(): expand crohn_metadata.csv to per-sample rows
  for responder modelling (oral/fecal per subject).
- _build_pooled_meta(): merge pooled covariate table with source-map for
  disease/PPI tasks and harmonised site/disease labels.
- _ensure_col(): enforce presence or auto-detection of required metadata
  columns with informative error messages.

CLI usage
---------
The module also provides a small command-line wrapper (main()) which:

  - Parses arguments:
      * --taxa, --meta, --site, --target, --add-covars, --rank
      * --out-X, --out-y, --out-meta, --out-groups
  - Calls load_X_y() with the requested settings.
  - Writes:
      * out-X      : feature matrix (X)
      * out-y      : label vector (column 'y')
      * out-meta   : aligned metadata table
      * out-groups : optional subject-level group labels (or empty structure)

Overall, this file defines the standard data-ingestion and preprocessing
pipeline feeding the ML scripts: it handles ID harmonisation, site/disease/PPI
labelling, CLR transformation of compositional taxa, inclusion of clinical
covariates, and construction of subject-level group labels in a robust and
reusable way.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

# -------------------------------
# Utilities
# -------------------------------
def _detect_sample_id_col(df: pd.DataFrame, preferred: str = "sample_id") -> Optional[str]:
    """
    Try to find the column that contains sample IDs.
    Priority:
      1) exact 'preferred'
      2) case/underscore variants like 'Sample_ID', 'sampleID', ...
      3) any column whose name contains both 'sample' and 'id'
    """
    cols = list(df.columns)
    if preferred in cols:
        return preferred

    norm = {c: c.lower().replace("-", "_") for c in cols}
    # common variants
    for cand in ["Sample_ID", "sample_ID", "SampleId", "sampleId", "sampleid", "SampleID"]:
        if cand in cols:
            return cand
    # case-insensitive preferred
    for c, nl in norm.items():
        if nl == preferred:
            return c
    # fuzzy: contains 'sample' and 'id'
    for c, nl in norm.items():
        if "sample" in nl and "id" in nl:
            return c
    return None


def _set_index_from_sample_id(meta: pd.DataFrame, preferred: str = "sample_id") -> pd.DataFrame:
    """
    Set index to the detected sample-id column if not already aligned.
    """
    if meta.index.name and meta.index.is_unique and meta.index.dtype == object:
        return meta  # assume already indexed by sample ids
    sid = _detect_sample_id_col(meta, preferred=preferred)
    if sid is None:
        return meta  # fallback: leave as-is (will be caught later by alignment guard)
    if sid in meta.columns:
        return meta.set_index(sid)
    return meta


def _as_str_lower(s: pd.Series) -> pd.Series:
    """Return a lowercase string version of a pandas Series, robust to numeric/categorical."""
    return s.astype(str).str.strip().str.lower()


def _binary_from_disease_col(s: pd.Series) -> pd.Series:
    """
    Map disease column to 0/1 robustly.
    - If numeric/bool: >0 → 1 else 0
    - Else: string labels → Crohn positives
    """
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return (s.astype(float) > 0).astype(int)
    sl = _as_str_lower(s)
    crohn_like   = {"crohn", "cd", "case", "1", "true", "yes"}
    healthy_like = {"healthy", "control", "0", "false", "no"}
    # default: crohn=1 if matches crohn_like; else 0
    return sl.isin(crohn_like).astype(int)


def _safe_log_clr(df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """
    Apply CLR transform with a small pseudocount to compositional data.
    Assumes rows are samples and columns are taxa/features in percentage (0-100) or proportion (0-1).
    """
    X = df.copy()
    # If percentages, convert to proportions
    if (X.max().max() > 1.0):
        X = X / 100.0
    X = X + pseudocount
    # Geometric mean across features per sample
    gmean = np.exp(np.log(X).mean(axis=1))
    clr = np.log(X.div(gmean, axis=0))
    return clr


def _one_hot(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """One-hot encode selected columns if present; ignore missing ones."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)
    oh = pd.get_dummies(df[cols].astype("category"), drop_first=False)
    return oh


def _select_covars(meta: pd.DataFrame, covars: List[str]) -> pd.DataFrame:
    """
    Build a covariate matrix from metadata:
    - numeric columns are kept as-is
    - non-numeric columns are one-hot encoded
    """
    covars = [c for c in covars if c in meta.columns]
    if not covars:
        return pd.DataFrame(index=meta.index)

    num = meta[covars].select_dtypes(include=[np.number]).copy()
    nonnum_cols = [c for c in covars if c not in num.columns]
    cat = _one_hot(meta, nonnum_cols) if nonnum_cols else pd.DataFrame(index=meta.index)
    out = pd.concat([num, cat], axis=1)
    # Ensure numeric dtype
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna(0.0)


# -------------------------------
# Auto-detect helper for site / disease / ppi / responder
# -------------------------------
def _detect_col(meta: pd.DataFrame, preferred: str, synonyms: List[str]) -> Optional[str]:
    """
    Generic helper: try preferred name, then synonyms, then fuzzy match on substrings.
    """
    cols = list(meta.columns)
    if preferred in cols:
        return preferred

    norm = {c: c.lower().replace("-", "_") for c in cols}
    # exact (case-insensitive) on preferred + synonyms
    targets = [preferred.lower()] + [s.lower() for s in synonyms]
    for c, nl in norm.items():
        if nl in targets:
            return c

    # fuzzy: contains all tokens of any target
    for target in targets:
        toks = [t for t in target.split("_") if t]
        for c, nl in norm.items():
            if all(t in nl for t in toks):
                return c

    return None
def _build_crohn_responder_meta(crohn_meta_csv: str) -> pd.DataFrame:
    """
    Build sample-level metadata for responder analysis from crohn_metadata.csv.

    For each Crohn subject we create up to two rows:
      - one for Oral_sample_ID (site='oral')
      - one for Fecal_sample_ID (site='fecal')
    Both share the same Responder label.
    """
    m = pd.read_csv(crohn_meta_csv)
    m.columns = [c.strip() for c in m.columns]

    required = {"STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID", "Responder"}
    if not required.issubset(m.columns):
        raise ValueError(
            f"crohn_metadata.csv must contain columns: {required}, got {m.columns}"
        )

    rows = []
    for _, r in m.iterrows():
        resp = r["Responder"]
        if pd.isna(resp):
            continue
        for site_name, col in [("oral", "Oral_sample_ID"), ("fecal", "Fecal_sample_ID")]:
            sid = r[col]
            if pd.isna(sid):
                continue
            rows.append({
                "sample_id": str(sid),
                "site": site_name,
                "Responder": resp,
                "STUDY_ID": r.get("STUDY_ID", None),
                "Age": r.get("Age", np.nan),
                "Sex": r.get("Sex", np.nan),
                "BMI": r.get("BMI", np.nan),
                "Smoking": r.get("Smoking", np.nan),
                "Antibiotics_3m": r.get("Antibiotics_3m", np.nan),
                "PPI_use": r.get("PPI_use", np.nan),
                "Steroids_ongoing": r.get("Steroids_ongoing", np.nan),
                "Immuno_ongoing": r.get("Immuno_ongoing", np.nan),
            })

    meta = pd.DataFrame(rows)
    if meta.empty:
        raise ValueError("No responder labels found in crohn_metadata.csv.")
    meta = meta.set_index("sample_id")
    return meta

def _build_pooled_meta(meta_csv: str,
                       source_map_csv: Optional[str],
                       sample_id_col: str) -> pd.DataFrame:
    """
    Build pooled metadata for disease/PPI tasks by merging:
      - model_table_pooled.csv (covariates)
      - da_pct_all_sources.csv (site_fallback, disease_fallback, cohort)
    """
    m = pd.read_csv(meta_csv)
    m.columns = [c.strip() for c in m.columns]

    # index by sample ID (Sample_ID or sample_id or similar)
    if sample_id_col in m.columns:
        m = m.set_index(sample_id_col)
    elif "Sample_ID" in m.columns:
        m = m.set_index("Sample_ID")
    else:
        m = _set_index_from_sample_id(m, preferred=sample_id_col)
    m.index = m.index.astype(str)

    if not source_map_csv:
        return m

    s = pd.read_csv(source_map_csv)
    s.columns = [c.strip() for c in s.columns]
    if "Sample_ID" not in s.columns:
        raise ValueError("source-map must contain 'Sample_ID' column.")
    s = s.rename(columns={"Sample_ID": "sample_id"})
    s["sample_id"] = s["sample_id"].astype(str)
    s = s.set_index("sample_id")

    # normalize fallback site a bit (Oral/Fecal)
    if "site_fallback" in s.columns:
        s["site_fallback"] = s["site_fallback"].astype(str).str.strip()

    meta = s.join(m, how="left")  # keep all samples from source-map
    return meta

def _ensure_col(meta: pd.DataFrame, col: str, role: str, synonyms: List[str]) -> str:
    """
    Ensure that a column exists; if not, try to auto-detect via synonyms.
    Returns the column name actually used.
    """
    if col in meta.columns:
        return col
    auto = _detect_col(meta, col, synonyms)
    if auto is not None:
        return auto
    raise ValueError(
        f"{role} column '{col}' not found in metadata, and no close alternative detected. "
        f"Available columns: {list(meta.columns)}"
    )

# -------------------------------
# Loader
# -------------------------------
def load_X_y(
    taxa_csv: str,
    meta_csv: str,
    site: Optional[str],
    target: str,
    add_covars: Optional[List[str]] = None,
    rank: Optional[str] = None,
    subject_id_col: str = "subject_id",
    sample_id_col: str = "sample_id",
    site_col: str = "site",          # e.g., "oral" / "fecal"
    disease_col: str = "disease",    # e.g., "Crohn" / "Healthy"
    ppi_col: str = "PPI_use",        # binary 0/1 or yes/no
    responder_col: str = "responder",# 0/1 if available (Crohn only)
    # *** NEW:
    source_map_csv: Optional[str] = None,  # da_pct_all_sources.csv
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.Series]]:

    """
    Load features and labels with flexible targeting and covariates.

    Parameters
    ----------
    taxa_csv : path to taxa-by-sample table (columns = taxa, rows = samples).
               Must contain an index column with unique sample IDs that match meta.
    meta_csv : path to pooled metadata, indexed by sample IDs (or containing sample_id_col).
    site     : filter to one body site ("oral" or "fecal") or None to keep all.
    target   : "disease" | "ppi" | "responder"
    add_covars : list of covariate column names to append to features (optional).
    rank     : optional, only for logging/paths (not used in logic).
    subject_id_col, sample_id_col, site_col, disease_col, ppi_col, responder_col : column names.

    Returns
    -------
    X : CLR-transformed taxa features (+ optional covariates concatenated)
    y : label vector
    meta_aligned : metadata aligned to X
    groups : optional Series of group labels (subject_id) aligned to X
    """
    # Read taxa table
    taxa = pd.read_csv(taxa_csv, index_col=0)

    # -----------------------------
    # Build metadata depending on target
    # -----------------------------
    if target == "responder":
        # meta_csv در این حالت crohn_metadata.csv است
        meta = _build_crohn_responder_meta(meta_csv)
    else:
        # disease / ppi → از مدل پولد + source-map
        meta = _build_pooled_meta(meta_csv, source_map_csv, sample_id_col)

    # --- Auto-detect orientation: rows vs columns are samples ---
    inter_rows = len(taxa.index.intersection(meta.index))
    inter_cols = len(taxa.columns.intersection(meta.index))
    if inter_rows == 0 and inter_cols > 0:
        taxa = taxa.T  # samples become rows

    # Align to intersection
    common_ids = taxa.index.intersection(meta.index)
    if common_ids.empty:
        raise ValueError(
            "No overlapping sample IDs between taxa and meta "
            f"(taxa rows={taxa.shape[0]}, meta rows={meta.shape[0]}). "
            f"Example taxa row labels: {taxa.index[:5].tolist()} | "
            f"meta index head: {meta.index[:5].tolist()}"
        )

    taxa = taxa.loc[common_ids].copy()
    meta = meta.loc[common_ids].copy()

    # Robust detection of key columns
    # (site only needed if site filter requested)
    # -----------------------------
    # Site filter (oral / fecal)
    # -----------------------------
    site_col_eff = None
    if site is not None:
        if target == "responder":
            # در meta ساخته شده برای responder ستون 'site' داریم
            site_col_eff = _ensure_col(
                meta, "site", role="Site",
                synonyms=["site"]
            )
        else:
            # اگر source-map داشتیم، از site_fallback استفاده کن
            if "site_fallback" in meta.columns:
                site_col_eff = "site_fallback"
            else:
                site_col_eff = _ensure_col(
                    meta, site_col, role="Site",
                    synonyms=["site", "body_site", "site_fallback"]
                )

        site_raw = meta[site_col_eff]
        site_norm = _as_str_lower(site_raw)

        oral_syn  = {"oral", "mouth", "saliva", "salivary", "oral_cavity", "buccal"}
        fecal_syn = {"fecal", "faecal", "stool", "feces", "faeces"}

        site_mapped = site_norm.copy()
        site_mapped = site_mapped.where(~site_norm.isin(oral_syn),  "oral")
        site_mapped = site_mapped.where(~site_norm.isin(fecal_syn), "fecal")

        keep_ids = meta.index[site_mapped == site.lower()]
        if len(keep_ids) == 0:
            uniq = sorted(site_norm.unique().tolist())
            raise ValueError(
                f"No samples left after site filter. "
                f"Requested site='{site}', but site values were: {uniq[:25]}"
            )
        taxa = taxa.loc[keep_ids]
        meta = meta.loc[keep_ids]

    # -----------------------------
    # Target construction
    # -----------------------------
    if target == "disease":
        # *** NEW: استفاده از disease_fallback اگر باشد
        if "disease_fallback" in meta.columns:
            y_raw = meta["disease_fallback"]
        else:
            dcol = _ensure_col(
                meta, disease_col, role="Disease",
                synonyms=["disease_status", "ibd_status", "diagnosis", "disease_fallback"]
            )
            y_raw = meta[dcol]
        y_num = pd.to_numeric(y_raw, errors="coerce")
        mask = y_num.isin([0, 1])
        taxa = taxa.loc[mask]
        meta = meta.loc[mask]
        y = y_num.loc[mask].astype(int)

    elif target == "ppi":
        # *** NEW: فقط نمونه‌هایی که PPI_use معلوم دارند
        pcol = _ensure_col(
            meta, ppi_col, role="PPI use",
            synonyms=["ppi", "ppi_use", "ppi_status"]
        )
        y_raw = meta[pcol]
        y_num = pd.to_numeric(y_raw, errors="coerce")
        mask = y_num.isin([0, 1])
        taxa = taxa.loc[mask]
        meta = meta.loc[mask]
        y = (y_num.loc[mask] > 0).astype(int)

    elif target == "responder":
        # meta از crohn_metadata آمده و ستون 'Responder' دارد
        if "Responder" not in meta.columns:
            raise ValueError("Responder column not found in crohn_metadata-derived meta.")
        y_raw = meta["Responder"]
        y_num = pd.to_numeric(y_raw, errors="coerce")
        mask = y_num.isin([0, 1])
        taxa = taxa.loc[mask]
        meta = meta.loc[mask]
        y = (y_num.loc[mask] > 0).astype(int)

    else:
        raise ValueError("target must be one of: 'disease', 'ppi', or 'responder'.")

    # CLR transform taxa features (leave scaling to the ML pipeline to avoid leakage)
    X_taxa = _safe_log_clr(taxa)

    # Optional covariates
    X_cov = pd.DataFrame(index=X_taxa.index)
    if add_covars:
        X_cov = _select_covars(meta, add_covars)

    # Concatenate features (early fusion)
    X = pd.concat([X_taxa, X_cov], axis=1)

    # Groups for group-aware CV (subject-level)
    groups = None
    if subject_id_col in meta.columns:
        groups = meta[subject_id_col].astype(str)

    # Align y to X index
    y = y.loc[X.index]
    meta_aligned = meta.loc[X.index]
    # Sanity guard: no samples?
    if X.shape[0] == 0 or y.shape[0] == 0:
        # helpful diagnostics
        msg = [
            f"No samples available after alignment/filtering.",
            f"taxa_csv='{taxa_csv}', meta_csv='{meta_csv}', site={site}, target={target}",
            f"X shape={X.shape}, y len={y.shape[0]}",
        ]
        if site_col in meta.columns:
            msg.append(f"site value counts: {_as_str_lower(meta[site_col]).value_counts().to_dict()}")
        raise ValueError(" | ".join(msg))

    return X, y, meta_aligned, groups


def main():
    parser = argparse.ArgumentParser(description="Build X, y matrices from taxa and metadata.")
    parser.add_argument("--taxa", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--site", choices=["oral", "fecal"], default=None)
    parser.add_argument("--target", choices=["disease", "ppi", "responder"], required=True)
    parser.add_argument("--add-covars", type=str, default="")
    parser.add_argument("--rank", type=str, default="species")
    parser.add_argument("--out-X", required=True)
    parser.add_argument("--out-y", required=True)
    parser.add_argument("--out-meta", required=True)
    parser.add_argument("--out-groups", required=False, default="")
    args = parser.parse_args()

    add_covars = [c.strip() for c in args.add_covars.split(",") if c.strip()] if args.add_covars else []

    X, y, meta_aligned, groups = load_X_y(
        taxa_csv=args.taxa,
        meta_csv=args.meta,
        site=args.site,
        target=args.target,
        add_covars=add_covars,
        rank=args.rank
    )

    X.to_csv(args.out_X)
    y.to_csv(args.out_y, header=["y"])
    meta_aligned.to_csv(args.out_meta)
    if args.out_groups:
        if groups is None:
            pd.Series(index=X.index, dtype=str).to_csv(args.out_groups, header=["group"])
        else:
            groups.to_csv(args.out_groups, header=["group"])

if __name__ == "__main__":
    main()
