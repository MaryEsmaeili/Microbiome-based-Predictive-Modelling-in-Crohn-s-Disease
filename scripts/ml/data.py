# ml/data.py
import argparse
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd


# -------------------------------
# Utilities
# -------------------------------

def _detect_sample_id_col(df: pd.DataFrame, preferred: str = "sample_id") -> Optional[str]:
    """
    Try to find the column that contains sample IDs.
    Priority:
      1) exact 'preferred'
      2) common variants like 'Sample_ID', 'sampleID', ...
      3) any column whose name contains both 'sample' and 'id' (case-insensitive)
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
    Ensure metadata index is sample IDs if possible.
    """
    if meta.index.name and meta.index.is_unique and meta.index.dtype == object:
        return meta  # assume already indexed by sample ids
    sid = _detect_sample_id_col(meta, preferred=preferred)
    if sid is None:
        return meta
    if sid in meta.columns:
        return meta.set_index(sid)
    return meta


def _as_str_lower(s: pd.Series) -> pd.Series:
    """Return a lowercase string version of a pandas Series, robust to numeric/categorical."""
    return s.astype(str).str.strip().str.lower()


def _binary_from_disease_col(s: pd.Series) -> pd.Series:
    """
    Map disease column to 0/1 robustly.
    Crohn-like → 1, Healthy-like → 0.
    """
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return (s.astype(float) > 0).astype(int)
    sl = _as_str_lower(s)
    crohn_like = {"crohn", "cd", "case", "1", "true", "yes", "ibd"}
    # healthy → 0; هر چیز دیگری هم به‌طور پیش‌فرض 0
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

    اگر نمی‌خواهی covariate استفاده کنی، add_covars را خالی بگذار.
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
    ppi_col: str = "PPI_use",        # optional; for target="ppi"
    responder_col: str = "responder" # optional; for target="responder"
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
                 *** برای تحلیل فعلی، این را خالی بگذار تا فقط میکروبیوم وارد مدل شود. ***
    rank     : optional, only for logging/paths (not used in logic).

    Returns
    -------
    X : CLR-transformed taxa features (+ optional covariates concatenated)
    y : label vector
    meta_aligned : metadata aligned to X
    groups : optional Series of group labels (subject_id) aligned to X
    """
    # Read taxa table (index = sample ids or taxa; تشخیص جهت پایین‌تر)
    taxa = pd.read_csv(taxa_csv, index_col=0)

    # Read metadata; allow sample_id_col as a column or index
    meta = pd.read_csv(meta_csv)
    meta = _set_index_from_sample_id(meta, preferred=sample_id_col)

    # --- Auto-detect orientation: rows vs columns are samples ---
    inter_rows = len(taxa.index.intersection(meta.index))
    inter_cols = len(taxa.columns.intersection(meta.index))
    if inter_rows == 0 and inter_cols > 0:
        taxa = taxa.T  # samples become rows

    # Align to intersection
    common_ids = taxa.index.intersection(meta.index)
    if common_ids.empty:
        raise ValueError(
            "No overlapping sample IDs between taxa and meta.\n"
            f"Example taxa row labels: {taxa.index[:5].tolist()}\n"
            f"Example taxa col labels: {taxa.columns[:5].tolist()}\n"
            f"Meta index head: {meta.index[:5].tolist()}"
        )

    taxa = taxa.loc[common_ids].copy()
    meta = meta.loc[common_ids].copy()

    # Filter by site if requested
    if site is not None:
        if site_col not in meta.columns:
            raise ValueError(
                f"Site column '{site_col}' not found in metadata. "
                f"Available columns: {list(meta.columns)}"
            )
        site_raw = meta[site_col]
        site_norm = _as_str_lower(site_raw)

        oral_syn = {"oral", "mouth", "saliva", "salivary", "oral_cavity", "buccal"}
        fecal_syn = {"fecal", "faecal", "stool", "feces", "faeces"}

        site_mapped = site_norm.copy()
        site_mapped = site_mapped.where(~site_norm.isin(oral_syn), "oral")
        site_mapped = site_mapped.where(~site_norm.isin(fecal_syn), "fecal")

        keep_ids = meta.index[site_mapped == site.lower()]
        if len(keep_ids) == 0:
            uniq = sorted(site_norm.unique().tolist())
            raise ValueError(
                f"No samples left after site filter. "
                f"Requested site='{site}', but '{site_col}' unique values were: {uniq[:25]}"
            )
        taxa = taxa.loc[keep_ids]
        meta = meta.loc[keep_ids]

    # Target construction
    if target == "disease":
        if disease_col not in meta.columns:
            raise ValueError(
                f"Disease column '{disease_col}' not found in metadata. "
                f"Available: {list(meta.columns)}"
            )
        y = _binary_from_disease_col(meta[disease_col])

    elif target == "ppi":
        if ppi_col not in meta.columns:
            raise ValueError(
                f"PPI column '{ppi_col}' not found in metadata. "
                f"Available: {list(meta.columns)}"
            )
        y_raw = meta[ppi_col]
        if pd.api.types.is_numeric_dtype(y_raw) or pd.api.types.is_bool_dtype(y_raw):
            y = (pd.to_numeric(y_raw, errors="coerce").fillna(0) > 0).astype(int)
        else:
            y = _as_str_lower(y_raw).map(
                {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
            ).fillna(0).astype(int)

    elif target == "responder":
        if disease_col not in meta.columns:
            raise ValueError(f"Disease column '{disease_col}' not found in metadata.")
        crohn_mask = _binary_from_disease_col(meta[disease_col]) == 1
        crohn_ids = meta.index[crohn_mask]

        taxa = taxa.loc[crohn_ids]
        meta = meta.loc[crohn_ids]

        if responder_col not in meta.columns:
            raise ValueError(
                f"Responder column '{responder_col}' not found in metadata (Crohn only)."
            )
        meta = meta[meta[responder_col].notna()]
        taxa = taxa.loc[meta.index]

        resp = meta[responder_col]
        if pd.api.types.is_numeric_dtype(resp) or pd.api.types.is_bool_dtype(resp):
            y = (pd.to_numeric(resp, errors="coerce").fillna(0) > 0).astype(int)
        else:
            y = _as_str_lower(resp).map(
                {
                    "yes": 1, "no": 0,
                    "responder": 1, "nonresponder": 0,
                    "1": 1, "0": 0
                }
            ).fillna(0).astype(int)
    else:
        raise ValueError("target must be one of: 'disease', 'ppi', or 'responder'.")

    # CLR transform taxa features
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
        msg = [
            "No samples available after alignment/filtering.",
            f"taxa_csv='{taxa_csv}', meta_csv='{meta_csv}', site={site}, target={target}",
            f"X shape={X.shape}, y len={y.shape[0]}",
        ]
        if site_col in meta.columns:
            msg.append(
                f"site value counts: {_as_str_lower(meta[site_col]).value_counts().to_dict()}"
            )
        raise ValueError(" | ".join(msg))

    return X, y, meta_aligned, groups


def main():
    parser = argparse.ArgumentParser(
        description="Build X, y matrices from taxa and metadata."
    )
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

    add_covars = [
        c.strip() for c in args.add_covars.split(",") if c.strip()
    ] if args.add_covars else []

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
            pd.Series(index=X.index, dtype=str).to_csv(
                args.out_groups, header=["group"]
            )
        else:
            groups.to_csv(args.out_groups, header=["group"])


if __name__ == "__main__":
    main()
