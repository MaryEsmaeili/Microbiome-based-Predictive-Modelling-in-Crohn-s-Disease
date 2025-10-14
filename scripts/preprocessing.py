#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing for microbiome pipeline (English-only comments):
- Read metadata Excel + merged MetaPhlAn tables (Crohn + Healthy oral/fecal).
- Normalize/clean sample IDs while preserving leading zeros (NO numeric casting).
- Utrecht (Crohn): extract Sdddd from column names, resolve duplicate sample columns via KEEP-ONE (max library size).
- Healthy: extract 6-digit IDs from headers, resolve duplicates via KEEP-ONE.
- Keep ONLY species-level rows (contain '|s__'), drop unclassified, renormalize columns to sum=1 when needed.
- Save split abundance tables + matched ID mapping + summary + mapping/collision logs.

This script is meant to be executed via Snakemake `script:` and uses `snakemake` globals.
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re
import json

import numpy as np
import pandas as pd

# -------------------------- small utils --------------------------

def ensure_dir_for_file(path: Path) -> None:
    """Create parent directories for a path if they do not exist."""
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric, preserving NaN on failures."""
    return df.apply(pd.to_numeric, errors="coerce")

def _write_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir_for_file(path)
    df.to_csv(path, sep="\t", index=False)

def _write_csv(df: pd.DataFrame, path: Path, index: bool) -> None:
    ensure_dir_for_file(path)
    df.to_csv(path, index=index)

# -------------------------- ID utilities --------------------------

def norm_metadata_id(x: str) -> str:
    """
    Normalize metadata IDs:
    - Trim and uppercase
    - Remove trailing '.rep' or '.<number>' if present
    - DO NOT cast to numeric (leading zeros are preserved)
    """
    s = str(x).strip()
    s = re.sub(r"\.(?:rep|[0-9]+)$", "", s, flags=re.IGNORECASE)
    return s.upper()

def norm_utrecht_colname(col: str) -> Optional[str]:
    """
    Extract 'S' followed by 4 digits (e.g., S0086) from Utrecht merged MetaPhlAn column names.
    Returns None if not found.
    """
    s = str(col).strip()
    m = re.search(r"(S\d{4})", s, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def extract_healthy_id(colname: str) -> str:
    """
    Extract a 6-digit ID from healthy MetaPhlAn column names.
    Tries patterns like '-012345_metaphlan' or the last 6 digits; falls back to a trimmed tail token.
    """
    s = str(colname).strip()
    m = re.search(r"-(\d{6})(?=_(?:rerun_)?metaphlan\b)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{6})(?!.*\d)", s)
    if m2:
        return m2.group(1)
    base = s.split("-")[-1]
    base = re.sub(r"(?i)_rerun", "", base)
    base = re.sub(r"(?i)_metaphlan", "", base)
    m3 = re.search(r"(\d{6})", base)
    if m3:
        return m3.group(1)
    return base[-12:]

# -------------------------- species-level & normalization --------------------------

def keep_species_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows containing '|s__' and drop 'unclassified' species."""
    idx = df.index.astype(str)
    mask_species = idx.str.contains(r"\|s__", regex=True, na=False)
    out = df.loc[mask_species].copy()
    bad = out.index.astype(str).str.contains(r"s__unclassified|unclassified", case=False, na=False)
    return out.loc[~bad]

def renorm_cols_to_one(df: pd.DataFrame) -> pd.DataFrame:
    """Renormalize columns to sum to 1 when sums are neither 0 nor ~1."""
    colsum = df.sum(axis=0)
    needs = ~colsum.round(6).isin([0.0, 1.0])
    if not needs.any():
        return df
    colsum_safe = colsum.replace(0, np.nan)
    return df.div(colsum_safe, axis=1).fillna(0.0)

# -------------------------- duplicate resolution --------------------------

def _choose_one_among_duplicates(df_num: pd.DataFrame, originals: List[str]) -> str:
    """
    Resolve duplicated sample columns by choosing the one with the largest library size (sum).
    On ties, keep the earliest column.
    """
    if len(originals) == 1:
        return originals[0]
    sums = {c: float(pd.to_numeric(df_num[c], errors="coerce").fillna(0).sum()) for c in originals}
    return max(originals, key=lambda c: (sums.get(c, 0.0), -originals.index(c)))

# -------------------------- parsers --------------------------

def parse_utrecht_merged(tsv_path: Path,
                         map_out: Path,
                         collisions_out: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Parse Utrecht merged MetaPhlAn (Crohn):
    - Map original columns to Sdddd keys.
    - Resolve duplicates via KEEP-ONE (max library size).
    - Return abundance (index=clade_name, cols=Sdddd) and kept_map (Sdddd -> kept original).
    - Write mapping and collision logs.
    """
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    if "#clade_name" in df.columns:
        df = df.rename(columns={"#clade_name": "clade_name"})
    rowdata = df.iloc[:, :2].copy()
    abund_raw = df.iloc[:, 2:].copy()

    abund_num = to_numeric_df(abund_raw).fillna(0.0)

    orig_cols = list(abund_raw.columns)
    norm_cols = [norm_utrecht_colname(c) for c in orig_cols]
    id_map_df = pd.DataFrame({"original": orig_cols, "normalized": norm_cols})
    _write_tsv(id_map_df, map_out)

    by_norm: Dict[str, List[str]] = {}
    for o, n in zip(orig_cols, norm_cols):
        if n is None:
            continue
        by_norm.setdefault(n, []).append(o)

    collisions_rows = []
    kept_map: Dict[str, str] = {}
    kept_originals = []
    for nkey, originals in by_norm.items():
        keep = _choose_one_among_duplicates(abund_num, originals)
        kept_map[nkey] = keep
        kept_originals.append(keep)
        dropped = [o for o in originals if o != keep]
        if dropped:
            collisions_rows.append({
                "normalized_key": nkey,
                "kept_original": keep,
                "dropped_originals": ",".join(dropped),
                "n_sources": len(originals),
            })

    collisions_df = pd.DataFrame(collisions_rows)
    if collisions_df.empty:
        collisions_df = pd.DataFrame(columns=["normalized_key", "kept_original", "dropped_originals", "n_sources"])
    _write_tsv(collisions_df, collisions_out)

    abund_kept = abund_raw[kept_originals].copy()
    inv_map = {v: k for k, v in kept_map.items()}  # kept original -> normalized Sdddd
    abund_kept.columns = [inv_map[c] for c in abund_kept.columns]

    abund_kept.index = rowdata.iloc[:, 0].values  # clade_name
    abund_kept = abund_kept.loc[abund_kept.sum(axis=1) > 0]

    abund_kept = keep_species_only(abund_kept)
    abund_kept = renorm_cols_to_one(abund_kept)

    return abund_kept, kept_map

def parse_metaphlan_healthy(tsv_path: Path,
                            map_out: Path,
                            collisions_out: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Parse healthy merged MetaPhlAn:
    - Drop NCBI_tax_id if present; index is clade_name (first column).
    - Extract 6-digit IDs from column names.
    - Resolve duplicates via KEEP-ONE (max library size).
    - Return abundance (index=clade_name, cols=6-digit) and kept_map.
    """
    df = pd.read_csv(tsv_path, sep="\t", comment="#", header=0, index_col=0, low_memory=False)
    df = df.drop(columns=["NCBI_tax_id"], errors="ignore")

    orig_cols = list(df.columns)
    extracted = [extract_healthy_id(c) for c in orig_cols]
    id_map_df = pd.DataFrame({"original": orig_cols, "extracted": extracted})
    _write_tsv(id_map_df, map_out)

    df_num = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    by_key: Dict[str, List[str]] = {}
    for o, k in zip(orig_cols, extracted):
        by_key.setdefault(k, []).append(o)

    collisions_rows = []
    kept_map: Dict[str, str] = {}
    kept_originals = []
    for key, originals in by_key.items():
        keep = _choose_one_among_duplicates(df_num, originals)
        kept_map[key] = keep
        kept_originals.append(keep)
        dropped = [o for o in originals if o != keep]
        if dropped:
            collisions_rows.append({
                "extracted_key": key,
                "kept_original": keep,
                "dropped_originals": ",".join(dropped),
                "n_sources": len(originals),
            })
    collisions_df = pd.DataFrame(collisions_rows)
    if collisions_df.empty:
        collisions_df = pd.DataFrame(columns=["extracted_key", "kept_original", "dropped_originals", "n_sources"])
    _write_tsv(collisions_df, collisions_out)

    kept = df_num[kept_originals].copy()
    inv_map = {v: k for k, v in kept_map.items()}  # kept original -> 6-digit id
    kept.columns = [inv_map[c] for c in kept.columns]
    kept = kept.loc[kept.sum(axis=1) > 0]

    kept = keep_species_only(kept)
    kept = renorm_cols_to_one(kept)

    return kept, kept_map

# -------------------------- metadata loader --------------------------

def load_metadata(metadata_file: Path) -> pd.DataFrame:
    """
    Load metadata Excel and ensure columns: STUDY_ID, Oral_sample_ID, Fecal_sample_ID.
    Supports normal header or row-oriented (variable names in first column).
    Leading zeros in IDs are preserved.
    """
    meta_raw = pd.read_excel(metadata_file, header=None)
    first_row = meta_raw.iloc[0].astype(str).str.lower().tolist()
    looks_like_header = all(any(k in c for k in ["study", "oral", "fecal"]) for c in first_row)

    if looks_like_header:
        meta = pd.read_excel(metadata_file)
    else:
        meta = pd.read_excel(metadata_file, header=None, index_col=0).T.reset_index(drop=True)

    meta.columns = meta.columns.astype(str).str.strip()
    required = ["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    meta = meta[required].copy()
    meta["Oral_clean"]  = meta["Oral_sample_ID"].map(norm_metadata_id)
    meta["Fecal_clean"] = meta["Fecal_sample_ID"].map(norm_metadata_id)
    return meta

# -------------------------- QC (optional) --------------------------

def qc_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Basic per-sample QC: nonzero species count, column sum, Shannon entropy."""
    nz = (df > 0).sum(axis=0)
    colsum = df.sum(axis=0)
    df_safe = df.replace(0, np.nan)
    shannon = -(df_safe * np.log(df_safe)).sum(axis=0).fillna(0.0)
    out = pd.DataFrame({
        "dataset": name,
        "sample": df.columns,
        "species_nonzero": nz.values,
        "colsum": colsum.values,
        "shannon": shannon.values
    })
    out["flag_low_richness"] = (out["species_nonzero"] < 10).astype(int)
    return out

# -------------------------- driver --------------------------

def run_preprocess(metadata_file: Path,
                   crohn_file: Path,
                   healthy_oral_in: Path,
                   healthy_fecal_in: Path,
                   out_oral_crohn: Path,
                   out_fecal_crohn: Path,
                   out_oral_healthy: Path,
                   out_fecal_healthy: Path,
                   matched_out: Path,
                   summary_json: Path,
                   u_map_out: Path,
                   u_collisions_out: Path,
                   ho_map_out: Path,
                   ho_collisions_out: Path,
                   hf_map_out: Path,
                   hf_collisions_out: Path,
                   extra_dir: Path):
    extra_dir.mkdir(parents=True, exist_ok=True)

    # 1) metadata
    meta = load_metadata(metadata_file)

    # 2) Crohn (Utrecht)
    abund_utrecht, u_kept_map = parse_utrecht_merged(
        crohn_file, map_out=u_map_out, collisions_out=u_collisions_out
    )

    # 3) match oral/fecal against available Sdddd columns
    available = set(abund_utrecht.columns)
    meta["Oral_col"]  = meta["Oral_clean"].where(meta["Oral_clean"].isin(available))
    meta["Fecal_col"] = meta["Fecal_clean"].where(meta["Fecal_clean"].isin(available))

    unmatched = meta[meta["Oral_col"].isna() | meta["Fecal_col"].isna()][
        ["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID", "Oral_col", "Fecal_col"]
    ].copy()
    if not unmatched.empty:
        _write_tsv(unmatched, extra_dir / "unmatched_ids.tsv")

    matched = meta.dropna(subset=["Oral_col", "Fecal_col"]).reset_index(drop=True)

    # 4) split Crohn (only matched)
    oral_crohn  = abund_utrecht.loc[:, matched["Oral_col"].unique()].copy()
    fecal_crohn = abund_utrecht.loc[:, matched["Fecal_col"].unique()].copy()

    # 5) Healthy
    oral_healthy,  _ = parse_metaphlan_healthy(healthy_oral_in,  map_out=ho_map_out, collisions_out=ho_collisions_out)
    fecal_healthy, _ = parse_metaphlan_healthy(healthy_fecal_in, map_out=hf_map_out, collisions_out=hf_collisions_out)

    # 6) save
    _write_csv(oral_crohn,    out_oral_crohn,    index=True)
    _write_csv(fecal_crohn,   out_fecal_crohn,   index=True)
    _write_csv(oral_healthy,  out_oral_healthy,  index=True)
    _write_csv(fecal_healthy, out_fecal_healthy, index=True)

    matched_save = matched[["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID", "Oral_col", "Fecal_col"]].copy()
    matched_save = matched_save.rename(columns={"Oral_col": "oral", "Fecal_col": "fecal"})
    matched_save["oral_original_kept"]  = matched_save["oral"].map(lambda k: u_kept_map.get(k, "NA"))
    matched_save["fecal_original_kept"] = matched_save["fecal"].map(lambda k: u_kept_map.get(k, "NA"))
    _write_csv(matched_save, matched_out, index=False)

    # 7) QC (optional)
    qc_rows = []
    if not oral_crohn.empty:    qc_rows.append(qc_table(oral_crohn,    "crohn_oral_matched"))
    if not fecal_crohn.empty:   qc_rows.append(qc_table(fecal_crohn,   "crohn_fecal_matched"))
    if not oral_healthy.empty:  qc_rows.append(qc_table(oral_healthy,  "healthy_oral"))
    if not fecal_healthy.empty: qc_rows.append(qc_table(fecal_healthy, "healthy_fecal"))
    if qc_rows:
        qc_all = pd.concat(qc_rows, ignore_index=True)
        _write_tsv(qc_all, extra_dir / "sample_qc.tsv")

    # 8) summary
    summary = {
        "crohn": {
            "n_cols_kept_total": int(len(abund_utrecht.columns)),
            "n_matched_rows": int(matched.shape[0]),
            "n_unmatched_rows": int(unmatched.shape[0]),
        },
        "healthy_oral":  {"n_cols_kept": int(len(oral_healthy.columns))},
        "healthy_fecal": {"n_cols_kept": int(len(fecal_healthy.columns))},
        "notes": {
            "duplicate_policy": "KEEP-ONE (max library size), no summation.",
            "feature_level": "species only (|s__), unclassified removed",
            "col_renorm": "each column ~ sum to 1 after cleaning",
        }
    }
    ensure_dir_for_file(summary_json)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("[preprocessing] Saved:")
    print(f" - {out_oral_crohn}")
    print(f" - {out_fecal_crohn}")
    print(f" - {out_oral_healthy}")
    print(f" - {out_fecal_healthy}")
    print(f" - {matched_out}")
    print(f"[preprocessing] Summary written to: {summary_json}")
    print(f"[preprocessing] Logs in: {extra_dir}")

# -------------------------- Snakemake entry --------------------------

if __name__ == "__main__":
    try:
        snakemake  # type: ignore  # noqa: F821
    except NameError:
        raise RuntimeError("Run this via Snakemake (script:).")

    md  = Path(snakemake.input["metadata"])        # noqa: F821
    cr  = Path(snakemake.input["crohn"])           # noqa: F821
    hor = Path(snakemake.input["healthy_oral"])    # noqa: F821
    hfe = Path(snakemake.input["healthy_fecal"])   # noqa: F821

    out_oral_c  = Path(snakemake.output["oral_crohn"])     # noqa: F821
    out_fecal_c = Path(snakemake.output["fecal_crohn"])    # noqa: F821
    out_oral_h  = Path(snakemake.output["healthy_oral"])   # noqa: F821
    out_fecal_h = Path(snakemake.output["healthy_fecal"])  # noqa: F821
    matched_out = Path(snakemake.output["matched"])        # noqa: F821

    summary_json      = Path(snakemake.output["preproc_summary"])      # noqa: F821
    u_map_out         = Path(snakemake.output["preproc_u_map"])        # noqa: F821
    u_collisions_out  = Path(snakemake.output["preproc_u_collisions"]) # noqa: F821
    ho_map_out        = Path(snakemake.output["preproc_ho_map"])       # noqa: F821
    ho_collisions_out = Path(snakemake.output["preproc_ho_collisions"])# noqa: F821
    hf_map_out        = Path(snakemake.output["preproc_hf_map"])       # noqa: F821
    hf_collisions_out = Path(snakemake.output["preproc_hf_collisions"])# noqa: F821

    extra_dir = Path("results/preprocessing")

    run_preprocess(
        metadata_file=md,
        crohn_file=cr,
        healthy_oral_in=hor,
        healthy_fecal_in=hfe,
        out_oral_crohn=out_oral_c,
        out_fecal_crohn=out_fecal_c,
        out_oral_healthy=out_oral_h,
        out_fecal_healthy=out_fecal_h,
        matched_out=matched_out,
        summary_json=summary_json,
        u_map_out=u_map_out,
        u_collisions_out=u_collisions_out,
        ho_map_out=ho_map_out,
        ho_collisions_out=ho_collisions_out,
        hf_map_out=hf_map_out,
        hf_collisions_out=hf_collisions_out,
        extra_dir=extra_dir
    )
