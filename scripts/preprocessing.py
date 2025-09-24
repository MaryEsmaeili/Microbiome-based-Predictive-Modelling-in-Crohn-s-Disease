#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snakemake-only preprocessing (improved, no-sum policy on duplicate columns):
- Reads metadata + merged MetaPhlAn tables (Crohn + Healthy oral/fecal).
- Normalizes/cleans sample IDs, detects collisions, writes detailed logs.
- IMPORTANT: When multiple original columns normalize to the same key,
  we KEEP ONLY ONE column (the one with the largest library size), and DROP the rest.
  We DO NOT sum duplicates (not appropriate for relative abundance).
- Emits:
    data/processed/oral_abund_crohn.csv
    data/processed/fecal_abund_crohn.csv
    data/processed/oral_abund_healthy.csv
    data/processed/fecal_abund_healthy.csv
    data/processed/matched_sample_ids.csv
- Logs (results/preprocessing/):
    Utrecht_id_map_original_to_norm.tsv
    Utrecht_id_collisions.tsv
    Healthy_oral_id_map.tsv
    Healthy_oral_id_collisions.tsv
    Healthy_fecal_id_map.tsv
    Healthy_fecal_id_collisions.tsv
    summary.json
NOTE: This script MUST be run via Snakemake. No CLI entrypoint.
"""

import re
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import pandas as pd
import numpy as np

# ---------------- small helpers (pure-Python) ----------------
def ensure_dir_for_file(path: Path) -> None:
    """Create parent directory of file path."""
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric, coercing errors to NaN."""
    return df.apply(pd.to_numeric, errors="coerce")

def norm_id_canonical(x: str, width: int = 5) -> str:
    """Normalize sample IDs to a canonical short ID:
    - strip whitespace
    - drop trailing '.0'
    - drop leading zeros ONLY if the string starts with zeros
    - truncate to `width`
    This works well for IDs like 'S0001_merged...' -> 'S0001' (first 5 chars).
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    # Do not strip a leading letter; only strip zeros if the first char is '0'
    if s and s[0] == "0":
        s = s.lstrip("0") or "0"
    return s[:width]

def _write_tsv(df: pd.DataFrame, path: Path):
    ensure_dir_for_file(path)
    df.to_csv(path, sep="\t", index=False)

# ---------------- I/O parsers (with robust duplicate handling) ----------------
def _choose_one_among_duplicates(df_num: pd.DataFrame, originals: List[str]) -> str:
    """Choose the 'best' original column among duplicates by largest library size (sum).
    In case of ties or non-numeric, fall back to the first appearance."""
    if len(originals) == 1:
        return originals[0]
    sums = {}
    for c in originals:
        series = pd.to_numeric(df_num[c], errors="coerce")
        sums[c] = float(series.fillna(0).sum())
    # pick column with maximum sum; if tie, pick first by appearance order
    best = max(originals, key=lambda c: (sums.get(c, 0.0), -originals.index(c)))
    return best

def parse_utrecht_merged(tsv_path: Path, logdir: Path, width: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str,str]]:
    """
    Expect columns: '#clade_name','NCBI_tax_id', then sample columns.
    Returns:
      - abundance matrix (rows=clade_name, cols=normalized IDs; duplicates resolved by KEEP-ONE)
      - id_map_df: columns [original, normalized]
      - collisions_df: columns [normalized_key, kept_original, dropped_originals, n_sources]
      - kept_map: dict normalized_key -> kept_original
    """
    df = pd.read_csv(tsv_path, sep="\t")
    rowdata = df.iloc[:, :2].copy()
    if "#clade_name" in rowdata.columns:
        rowdata = rowdata.rename(columns={"#clade_name": "clade_name"})
    abund_raw = df.iloc[:, 2:].copy()

    # Drop accidental merge-suffix columns like *_y only if they are exact dup names.
    # We'll still handle true duplicates after normalization below.
    drop_cols = [c for c in abund_raw.columns if str(c).endswith("_y")]
    abund_raw = abund_raw.drop(columns=drop_cols, errors="ignore")

    # Make numeric view for library size comparisons
    abund_num = to_numeric_df(abund_raw)

    # Build original->normalized map
    orig_cols = list(abund_raw.columns)
    norm_cols = [norm_id_canonical(c, width=width) for c in orig_cols]
    id_map_df = pd.DataFrame({"original": orig_cols, "normalized": norm_cols})
    _write_tsv(id_map_df, logdir / "Utrecht_id_map_original_to_norm.tsv")

    # Resolve duplicates by keeping ONE best original per normalized key
    collisions_rows = []
    kept_map: Dict[str, str] = {}
    # group originals by normalized key
    by_norm: Dict[str, List[str]] = {}
    for o, n in zip(orig_cols, norm_cols):
        by_norm.setdefault(n, []).append(o)

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
    if not collisions_df.empty:
        _write_tsv(collisions_df, logdir / "Utrecht_id_collisions.tsv")

        # Always write a collisions file, even if empty (Snakemake expects it)
    collisions_path = logdir / "Utrecht_id_collisions.tsv"
    if collisions_df.empty:
        _write_tsv(pd.DataFrame(columns=["normalized_key", "kept_original", "dropped_originals", "n_sources"]),
                   collisions_path)
    else:
        _write_tsv(collisions_df, collisions_path)

    # Build final abundance with KEPT columns only, rename them to normalized keys
    abund_kept = abund_raw[kept_originals].copy()
    # Map kept original -> normalized key (inverse map of kept_map)
    inv_map = {v: k for k, v in kept_map.items()}
    abund_kept.columns = [inv_map[c] for c in abund_kept.columns]

    # Attach clade_name as index and drop all-zero rows
    abund_kept.index = rowdata["clade_name"].values
    abund_kept = abund_kept.loc[abund_kept.sum(axis=1) > 0]

    return abund_kept, id_map_df, collisions_df, kept_map

def extract_healthy_id(colname: str) -> str:
    """Heuristics to extract 6-digit ID from healthy filenames."""
    s = str(colname).strip()
    m = re.search(r"-(\d{6})(?=_(?:rerun_)?metaphlan$)", s, flags=re.IGNORECASE)
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
    return base[:12]

def parse_metaphlan_healthy(tsv_path: Path, logdir: Path, log_prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str,str]]:
    """
    Load healthy MetaPhlAn table, numericize, collapse duplicate columns by KEEP-ONE policy.
    Returns:
      - abundance matrix (rows=clade_name, cols=extracted 6-digit IDs; duplicates resolved by KEEP-ONE)
      - id_map_df: [original, extracted]
      - collisions_df: [extracted_key, kept_original, dropped_originals, n_sources]
      - kept_map: dict extracted_key -> kept_original
    """
    df = pd.read_csv(tsv_path, sep="\t", comment="#", header=0, index_col=0, low_memory=False)
    df = df.drop(columns=["NCBI_tax_id"], errors="ignore")

    orig_cols = list(df.columns)
    extracted = [extract_healthy_id(c) for c in orig_cols]
    id_map_df = pd.DataFrame({"original": orig_cols, "extracted": extracted})
    _write_tsv(id_map_df, logdir / f"{log_prefix}_id_map.tsv")

    # numeric view for library sizes; also keep only rows with some signal later
    df_num = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # resolve duplicates by KEEP-ONE policy
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
    if not collisions_df.empty:
        _write_tsv(collisions_df, logdir / f"{log_prefix}_id_collisions.tsv")

        # Always write a collisions file, even if empty
    collisions_path = logdir / f"{log_prefix}_id_collisions.tsv"
    if collisions_df.empty:
        _write_tsv(pd.DataFrame(columns=["extracted_key", "kept_original", "dropped_originals", "n_sources"]),
                   collisions_path)
    else:
        _write_tsv(collisions_df, collisions_path)

    kept = df_num[kept_originals].copy()
    inv_map = {v: k for k, v in kept_map.items()}
    kept.columns = [inv_map[c] for c in kept.columns]
    kept = kept.loc[kept.sum(axis=1) > 0]
    return kept, id_map_df, collisions_df, kept_map

# ---------------- metadata handling ----------------
def load_metadata(metadata_file: Path) -> pd.DataFrame:
    """
    Loads metadata Excel and enforces columns: STUDY_ID, Oral_sample_ID, Fecal_sample_ID.
    Note: header detection is heuristic; keep your sheet consistent if possible.
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
    meta["Oral_clean"]  = meta["Oral_sample_ID"].apply(lambda x: norm_id_canonical(x, width=5))
    meta["Fecal_clean"] = meta["Fecal_sample_ID"].apply(lambda x: norm_id_canonical(x, width=5))
    return meta

# ---------------- core logic ----------------
def run_preprocess(metadata_file: Path,
                   crohn_file: Path,
                   healthy_oral_in: Path,
                   healthy_fecal_in: Path,
                   outdir: Path,
                   logdir: Path):
    # dirs
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    # 1) read metadata
    meta = load_metadata(metadata_file)

    # 2) Crohn merged → normalized cols with KEEP-ONE policy
    abund_utrecht, u_id_map, u_collisions, u_kept_map = parse_utrecht_merged(crohn_file, logdir, width=5)

    # 3) match oral/fecal against available normalized cols (post-keep)
    available_cols = set(abund_utrecht.columns)
    meta["Oral_col"]  = meta["Oral_clean"].where(meta["Oral_clean"].isin(available_cols))
    meta["Fecal_col"] = meta["Fecal_clean"].where(meta["Fecal_clean"].isin(available_cols))

    unmatched = pd.DataFrame({
        "STUDY_ID": meta["STUDY_ID"],
        "Oral_sample_ID": meta["Oral_sample_ID"],
        "Fecal_sample_ID": meta["Fecal_sample_ID"],
        "Oral_col": meta["Oral_col"],
        "Fecal_col": meta["Fecal_col"],
    })
    unmatched = unmatched[unmatched["Oral_col"].isna() | unmatched["Fecal_col"].isna()]
    if not unmatched.empty:
        _write_tsv(unmatched, logdir / "unmatched_ids.tsv")

    matched = meta.dropna(subset=["Oral_col", "Fecal_col"]).reset_index(drop=True)

    # 4) build Crohn oral/fecal matrices ONLY on matched columns (no all-zero taxa)
    oral_crohn  = abund_utrecht.loc[:, matched["Oral_col"].unique()].copy()
    oral_crohn  = oral_crohn.loc[oral_crohn.sum(axis=1) > 0]
    fecal_crohn = abund_utrecht.loc[:, matched["Fecal_col"].unique()].copy()
    fecal_crohn = fecal_crohn.loc[fecal_crohn.sum(axis=1) > 0]

    # 5) healthy oral/fecal with KEEP-ONE policy and logs
    oral_healthy,  ho_id_map, ho_collisions, ho_kept_map = parse_metaphlan_healthy(healthy_oral_in,  logdir, "Healthy_oral")
    fecal_healthy, hf_id_map, hf_collisions, hf_kept_map = parse_metaphlan_healthy(healthy_fecal_in, logdir, "Healthy_fecal")

    # 6) save outputs
    o1 = outdir / "oral_abund_crohn.csv"
    o2 = outdir / "fecal_abund_crohn.csv"
    o3 = outdir / "oral_abund_healthy.csv"
    o4 = outdir / "fecal_abund_healthy.csv"
    o5 = outdir / "matched_sample_ids.csv"
    for p in (o1, o2, o3, o4, o5):
        ensure_dir_for_file(p)

    oral_crohn.to_csv(o1, index=True)
    fecal_crohn.to_csv(o2, index=True)
    oral_healthy.to_csv(o3, index=True)
    fecal_healthy.to_csv(o4, index=True)

    # matched file: include original kept columns for traceability
    matched_out = matched[["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID", "Oral_col", "Fecal_col"]].copy()
    matched_out = matched_out.rename(columns={"Oral_col": "oral", "Fecal_col": "fecal"})
    matched_out["oral_original_kept"] = matched_out["oral"].map(u_kept_map)
    matched_out["fecal_original_kept"] = matched_out["fecal"].map(u_kept_map)
    matched_out.to_csv(o5, index=False)

    # 7) write a compact JSON summary
    summary = {
        "crohn": {
            "n_columns_original": int(u_id_map.shape[0]),
            "n_columns_kept": int(len(abund_utrecht.columns)),
            "n_collisions": int(u_collisions.shape[0]) if not u_collisions.empty else 0,
        },
        "healthy_oral": {
            "n_columns_original": int(ho_id_map.shape[0]),
            "n_columns_kept": int(len(oral_healthy.columns)),
            "n_collisions": int(ho_collisions.shape[0]) if not ho_collisions.empty else 0,
        },
        "healthy_fecal": {
            "n_columns_original": int(hf_id_map.shape[0]),
            "n_columns_kept": int(len(fecal_healthy.columns)),
            "n_collisions": int(hf_collisions.shape[0]) if not hf_collisions.empty else 0,
        },
        "matching": {
            "n_total_metadata_rows": int(meta.shape[0]),
            "n_matched_pairs": int(matched.shape[0]),
            "n_unmatched_rows": int(unmatched.shape[0]),
        },
        "notes": {
            "duplicate_policy": "KEEP-ONE (max library size), no summation.",
            "width_norm_crohn": 5,
            "healthy_id_extraction": "6-digit heuristic; collisions resolved by KEEP-ONE.",
        }
    }
    ensure_dir_for_file(logdir / "summary.json")
    with open(logdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[preprocessing] Saved:")
    print(f" - {o1}")
    print(f" - {o2}")
    print(f" - {o3}")
    print(f" - {o4}")
    print(f" - {o5}")
    print(f"[preprocessing] Summary written to: {logdir / 'summary.json'}")

# ---------------- Snakemake entrypoint ONLY ----------------
if "snakemake" not in globals():
    raise RuntimeError("This script must be executed via Snakemake (no CLI supported).")

# Resolve inputs/outputs from the Snakemake rule
md  = Path(snakemake.input["metadata"])        # noqa: F821
cr  = Path(snakemake.input["crohn"])           # noqa: F821
hor = Path(snakemake.input["healthy_oral"])    # noqa: F821
hfe = Path(snakemake.input["healthy_fecal"])   # noqa: F821

out_any = Path(snakemake.output["oral_crohn"])  # derive outdir from outputs  # noqa: F821
outdir  = out_any.parent
logdir  = Path("results/preprocessing")

run_preprocess(md, cr, hor, hfe, outdir, logdir)
