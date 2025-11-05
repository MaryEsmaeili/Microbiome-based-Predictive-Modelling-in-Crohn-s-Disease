#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing for microbiome pipeline:
- Read metadata Excel + merged MetaPhlAn tables (Crohn + Healthy oral/fecal + HMP oral).
- Normalize/clean sample IDs while preserving leading zeros (no numeric casting).
- Utrecht (Crohn): extract Sdddd from column names, resolve duplicate sample columns via KEEP-ONE (max library size).
- Healthy: extract 6-digit IDs from headers, resolve duplicates via KEEP-ONE.
- HMP oral: keep only SRR+6digits as sample id (strip any HMP_ prefix and _metaphlan suffix), ensure taxa x samples.
- Keep ONLY species-level rows (contain '|s__'), drop unclassified, renormalize columns to sum=1 when needed.
- Save split abundance tables + matched ID mapping + summary + mapping/collision logs.
- Write all CSVs with index_label='clade_name'.

Run only via Snakemake.
"""
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re
import json

import numpy as np
import pandas as pd

# -------------------------- small utils --------------------------
def ensure_dir_for_file(path: Path) -> None:
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")

def _write_csv(df: pd.DataFrame, path: Path, with_index: bool = True) -> None:
    ensure_dir_for_file(path)
    if with_index:
        df.to_csv(path, index=True, index_label="clade_name")
    else:
        df.to_csv(path, index=False)

def _write_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir_for_file(path)
    df.to_csv(path, sep="\t", index=False)

# -------------------------- ID utilities --------------------------
def norm_metadata_id(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"\.(?:rep|[0-9]+)$", "", s, flags=re.IGNORECASE)
    return s.upper()

def norm_utrecht_colname(col: str) -> Optional[str]:
    s = str(col).strip()
    m = re.search(r"(S\d{4})", s, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def extract_healthy_id(colname: str) -> str:
    s = str(colname).strip()
    m = re.search(r"-(\d{6})(?=_(?:rerun_)?metaphlan\b)", s, flags=re.IGNORECASE)
    if m: return m.group(1)
    m2 = re.search(r"(\d{6})(?!.*\d)", s)
    if m2: return m2.group(1)
    base = s.split("-")[-1]
    base = re.sub(r"(?i)_rerun", "", base)
    base = re.sub(r"(?i)_metaphlan", "", base)
    m3 = re.search(r"(\d{6})", base)
    return m3.group(1) if m3 else base[-12:]

def standardize_hmp_sample_id(x: str) -> Optional[str]:
    """
    Keep strictly 'SRR' + 6 digits. Strip any 'HMP_' prefix or '_metaphlan' suffix.
    Return None if no SRR###### found (column dropped).
    """
    s = str(x).strip()
    s = re.sub(r"(?i)^HMP_", "", s)
    s = re.sub(r"(?i)_metaphlan.*$", "", s)
    m = re.search(r"(SRR\d{6})", s, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

# -------------------------- species-level & normalization --------------------------
def keep_species_only(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index.astype(str)
    mask_species = idx.str.contains(r"\|s__", regex=True, na=False)
    out = df.loc[mask_species].copy()
    bad = out.index.astype(str).str.contains(r"s__unclassified|unclassified", case=False, na=False)
    return out.loc[~bad]

def renorm_cols_to_one(df: pd.DataFrame) -> pd.DataFrame:
    colsum = df.sum(axis=0)
    needs = ~colsum.round(6).isin([0.0, 1.0])
    if not needs.any():
        return df
    colsum_safe = colsum.replace(0, np.nan)
    return df.div(colsum_safe, axis=1).fillna(0.0)

# -------------------------- duplicate resolution --------------------------
def _choose_one_among_duplicates(df_num: pd.DataFrame, originals: List[str]) -> str:
    if len(originals) == 1:
        return originals[0]
    sums = {c: float(pd.to_numeric(df_num[c], errors="coerce").fillna(0).sum()) for c in originals}
    return max(originals, key=lambda c: (sums.get(c, 0.0), -originals.index(c)))

# -------------------------- parsers --------------------------
def parse_utrecht_merged(tsv_path: Path,
                         map_out: Path,
                         collisions_out: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
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
        if n is None: continue
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
    inv_map = {v: k for k, v in kept_map.items()}
    abund_kept.columns = [inv_map[c] for c in abund_kept.columns]

    abund_kept.index = rowdata.iloc[:, 0].values  # clade_name
    abund_kept = abund_kept.loc[abund_kept.sum(axis=1) > 0]
    abund_kept = keep_species_only(abund_kept)
    abund_kept = renorm_cols_to_one(abund_kept)
    return abund_kept, kept_map

def parse_metaphlan_healthy(tsv_path: Path,
                            map_out: Path,
                            collisions_out: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
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
    inv_map = {v: k for k, v in kept_map.items()}
    kept.columns = [inv_map[c] for c in kept.columns]
    kept = kept.loc[kept.sum(axis=1) > 0]

    kept = keep_species_only(kept)
    kept = renorm_cols_to_one(kept)
    return kept, kept_map

def parse_hmp_oral(csv_path: Path) -> pd.DataFrame:
    """
    HMP oral CSV has first column header 'sample' then many clade columns.
    We need taxa x samples with species rows only and columns named SRR######.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    # Expect a 'sample' column; if not, try to guess
    if "sample" not in df.columns:
        # try: first column is sample id
        df.rename(columns={df.columns[0]: "sample"}, inplace=True)

    # set sample index and transpose to taxa x samples
    df = df.set_index("sample")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df_t = df.T  # taxa x samples

    # standardize HMP sample ids to SRR###### and drop others
    new_cols = [standardize_hmp_sample_id(c) for c in df_t.columns]
    keep_mask = [c is not None for c in new_cols]
    df_t = df_t.loc[:, keep_mask]
    df_t.columns = [c for c in new_cols if c is not None]

    # species-only + renorm
    df_t.index = df_t.index.astype(str)
    df_t = keep_species_only(df_t)
    df_t = renorm_cols_to_one(df_t)

    # drop empty rows
    df_t = df_t.loc[df_t.sum(axis=1) > 0]
    return df_t

# -------------------------- metadata loader --------------------------
def load_metadata(metadata_file: Path) -> pd.DataFrame:
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
                   hmp_oral_in: Path,
                   out_oral_crohn: Path,
                   out_fecal_crohn: Path,
                   out_oral_healthy: Path,
                   out_fecal_healthy: Path,
                   hmp_oral_out: Path,
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

    # 6) HMP oral (separate table; merge در مرحلهٔ filtering انجام می‌شود)
    hmp_oral_df = parse_hmp_oral(hmp_oral_in)

    # 7) save core CSVs (با هدر clade_name)
    _write_csv(oral_crohn,    out_oral_crohn,    with_index=True)
    _write_csv(fecal_crohn,   out_fecal_crohn,   with_index=True)
    _write_csv(oral_healthy,  out_oral_healthy,  with_index=True)
    _write_csv(fecal_healthy, out_fecal_healthy, with_index=True)
    _write_csv(hmp_oral_df,   hmp_oral_out,      with_index=True)

    # matched pairs (to data/processed)
    matched_save = matched[["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID", "Oral_col", "Fecal_col"]].copy()
    matched_save = matched_save.rename(columns={"Oral_col": "oral", "Fecal_col": "fecal"})
    matched_save["oral_original_kept"]  = matched_save["oral"].map(lambda k: u_kept_map.get(k, "NA"))
    matched_save["fecal_original_kept"] = matched_save["fecal"].map(lambda k: u_kept_map.get(k, "NA"))
    _write_csv(matched_save, matched_out, with_index=False)

    # 8) QC (optional)
    qc_rows = []
    if not oral_crohn.empty:    qc_rows.append(qc_table(oral_crohn,    "crohn_oral_matched"))
    if not fecal_crohn.empty:   qc_rows.append(qc_table(fecal_crohn,   "crohn_fecal_matched"))
    if not oral_healthy.empty:  qc_rows.append(qc_table(oral_healthy,  "healthy_oral"))
    if not fecal_healthy.empty: qc_rows.append(qc_table(fecal_healthy, "healthy_fecal"))
    if qc_rows:
        qc_all = pd.concat(qc_rows, ignore_index=True)
        _write_tsv(qc_all, extra_dir / "sample_qc.tsv")

    # 9) summary
    summary = {
        "crohn": {
            "n_cols_kept_total": int(len(abund_utrecht.columns)),
            "n_matched_rows": int(matched.shape[0]),
            "n_unmatched_rows": int(unmatched.shape[0]),
        },
        "healthy_oral":  {"n_cols_kept": int(len(oral_healthy.columns))},
        "healthy_fecal": {"n_cols_kept": int(len(fecal_healthy.columns))},
        "hmp_oral":      {"n_cols_kept": int(len(hmp_oral_df.columns))},
        "notes": {
            "duplicate_policy": "KEEP-ONE (max library size), no summation.",
            "feature_level": "species only (|s__), unclassified removed",
            "col_renorm": "each column ~ sum to 1 after cleaning",
            "hmp_ids": "columns standardized to SRR######",
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
    print(f" - {hmp_oral_out}")
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
    hmp = Path(snakemake.input["hmp_oral"])        # noqa: F821

    out_oral_c  = Path(snakemake.output["oral_crohn"])     # noqa: F821
    out_fecal_c = Path(snakemake.output["fecal_crohn"])    # noqa: F821
    out_oral_h  = Path(snakemake.output["healthy_oral"])   # noqa: F821
    out_fecal_h = Path(snakemake.output["healthy_fecal"])  # noqa: F821
    hmp_out     = Path(snakemake.output["hmp_oral_out"])   # noqa: F821
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
        hmp_oral_in=hmp,
        out_oral_crohn=out_oral_c,
        out_fecal_crohn=out_fecal_c,
        out_oral_healthy=out_oral_h,
        out_fecal_healthy=out_fecal_h,
        hmp_oral_out=hmp_out,
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