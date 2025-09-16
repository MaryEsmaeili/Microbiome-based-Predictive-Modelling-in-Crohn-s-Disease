#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocesses metadata + abundance tables into site-specific matrices
with robust ID normalization, duplicate detection/aggregation,
and explicit logging for traceability.

Output:
- Crohn oral/fecal abundance (columns = normalized sample IDs)
- Healthy oral/fecal abundance (columns = normalized sample IDs)
- matched_sample_ids.csv (STUDY_ID, Oral_col, Fecal_col)
- logs: id_map_original_to_norm.tsv, id_collisions.tsv, unmatched_ids.tsv
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")

def norm_id_canonical(x: str, width: int = 5) -> str:
    """
    Canonical sample ID normalizer used everywhere:
    - strip whitespace
    - remove trailing '.0'
    - strip leading zeros
    - truncate/pad to `width` characters if desired (here: 5 by project convention)
    NOTE: If result shorter than width, do NOT pad with zeros (keeps human readability).
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = s.lstrip("0") or "0"  # keep at least one char
    return s[:width]

def normalize_columns_with_report(df: pd.DataFrame, width: int = 5, tag: str = "Utrecht") -> pd.DataFrame:
    """
    Normalize sample column labels using the canonical function.
    - Detect collisions (two original cols map to same normalized label)
    - Aggregate duplicates by sum (compositional, pre-normalization)
    - Write reports for mapping and collisions
    """
    orig_cols = list(df.columns)
    norm_cols = [norm_id_canonical(c, width=width) for c in orig_cols]

    # Save mapping
    map_df = pd.DataFrame({"original": orig_cols, "normalized": norm_cols})
    ensure_dir("results/preprocessing/_logs/_tmp.txt")
    map_df.to_csv(f"results/preprocessing/{tag}_id_map_original_to_norm.tsv", sep="\t", index=False)

    # Detect collisions
    counts = Counter(norm_cols)
    collisions = [c for c, n in counts.items() if n > 1]
    if collisions:
        # report all originals mapping to each duplicate key
        rows = []
        for key in collisions:
            idxs = [i for i, v in enumerate(norm_cols) if v == key]
            rows.append({
                "normalized_key": key,
                "original_columns": ",".join(orig_cols[i] for i in idxs),
                "n_sources": len(idxs)
            })
        coll_df = pd.DataFrame(rows)
        coll_df.to_csv(f"results/preprocessing/{tag}_id_collisions.tsv", sep="\t", index=False)

    # Aggregate columns by normalized key (sum across duplicates)
    df_agg = df.copy()
    df_agg.columns = norm_cols
    df_agg = df_agg.groupby(level=0, axis=1).sum()
    return df_agg

def parse_utrecht_merged(tsv_path: str, width: int = 5):
    """
    Expect columns: '#clade_name', 'NCBI_tax_id', then sample columns.
    Keep clade_name as index and abundance matrix as float.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    rowdata = df.iloc[:, :2].copy()
    if "#clade_name" in rowdata.columns:
        rowdata = rowdata.rename(columns={"#clade_name": "clade_name"})
    abund = df.iloc[:, 2:].copy()

    # Drop accidental merge suffixes
    drop_cols = [c for c in abund.columns if c.endswith("_y")]
    abund = abund.drop(columns=drop_cols, errors="ignore")

    # Normalize columns consistently
    abund = to_numeric_df(abund)
    abund = normalize_columns_with_report(abund, width=width, tag="Utrecht")

    # Set index to clade names once, then filter later
    abund.index = rowdata["clade_name"].values
    return abund

def parse_metaphlan_single(tsv_path: str, width: int = 5):
    """
    Generic MetaPhlAn table (rows=taxa, cols=samples).
    - Drop NCBI_tax_id if present
    - Normalize sample ids with the same canonical function
    - Coerce to numeric; drop all-zero taxa
    """
    df = pd.read_csv(tsv_path, sep="\t", comment="#", header=0, index_col=0, low_memory=False)
    df = df.drop(columns=["NCBI_tax_id"], errors="ignore")
    df = to_numeric_df(df)

    # Normalize columns canonically (strip leading zeros, 5 chars)
    df = normalize_columns_with_report(df, width=width, tag=os.path.splitext(os.path.basename(tsv_path))[0])
    df = df.loc[df.sum(axis=1) > 0]
    return df

# ------------------------------------------------------------
# Snakemake I/O
# ------------------------------------------------------------
metadata_file    = snakemake.input.metadata
crohn_file       = snakemake.input.crohn
healthy_oral_in  = snakemake.input.healthy_oral
healthy_fecal_in = snakemake.input.healthy_fecal

oral_abund_crohn   = snakemake.output.oral_crohn
fecal_abund_crohn  = snakemake.output.fecal_crohn
oral_abund_healthy = snakemake.output.healthy_oral
fecal_abund_healthy= snakemake.output.healthy_fecal
matched_ids        = snakemake.output.matched

for p in [oral_abund_crohn, fecal_abund_crohn, oral_abund_healthy,
          fecal_abund_healthy, matched_ids]:
    ensure_dir(p)

# ------------------------------------------------------------
# 1) Load metadata (robust to headerless vs. headered Excel)
# ------------------------------------------------------------
meta_raw = pd.read_excel(metadata_file, header=None)
# Heuristic: if first row looks like column names, use it as header
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

# Normalize sample IDs canonically (same rule everywhere)
meta["Oral_clean"]  = meta["Oral_sample_ID"].apply(lambda x: norm_id_canonical(x, width=5))
meta["Fecal_clean"] = meta["Fecal_sample_ID"].apply(lambda x: norm_id_canonical(x, width=5))

# ------------------------------------------------------------
# 2) Load Crohn merged (Utrecht) and split oral/fecal by metadata matches
# ------------------------------------------------------------
abund_utrecht = parse_utrecht_merged(crohn_file, width=5)

# Match against normalized columns
meta["Oral_col"]  = meta["Oral_clean"].where(meta["Oral_clean"].isin(abund_utrecht.columns))
meta["Fecal_col"] = meta["Fecal_clean"].where(meta["Fecal_clean"].isin(abund_utrecht.columns))

# Save unmatched IDs for traceability
unmatched = pd.DataFrame({
    "Oral_unmatched": meta.loc[meta["Oral_col"].isna(), "Oral_sample_ID"],
    "Fecal_unmatched": meta.loc[meta["Fecal_col"].isna(), "Fecal_sample_ID"]
})
unmatched.to_csv("results/preprocessing/unmatched_ids.tsv", sep="\t", index=False)

matched = meta.dropna(subset=["Oral_col", "Fecal_col"]).reset_index(drop=True)

# Build Crohn oral/fecal tables
oral_crohn = abund_utrecht.loc[:, matched["Oral_col"].unique()].copy()
oral_crohn = oral_crohn.loc[oral_crohn.sum(axis=1) > 0]

fecal_crohn = abund_utrecht.loc[:, matched["Fecal_col"].unique()].copy()
fecal_crohn = fecal_crohn.loc[fecal_crohn.sum(axis=1) > 0]

# ------------------------------------------------------------
# 3) Healthy oral + Healthy fecal (same normalization)
# ------------------------------------------------------------
oral_healthy  = parse_metaphlan_single(healthy_oral_in,  width=5)
fecal_healthy = parse_metaphlan_single(healthy_fecal_in, width=5)

# ------------------------------------------------------------
# 4) Save
# ------------------------------------------------------------
oral_crohn.to_csv(oral_abund_crohn, index=True)
fecal_crohn.to_csv(fecal_abund_crohn, index=True)
oral_healthy.to_csv(oral_abund_healthy, index=True)
fecal_healthy.to_csv(fecal_abund_healthy, index=True)

matched[["STUDY_ID", "Oral_col", "Fecal_col"]].to_csv(matched_ids, index=False)

print(
    "Saved:\n"
    f" - {oral_abund_crohn}\n"
    f" - {fecal_abund_crohn}\n"
    f" - {oral_abund_healthy}\n"
    f" - {fecal_abund_healthy}\n"
    f" - {matched_ids}\n"
    "Logs in results/preprocessing/: *_id_map_original_to_norm.tsv, *_id_collisions.tsv, unmatched_ids.tsv"
)
