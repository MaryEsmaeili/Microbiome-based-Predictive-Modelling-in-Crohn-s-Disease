#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")

def norm_id_5(x):
    """Keep first 5 chars (your Utrecht merged table logic)."""
    if pd.isna(x):
        return x
    s = str(x).strip()
    return s[:5]

def parse_utrecht_merged(tsv_path: str):
    """
    Expect columns: '#clade_name', 'NCBI_tax_id', then sample columns.
    We keep clade_name as index and abundance matrix as float.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    # row metadata
    rowdata = df.iloc[:, :2].copy()
    if "#clade_name" in rowdata.columns:
        rowdata = rowdata.rename(columns={"#clade_name": "clade_name"})
    # abundance part
    abund = df.iloc[:, 2:].copy()

    # Drop accidental merge suffixes
    drop_cols = [c for c in abund.columns if c.endswith("_y")]
    abund = abund.drop(columns=drop_cols, errors="ignore")

    # Normalize sample column ids to first 5 chars (to match metadata)
    abund.columns = [norm_id_5(c) for c in abund.columns]

    abund = to_numeric_df(abund)
    rowdata = rowdata.reset_index(drop=True)
    clade = rowdata["clade_name"]
    return clade, abund

def parse_metaphlan_single(tsv_path: str, id_slice: int = 6):
    """
    Generic MetaPhlAn table (rows=taxa, cols=samples). Often has NCBI_tax_id.
    We:
      - drop NCBI_tax_id if present
      - normalize sample ids (last token after '-' and strip '_metaphlan'), then slice first `id_slice` chars
      - coerce to numeric and keep only rows with sum>0
    """
    df = pd.read_csv(tsv_path, sep="\t", comment="#", header=0, index_col=0, low_memory=False)
    df = df.drop(columns=["NCBI_tax_id"], errors="ignore")

    def clean_col(c):
        base = str(c).split("-")[-1].replace("_metaphlan", "")
        return base[:id_slice]

    df.columns = [clean_col(c) for c in df.columns]
    df = to_numeric_df(df)
    df = df.loc[df.sum(axis=1) > 0]
    return df

def match_sample_col(sample_id, columns):
    if pd.isnull(sample_id) or str(sample_id).strip() == "":
        return np.nan
    sid = norm_id_5(sample_id)
    if sid in columns:
        return sid
    print(f"[WARN] No match for sample {sample_id} -> {sid}")
    return np.nan

# ------------------------------------------------------------
# Snakemake I/O
# ------------------------------------------------------------
metadata_file    = snakemake.input.metadata
crohn_file       = snakemake.input.crohn
healthy_oral_in  = snakemake.input.healthy_oral
healthy_fecal_in = snakemake.input.healthy_fecal

oral_abund_crohn_out   = snakemake.output.oral_crohn
fecal_abund_crohn_out  = snakemake.output.fecal_crohn
oral_abund_healthy_out = snakemake.output.oral_healthy
fecal_abund_healthy_out= snakemake.output.fecal_healthy
matched_ids_out        = snakemake.output.matched

for p in [oral_abund_crohn_out, fecal_abund_crohn_out, oral_abund_healthy_out,
          fecal_abund_healthy_out, matched_ids_out]:
    ensure_dir(p)

# ------------------------------------------------------------
# 1) Load metadata
#    Your sheet is headerless; first column holds keys -> transpose.
# ------------------------------------------------------------
meta = pd.read_excel(metadata_file, header=None, index_col=0)
meta = meta.T.reset_index(drop=True)
# Keep only needed cols (robust to stray whitespace / dtype)
meta.columns = meta.columns.astype(str).str.strip()
required = ["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID"]
missing = [c for c in required if c not in meta.columns]
if missing:
    raise ValueError(f"Missing required metadata columns: {missing}")
meta = meta[required].copy()

# Clean for matching against Utrecht-abundance (use first 5 chars)
meta["Oral_clean"]  = meta["Oral_sample_ID"].apply(norm_id_5)
meta["Fecal_clean"] = meta["Fecal_sample_ID"].apply(norm_id_5)

# ------------------------------------------------------------
# 2) Load Crohn merged (Utrecht) and split oral/fecal by metadata matches
# ------------------------------------------------------------
clade_names, abund_utrecht = parse_utrecht_merged(crohn_file)

# Find matches
meta["Oral_col"]  = meta["Oral_clean"].apply(lambda x: match_sample_col(x, abund_utrecht.columns))
meta["Fecal_col"] = meta["Fecal_clean"].apply(lambda x: match_sample_col(x, abund_utrecht.columns))
matched = meta.dropna(subset=["Oral_col", "Fecal_col"]).reset_index(drop=True)

# Build Crohn oral/fecal tables
oral_crohn = abund_utrecht[matched["Oral_col"].unique()].copy()
oral_crohn.index = clade_names
oral_crohn = oral_crohn.loc[oral_crohn.sum(axis=1) > 0]

fecal_crohn = abund_utrecht[matched["Fecal_col"].unique()].copy()
fecal_crohn.index = clade_names
fecal_crohn = fecal_crohn.loc[fecal_crohn.sum(axis=1) > 0]

# ------------------------------------------------------------
# 3) Healthy oral + NEW healthy fecal
# ------------------------------------------------------------
oral_healthy   = parse_metaphlan_single(healthy_oral_in,  id_slice=6)
fecal_healthy  = parse_metaphlan_single(healthy_fecal_in, id_slice=6)

# ------------------------------------------------------------
# 4) Save
# ------------------------------------------------------------
oral_crohn.to_csv(oral_abund_crohn_out, index=True)
fecal_crohn.to_csv(fecal_abund_crohn_out, index=True)
oral_healthy.to_csv(oral_abund_healthy_out, index=True)
fecal_healthy.to_csv(fecal_abund_healthy_out, index=True)

matched[["STUDY_ID", "Oral_col", "Fecal_col"]].to_csv(matched_ids_out, index=False)

print(
    "Saved:\n"
    f" - {oral_abund_crohn_out}\n"
    f" - {fecal_abund_crohn_out}\n"
    f" - {oral_abund_healthy_out}\n"
    f" - {fecal_abund_healthy_out}\n"
    f" - {matched_ids_out}"
)
