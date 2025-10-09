#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral–Fecal overlap & group comparisons (genus + species)

Inputs:
  - --metadata : data/meta/crohn_metadata_extended.csv  (must contain STUDY_ID, Oral_sample_ID, Fecal_sample_ID; optional: Responder, PPI_use)
  - --oral     : oral abundance matrix (filtered/processed), first column = taxonomy OR any name-like column; sample columns like S00xx
  - --fecal    : fecal abundance matrix (filtered/processed), same format as oral
  - --outdir   : results/oral_fecal_overlap
  - --presence-threshold : float, abundance considered "present" if > threshold (default = 1e-12)

Outputs (all into --outdir):
  - overlap_species_per_sample.csv
  - overlap_species_summary.csv
  - overlap_species_per_patient.png
  - overlap_species_top20.png
  - overlap_genus_per_sample.csv
  - overlap_genus_summary.csv
  - overlap_genus_per_patient.png
  - overlap_genus_top20.png
  - overlap_unmatched_samples.csv
  - species_ppi_group_summary.csv / species_resp_group_summary.csv
  - genus_ppi_group_summary.csv   / genus_resp_group_summary.csv
  - species_taxa_counts_by_group.csv / genus_taxa_counts_by_group.csv
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

# --------------------------- Utilities ---------------------------

def norm_sample_id(x: str) -> str:
    """
    Normalize sample column names:
      - strip spaces/tabs
      - uppercase
      - ensure pattern 'Sdddd' (zero-pad 4 digits)
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper().replace("\t", "")
    # remove internal spaces
    s = re.sub(r"\s+", "", s)
    m = re.match(r"^S(\d+)$", s)
    if m:
        d = m.group(1)
        s = "S" + d.zfill(4)  # S94 -> S0094
    return s

def detect_taxcol(df: pd.DataFrame) -> str:
    """
    Find taxonomy column name in a long/filtered file (first col not necessarily named 'clade_name').
    """
    first = df.columns[0]
    patt = r"(?:^|[|])(k__|p__|c__|o__|f__|g__|s__)"
    frac = df[first].astype(str).str.contains(patt, regex=True).mean()
    if frac > 0.5:
        return first
    # try common names
    for cand in ["clade_name", "taxon", "taxonomy", "name", "feature"]:
        if cand in df.columns:
            return cand
    raise RuntimeError("Cannot find a taxonomy column (clade_name/taxon/feature/name).")

def keep_level(df_long: pd.DataFrame, level: str) -> pd.DataFrame:
    """
    From long table -> keep only rows at requested level (genus/species).
    Returns WIDE matrix indexed by taxon with sample columns.
    """
    taxcol = detect_taxcol(df_long)
    mat = df_long.copy()
    # clean sample cols
    valcols = [c for c in mat.columns if c != taxcol]
    # strip weird spaces that sometimes concatenate numbers
    mat[valcols] = mat[valcols].astype(str).replace({" ": "", "\t": ""}, regex=True)
    # convert to numeric safely
    mat[valcols] = mat[valcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def pick_taxon(t):
        parts = str(t).split("|")
        key = "s__" if level == "species" else "g__"
        hit = [p for p in parts if p.startswith(key)]
        return hit[0] if hit else np.nan

    mat["taxon"] = mat[taxcol].map(pick_taxon)
    mat = mat.dropna(subset=["taxon"])
    mat = mat.drop(columns=[taxcol]).set_index("taxon")
    # collapse duplicated taxa if any
    mat = mat.groupby(level=0).sum()
    # normalize column names (S0094 etc.)
    mat.columns = [norm_sample_id(c) for c in mat.columns]
    return mat

def load_abund(path_csv: str, level: str) -> pd.DataFrame:
    raw = pd.read_csv(path_csv, low_memory=False)
    return keep_level(raw, level)  # taxon × samples

def build_mapping(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Use crohn_metadata_extended.csv to map STUDY_ID -> (Oral_col, Fecal_col).
    """
    need = ["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID"]
    for c in need:
        if c not in meta.columns:
            raise RuntimeError(f"Metadata missing column '{c}'.")

    m = meta[need + [c for c in ["Responder", "PPI_use"] if c in meta.columns]].copy()
    m["Oral_col"]  = m["Oral_sample_ID"].map(norm_sample_id)
    m["Fecal_col"] = m["Fecal_sample_ID"].map(norm_sample_id)
    m = m.drop_duplicates(subset=["STUDY_ID"])
    return m

def fisher_summary_from_counts(a, b, c, d):
    """
    Build Fisher exact summary from a 2×2 table:
    [[a(Resp1&Overlap1), b(Resp1&Overlap0)],
     [c(Resp0&Overlap1), d(Resp0&Overlap0)]]
    Returns odds_ratio (with Haldane correction) and p_value.
    """
    table = np.array([[a, b], [c, d]], dtype=float)
    # p-value (exact)
    _, p = fisher_exact(table, alternative="two-sided")
    # odds ratio with Haldane–Anscombe correction (avoid 0)
    table += 0.5
    or_ = (table[0,0] * table[1,1]) / (table[0,1] * table[1,0])
    return float(or_), float(p)

# --------------------------- Core overlap ---------------------------

def compute_per_sample_overlap(mapping, oral_mat, fecal_mat, meta, level, thr):
    """
    For each STUDY_ID with both columns present in matrices:
      - Convert to presence/absence via > thr
      - Overlap = taxa present in both
      - Save count and '|' joined list of taxa names
    """
    rows = []
    unmatched = []

    # presence matrices (align taxa first)
    taxa = sorted(set(oral_mat.index) | set(fecal_mat.index))
    oral_bin  = (oral_mat.reindex(taxa).fillna(0.0)  > thr).astype(bool)
    fecal_bin = (fecal_mat.reindex(taxa).fillna(0.0) > thr).astype(bool)

    for _, r in mapping.iterrows():
        sid, oc, fc = r["STUDY_ID"], r["Oral_col"], r["Fecal_col"]
        # check available columns
        miss_oral  = oc not in oral_bin.columns
        miss_fecal = fc not in fecal_bin.columns
        if miss_oral or miss_fecal:
            unmatched.append({
                "STUDY_ID": sid,
                "Oral_col": oc,
                "Fecal_col": fc,
                "reason": "missing_oral_col" if miss_oral and not miss_fecal
                          else "missing_fecal_col" if miss_fecal and not miss_oral
                          else "missing_both_cols"
            })
            continue

        overlap_mask = (oral_bin[oc] & fecal_bin[fc])
        n = int(overlap_mask.sum())
        taxa_list = "|".join(pd.Index(taxa)[overlap_mask.values])

        row = {
            "STUDY_ID": sid,
            "Oral_col": oc,
            "Fecal_col": fc,
            f"Overlap_n_{level}": n,
            f"Overlap_taxa_{level}": taxa_list
        }
        # attach groups if present
        if "Responder" in meta.columns:
            row["Responder"] = int(pd.to_numeric(
                meta.loc[meta["STUDY_ID"] == sid, "Responder"], errors="coerce"
            ).fillna(0).astype(int).values[0])
        if "PPI_use" in meta.columns:
            row["PPI_use"] = int(pd.to_numeric(
                meta.loc[meta["STUDY_ID"] == sid, "PPI_use"], errors="coerce"
            ).fillna(0).astype(int).values[0])
        rows.append(row)

    per = pd.DataFrame(rows)
    unmatched_df = pd.DataFrame(unmatched)
    return per, unmatched_df

def summarize_overlap(per_df, level):
    """
    Build:
      - per-patient bar chart
      - top-20 taxa across patients
      - summary table with taxa frequency
    """
    out = {}
    cnt_col = f"Overlap_n_{level}"
    taxa_col = f"Overlap_taxa_{level}"

    # ---- per-patient counts (bar)
    if not per_df.empty:
        fig, ax = plt.subplots(figsize=(12,4))
        per_plot = per_df[["STUDY_ID", cnt_col]].sort_values(cnt_col, ascending=False)
        ax.bar(per_plot["STUDY_ID"], per_plot[cnt_col])
        ax.set_title(f"{level.capitalize()} overlap per patient (n={len(per_plot)})")
        ax.set_ylabel("# overlapping taxa")
        plt.xticks(rotation=90)
        plt.tight_layout()
        out["per_patient_fig"] = fig
    else:
        out["per_patient_fig"] = None

    # ---- taxa frequencies across patients
    taxa_counts = {}
    for s in per_df.get(taxa_col, pd.Series([], dtype=str)).fillna(""):
        if not s: 
            continue
        for t in str(s).split("|"):
            if t:
                taxa_counts[t] = taxa_counts.get(t, 0) + 1

    summ = (pd.Series(taxa_counts, name="patients_with_overlap")
              .sort_values(ascending=False)
              .to_frame())
    out["summary_df"] = summ

    # ---- top-20 taxa (barh)
    if not summ.empty:
        fig2, ax2 = plt.subplots(figsize=(8,6))
        top = summ.head(20).sort_values("patients_with_overlap")
        ax2.barh(top.index, top["patients_with_overlap"])
        ax2.set_title(f"Top-20 overlapping {level} across patients")
        ax2.set_xlabel("# patients with oral∩fecal")
        plt.tight_layout()
        out["top20_fig"] = fig2
    else:
        out["top20_fig"] = None

    return out

def group_fishers(per_df, level):
    """
    Build 2×2 tables and Fisher p-values for:
      - Responder (if present)
      - PPI_use  (if present)
    """
    res = {}
    cnt_col = f"Overlap_n_{level}"
    per_df = per_df.copy()
    per_df["has_overlap"] = (per_df[cnt_col] > 0).astype(int)

    def _one(var, fname):
        if var not in per_df.columns:
            return None
        a = int(((per_df[var] == 1) & (per_df["has_overlap"] == 1)).sum())
        b = int(((per_df[var] == 1) & (per_df["has_overlap"] == 0)).sum())
        c = int(((per_df[var] == 0) & (per_df["has_overlap"] == 1)).sum())
        d = int(((per_df[var] == 0) & (per_df["has_overlap"] == 0)).sum())
        or_, p = fisher_summary_from_counts(a, b, c, d)
        df = pd.DataFrame([{
            "label": fname,
            "a_resp1_overlap1": a,
            "b_resp1_overlap0": b,
            "c_resp0_overlap1": c,
            "d_resp0_overlap0": d,
            "odds_ratio": or_,
            "p_value": p
        }])
        return df

    out = []
    x = _one("Responder", "Responder")
    if x is not None: out.append(x)
    y = _one("PPI_use", "PPI_use")
    if y is not None: out.append(y)
    if out:
        return pd.concat(out, ignore_index=True)
    return pd.DataFrame()

def taxa_counts_by_group(per_df, level):
    """
    Count each overlapping taxon separately within groups:
      - Responder=0/1 and PPI_use=0/1 (if columns available)
    """
    taxa_col = f"Overlap_taxa_{level}"
    df = per_df.copy()
    for col in ["Responder", "PPI_use"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    rows = []
    for _, r in df.iterrows():
        taxa = [t for t in str(r.get(taxa_col, "")).split("|") if t]
        if not taxa:
            continue
        for t in taxa:
            rows.append({"taxon": t,
                         "Responder": r.get("Responder", np.nan),
                         "PPI_use": r.get("PPI_use", np.nan)})
    if not rows:
        return pd.DataFrame(columns=["taxon","resp_1","resp_0","ppi_1","ppi_0"])
    tc = pd.DataFrame(rows)
    out = tc.groupby("taxon").agg(
        resp_1=("Responder", lambda s: int((s==1).sum())) if "Responder" in tc.columns else ("Responder","size"),
        resp_0=("Responder", lambda s: int((s==0).sum())) if "Responder" in tc.columns else ("Responder","size"),
        ppi_1 =("PPI_use",  lambda s: int((s==1).sum()))  if "PPI_use"  in tc.columns else ("PPI_use","size"),
        ppi_0 =("PPI_use",  lambda s: int((s==0).sum()))  if "PPI_use"  in tc.columns else ("PPI_use","size"),
    )
    # if columns are missing, fill zeros
    for c in ["resp_1","resp_0","ppi_1","ppi_0"]:
        if c not in out.columns:
            out[c] = 0
    return out.sort_values(["resp_1","ppi_1","resp_0","ppi_0"], ascending=False)

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Oral–Fecal overlap & group comparisons")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--oral", required=True)
    ap.add_argument("--fecal", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--presence-threshold", type=float, default=1e-12)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load
    meta  = pd.read_csv(args.metadata, dtype=str, low_memory=False)
    mapping = build_mapping(meta)

    oral_species  = load_abund(args.oral,  "species")
    fecal_species = load_abund(args.fecal, "species")
    oral_genus    = load_abund(args.oral,  "genus")
    fecal_genus   = load_abund(args.fecal, "genus")

    # species
    per_sp, un_sp = compute_per_sample_overlap(mapping, oral_species, fecal_species, meta, "species", args.presence_threshold)
    per_sp.to_csv(outdir/"overlap_species_per_sample.csv", index=False)
    # genus
    per_ge, un_ge = compute_per_sample_overlap(mapping, oral_genus, fecal_genus, meta, "genus", args.presence_threshold)
    per_ge.to_csv(outdir/"overlap_genus_per_sample.csv", index=False)

    # unmatched (merge & save once)
    unmatched = pd.concat([un_sp.assign(level="species"), un_ge.assign(level="genus")], ignore_index=True)
    if not unmatched.empty:
        unmatched.to_csv(outdir/"overlap_unmatched_samples.csv", index=False)

    # summaries & plots
    for level, per in [("species", per_sp), ("genus", per_ge)]:
        s = summarize_overlap(per, level)
        s["summary_df"].to_csv(outdir/f"overlap_{level}_summary.csv")

        if s["per_patient_fig"] is not None:
            s["per_patient_fig"].savefig(outdir/f"overlap_{level}_per_patient.png", dpi=150)
            plt.close(s["per_patient_fig"])
        if s["top20_fig"] is not None:
            s["top20_fig"].savefig(outdir/f"overlap_{level}_top20.png", dpi=150)
            plt.close(s["top20_fig"])

        # fisher summaries
        fs = group_fishers(per, level)
        if not fs.empty:
            if "Responder" in meta.columns:
                fs[fs["label"]=="Responder"].to_csv(outdir/f"{level}_resp_group_summary.csv", index=False)
            if "PPI_use" in meta.columns:
                fs[fs["label"]=="PPI_use"].to_csv(outdir/f"{level}_ppi_group_summary.csv", index=False)

        # taxa-by-group counts
        tg = taxa_counts_by_group(per, level)
        tg.to_csv(outdir/f"{level}_taxa_counts_by_group.csv")

    print("[OK] overlap completed.")

if __name__ == "__main__":
    main()
