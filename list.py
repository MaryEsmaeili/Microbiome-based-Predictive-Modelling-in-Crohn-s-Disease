#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create shortlists of important taxa from DA outputs:
- genus/species × (Oral CH, Fecal CH, Paired OC↔FC)
Filters by q-value and effect size / log2FC and writes CSVs.

Usage:
  python shortlist_taxa.py --base results/taxa_compare \
      --q 0.05 --l2fc 1.0 --cliffs 0.33 --r 0.30
"""

import pandas as pd
from pathlib import Path
import argparse

def shortlist_group(path_csv, rank, site_label, q_cut, l2fc_cut, cliffs_cut):
    """Filter CH (unpaired) DA table by q and effect size/log2FC."""
    if not path_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(path_csv)
    if df.empty:
        return df

    # Compute helper columns *before* filtering/selection
    df["abs_log2FC"] = df["log2FC"].abs()
    df["abs_eff"] = df["effect_size"].abs()  # Cliff's delta here

    keep = (df["q"] <= q_cut) & ((df["abs_log2FC"] >= l2fc_cut) | (df["abs_eff"] >= cliffs_cut))
    if not keep.any():
        return pd.DataFrame()

    # Sort while helper columns are still present
    df2 = df.loc[keep].copy()
    df2 = df2.sort_values(["q", "abs_log2FC"], ascending=[True, False])

    # Now trim columns for output
    out = df2.loc[:, ["pretty_taxon","taxon","q","log2FC","effect_size","test"]].rename(
        columns={"effect_size": "Cliffs_delta"}
    )

    # Annotate
    out.insert(0, "Rank", rank)
    out.insert(1, "Site", site_label)      # Oral or Fecal
    out.insert(2, "Contrast", "Crohn vs Healthy")
    return out


def shortlist_paired(path_csv, rank, q_cut, r_cut):
    """Filter paired OC↔FC table by q and rank-biserial r."""
    if not path_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(path_csv)
    if df.empty:
        return df

    df["abs_r"] = df["effect_size"].abs()  # rank-biserial r stored in 'effect_size' here
    keep = (df["q"] <= q_cut) & (df["abs_r"] >= r_cut)
    if not keep.any():
        return pd.DataFrame()

    df2 = df.loc[keep].copy().sort_values(["q", "abs_r"], ascending=[True, False])

    out = df2.loc[:, ["pretty_taxon","taxon","q","effect_size","delta_median","n_pairs","test"]].rename(
        columns={"effect_size": "rank_biserial_r"}
    )
    out.insert(0, "Rank", rank)
    out.insert(1, "Contrast", "Paired OC↔FC")
    return out


def run_rank(base: Path, rank: str, q_cut, l2fc_cut, cliffs_cut, r_cut):
    p_oral  = base / rank / f"da_oral_CH_{rank}.csv"
    p_fecal = base / rank / f"da_fecal_CH_{rank}.csv"
    p_pair  = base / rank / f"da_paired_OC_FC_{rank}.csv"

    parts = []
    parts.append(shortlist_group(p_oral,  rank, "Oral",  q_cut, l2fc_cut, cliffs_cut))
    parts.append(shortlist_group(p_fecal, rank, "Fecal", q_cut, l2fc_cut, cliffs_cut))
    parts.append(shortlist_paired(p_pair, rank, q_cut, r_cut))

    parts = [x for x in parts if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    cols = [c for c in ["Rank","Site","Contrast","pretty_taxon","q","log2FC","Cliffs_delta",
                        "rank_biserial_r","delta_median","n_pairs","test","taxon"]
            if c in out.columns]
    return out[cols]

def main():
    ap = argparse.ArgumentParser(description="Shortlist significant taxa from DA outputs.")
    ap.add_argument("--base", default="results/taxa_compare",
                    help="Base directory with /genus and /species (default: results/taxa_compare)")
    ap.add_argument("--q", type=float, default=0.05, help="q-value cutoff (default 0.05)")
    ap.add_argument("--l2fc", type=float, default=1.0, help="|log2FC| cutoff for CH (default 1.0 ≈ 2x)")
    ap.add_argument("--cliffs", type=float, default=0.33, help="|Cliff's delta| cutoff (default 0.33)")
    ap.add_argument("--r", type=float, default=0.30, help="|rank-biserial r| cutoff for paired (default 0.30)")
    args = ap.parse_args()

    base = Path(args.base)
    outdir = base / "shortlists"
    outdir.mkdir(parents=True, exist_ok=True)

    genus_hits   = run_rank(base, "genus",   args.q, args.l2fc, args.cliffs, args.r)
    species_hits = run_rank(base, "species", args.q, args.l2fc, args.cliffs, args.r)

    if not genus_hits.empty:
        genus_hits.to_csv(outdir / "shortlist_genus.csv", index=False)
        print(f"[OK] wrote {outdir/'shortlist_genus.csv'}  ({len(genus_hits)} rows)")
        print(genus_hits.head(15).to_string(index=False))
    else:
        print("[WARN] no genus hits under current thresholds.")

    if not species_hits.empty:
        species_hits.to_csv(outdir / "shortlist_species.csv", index=False)
        print(f"[OK] wrote {outdir/'shortlist_species.csv'}  ({len(species_hits)} rows)")
        print(species_hits.head(15).to_string(index=False))
    else:
        print("[WARN] no species hits under current thresholds.")

if __name__ == "__main__":
    main()
