#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build paired Δ features for Crohn: Δ = Fecal − Oral per matched pair on CLR matrix.

Inputs:
  --wide-clr  results/taxa_compare/{rank}/ml_{rank}_wide_clr.csv
  --pairs     data/processed/matched_sample_ids.csv (two columns: oral, fecal; names auto-detected)
Output:
  --out       results/ml/features/deltas_{rank}.csv
"""

import argparse, pandas as pd, numpy as np, re, warnings

def _guess_pair_columns(df):
    cols = [str(c) for c in df.columns]
    low = [c.lower() for c in cols]
    oral_keys  = ("oral","oc","mouth")
    fecal_keys = ("fecal","fc","stool")
    oi = next((i for i,c in enumerate(low) if any(k in c for k in oral_keys)), 0)
    fi = next((i for i,c in enumerate(low) if any(k in c for k in fecal_keys)), 1 if len(cols)>1 else 0)
    return cols[oi], cols[fi]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide-clr", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--rank", required=True, choices=["genus","species"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    W = pd.read_csv(args.wide_clr)     # samples × taxa + Group + Site
    tax_cols = [c for c in W.columns if c not in ["Sample","Group","Site"]]
    if "Sample" not in W.columns:
        raise RuntimeError("wide_clr needs a 'Sample' column.")
    W = W.set_index("Sample")

    P = pd.read_csv(args.pairs)
    oc_col, fc_col = _guess_pair_columns(P)
    P = P[[oc_col, fc_col]].dropna()
    P.columns = ["oral","fecal"]

    rows = []
    for _, r in P.iterrows():
        o, f = str(r["oral"]), str(r["fecal"])
        if o in W.index and f in W.index:
            if W.loc[o, "Site"] != "Oral" or W.loc[f, "Site"] != "Fecal":
                # skip mismatched site
                continue
            # Crohn only
            if W.loc[o, "Group"] != "Crohn" or W.loc[f, "Group"] != "Crohn":
                continue
            delta = W.loc[f, tax_cols].astype(float).values - W.loc[o, tax_cols].astype(float).values
            row = {"Pair": f"{o}__{f}", "Group":"Crohn", "Site":"Delta(F−O)"}
            row.update({tax: delta[i] for i, tax in enumerate(tax_cols)})
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
