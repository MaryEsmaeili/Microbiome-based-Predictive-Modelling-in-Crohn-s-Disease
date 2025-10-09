#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Δ (Fecal − Oral) features on CLR for matched Crohn pairs.

Inputs
------
--pct-all : path to <results/taxa_compare/{rank}/pct_all.csv> (taxa x samples, %)
--pairs   : CSV with Oral/Fecal sample IDs (any of: Oral_ID/Oral/OC, Fecal_ID/Fecal/FC)
--rank    : "genus" or "species"
--out     : output CSV (wide), rows=pairs, cols=taxa (values are CLR(FC) - CLR(OC))

Notes
-----
- We do an in-script CLR transform with a small pseudocount.
- Pairs without both columns present in pct_all are skipped.
- First column in the output is 'pair_id' (format: OCID__FCID).
"""

import os, argparse, warnings
import numpy as np
import pandas as pd

def clr_from_pct(df_pct, pseudocount=1e-6):
    X = (df_pct.astype(float) / 100.0) + pseudocount     # taxa x samples
    logX = np.log(X)
    gm = logX.mean(axis=0)                               # per-sample mean(log)
    clr = logX.sub(gm, axis=1)
    return clr

def _pick_col(pairs_df, *cands):
    low = {c.lower(): c for c in pairs_df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct-all", required=True)
    ap.add_argument("--pairs",    required=True)
    ap.add_argument("--rank",     required=True, choices=["genus","species"])
    ap.add_argument("--out",      required=True)
    return ap.parse_args()

def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    # Load matrices
    pct = pd.read_csv(a.pct_all, index_col=0)
    pct.columns = pct.columns.astype(str)
    clr = clr_from_pct(pct, pseudocount=1e-6)            # taxa x samples

    # Load pairs
    pairs_df = pd.read_csv(a.pairs)
    oc_col = _pick_col(pairs_df, "Oral_ID","Oral","OC","oral_id","oral")
    fc_col = _pick_col(pairs_df, "Fecal_ID","Fecal","FC","fecal_id","fecal")
    if oc_col is None or fc_col is None:
        raise ValueError("Pairs file must have Oral and Fecal ID columns (e.g., Oral_ID/Fecal_ID).")

    pairs = pairs_df[[oc_col, fc_col]].dropna().astype(str).values.tolist()
    # Keep only those present in clr matrix
    colnames = set(clr.columns.astype(str))
    pairs = [(o,f) for (o,f) in pairs if o in colnames and f in colnames]

    if len(pairs) == 0:
        warnings.warn("No usable pairs overlap with pct_all columns; writing empty output.")
        pd.DataFrame(columns=["pair_id"]).to_csv(a.out, index=False)
        return

    # Build wide Δ table: rows = pair, cols = taxa
    deltas = []
    pair_ids = []
    for (o,f) in pairs:
        d = (clr[f] - clr[o]).rename(None)               # pandas Series (taxa)
        deltas.append(d.values)
        pair_ids.append(f"{o}__{f}")

    mat = np.vstack(deltas)                              # n_pairs x n_taxa
    out = pd.DataFrame(mat, columns=clr.index.tolist())
    out.insert(0, "pair_id", pair_ids)
    out.to_csv(a.out, index=False)
    print(f"[INFO] Δ features written → {a.out}  (pairs={len(pairs)}, taxa={out.shape[1]-1})")

if __name__ == "__main__":
    main()
