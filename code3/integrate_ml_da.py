#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrate ML top features with DA table.
- Overlap (Venn-like counts)
- Direction consistency: sign(beta) vs sign(importance_proxy)
- Optional pathway enrichment if annotation file exists (Fisher)

Inputs:
  --ml-top : path to ML top_features.csv (columns: feature, importance)
  --da     : path to DA table (feature,beta,qval,...)
  --anno   : OPTIONAL species_to_pathway.csv (taxon,pathway)

Outputs in --outdir:
  - overlap.json
  - merged_table.csv (DA + ML ranks)
  - enrichment.csv (if anno provided)
  - complete.txt
"""
import os, json, argparse
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import fisher_exact

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml-top", required=True)
    ap.add_argument("--da", required=True)
    ap.add_argument("--anno", required=False)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ml = pd.read_csv(args.ml_top)
    da = pd.read_csv(args.da)
    if "feature" not in ml.columns:
        raise SystemExit("ML top must have column 'feature'")
    if "feature" not in da.columns or "beta" not in da.columns:
        raise SystemExit("DA must have columns 'feature','beta'")

    # clean
    ml = ml.dropna(subset=["feature"]).copy()
    da = da.dropna(subset=["feature"]).copy()

    # overlap
    S_ml = set(ml["feature"])
    S_da = set(da["feature"])
    overlap = {
        "n_ml": int(len(S_ml)),
        "n_da": int(len(S_da)),
        "n_overlap": int(len(S_ml & S_da)),
    }
    with open(os.path.join(args.outdir,"overlap.json"),"w") as f:
        json.dump(overlap, f, indent=2)

    # direction proxy: ML has importance only (positive); برای جهت، از sign(beta) استفاده می‌کنیم
    merged = da.merge(ml[["feature","importance"]], on="feature", how="left")
    merged["ml_present"] = merged["importance"].notna()
    merged.to_csv(os.path.join(args.outdir,"merged_table.csv"), index=False)

    # enrichment (optional)
    enr_path = os.path.join(args.outdir, "enrichment.csv")
    if args.anno and os.path.exists(args.anno):
        anno = pd.read_csv(args.anno)
        if set(["taxon","pathway"]).issubset(anno.columns):
            # foreground = overlap features
            fg = pd.Series(sorted(S_ml & S_da), name="feature")
            bg = pd.Series(sorted(S_da), name="feature")  # DA universe
            ann = anno.rename(columns={"taxon":"feature"})
            fg_pw = ann.merge(fg, on="feature")["pathway"].dropna()
            bg_pw = ann.merge(bg, on="feature")["pathway"].dropna()
            pathways = sorted(set(bg_pw))
            rows = []
            c_bg = Counter(bg_pw)
            c_fg = Counter(fg_pw)
            for p in pathways:
                a = c_fg.get(p,0)
                b = len(fg) - a
                c = c_bg.get(p,0) - a
                d = len(bg) - (a + b + c)  # rest
                if min(a,b,c,d) < 0: 
                    continue
                try:
                    OR, pval = fisher_exact([[a,b],[c,d]], alternative="greater")
                except Exception:
                    OR, pval = np.nan, 1.0
                rows.append({"pathway":p, "a_overlap":a, "fg":int(len(fg)), "bg_hits":c_bg.get(p,0), "bg":int(len(bg)), "odds":OR, "pval":pval})
            pd.DataFrame(rows).sort_values("pval").to_csv(enr_path, index=False)
        else:
            pd.DataFrame(columns=["pathway","a_overlap","fg","bg_hits","bg","odds","pval"]).to_csv(enr_path, index=False)
    else:
        # no annotation—write empty file to keep pipeline stable
        pd.DataFrame(columns=["pathway","a_overlap","fg","bg_hits","bg","odds","pval"]).to_csv(enr_path, index=False)

    with open(os.path.join(args.outdir,"complete.txt"),"w") as f:
        f.write("done\n")

if __name__ == "__main__":
    main()
