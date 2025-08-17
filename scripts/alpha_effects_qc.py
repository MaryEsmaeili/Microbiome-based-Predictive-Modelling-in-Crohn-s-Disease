#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

METRICS = ["Richness", "Shannon", "Simpson"]

def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0: return np.nan
    # O(n*m) is fine for these sizes
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (n1 * n2)

def rank_biserial_from_pairs(a, b):
    """For paired data (Wilcoxon): r_rb = (n_pos - n_neg) / (n_pos + n_neg)"""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    if d.size == 0: return np.nan
    n_pos = (d > 0).sum()
    n_neg = (d < 0).sum()
    return (n_pos - n_neg) / (n_pos + n_neg)

def iqr_outliers(df, group_col, value_col):
    q1 = df[value_col].quantile(0.25)
    q3 = df[value_col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    flags = df[(df[value_col] < low) | (df[value_col] > high)].copy()
    if flags.empty: 
        flags = pd.DataFrame(columns=df.columns)
    flags["Q1"] = q1; flags["Q3"] = q3; flags["IQR"] = iqr
    flags["fence_low"] = low; flags["fence_high"] = high
    flags["metric"] = group_col  # we'll rename later
    return flags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha-oral", required=True)
    ap.add_argument("--alpha-fecal", required=True)
    ap.add_argument("--alpha-healthy", required=True)
    ap.add_argument("--matched", required=True)
    ap.add_argument("--paired-stats", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load tables
    oral    = pd.read_csv(args.alpha_oral, index_col=0)
    fecal   = pd.read_csv(args.alpha_fecal, index_col=0)
    healthy = pd.read_csv(args.alpha_healthy, index_col=0)
    matched = pd.read_csv(args.matched)
    paired_stats = pd.read_csv(args.paired_stats)

    # ---------- Effect sizes ----------
    # 1) Crohn-Oral vs Healthy-Oral (independent)
    fx_rows = []
    for m in METRICS:
        # p-value (recompute here for completeness)
        u, p = mannwhitneyu(oral[m], healthy[m], alternative="two-sided")
        cd = cliffs_delta(oral[m], healthy[m])
        fx_rows.append({
            "contrast": "Crohn-Oral vs Healthy-Oral",
            "metric": m,
            "p_value": p,
            "effect_size": cd,
            "effect_name": "Cliffs_delta"
        })

    # 2) Paired Fecal vs Oral (Wilcoxon) — rank-biserial r
    # read q-values already computed by your alpha_diversity
    q_map = paired_stats.set_index("metric")[["p_value", "q_value"]].to_dict("index")

    for m in METRICS:
        a = matched[f"Oral_{m}"]
        b = matched[f"Fecal_{m}"]
        r_rb = rank_biserial_from_pairs(a, b)
        p = q_map[m]["p_value"]; q = q_map[m]["q_value"]
        fx_rows.append({
            "contrast": "Fecal vs Oral (paired)",
            "metric": m,
            "p_value": p,
            "q_value": q,
            "effect_size": r_rb,
            "effect_name": "rank_biserial_r"
        })

    fx = pd.DataFrame(fx_rows)
    fx.to_csv(os.path.join(args.outdir, "alpha_effect_sizes.csv"), index=False)

    # ---------- IQR outliers ----------
    oral_c = oral.copy();    oral_c["Group"] = "Crohn-Oral";    oral_c["Sample"] = oral_c.index
    fecal_c = fecal.copy();  fecal_c["Group"] = "Crohn-Fecal";  fecal_c["Sample"] = fecal_c.index
    healthy_c = healthy.copy(); healthy_c["Group"] = "Healthy-Oral"; healthy_c["Sample"] = healthy_c.index
    all_df = pd.concat([oral_c, fecal_c, healthy_c], axis=0, ignore_index=True)

    out_rows = []
    for g in ["Crohn-Oral", "Crohn-Fecal", "Healthy-Oral"]:
        sub = all_df[all_df["Group"] == g]
        for m in METRICS:
            flags = iqr_outliers(sub[["Sample", m]].rename(columns={m:"Value"}), m, "Value")
            if not flags.empty:
                flags["Group"] = g
                flags = flags.rename(columns={"metric":"Metric"})
                out_rows.append(flags)

    out_df = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(
        columns=["Sample","Value","Q1","Q3","IQR","fence_low","fence_high","Metric","Group"]
    )
    out_df.to_csv(os.path.join(args.outdir, "alpha_outliers_iqr.csv"), index=False)

    # ---------- Leave-one-out stability (paired Wilcoxon) ----------
    lines = []
    lines.append("Alpha paired Wilcoxon p-value stability (LOO over matched pairs)\n")
    for m in METRICS:
        x = matched[f"Oral_{m}"].to_numpy()
        y = matched[f"Fecal_{m}"].to_numpy()
        pvals = []
        for i in range(len(x)):
            xi = np.delete(x, i); yi = np.delete(y, i)
            try:
                _, p = wilcoxon(xi, yi, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                p = np.nan
            pvals.append(p)
        pvals = np.array(pvals, dtype=float)
        lines.append(
            f"- {m}: n={len(x)}, p(min/median/max) = "
            f"{np.nanmin(pvals):.3g}/{np.nanmedian(pvals):.3g}/{np.nanmax(pvals):.3g}; "
            f"non-significant runs (p>=0.05): {(pvals>=0.05).sum()}\n"
        )
    with open(os.path.join(args.outdir, "alpha_stability_summary.txt"), "w") as fh:
        fh.writelines(lines)

    print("[alpha_effects_qc] Done.")

if __name__ == "__main__":
    main()

