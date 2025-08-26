#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, mannwhitneyu, wilcoxon, kruskal
import yaml

# ---------- colors strictly from YAML ----------
def load_colors_strict(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    colors = (cfg.get("colors") or {})

    req = ["Crohn-Oral", "Crohn-Fecal", "Healthy-Oral", "Healthy-Fecal"]
    miss = [k for k in req if k not in colors]
    if miss:
        raise KeyError(f"Missing color keys in config.yaml: {miss}")

    group_order   = ["Crohn-Oral", "Crohn-Fecal", "Healthy-Oral", "Healthy-Fecal"]
    group_palette = [colors[g] for g in group_order]

    oh_order   = ["Crohn-Oral", "Healthy-Oral"]
    oh_palette = [colors[g] for g in oh_order]

    fh_order   = ["Crohn-Fecal", "Healthy-Fecal"]
    fh_palette = [colors[g] for g in fh_order]

    oral_col  = colors.get("Oral",  colors["Crohn-Oral"])
    fecal_col = colors.get("Fecal", colors["Crohn-Fecal"])
    of_order   = ["Oral", "Fecal"]
    of_palette = [oral_col, fecal_col]

    return (group_order, group_palette), (oh_order, oh_palette), (of_order, of_palette), (fh_order, fh_palette)

# ---------- alpha metrics ----------
def _safe_array(x):
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]

def shannon(col):
    a = _safe_array(col); a = a[a > 0]
    if a.size == 0: return 0.0
    p = a / a.sum()
    return float(entropy(p, base=np.e))

def simpson(col):
    a = _safe_array(col); a = a[a > 0]
    if a.size == 0: return 0.0
    p = a / a.sum()
    return float(1.0 - np.sum(p**2))

def observed(col):
    a = _safe_array(col)
    return int(np.sum(a > 0))

def alpha_df(abund: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Shannon":  abund.apply(shannon, axis=0),
        "Simpson":  abund.apply(simpson, axis=0),
        "Richness": abund.apply(observed, axis=0),
    })

# ---------- BH-FDR ----------
def bh_qvalues(pvals):
    p = np.array(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i, idx in enumerate(order[::-1], start=1):
        rank = n - i + 1
        val = min(prev, p[idx] * n / rank)
        q[idx] = val
        prev = val
    return q.tolist()

# ---------- helpers ----------
def annotate_title(ax, title, p=None, q=None, stat=None):
    parts = [title]
    if p is not None:   parts.append(f"p={p:.3g}")
    if q is not None:   parts.append(f"q={q:.3g}")
    if stat is not None:parts.append(f"W={stat:.2f}")
    ax.set_title(" | ".join(parts))

# ---------- main ----------
def main():
    # snakemake IO
    oral_file       = snakemake.input.oral_crohn
    fecal_file      = snakemake.input.fecal_crohn
    healthy_oral_file  = snakemake.input.oral_healthy
    healthy_fecal_file = snakemake.input.healthy_fecal
    matched_file    = snakemake.input.matched_ids

    out_oral        = snakemake.output.oral
    out_fecal       = snakemake.output.fecal
    out_healthy     = snakemake.output.healthy              # Healthy-Oral
    out_healthy_fecal = snakemake.output.healthy_fecal

    results_dir   = os.path.dirname(out_oral)
    os.makedirs(results_dir, exist_ok=True)

    (group_order, group_palette), (oh_order, oh_palette), (of_order, of_palette), (fh_order, fh_palette) = load_colors_strict("config.yaml")

    # data
    oral          = pd.read_csv(oral_file, index_col=0)
    fecal         = pd.read_csv(fecal_file, index_col=0)
    healthy_oral  = pd.read_csv(healthy_oral_file, index_col=0)
    healthy_fecal = pd.read_csv(healthy_fecal_file, index_col=0)
    matched       = pd.read_csv(matched_file)

    # alpha tables
    alpha_oral = alpha_df(oral);           alpha_oral.index.name = "Sample";         alpha_oral.to_csv(out_oral)
    alpha_fecal = alpha_df(fecal);         alpha_fecal.index.name = "Sample";        alpha_fecal.to_csv(out_fecal)
    alpha_healthy = alpha_df(healthy_oral);alpha_healthy.index.name = "Sample";      alpha_healthy.to_csv(out_healthy)
    alpha_healthy_fecal = alpha_df(healthy_fecal); alpha_healthy_fecal.index.name = "Sample"; alpha_healthy_fecal.to_csv(out_healthy_fecal)

    # matched checks (Crohn only)
    for c in ["STUDY_ID","Oral_col","Fecal_col"]:
        if c not in matched.columns:
            raise KeyError(f"Missing column '{c}' in {matched_file}")
    assert set(matched["Oral_col"]).issubset(oral.columns)
    assert set(matched["Fecal_col"]).issubset(fecal.columns)

    oral_m   = oral[matched["Oral_col"]]
    fecal_m  = fecal[matched["Fecal_col"]]

    matched_alpha = pd.DataFrame({
        "STUDY_ID":     matched["STUDY_ID"].values,
        "Oral_Sample":  matched["Oral_col"].values,
        "Fecal_Sample": matched["Fecal_col"].values,
        "Oral_Shannon":  [shannon(oral_m[c])  for c in oral_m.columns],
        "Fecal_Shannon": [shannon(fecal_m[c]) for c in fecal_m.columns],
        "Oral_Simpson":  [simpson(oral_m[c])  for c in oral_m.columns],
        "Fecal_Simpson": [simpson(fecal_m[c]) for c in fecal_m.columns],
        "Oral_Richness": [observed(oral_m[c])  for c in oral_m.columns],
        "Fecal_Richness":[observed(fecal_m[c]) for c in fecal_m.columns],
    })
    matched_alpha.to_csv(os.path.join(results_dir, "alpha_matched.csv"), index=False)

    METRICS = ["Richness","Shannon","Simpson"]

    # ---- stats text (MWU / Wilcoxon / Kruskal) ----
    lines = []
    lines.append("== Crohn Oral vs Healthy Oral (independent) ==\n")
    for m in METRICS:
        u, p = mannwhitneyu(alpha_oral[m], alpha_healthy[m], alternative="two-sided")
        lines.append(f"Mann-Whitney U ({m}): U={u:.2f}, p={p:.4g}\n")
    lines.append("\n")

    lines.append("== Crohn Fecal vs Healthy Fecal (independent) ==\n")
    for m in METRICS:
        u, p = mannwhitneyu(alpha_fecal[m], alpha_healthy_fecal[m], alternative="two-sided")
        lines.append(f"Mann-Whitney U ({m}): U={u:.2f}, p={p:.4g}\n")
    lines.append("\n")

    # paired wilcoxon + BH (Crohn Oral vs Crohn Fecal)
    paired_rows, pvals = [], []
    for m in METRICS:
        x = matched_alpha[f"Oral_{m}"]
        y = matched_alpha[f"Fecal_{m}"]
        w, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        paired_rows.append({
            "metric": m,
            "n_pairs": int(len(x)),
            "median_oral":  float(np.median(x)),
            "median_fecal": float(np.median(y)),
            "delta_median": float(np.median(y) - np.median(x)),
            "wilcoxon_stat": float(w),
            "p_value": float(p),
        })
        pvals.append(p)
    qvals = bh_qvalues(pvals)
    for i, q in enumerate(qvals):
        paired_rows[i]["q_value"] = float(q)

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(os.path.join(results_dir, "paired_alpha_stats.csv"), index=False)
    pmap = paired_df.set_index("metric")[["p_value","q_value","wilcoxon_stat"]].to_dict("index")

    lines.append("== Crohn Oral vs Crohn Fecal (paired) ==\n")
    for m in METRICS:
        info = pmap[m]
        lines.append(f"Wilcoxon ({m}): W={info['wilcoxon_stat']:.2f}, p={info['p_value']:.4g}, q={info['q_value']:.4g}\n")
    lines.append("\n")

    lines.append("== All groups Kruskal-Wallis (independent, 4 groups) ==\n")
    for m in METRICS:
        h, p = kruskal(alpha_oral[m], alpha_fecal[m], alpha_healthy[m], alpha_healthy_fecal[m])
        lines.append(f"Kruskal-Wallis ({m}): H={h:.2f}, p={p:.4g}\n")
    lines.append("\n")
    with open(os.path.join(results_dir, "alpha_stats.txt"), "w") as fh:
        fh.writelines(lines)

    # ---- deltas (wide + long) ----
    deltas = pd.DataFrame({
        "STUDY_ID":     matched_alpha["STUDY_ID"],
        "sample_oral":  matched_alpha["Oral_Sample"],
        "sample_fecal": matched_alpha["Fecal_Sample"],
        "delta_richness": matched_alpha["Fecal_Richness"] - matched_alpha["Oral_Richness"],
        "delta_shannon":  matched_alpha["Fecal_Shannon"]  - matched_alpha["Oral_Shannon"],
        "delta_simpson":  matched_alpha["Fecal_Simpson"]  - matched_alpha["Oral_Simpson"],
    })
    out_delta_csv = os.path.join(results_dir, "paired_alpha_deltas.csv")
    deltas.to_csv(out_delta_csv, index=False)

    d_long = deltas.melt(
        id_vars=["STUDY_ID","sample_oral","sample_fecal"],
        value_vars=["delta_richness","delta_shannon","delta_simpson"],
        var_name="Delta", value_name="Value"
    )

    # ---------- plotting ----------
    sns.set(style="whitegrid", font_scale=1.1)

    # 1) all groups (4 groups)
    for m in METRICS:
        df_all = pd.DataFrame({
            "Value": pd.concat([alpha_oral[m], alpha_fecal[m], alpha_healthy[m], alpha_healthy_fecal[m]]),
            "Group": (["Crohn-Oral"]*len(alpha_oral)
                    + ["Crohn-Fecal"]*len(alpha_fecal)
                    + ["Healthy-Oral"]*len(alpha_healthy)
                    + ["Healthy-Fecal"]*len(alpha_healthy_fecal))
        })
        plt.figure(figsize=(8,5))
        ax = sns.boxplot(
            data=df_all, x="Group", y="Value",
            hue="Group", order=group_order, dodge=False,
            palette=group_palette, legend=False
        )
        sns.stripplot(data=df_all, x="Group", y="Value", order=group_order, color="k", alpha=0.45, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel(m)
        ax.set_title(f"Alpha Diversity ({m}) Across Groups")
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, f"boxplot_{m.lower()}_allgroups.png"), dpi=180); plt.close()

    # 2) paired oral vs fecal (Crohn only)
    def plot_paired(m):
        dfp = pd.DataFrame({"Oral": matched_alpha[f"Oral_{m}"], "Fecal": matched_alpha[f"Fecal_{m}"]})
        mlong = dfp.melt(var_name="SampleType", value_name=m)
        plt.figure(figsize=(7,5))
        of_map = dict(zip(["Oral","Fecal"], of_palette))
        ax = sns.boxplot(
            data=mlong, x="SampleType", y=m,
            order=["Oral","Fecal"],
            hue="SampleType", palette=of_map,
            dodge=False, legend=False
        )

        sns.stripplot(data=mlong, x="SampleType", y=m, order=["Oral","Fecal"], color="k", alpha=0.45, jitter=0.2, ax=ax)
        for i in range(len(dfp)):
            plt.plot(["Oral","Fecal"], [dfp.iloc[i,0], dfp.iloc[i,1]], color="gray", alpha=0.35, linewidth=1)
        info = pmap[m]
        annotate_title(ax, f"Paired Alpha Diversity: Crohn Oral vs Fecal ({m})",
                       p=info["p_value"], q=info["q_value"], stat=info["wilcoxon_stat"])
        ax.set_xlabel(""); ax.set_ylabel(m)
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, f"paired_boxplot_{m.lower()}_oral_fecal.png"), dpi=180); plt.close()
    for m in METRICS:
        plot_paired(m)

    # 3) Crohn Oral vs Healthy Oral (independent)
    for m in METRICS:
        df_oh = pd.DataFrame({
            "Value": pd.concat([alpha_oral[m], alpha_healthy[m]]),
            "Group": (["Crohn-Oral"]*len(alpha_oral) + ["Healthy-Oral"]*len(alpha_healthy))
        })
        plt.figure(figsize=(6,5))
        ax = sns.boxplot(
            data=df_oh, x="Group", y="Value",
            hue="Group", order=oh_order, dodge=False,
            palette=oh_palette, legend=False
        )
        sns.stripplot(data=df_oh, x="Group", y="Value", order=oh_order, color="k", alpha=0.45, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel(m)
        ax.set_title(f"Alpha Diversity ({m}) Crohn Oral vs Healthy Oral")
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, f"boxplot_{m.lower()}_oral_vs_healthy.png"), dpi=180); plt.close()

    # 4) Crohn Fecal vs Healthy Fecal (independent)
    for m in METRICS:
        df_fh = pd.DataFrame({
            "Value": pd.concat([alpha_fecal[m], alpha_healthy_fecal[m]]),
            "Group": (["Crohn-Fecal"]*len(alpha_fecal) + ["Healthy-Fecal"]*len(alpha_healthy_fecal))
        })
        plt.figure(figsize=(6,5))
        ax = sns.boxplot(
            data=df_fh, x="Group", y="Value",
            hue="Group", order=fh_order, dodge=False,
            palette=fh_palette, legend=False
        )
        sns.stripplot(data=df_fh, x="Group", y="Value", order=fh_order, color="k", alpha=0.45, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel(m)
        ax.set_title(f"Alpha Diversity ({m}) Crohn Fecal vs Healthy Fecal")
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, f"boxplot_{m.lower()}_fecal_vs_healthyfecal.png"), dpi=180); plt.close()

    # 5) delta box (Fecal − Oral, Crohn pairs)
    d_long = deltas.melt(
        id_vars=["STUDY_ID","sample_oral","sample_fecal"],
        value_vars=["delta_richness","delta_shannon","delta_simpson"],
        var_name="Delta", value_name="Value"
    )
    order = ["delta_richness", "delta_shannon", "delta_simpson"]

    g = sns.catplot(
        data=d_long, col="Delta", kind="box", y="Value",
        col_order=order, sharey=False, height=3.6, aspect=0.9
    )
    
    for ax, metric in zip(g.axes.flat, order):
        sub = d_long[d_long["Delta"] == metric]
        sns.stripplot(data=sub, y="Value", ax=ax, alpha=0.45, jitter=0.18, color="k")
        ax.axhline(0, ls="--", lw=1)
        ax.set_xlabel("")
        ax.set_title(metric.replace("delta_", "Δ ").title())

    # --- save both new and legacy filenames so Snakefile is happy ---
    plt.tight_layout()

    out_legacy = os.path.join(results_dir, "paired_alpha_delta_boxplot.png")

    plt.savefig(out_legacy, dpi=180)
    plt.close()

    print(f"[INFO] Alpha diversity exports complete → {results_dir}")

if __name__ == "__main__":
    main()
