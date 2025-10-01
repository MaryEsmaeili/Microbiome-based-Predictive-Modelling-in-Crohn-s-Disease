#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha diversity — matched (with covariates) + raw (no covariates)
=================================================================

What this script does
---------------------
1) Reads four abundance files (oral/fecal × Crohn/Healthy) and computes:
   - Shannon, Richness, Evenness per sample
   - Conservative ID normalization (only strip leading zeros for numeric-only IDs)
2) Builds a unified alpha table across all groups (alpha_all).
3) Reads pooled covariates (model_table_pooled.csv), normalizes Sample_ID the
   same way, and MERGEs **by Sample_ID only** (no inner-join on site).
   - All WITH-covariate models/figures are computed on the *matched set*.
   - All RAW panels (no covariates) use alpha_all (unmatched allowed).
4) Outputs:
   - Matched models (HC3 OLS), per-metric FDR + across-metrics FDR for "disease"
   - PPI models in Crohn-only (per site)
   - Disease×Site interaction model
   - Unpaired Oral vs Fecal (by group)
   - Paired Wilcoxon (Crohn) if pairs provided; fallback = intersection by Sample_ID
   - Figures:
       • Per-site 4-up panels WITH covariates (matched counts)
       • Per-site 4-up panels RAW (unmatched counts)
       • Single-metric boxplots WITH covariates (per site)
       • NEW: Four-group panel (Healthy/Crohn × Oral/Fecal), RAW + matched
       • NEW: Crohn paired Oral↔Fecal lines panel (3-up)
     Each panel includes captions with n, p, q(FDR), Cliff’s δ as applicable.

CLI
---
  --oral-crohn PATH
  --fecal-crohn PATH
  --oral-healthy PATH
  --fecal-healthy PATH
  --covars-pool PATH                 # model_table_pooled.csv
  --pairs PATH                       # optional pairs list (oral/fecal columns)
  --colors PATH                      # config/colors.yml
  --outdir PATH
  --level {genus,species}            # optional, for Snakefile compatibility

Dependencies: pandas, numpy, scipy, statsmodels, matplotlib, seaborn, pyyaml
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests

# ------------------------------ Config ---------------------------------
METRICS = ["Shannon", "Richness", "Evenness"]

# Covariates we *wish* to include if present in pooled covars
WISH_COVARS = [
    "Age", "Sex", "BMI", "Smoking", "Antibiotics_3m",
    "PPI_use", "Steroids_ongoing", "Immuno_ongoing", "Responder",
]

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------ Utils ----------------------------------
def ensure_dir(p: str | Path) -> None:
    """Create directory if not exists."""
    Path(p).mkdir(parents=True, exist_ok=True)

def normalize_sample_id(s: str) -> str:
    """
    Normalize sample identifiers consistently:
    - Strip whitespace
    - Drop replicate suffix like '.1' or '.2'
    - If ID is numeric-only, strip leading zeros by casting to int
    - Uppercase for stability
    """
    s = str(s).strip()
    # remove a trailing ".digits"
    s = pd.Series([s]).str.replace(r"\.\d+$", "", regex=True).iloc[0]
    if s.isdigit():
        try:
            s = str(int(s))
        except Exception:
            pass
    return s.upper()

def _norm_sex(x) -> Optional[float]:
    """Map F/female→0, M/male→1; else try numeric."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"f", "female"}:
        return 0.0
    if s in {"m", "male"}:
        return 1.0
    try:
        return float(x)
    except Exception:
        return None

def placeholder_plot(path: Path, title_txt: str) -> None:
    """Write a placeholder PNG when panel has no data."""
    ensure_dir(path.parent)
    plt.figure(figsize=(5.8, 4.2))
    plt.axis("off")
    plt.text(0.5, 0.5, title_txt, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (n1 * n2)

def fdr(p: List[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR over a list (NaNs are kept)."""
    p = np.array([np.nan if v is None else v for v in p], dtype=float)
    mask = ~np.isnan(p)
    q = np.full_like(p, np.nan, dtype=float)
    if mask.sum():
        q[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return q

def format_annot_pairs(n_pairs: int, p: float | None = None, q: float | None = None) -> str:
    """Caption for paired lines plots."""
    parts = [f"n(pairs)={n_pairs}"]
    if p is not None and not pd.isna(p):
        parts.append(f"p={p:.2e}")
    if q is not None and not pd.isna(q):
        parts.append(f"q={q:.2e}")
    return " | ".join(parts)

def format_annot_2x(nH: int, nC: int, p: float | None = None,
                    q: float | None = None, cd: float | None = None) -> str:
    """Caption for two-group (Healthy vs Crohn) comparisons."""
    parts = [f"n(Healthy)={nH}", f"n(Crohn)={nC}"]
    if p is not None and not pd.isna(p):
        parts.append(f"p={p:.2e}")
    if q is not None and not pd.isna(q):
        parts.append(f"q={q:.2e}")
    if cd is not None and not pd.isna(cd):
        parts.append(f"δ={round(float(cd), 2)}")
    return " | ".join(parts)

# ------------------------------ Colors ---------------------------------
def load_colors(path: str | None) -> dict:
    """
    Load color palette from YAML.
    Supports:
      1) legacy: {colors: {Crohn-Oral: "#...", Healthy-Oral: "#...", ...}}
      2) new: group/synonyms schema shown in README (preferred)
    Returns:
      {
        "oral":  [crohn_color, healthy_color],
        "fecal": [crohn_color, healthy_color],
      }
    """
    fallback = {
        "oral":  ["#edae49", "#00798c"],  # [Crohn, Healthy]
        "fecal": ["#30638e", "#d1495b"],
    }
    if not path or not Path(path).exists():
        return fallback
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    # legacy block
    if "colors" in cfg and isinstance(cfg["colors"], dict):
        c = cfg["colors"]
        return {
            "oral":  [c.get("Crohn-Oral",  fallback["oral"][0]),  c.get("Healthy-Oral",  fallback["oral"][1])],
            "fecal": [c.get("Crohn-Fecal", fallback["fecal"][0]), c.get("Healthy-Fecal", fallback["fecal"][1])],
        }

    # group/synonyms
    group = cfg.get("group", {}) or {}
    syn = cfg.get("synonyms", {}) or {}

    def get(key: str, default: str) -> str:
        if key in group and group[key]:
            return group[key]
        for alias in syn.get(key, []):
            if alias in group and group[alias]:
                return group[alias]
        return default

    return {
        "oral":  [get("Oral_Crohn",  fallback["oral"][0]),  get("Oral_Healthy",  fallback["oral"][1])],
        "fecal": [get("Fecal_Crohn", fallback["fecal"][0]), get("Fecal_Healthy", fallback["fecal"][1])],
    }

# ------------------------- Alpha from abundance -------------------------
def to_sample_by_taxa(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert table to matrix with samples as index.
    If a 'sample_id' column exists, assume rows=samples; otherwise assume
    first column is taxa and columns are samples.
    """
    cols_lower = [c.lower() for c in df.columns]
    sid_col = None
    for cand in ("sample_id", "sample", "id"):
        if cand in cols_lower:
            sid_col = df.columns[cols_lower.index(cand)]
            break
    if sid_col is not None:
        df = df.copy().set_index(sid_col)
        num = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
        return num, list(num.index.astype(str))
    df = df.copy().set_index(df.columns[0])
    num = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    return num.T, list(num.columns.astype(str))

def compute_alpha_from_abundance(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read an abundance CSV and compute Shannon/Richness/Evenness per sample.
    Returns:
      alpha_df with columns: Sample_ID, Shannon, Richness, Evenness
      idmap original→normalized for transparency.
    """
    df = pd.read_csv(path)
    mat, _ = to_sample_by_taxa(df)
    # normalize IDs
    norm_ids = [normalize_sample_id(x) for x in mat.index.astype(str)]
    idmap = pd.DataFrame({"original_id": mat.index.astype(str), "Sample_ID": norm_ids})
    mat.index = norm_ids
    # convert to proportions per sample
    row_sums = mat.sum(axis=1).replace(0.0, np.nan)
    props = mat.div(row_sums, axis=0).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = -(props.replace(0, np.nan) * np.log(props.replace(0, np.nan))).sum(axis=1).fillna(0.0)
    rich = (mat > 0).sum(axis=1)
    even = [float(s / np.log(r)) if r > 1 else float("nan") for s, r in zip(sh.values, rich.values)]
    alpha = pd.DataFrame({
        "Sample_ID": mat.index.astype(str),
        "Shannon": sh.values.astype(float),
        "Richness": rich.values.astype(int),
        "Evenness": np.array(even, dtype=float),
    })
    return alpha, idmap

def build_alpha_table(abund_path: str, site: str, disease: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute alpha, add site/disease columns."""
    a, idmap = compute_alpha_from_abundance(abund_path)
    a["site"], a["disease"] = site, float(disease)
    a = a[["Sample_ID", "site", "disease", "Shannon", "Richness", "Evenness"]]
    return a, idmap.assign(site=site, disease=int(disease))

# ---------------------------- Plot helpers ------------------------------
def disease_legend_handles(pal: Dict[str, str]):
    """Legend for Healthy/Crohn given a palette dict."""
    return [
        mpatches.Patch(facecolor=pal["Healthy"], label="Healthy"),
        mpatches.Patch(facecolor=pal["Crohn"],   label="Crohn"),
    ]

def box_and_scatter_panel(site_name: str,
                          df: pd.DataFrame,
                          out_png: Path,
                          crohn_color: str, healthy_color: str,
                          title_suffix: str,
                          metric_annots: Optional[Dict[str, str]] = None):
    """
    4-up panel (two-group per site) with captions under axes.
    metric_annots: optional dict {metric -> caption string returned by format_annot_2x()}
    """
    if df.empty:
        placeholder_plot(out_png, f"{site_name.capitalize()} — Alpha (4-up {title_suffix})\n(No data)")
        return

    disease_labels = {0.0: "Healthy", 1.0: "Crohn"}
    pal = {"Crohn": crohn_color, "Healthy": healthy_color}
    metric_annots = metric_annots or {}

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))

    # Top-left 3: boxplots for metrics
    for ax, met in zip(axes.flat[:3], METRICS):
        dd = df[["disease", met]].dropna()
        if dd.empty or dd["disease"].nunique() < 2:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No data: {met}", ha="center")
            continue
        dd = dd.rename(columns={met: "value"})
        dd["group"] = dd["disease"].map(disease_labels)

        sns.boxplot(data=dd, x="group", y="value", order=["Healthy", "Crohn"],
                    hue="group", palette=pal, dodge=False, legend=False, ax=ax)
        sns.stripplot(data=dd, x="group", y="value", order=["Healthy", "Crohn"],
                      color="black", alpha=0.6, jitter=0.15, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(met)
        ax.set_title(met, fontsize=11)

        # caption under axis
        nH = int((dd["group"] == "Healthy").sum())
        nC = int((dd["group"] == "Crohn").sum())
        cap = metric_annots.get(met, format_annot_2x(nH, nC))
        ax.text(0.5, -0.25, cap, transform=ax.transAxes, ha="center", va="top", fontsize=9)

    # Bottom-right: scatter Shannon vs Richness
    ax = axes.flat[3]
    sub = df[["Shannon", "Richness", "disease"]].dropna()
    if sub.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data: scatter", ha="center")
    else:
        sub["group"] = sub["disease"].map(disease_labels)
        for g, ddg in sub.groupby("group"):
            ax.scatter(ddg["Shannon"], ddg["Richness"], label=g, s=28, alpha=0.85, c=pal[g])
        sns.regplot(x="Shannon", y="Richness", data=sub, scatter=False, ci=95, ax=ax)
        ax.set_xlabel("Shannon")
        ax.set_ylabel("Richness")
        ax.set_title("Shannon vs Richness")
        ax.legend(frameon=False, loc="best")
        nH = int((sub["group"] == "Healthy").sum())
        nC = int((sub["group"] == "Crohn").sum())
        ax.text(0.5, -0.25, format_annot_2x(nH, nC), transform=ax.transAxes, ha="center", va="top", fontsize=9)

    # global legend
    handles = disease_legend_handles(pal)
    labels  = ["Healthy", "Crohn"]
    fig.legend(handles=handles, labels=labels, frameon=False, loc="upper right")
    plt.suptitle(f"{site_name.capitalize()} — Alpha (4-up {title_suffix})", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300)
    plt.close()

# ---- NEW: Four-group panel (Healthy/Crohn × Oral/Fecal), RAW + matched ----
def four_group_panel(df_all: pd.DataFrame, out_png: Path, palettes: Dict[str, List[str]], title_suffix: str):
    """
    Build a 4-group panel across all samples:
        Groups = ["Healthy-Oral","Crohn-Oral","Healthy-Fecal","Crohn-Fecal"]
    For each metric: a 4-group boxplot with per-site two-group stats (Crohn vs Healthy)
    printed in the caption: Oral(p,q,δ) | Fecal(p,q,δ).
    Bottom-right: Shannon vs Richness scatter colored by 4 groups.
    """
    if df_all.empty:
        placeholder_plot(out_png, f"Alpha (4-group {title_suffix})\n(No data)")
        return

    # Construct group labels and palette
    lab_map = {0.0: "Healthy", 1.0: "Crohn"}
    df = df_all.copy()
    df["disease_label"] = df["disease"].map(lab_map)
    df["group4"] = df["disease_label"] + "-" + df["site"].str.capitalize()  # e.g., Healthy-Oral
    order4 = ["Healthy-Oral", "Crohn-Oral", "Healthy-Fecal", "Crohn-Fecal"]
    pal4 = {
        "Healthy-Oral": palettes["oral"][1],
        "Crohn-Oral":   palettes["oral"][0],
        "Healthy-Fecal":palettes["fecal"][1],
        "Crohn-Fecal":  palettes["fecal"][0],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))

    # For each metric, compute within-site stats and draw boxplot
    for ax, met in zip(axes.flat[:3], METRICS):
        dd = df[["group4", "site", "disease", met]].dropna()
        if dd.empty or dd["group4"].nunique() < 2:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No data: {met}", ha="center")
            continue

        sns.boxplot(data=dd, x="group4", y=met, order=order4, palette=pal4, ax=ax)
        sns.stripplot(data=dd, x="group4", y=met, order=order4,
                      color="black", alpha=0.5, jitter=0.15, ax=ax)
        ax.set_xlabel("")
        ax.set_title(met)

        # per-site two-group stats (Crohn vs Healthy)
        parts = []
        for site_name in ["oral", "fecal"]:
            sub = dd[dd["site"] == site_name]
            nH = int((sub["disease"] == 0.0).sum())
            nC = int((sub["disease"] == 1.0).sum())
            if sub["disease"].nunique() < 2:
                parts.append(f"{site_name.capitalize()}(nH={nH}, nC={nC})")
                continue
            x = sub.loc[sub["disease"] == 1.0, met].values
            y = sub.loc[sub["disease"] == 0.0, met].values
            try:
                p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            except Exception:
                p = np.nan
            cd = cliffs_delta(x, y)
            # FDR across the two site-tests per metric
            # (compute both p's first)
            parts.append((site_name, nH, nC, p, cd))
        # compute FDR for the two site p-values
        pvals = [t[3] for t in parts if isinstance(t, tuple)]
        qvals = list(fdr(pvals)) if pvals else []
        qi = 0
        cap_chunks = []
        for t in parts:
            if isinstance(t, tuple):
                site_name, nH, nC, p, cd = t
                qv = qvals[qi] if qi < len(qvals) else np.nan
                qi += 1
                cap_chunks.append(f"{site_name.capitalize()}(nH={nH}, nC={nC}, p={p:.2e}, q={qv:.2e}, δ={np.nan if pd.isna(cd) else round(float(cd),2)})")
            else:
                cap_chunks.append(t)
        cap = " | ".join(cap_chunks)
        ax.text(0.5, -0.28, cap, transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.tick_params(axis='x', rotation=20)

    # Scatter: Shannon vs Richness
    ax = axes.flat[3]
    sub = df[["Shannon", "Richness", "group4"]].dropna()
    if sub.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data: scatter", ha="center")
    else:
        for g, gg in sub.groupby("group4"):
            ax.scatter(gg["Shannon"], gg["Richness"], s=26, alpha=0.85,
                       c=pal4.get(g, "#777777"), label=g)
        sns.regplot(x="Shannon", y="Richness", data=sub, scatter=False, ci=95, ax=ax)
        ax.set_xlabel("Shannon")
        ax.set_ylabel("Richness")
        ax.set_title("Shannon vs Richness (4 groups)")
        ax.legend(frameon=False, loc="best")

    plt.suptitle(f"Alpha (4 groups — {title_suffix})", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300)
    plt.close()

# ---- NEW: Paired lines panel for Crohn (Oral ↔ Fecal) -----------------
def crohn_paired_lines_panel(alpha_all: pd.DataFrame,
                             pairs: List[Tuple[str, str]],
                             out_png: Path,
                             palettes: Dict[str, List[str]]):
    """
    Draw a 3-up panel (Shannon, Richness, Evenness) connecting paired Crohn samples
    (oral vs fecal) with lines. Caption shows n_pairs, Wilcoxon p and FDR q per metric.
    """
    # Build two site tables for Crohn
    crohn = alpha_all[alpha_all["disease"] == 1.0].copy()
    oral  = crohn[crohn["site"] == "oral"].set_index("Sample_ID")
    fecal = crohn[crohn["site"] == "fecal"].set_index("Sample_ID")

    # Collect paired values
    pairs = pairs or []
    if not pairs:
        inter = oral.index.intersection(fecal.index)
        pairs = [(sid, sid) for sid in inter]

    if not pairs:
        placeholder_plot(out_png, "Crohn Oral↔Fecal Paired (No pairs)")
        return

    # Compute Wilcoxon and q across metrics
    wilco_p = []
    series_by_metric = {}  # met -> (xo, xf)
    for m in METRICS:
        xo, xf = [], []
        for o_id, f_id in pairs:
            vo = oral[m].get(o_id) if m in oral.columns and o_id in oral.index else np.nan
            vf = fecal[m].get(f_id) if m in fecal.columns and f_id in fecal.index else np.nan
            if pd.notna(vo) and pd.notna(vf):
                xo.append(float(vo)); xf.append(float(vf))
        series_by_metric[m] = (xo, xf)
        if len(xo) >= 1:
            try:
                stat, p = wilcoxon(xo, xf, zero_method="wilcox",
                                   alternative="two-sided", correction=False, mode="auto")
            except Exception:
                p = np.nan
        else:
            p = np.nan
        wilco_p.append(p)
    wilco_q = fdr(wilco_p) if any([not pd.isna(p) for p in wilco_p]) else np.array([np.nan]*len(METRICS))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=False)
    x_oral, x_fecal = 0, 1
    for i, m in enumerate(METRICS):
        ax = axes[i]
        xo, xf = series_by_metric[m]
        if len(xo) == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No pairs: {m}", ha="center")
            continue
        # draw paired lines
        for a, b in zip(xo, xf):
            ax.plot([x_oral, x_fecal], [a, b], alpha=0.5, lw=1.0, color="#7f7f7f")
        # scatter endpoints
        ax.scatter([x_oral]*len(xo), xo, s=24, alpha=0.9, c=palettes["oral"][0], label="Crohn-Oral")
        ax.scatter([x_fecal]*len(xf), xf, s=24, alpha=0.9, c=palettes["fecal"][0], label="Crohn-Fecal")
        ax.set_xticks([x_oral, x_fecal]); ax.set_xticklabels(["Oral", "Fecal"])
        ax.set_title(m)
        ax.set_xlim(-0.3, 1.3)
        # caption
        p = wilco_p[i]; q = wilco_q[i] if i < len(wilco_q) else np.nan
        ax.text(0.5, -0.22, format_annot_pairs(len(xo), p=p, q=q),
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
    handles = [mpatches.Patch(color=palettes["oral"][0], label="Crohn-Oral"),
               mpatches.Patch(color=palettes["fecal"][0], label="Crohn-Fecal")]
    fig.legend(handles=handles, frameon=False, loc="upper right")
    plt.suptitle("Crohn — Oral vs Fecal (Paired lines)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300)
    plt.close()

# --------------------------- Stats per site -----------------------------
def summarize_groups(df_site: pd.DataFrame, site_name: str) -> pd.DataFrame:
    """Simple descriptive stats per disease group for a site."""
    rows = []
    for m in METRICS:
        if m not in df_site.columns:
            continue
        for g in [0.0, 1.0]:
            vals = df_site.loc[df_site["disease"] == g, m].dropna().values
            if vals.size == 0:
                rows.append({"site": site_name, "group": int(g), "metric": m, "n": 0,
                             "mean": np.nan, "sd": np.nan, "median": np.nan, "iqr": np.nan})
                continue
            q1, q3 = np.quantile(vals, [0.25, 0.75])
            rows.append({"site": site_name, "group": int(g), "metric": m, "n": int(vals.size),
                         "mean": float(np.mean(vals)),
                         "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                         "median": float(np.median(vals)),
                         "iqr": float(q3 - q1)})
    return pd.DataFrame(rows)

def run_site_with_cov(site_name: str,
                      alpha_all: pd.DataFrame,
                      covars_df: pd.DataFrame,
                      outdir: Path,
                      palettes: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Matched analysis (WITH covariates):
    - Merge by Sample_ID
    - MWU on matched set (p, q, Cliff's δ)
    - OLS (HC3) per metric with chosen covariates + disease
    - Save models and group summaries
    - Draw 4-up panel and single-metric boxplots with rich captions
    """
    site_mask = alpha_all["site"] == site_name
    d_raw = alpha_all.loc[site_mask].copy()
    d_cov = d_raw.merge(covars_df, on="Sample_ID", how="inner", suffixes=("", "_cov"))

    # Mann–Whitney U on matched set
    wilx_rows = []
    for m in METRICS:
        dd = d_cov[["disease", m]].dropna()
        if dd.empty or dd["disease"].nunique() < 2:
            wilx_rows.append({"metric": m, "p_wilcox": np.nan, "cliffs_delta": np.nan})
            continue
        x = dd.loc[dd["disease"] == 1.0, m].values
        y = dd.loc[dd["disease"] == 0.0, m].values
        try:
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        cd = cliffs_delta(x, y)
        wilx_rows.append({"metric": m, "p_wilcox": p, "cliffs_delta": cd})
    wilx_df = pd.DataFrame(wilx_rows)
    wilx_df["q_wilcox"] = fdr(wilx_df["p_wilcox"].tolist())

    # Build captions per metric (matched)
    def _annot_for(met: str) -> str:
        dd = d_cov[["disease", met]].dropna()
        nH = int((dd["disease"] == 0.0).sum())
        nC = int((dd["disease"] == 1.0).sum())
        row = wilx_df[wilx_df["metric"] == met]
        p = q = cd = None
        if len(row):
            p = row["p_wilcox"].iloc[0]
            q = row["q_wilcox"].iloc[0]
            cd = row["cliffs_delta"].iloc[0]
        return format_annot_2x(nH, nC, p=p, q=q, cd=cd)

    annots = {m: _annot_for(m) for m in METRICS}

    # OLS HC3 models per metric with covariates
    covar_pool = [c for c in WISH_COVARS if c in d_cov.columns]
    for c in covar_pool + ["disease"]:
        if c == "Sex":
            d_cov[c] = d_cov[c].apply(_norm_sex)
        else:
            d_cov[c] = pd.to_numeric(d_cov[c], errors="coerce")

    rows = []
    for m in METRICS:
        if m not in d_cov.columns:
            rows.append({"metric": m, "term": "(missing_metric)",
                         "estimate": np.nan, "std.error": np.nan,
                         "statistic": np.nan, "p.value": np.nan})
            continue
        dd = d_cov[[m, "disease"] + covar_pool].dropna()
        if len(dd) < 6 or dd["disease"].nunique() < 2:
            rows.append({"metric": m, "term": "(skipped_small_n)",
                         "estimate": np.nan, "std.error": np.nan,
                         "statistic": np.nan, "p.value": np.nan})
            continue
        X = add_constant(dd[["disease"] + covar_pool], has_constant="add")
        y = dd[m].values
        fit = OLS(y, X).fit(cov_type="HC3")
        for term, est, se, tval, pval in zip(fit.params.index, fit.params.values,
                                             fit.bse.values, fit.tvalues.values, fit.pvalues.values):
            rows.append({"metric": m, "term": term, "estimate": est,
                         "std.error": se, "statistic": tval, "p.value": pval})
    model_df = pd.DataFrame(rows)
    if not model_df.empty:
        # per-metric FDR
        model_df["q.value"] = np.nan
        for m in METRICS:
            idx = model_df["metric"] == m
            if idx.any():
                model_df.loc[idx, "q.value"] = fdr(model_df.loc[idx, "p.value"].tolist())
        # across-metrics FDR for disease term
        dis_p = []
        for m in METRICS:
            tmp = model_df[(model_df["metric"] == m) & (model_df["term"] == "disease")]
            dis_p.append(tmp["p.value"].iloc[0] if len(tmp) else np.nan)
        dq = fdr(dis_p)
        model_df["q_disease_across_metrics"] = model_df["metric"].map({m: q for m, q in zip(METRICS, dq)})

    # Save tables
    ensure_dir(outdir)
    model_df.to_csv(outdir / f"alpha_models_{site_name}.csv", index=False)
    desc = summarize_groups(d_cov[["disease"] + METRICS], site_name)
    desc["group_label"] = desc["group"].map({0: "Healthy", 1: "Crohn"})
    desc.to_csv(outdir / f"group_summary_{site_name}.csv", index=False)

    # Matched per-site 4-up panel
    colors = palettes[site_name]  # [Crohn, Healthy]
    panel_png = outdir / f"panel_{site_name}_4up_with_covariates.png"
    box_and_scatter_panel(site_name,
                          d_cov[["disease"] + METRICS].copy(),
                          panel_png,
                          colors[0], colors[1],
                          "with covariates",
                          metric_annots=annots)

    # Single metric boxplots (matched) with caption
    disease_labels = {0.0: "Healthy", 1.0: "Crohn"}
    for m in METRICS:
        dd = d_cov[["disease", m]].dropna()
        out_png = outdir / f"boxplot_{site_name}_{m}_with_covariates.png"
        if dd.empty or dd["disease"].nunique() < 2:
            placeholder_plot(out_png, f"{site_name.capitalize()} — {m} (with covariates)\n(No data)")
            continue
        dd = dd.rename(columns={m: "value"})
        dd["group"] = dd["disease"].map(disease_labels)
        plt.figure(figsize=(6.6, 4.8))
        pal = {"Crohn": colors[0], "Healthy": colors[1]}
        ax = sns.boxplot(data=dd, x="group", y="value", order=["Healthy", "Crohn"],
                         hue="group", palette=pal, dodge=False, legend=False)
        sns.stripplot(data=dd, x="group", y="value", order=["Healthy", "Crohn"],
                      color="black", alpha=0.6, jitter=0.15)
        ax.set_xlabel("")
        ax.set_ylabel(m)
        ax.set_title(f"{site_name.capitalize()} — {m}", fontsize=12)
        # caption under axis
        row = wilx_df[wilx_df["metric"] == m]
        p = q = cd = None
        if len(row):
            p = row["p_wilcox"].iloc[0]
            q = row["q_wilcox"].iloc[0]
            cd = row["cliffs_delta"].iloc[0]
        nH = int((dd["group"] == "Healthy").sum())
        nC = int((dd["group"] == "Crohn").sum())
        ax.text(0.5, -0.22, format_annot_2x(nH, nC, p=p, q=q, cd=cd),
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()

    # Return matched table (for 4-group matched panel upstream)
    return d_cov[["Sample_ID","site","disease"]+METRICS].copy(), model_df

# ------------------------ PPI & Interaction etc. -----------------------
def ppi_models(site_name: str,
               alpha_all: pd.DataFrame,
               covars_df: pd.DataFrame,
               outdir: Path) -> pd.DataFrame:
    """Crohn-only PPI models per site using matched set."""
    df = alpha_all.merge(covars_df, on="Sample_ID", how="inner", suffixes=("", "_cov"))
    df = df[(df["site"] == site_name) & (df["disease"] == 1.0)].copy()

    covars = [c for c in WISH_COVARS if c in df.columns]
    for c in covars:
        if c == "Sex":
            df[c] = df[c].apply(_norm_sex)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for m in METRICS:
        if m not in df.columns:
            continue
        rhs = [v for v in covars if v != "PPI_use"]
        use = ["PPI_use"] + rhs
        dd = df[[m] + use].dropna()
        if len(dd) < 6 or dd["PPI_use"].nunique() < 2:
            rows.append({"metric": m, "term": "(skipped_small_n)", "estimate": np.nan,
                         "std.error": np.nan, "statistic": np.nan, "p.value": np.nan})
            continue
        X = add_constant(dd[use], has_constant="add")
        y = dd[m].values
        fit = OLS(y, X).fit(cov_type="HC3")
        for term, est, se, tval, pval in zip(fit.params.index, fit.params.values,
                                             fit.bse.values, fit.tvalues.values, fit.pvalues.values):
            rows.append({"metric": m, "term": term, "estimate": est,
                         "std.error": se, "statistic": tval, "p.value": pval})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q.value"] = np.nan
        for m in METRICS:
            idx = out["metric"] == m
            if idx.any():
                out.loc[idx, "q.value"] = fdr(out.loc[idx, "p.value"].tolist())
    out.to_csv(outdir / f"alpha_ppi_{site_name}.csv", index=False)
    return out

def site_disease_interaction(alpha_all: pd.DataFrame,
                             covars_df: pd.DataFrame,
                             outdir: Path) -> pd.DataFrame:
    """Global disease×site interaction model on matched set."""
    df = alpha_all.merge(covars_df, on="Sample_ID", how="inner", suffixes=("", "_cov")).copy()
    df["site_bin"] = df["site"].map({"fecal": 0.0, "oral": 1.0})
    covar_pool = [c for c in WISH_COVARS if c in df.columns]
    for c in covar_pool + ["disease", "site_bin"]:
        if c == "Sex":
            df[c] = df[c].apply(_norm_sex)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for m in METRICS:
        if m not in df.columns:
            continue
        dd = df[[m, "disease", "site_bin"] + covar_pool].dropna()
        if len(dd) < 8 or dd["disease"].nunique() < 2 or dd["site_bin"].nunique() < 2:
            rows.append({"metric": m, "term": "(skipped_small_n)", "estimate": np.nan,
                         "std.error": np.nan, "statistic": np.nan, "p.value": np.nan})
            continue
        X = dd[["disease", "site_bin"]].copy()
        X["disease:site_bin"] = dd["disease"] * dd["site_bin"]
        for c in covar_pool:
            X[c] = dd[c].values
        X = add_constant(X, has_constant="add")
        y = dd[m].values
        fit = OLS(y, X).fit(cov_type="HC3")
        for term, est, se, tval, pval in zip(fit.params.index, fit.params.values,
                                             fit.bse.values, fit.tvalues.values, fit.pvalues.values):
            rows.append({"metric": m, "term": term, "estimate": est,
                         "std.error": se, "statistic": tval, "p.value": pval})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q.value"] = np.nan
        for m in METRICS:
            idx = out["metric"] == m
            if idx.any():
                out.loc[idx, "q.value"] = fdr(out.loc[idx, "p.value"].tolist())
    out.to_csv(outdir / "alpha_site_disease_interaction.csv", index=False)
    return out

def oral_vs_fecal_unpaired(alpha_all: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Unpaired oral vs fecal within each disease group (MWU + FDR)."""
    rows = []
    for status, lab in [(1.0, "Crohn"), (0.0, "Healthy")]:
        sub = alpha_all[alpha_all["disease"] == status]
        for m in METRICS:
            dd = sub[["site", m]].dropna()
            if dd["site"].nunique() < 2:
                rows.append({"group": lab, "metric": m, "p_mwu": np.nan})
                continue
            x = dd.loc[dd["site"] == "oral", m].values
            y = dd.loc[dd["site"] == "fecal", m].values
            try:
                p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            except Exception:
                p = np.nan
            rows.append({"group": lab, "metric": m, "p_mwu": p})
    out = pd.DataFrame(rows)
    out["q_mwu"] = fdr(out["p_mwu"].tolist())
    out.to_csv(outdir / "oral_fecal_unpaired_by_group.csv", index=False)
    return out

# -------------------------- RAW annotations -----------------------------
def raw_annots(df_site: pd.DataFrame) -> Dict[str, str]:
    """
    Build {metric -> caption} on RAW (no covariates) data.
    Uses MWU p and FDR q across metrics; includes Cliff's δ.
    """
    rows_p = []
    tmp_store: Dict[str, Tuple[int, int, float, float]] = {}
    for m in METRICS:
        dd = df_site[["disease", m]].dropna()
        nH = int((dd["disease"] == 0.0).sum())
        nC = int((dd["disease"] == 1.0).sum())
        if dd["disease"].nunique() < 2:
            tmp_store[m] = (nH, nC, np.nan, np.nan)
            continue
        x = dd.loc[dd["disease"] == 1.0, m].values
        y = dd.loc[dd["disease"] == 0.0, m].values
        try:
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        cd = cliffs_delta(x, y)
        rows_p.append(p)
        tmp_store[m] = (nH, nC, p, cd)
    qv = fdr(rows_p) if rows_p else np.array([])
    ann: Dict[str, str] = {}
    k = 0
    for m in METRICS:
        nH, nC, p, cd = tmp_store.get(m, (0, 0, np.nan, np.nan))
        if not np.isnan(p):
            q = qv[k] if k < len(qv) else np.nan
            ann[m] = format_annot_2x(nH, nC, p=p, q=q, cd=cd)
            k += 1
        else:
            ann[m] = format_annot_2x(nH, nC)
    return ann

# ------------------------------- Main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oral-crohn", required=True)
    ap.add_argument("--fecal-crohn", required=True)
    ap.add_argument("--oral-healthy", required=True)
    ap.add_argument("--fecal-healthy", required=True)
    ap.add_argument("--covars-pool", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--colors", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--level", required=False, default=None)  # optional for Snakefile compatibility
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Colors: palettes["oral"] = [Crohn, Healthy], palettes["fecal"] = [Crohn, Healthy]
    palettes = load_colors(args.colors)
    with open(outdir / "colors_used.json", "w") as f:
        json.dump(palettes, f, indent=2)

    # 1) Alpha tables + ID maps
    alpha_oral, idmap_oo = build_alpha_table(args.oral_crohn, "oral", 1)
    alpha_fecal, idmap_of = build_alpha_table(args.fecal_crohn, "fecal", 1)
    alpha_h_oral, idmap_ho = build_alpha_table(args.oral_healthy, "oral", 0)
    alpha_h_fecal, idmap_hf = build_alpha_table(args.fecal_healthy, "fecal", 0)

    iddir = outdir / "idmaps"
    ensure_dir(iddir)
    idmap_oo.to_csv(iddir / "idmap_oral_crohn.tsv", sep="\t", index=False)
    idmap_of.to_csv(iddir / "idmap_fecal_crohn.tsv", sep="\t", index=False)
    idmap_ho.to_csv(iddir / "idmap_oral_healthy.tsv", sep="\t", index=False)
    idmap_hf.to_csv(iddir / "idmap_fecal_healthy.tsv", sep="\t", index=False)

    alpha_all = (
        pd.concat([alpha_oral, alpha_fecal, alpha_h_oral, alpha_h_fecal],
                  axis=0, ignore_index=True)
        .drop_duplicates(subset=["Sample_ID", "site"], keep="first")
    )

    # 2) Covariates pool — must contain Sample_ID; merge ONLY by Sample_ID
    covars = pd.read_csv(args.covars_pool, dtype=str)
    if "Sample_ID" not in covars.columns:
        raise RuntimeError("Covariates file must contain 'Sample_ID'.")
    covars["Sample_ID"] = covars["Sample_ID"].astype(str).map(normalize_sample_id)
    if "Sex" in covars.columns:
        covars["Sex"] = covars["Sex"].apply(_norm_sex)

    # Diagnostics: who matches / who doesn’t
    j = alpha_all.merge(covars[["Sample_ID"]].drop_duplicates(), on="Sample_ID", how="left", indicator=True)
    with open(outdir / "join_report.txt", "w") as fh:
        fh.write(f"Alpha rows (raw): {len(alpha_all)}\n")
        fh.write(f"Matched to covariates by Sample_ID: {int((j['_merge']=='both').sum())}\n")
        fh.write(f"Missing in covariates: {int((j['_merge']=='left_only').sum())}\n")
    j.loc[j["_merge"] == "left_only", ["Sample_ID", "site", "disease"]].to_csv(
        outdir / "alpha_unmatched_in_covariates.csv", index=False
    )
    ccheck = covars.merge(alpha_all[["Sample_ID"]].drop_duplicates(), on="Sample_ID", how="left", indicator=True)
    ccheck.loc[ccheck["_merge"] == "left_only", ["Sample_ID"]].to_csv(
        outdir / "covariates_unmatched_in_alpha.csv", index=False
    )

    # Save consolidated matched table (for other stages)
    alpha_with_covariates = alpha_all.merge(covars, on="Sample_ID", how="inner", suffixes=("", "_cov"))
    alpha_with_covariates.to_csv(outdir / "alpha_with_covariates.csv", index=False)

    # 3) Per-site — WITH covariates (matched)
    matched_oral, _ = run_site_with_cov("oral", alpha_all, covars, outdir, palettes)
    matched_fecal, _ = run_site_with_cov("fecal", alpha_all, covars, outdir, palettes)
    matched_all = pd.concat([matched_oral, matched_fecal], axis=0, ignore_index=True)

    # 4) Per-site — RAW (no covariates) with captions
    raw_oral = alpha_all[alpha_all["site"] == "oral"][["disease"] + METRICS].copy()
    box_and_scatter_panel("oral", raw_oral, outdir / "panel_oral_4up_raw.png",
                          palettes["oral"][0], palettes["oral"][1],
                          "RAW (no covariates)",
                          metric_annots=raw_annots(raw_oral))

    raw_fecal = alpha_all[alpha_all["site"] == "fecal"][["disease"] + METRICS].copy()
    box_and_scatter_panel("fecal", raw_fecal, outdir / "panel_fecal_4up_raw.png",
                          palettes["fecal"][0], palettes["fecal"][1],
                          "RAW (no covariates)",
                          metric_annots=raw_annots(raw_fecal))

    # 5) PPI (Crohn-only matched), Interaction, and cross-site tests
    ppi_models("oral", alpha_all, covars, outdir)
    ppi_models("fecal", alpha_all, covars, outdir)
    site_disease_interaction(alpha_all, covars, outdir)
    oral_vs_fecal_unpaired(alpha_all, outdir)

    # 6) Paired Wilcoxon (Crohn) + save CSV (as before)
    try:
        matched_df_file = pd.read_csv(args.pairs)
    except Exception:
        matched_df_file = None
    crohn = alpha_all[alpha_all["disease"] == 1.0].copy()
    oral = crohn[crohn["site"] == "oral"].set_index("Sample_ID")
    fecal = crohn[crohn["site"] == "fecal"].set_index("Sample_ID")

    pairs: List[Tuple[str, str]] = []
    if matched_df_file is not None and not matched_df_file.empty:
        oc = fc = None
        for c in matched_df_file.columns:
            lc = c.lower()
            if oc is None and ("oral" in lc or lc in {"oral_id", "sample_id_oral"}):
                oc = c
            if fc is None and ("fecal" in lc or "faecal" in lc or lc in {"fecal_id", "stool", "sample_id_fecal"}):
                fc = c
        if oc and fc:
            for _, r in matched_df_file[[oc, fc]].dropna().iterrows():
                pairs.append((normalize_sample_id(str(r[oc])), normalize_sample_id(str(r[fc]))))
    if not pairs:
        inter = oral.index.intersection(fecal.index)
        pairs = [(sid, sid) for sid in inter]

    rows = []
    for m in METRICS:
        xo, xf = [], []
        for o_id, f_id in pairs:
            vo = oral.get(m).get(o_id) if m in oral.columns and o_id in oral.index else np.nan
            vf = fecal.get(m).get(f_id) if m in fecal.columns and f_id in fecal.index else np.nan
            if pd.notna(vo) and pd.notna(vf):
                xo.append(float(vo))
                xf.append(float(vf))
        if len(xo) == 0:
            rows.append({"metric": m, "n_pairs": 0, "wilcoxon_stat": np.nan, "p_value": np.nan})
            continue
        stat, p = wilcoxon(xo, xf, zero_method="wilcox", alternative="two-sided",
                           correction=False, mode="auto")
        rows.append({"metric": m, "n_pairs": len(xo), "wilcoxon_stat": float(stat), "p_value": float(p)})
    pd.DataFrame(rows).assign(q_value=lambda d: fdr(d["p_value"].tolist())) \
        .to_csv(outdir / "paired_wilcoxon_summary.csv", index=False)

    # 7) NEW figures
    # 7a) Four-group panels (matched + raw)
    four_group_panel(matched_all, outdir / "panel_4groups_with_covariates.png",
                     palettes, "with covariates")
    four_group_panel(alpha_all[["Sample_ID","site","disease"]+METRICS].copy(),
                     outdir / "panel_4groups_raw.png",
                     palettes, "RAW (no covariates)")

    # 7b) Paired lines panel for Crohn (Oral ↔ Fecal)
    crohn_paired_lines_panel(alpha_all, pairs, outdir / "panel_paired_crohn_oral_vs_fecal.png", palettes)

    print(f"[INFO] Alpha analysis complete (level={args.level or 'n/a'}) → {outdir}")

if __name__ == "__main__":
    main()
