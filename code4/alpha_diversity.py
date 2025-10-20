#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha diversity — matched (with covariates) + raw (no covariates)
=================================================================

This script:
  1) Reads four abundance files (oral/fecal × Crohn/Healthy) and computes
     Shannon, Richness, Evenness per sample (ID normalization included).
  2) Concatenates to one alpha table (alpha_all) with columns:
       Sample_ID, site{"oral","fecal"}, disease{0,1}, Shannon, Richness, Evenness
  3) Reads pooled covariates (model_table_pooled.csv), normalizes Sample_ID
     the same way, and MERGEs **by Sample_ID only**.
     - WITH-covariate models/figures: run on matched set.
     - RAW panels: run on alpha_all (unmatched allowed).
  4) Outputs:
     - Per-site matched OLS(HC3) tables; per-metric FDR + disease-term FDR across metrics
     - PPI models in Crohn-only (per site)
     - Crohn-only OLS with severity/therapy covariates (if available & variable)
     - Disease×Site interaction model
     - Unpaired Oral vs Fecal (by group)
     - Paired Wilcoxon (Crohn) if pairs provided; fallback = intersection by Sample_ID
     - Figures: per-site 4-up (raw & matched), single-metric boxplots (matched),
                four-group panels (raw & matched), Crohn paired lines panel.

Assumptions about pooled covariates (model_table_pooled.csv):
  - disease is 0/1 (int), but we DO NOT trust it; we use disease from alpha_all
  - site in pooled is 0/1; we ignore it for merging (merge is by Sample_ID only)
  - Binary columns may be Int64/0/1/NA; we coerce on-the-fly before modeling
  - BMI is numeric (we keep as float; pooling might have 1 decimal)

CLI
---
  --oral-crohn PATH
  --fecal-crohn PATH
  --oral-healthy PATH
  --fecal-healthy PATH
  --covars-pool PATH                 # data/meta/model_table_pooled.csv
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

# ------------------------------------------------------------------ Config
METRICS = ["Shannon", "Richness", "Evenness"]

# Candidate covariates to include if present in pooled covariates
WISH_COVARS = [
    "Age", "Sex", "BMI", "Smoking", "Antibiotics_3m",
    "PPI_use", "Steroids_ongoing", "Immuno_ongoing", "Responder",
]

BIN_SYMS_TRUE  = {"1","true","t","yes","y"}
BIN_SYMS_FALSE = {"0","false","f","no","n"}

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------------------------------------------- Utilities
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def normalize_sample_id(s: str) -> str:
    """
    Normalize sample identifiers:
      - strip whitespace
      - drop replicate suffix like '.1' or '.2'
      - for purely numeric IDs, drop leading zeros (cast to int)
      - uppercase for stability
    """
    s = str(s).strip()
    s = pd.Series([s]).str.replace(r"\.\d+$", "", regex=True).iloc[0]
    if s.isdigit():
        try:
            s = str(int(s))
        except Exception:
            pass
    return s.upper()

def _norm_sex(x) -> Optional[float]:
    """Map F/female→0, M/male→1; else try numeric; return None if unknown."""
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

def _to_bin01(x) -> Optional[float]:
    """Robust YES/NO/1/0 to {0.0,1.0,None} (safe for pooled Int64 binaries)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in BIN_SYMS_TRUE:
        return 1.0
    if s in BIN_SYMS_FALSE:
        return 0.0
    try:
        v = float(s)
        if v in (0.0, 1.0):
            return v
    except Exception:
        pass
    return None

def safe_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce to numeric float Series; keep index and name; drop inf."""
    out = pd.to_numeric(s, errors="coerce").astype(float)
    out[~np.isfinite(out)] = np.nan
    return out

def safe_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric float; drop inf."""
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    X[~np.isfinite(X)] = np.nan
    return X

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
    """Benjamini-Hochberg FDR (NaNs kept)."""
    p = np.array([np.nan if v is None else v for v in p], dtype=float)
    mask = ~np.isnan(p)
    q = np.full_like(p, np.nan, dtype=float)
    if mask.sum():
        q[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return q

def format_annot_pairs(n_pairs: int, p: float | None = None, q: float | None = None) -> str:
    parts = [f"n(pairs)={n_pairs}"]
    if p is not None and not pd.isna(p):
        parts.append(f"p={p:.2e}")
    if q is not None and not pd.isna(q):
        parts.append(f"q={q:.2e}")
    return " | ".join(parts)

def format_annot_2x(nH: int, nC: int, p: float | None = None,
                    q: float | None = None, cd: float | None = None) -> str:
    parts = [f"n(Healthy)={nH}", f"n(Crohn)={nC}"]
    if p is not None and not pd.isna(p):
        parts.append(f"p={p:.2e}")
    if q is not None and not pd.isna(q):
        parts.append(f"q={q:.2e}")
    if cd is not None and not pd.isna(cd):
        parts.append(f"δ={round(float(cd), 2)}")
    return " | ".join(parts)

# --------------------------- Small numeric helpers ---------------------------
def _to_bin01(x) -> Optional[int]:
    """Map common truthy/falsey strings/numbers → 1/0 (int); unknown → NaN."""
    if x is None:
        return np.nan
    s = str(x).strip().lower()
    if s in {"1","true","t","yes","y"}:
        return 1
    if s in {"0","false","f","no","n"}:
        return 0
    try:
        v = float(s)
        if np.isfinite(v) and v in (0.0,1.0):
            return int(v)
    except Exception:
        pass
    return np.nan

def safe_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce to float Series; keep index/ name."""
    out = pd.to_numeric(s, errors="coerce").astype(float)
    return out

def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to float; keep column names and index."""
    out = df.apply(pd.to_numeric, errors="coerce").astype(float)
    return out
# ----------------------------------------------------------- Colors loader
def load_colors(path: str | None) -> dict:
    """
    Load palette from YAML.
    Supports:
      legacy: {colors: {Crohn-Oral, Healthy-Oral, Crohn-Fecal, Healthy-Fecal}}
      group/synonyms (preferred).
    Returns:
      palettes = {
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

    if "colors" in cfg and isinstance(cfg["colors"], dict):
        c = cfg["colors"]
        return {
            "oral":  [c.get("Crohn-Oral",  fallback["oral"][0]),  c.get("Healthy-Oral",  fallback["oral"][1])],
            "fecal": [c.get("Crohn-Fecal", fallback["fecal"][0]), c.get("Healthy-Fecal", fallback["fecal"][1])],
        }

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


# --------------------------------------- Alpha computation from abundance
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
      alpha_df: Sample_ID, Shannon, Richness, Evenness
      idmap: original_id → Sample_ID
    """
    df = pd.read_csv(path)
    mat, _ = to_sample_by_taxa(df)
    # normalize IDs
    norm_ids = [normalize_sample_id(x) for x in mat.index.astype(str)]
    idmap = pd.DataFrame({"original_id": mat.index.astype(str), "Sample_ID": norm_ids})
    mat.index = norm_ids
    # proportions
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
    a, idmap = compute_alpha_from_abundance(abund_path)
    a["site"], a["disease"] = site, int(disease)
    return a[["Sample_ID", "site", "disease", "Shannon", "Richness", "Evenness"]], \
           idmap.assign(site=site, disease=int(disease))


# --------------------------------------------------------------- Plotting
def disease_legend_handles(pal: Dict[str, str]):
    return [
        mpatches.Patch(facecolor=pal["Healthy"], label="Healthy"),
        mpatches.Patch(facecolor=pal["Crohn"],   label="Crohn"),
    ]

def placeholder_plot(path: Path, title_txt: str) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(5.8, 4.2)); plt.axis("off")
    plt.text(0.5, 0.5, title_txt, ha="center", va="center"); plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()

def box_and_scatter_panel(site_name: str,
                          df: pd.DataFrame,
                          out_png: Path,
                          crohn_color: str, healthy_color: str,
                          title_suffix: str,
                          metric_annots: Optional[Dict[str, str]] = None):
    if df.empty:
        placeholder_plot(out_png, f"{site_name.capitalize()} — Alpha (4-up {title_suffix})\n(No data)")
        return

    disease_labels = {0: "Healthy", 1: "Crohn"}
    pal = {"Crohn": crohn_color, "Healthy": healthy_color}
    metric_annots = metric_annots or {}

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))

    for ax, met in zip(axes.flat[:3], METRICS):
        dd = df[["disease", met]].dropna()
        if dd.empty or dd["disease"].nunique() < 2:
            ax.axis("off"); ax.text(0.5, 0.5, f"No data: {met}", ha="center"); continue
        dd = dd.rename(columns={met: "value"})
        dd["group"] = dd["disease"].map(disease_labels)

        sns.boxplot(data=dd, x="group", y="value", order=["Healthy", "Crohn"],
                    hue="group", palette=pal, dodge=False, legend=False, ax=ax)
        sns.stripplot(data=dd, x="group", y="value", order=["Healthy", "Crohn"],
                      color="black", alpha=0.6, jitter=0.15, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel(met); ax.set_title(met, fontsize=11)

        nH = int((dd["group"] == "Healthy").sum())
        nC = int((dd["group"] == "Crohn").sum())
        cap = metric_annots.get(met, format_annot_2x(nH, nC))
        ax.text(0.5, -0.25, cap, transform=ax.transAxes, ha="center", va="top", fontsize=9)

    ax = axes.flat[3]
    sub = df[["Shannon", "Richness", "disease"]].dropna()
    if sub.empty:
        ax.axis("off"); ax.text(0.5, 0.5, "No data: scatter", ha="center")
    else:
        sub["group"] = sub["disease"].map(disease_labels)
        for g, ddg in sub.groupby("group"):
            ax.scatter(ddg["Shannon"], ddg["Richness"], label=g, s=28, alpha=0.85,
                       c=pal[g])
        sns.regplot(x="Shannon", y="Richness", data=sub, scatter=False, ci=95, ax=ax)
        ax.set_xlabel("Shannon"); ax.set_ylabel("Richness"); ax.set_title("Shannon vs Richness")
        ax.legend(frameon=False, loc="best")
        nH = int((sub["group"] == "Healthy").sum()); nC = int((sub["group"] == "Crohn").sum())
        ax.text(0.5, -0.25, format_annot_2x(nH, nC), transform=ax.transAxes, ha="center", va="top", fontsize=9)

    handles = disease_legend_handles(pal); labels = ["Healthy", "Crohn"]
    fig.legend(handles=handles, labels=labels, frameon=False, loc="upper right")
    plt.suptitle(f"{site_name.capitalize()} — Alpha (4-up {title_suffix})", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300); plt.close()

def four_group_panel(df_all: pd.DataFrame, out_png: Path, palettes: Dict[str, List[str]], title_suffix: str):
    if df_all.empty:
        placeholder_plot(out_png, f"Alpha (4-group {title_suffix})\n(No data)"); return

    lab_map = {0: "Healthy", 1: "Crohn"}
    df = df_all.copy()
    df["disease_label"] = df["disease"].map(lab_map)
    df["group4"] = df["disease_label"] + "-" + df["site"].str.capitalize()
    order4 = ["Healthy-Oral", "Crohn-Oral", "Healthy-Fecal", "Crohn-Fecal"]
    pal4 = {
        "Healthy-Oral": palettes["oral"][1], "Crohn-Oral": palettes["oral"][0],
        "Healthy-Fecal": palettes["fecal"][1], "Crohn-Fecal": palettes["fecal"][0],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))

    for ax, met in zip(axes.flat[:3], METRICS):
        dd = df[["group4", "site", "disease", met]].dropna()
        if dd.empty or dd["group4"].nunique() < 2:
            ax.axis("off"); ax.text(0.5, 0.5, f"No data: {met}", ha="center"); continue

        sns.boxplot(data=dd, x="group4", y=met, order=order4, palette=pal4, ax=ax)
        sns.stripplot(data=dd, x="group4", y=met, order=order4,
                      color="black", alpha=0.5, jitter=0.15, ax=ax)
        ax.set_xlabel(""); ax.set_title(met); ax.tick_params(axis='x', rotation=20)

        parts = []
        for site_name in ["oral", "fecal"]:
            sub = dd[dd["site"] == site_name]
            nH = int((sub["disease"] == 0).sum()); nC = int((sub["disease"] == 1).sum())
            if sub["disease"].nunique() < 2:
                parts.append(f"{site_name.capitalize()}(nH={nH}, nC={nC})"); continue
            x = sub.loc[sub["disease"] == 1, met].values
            y = sub.loc[sub["disease"] == 0, met].values
            try:
                p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            except Exception:
                p = np.nan
            cd = cliffs_delta(x, y)
            parts.append((site_name, nH, nC, p, cd))
        pvals = [t[3] for t in parts if isinstance(t, tuple)]
        qvals = list(fdr(pvals)) if pvals else []
        qi = 0; chunks = []
        for t in parts:
            if isinstance(t, tuple):
                site_name, nH, nC, p, cd = t
                qv = qvals[qi] if qi < len(qvals) else np.nan; qi += 1
                chunks.append(f"{site_name.capitalize()}(nH={nH}, nC={nC}, p={p:.2e}, q={qv:.2e}, δ={np.nan if pd.isna(cd) else round(float(cd),2)})")
            else:
                chunks.append(t)
        ax.text(0.5, -0.28, " | ".join(chunks), transform=ax.transAxes, ha="center", va="top", fontsize=9)

    ax = axes.flat[3]
    sub = df[["Shannon", "Richness", "group4"]].dropna()
    if sub.empty:
        ax.axis("off"); ax.text(0.5, 0.5, "No data: scatter", ha="center")
    else:
        for g, gg in sub.groupby("group4"):
            ax.scatter(gg["Shannon"], gg["Richness"], s=26, alpha=0.85,
                       c=pal4.get(g, "#777777"), label=g)
        sns.regplot(x="Shannon", y="Richness", data=sub, scatter=False, ci=95, ax=ax)
        ax.set_xlabel("Shannon"); ax.set_ylabel("Richness"); ax.set_title("Shannon vs Richness (4 groups)")
        ax.legend(frameon=False, loc="best")

    plt.suptitle(f"Alpha (4 groups — {title_suffix})", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300); plt.close()

def crohn_paired_lines_panel(alpha_all: pd.DataFrame,
                             pairs: List[Tuple[str, str]],
                             out_png: Path,
                             palettes: Dict[str, List[str]]):
    crohn = alpha_all[alpha_all["disease"] == 1].copy()
    oral  = crohn[crohn["site"] == "oral"].set_index("Sample_ID")
    fecal = crohn[crohn["site"] == "fecal"].set_index("Sample_ID")

    pairs = pairs or []
    if not pairs:
        inter = oral.index.intersection(fecal.index)
        pairs = [(sid, sid) for sid in inter]
    if not pairs:
        placeholder_plot(out_png, "Crohn Oral↔Fecal Paired (No pairs)"); return

    wilco_p = []; series_by_metric = {}
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
                stat, p = wilcoxon(xo, xf, zero_method="wilcox", alternative="two-sided",
                                   correction=False, mode="auto")
            except Exception:
                p = np.nan
        else:
            p = np.nan
        wilco_p.append(p)
    wilco_q = fdr(wilco_p) if any([not pd.isna(p) for p in wilco_p]) else np.array([np.nan]*len(METRICS))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=False)
    for i, m in enumerate(METRICS):
        ax = axes[i]
        xo, xf = series_by_metric[m]
        if len(xo) == 0:
            ax.axis("off"); ax.text(0.5, 0.5, f"No pairs: {m}", ha="center"); continue
        for a, b in zip(xo, xf):
            ax.plot([0, 1], [a, b], alpha=0.5, lw=1.0, color="#7f7f7f")
        ax.scatter([0]*len(xo), xo, s=24, alpha=0.9, c=palettes["oral"][0], label="Crohn-Oral")
        ax.scatter([1]*len(xf), xf, s=24, alpha=0.9, c=palettes["fecal"][0], label="Crohn-Fecal")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Oral", "Fecal"])
        ax.set_title(m); ax.set_xlim(-0.3, 1.3)
        p = wilco_p[i]; q = wilco_q[i] if i < len(wilco_q) else np.nan
        ax.text(0.5, -0.22, format_annot_pairs(len(xo), p=p, q=q),
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
    handles = [mpatches.Patch(color=palettes["oral"][0], label="Crohn-Oral"),
               mpatches.Patch(color=palettes["fecal"][0], label="Crohn-Fecal")]
    fig.legend(handles=handles, frameon=False, loc="upper right")
    plt.suptitle("Crohn — Oral vs Fecal (Paired lines)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300); plt.close()


# -------------------------------------------------------------- Summaries
def summarize_groups(df_site: pd.DataFrame, site_name: str) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        if m not in df_site.columns:
            continue
        for g in [0, 1]:
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


# -------------------------------------------------------------- Modeling
def _prep_design(dd: pd.DataFrame, ycol: str, xcols: List[str]) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Create y (float Series) and X (float DataFrame with constant),
    drop rows with any NaN/Inf; keep index/colnames for statsmodels.
    """
    y = safe_numeric_series(dd[ycol])
    X = add_constant(dd[xcols], has_constant="add")
    X = safe_numeric_df(X)
    mask = np.isfinite(y.values) & np.all(np.isfinite(X.values), axis=1)
    y = y.loc[mask]
    X = X.loc[mask]
    return y, X

def run_site_with_cov(site_name: str,
                      alpha_all: pd.DataFrame,
                      covars_df: pd.DataFrame,
                      outdir: Path,
                      palettes: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Matched analysis (WITH covariates):
      - Merge by Sample_ID (site/disease from alpha_all)
      - MWU on matched set (p, q, Cliff's δ)
      - OLS (HC3) per metric with chosen covariates + disease
      - Save models and group summaries
      - Draw 4-up panel and single-metric boxplots with captions
    """
    site_mask = alpha_all["site"] == site_name
    d_raw = alpha_all.loc[site_mask].copy()
    d_cov = d_raw.merge(covars_df, on="Sample_ID", how="inner", suffixes=("", "_cov"))

    # Matched MWU
    wilx_rows = []
    for m in METRICS:
        dd = d_cov[["disease", m]].dropna()
        if dd.empty or dd["disease"].nunique() < 2:
            wilx_rows.append({"metric": m, "p_wilcox": np.nan, "cliffs_delta": np.nan})
            continue
        x = dd.loc[dd["disease"] == 1, m].values
        y = dd.loc[dd["disease"] == 0, m].values
        try:
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        cd = cliffs_delta(x, y)
        wilx_rows.append({"metric": m, "p_wilcox": p, "cliffs_delta": cd})
    wilx_df = pd.DataFrame(wilx_rows)
    wilx_df["q_wilcox"] = fdr(wilx_df["p_wilcox"].tolist())

    # Captions
    def _annot_for(met: str) -> str:
        dd = d_cov[["disease", met]].dropna()
        nH = int((dd["disease"] == 0).sum()); nC = int((dd["disease"] == 1).sum())
        row = wilx_df[wilx_df["metric"] == met]
        p = q = cd = None
        if len(row):
            p = row["p_wilcox"].iloc[0]; q = row["q_wilcox"].iloc[0]; cd = row["cliffs_delta"].iloc[0]
        return format_annot_2x(nH, nC, p=p, q=q, cd=cd)
    annots = {m: _annot_for(m) for m in METRICS}

    # OLS with covariates
    covar_pool = [c for c in WISH_COVARS if c in d_cov.columns]
    # Coerce covariates: binary via _to_bin01, Sex via _norm_sex, others numeric
    for c in covar_pool + ["disease"]:
        if c == "Sex":
            d_cov[c] = d_cov[c].apply(_norm_sex)
        elif c in {"Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing","Responder"}:
            d_cov[c] = d_cov[c].apply(_to_bin01)
        else:
            d_cov[c] = pd.to_numeric(d_cov[c], errors="coerce")

    rows = []
    for m in METRICS:
        if m not in d_cov.columns:
            rows.append({"metric": m, "term": "(missing_metric)", "estimate": np.nan,
                         "std.error": np.nan, "statistic": np.nan, "p.value": np.nan})
            continue
        cols = ["disease"] + covar_pool
        dd = d_cov[[m] + cols].copy().dropna(subset=[m])
        if dd["disease"].nunique() < 2:
            rows.append({"metric": m, "term": "(skipped_small_n)", "estimate": np.nan,
                         "std.error": np.nan, "statistic": np.nan, "p.value": np.nan})
            continue
        y, X = _prep_design(dd, m, cols)
        if len(y) < 6 or X.shape[1] < 2:
            rows.append({"metric": m, "term": "(skipped_small_n)", "estimate": np.nan,
                         "std.error": np.nan, "statistic": np.nan, "p.value": np.nan})
            continue
        fit = OLS(y, X).fit(cov_type="HC3")  # pandas in → pandas out
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
    box_and_scatter_panel_always_p(
        site_name=site_name,
        df=d_cov[["disease"] + METRICS].copy(),
        out_png=panel_png,
        crohn_color=colors[0],
        healthy_color=colors[1],
        title_suffix="with covariates"
    )



    # Single metric boxplots (matched) with caption
    disease_labels = {0: "Healthy", 1: "Crohn"}
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
    df = df[(df["site"] == site_name) & (df["disease"] == 1)].copy()

    covars = [c for c in WISH_COVARS if c in df.columns]
    # Coercions
    for c in covars:
        if c == "Sex":
            df[c] = df[c].apply(_norm_sex)
        elif c in {"Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing","Responder"}:
            df[c] = df[c].apply(_to_bin01)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for m in METRICS:
        if m not in df.columns:
            continue
        rhs = [v for v in covars if v != "PPI_use"]
        use = ["PPI_use"] + rhs
        dd = df[[m] + use].dropna()
        y, X = _prep_design(dd, m, use)
        if len(y) < 6 or X.shape[1] < 2 or dd["PPI_use"].nunique() < 2:
            rows.append({"metric": m, "term": "(skipped_small_n)",
                        "estimate": np.nan, "std.error": np.nan,
                        "statistic": np.nan, "p.value": np.nan})
            continue
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

def crohn_only_models(site_name: str,
                      alpha_all: pd.DataFrame,
                      covars_df: pd.DataFrame,
                      outdir: Path) -> pd.DataFrame:
    """OLS (HC3) inside Crohn, per site, with severity/therapy covariates if available."""
    df = alpha_all.merge(covars_df, on="Sample_ID", how="inner", suffixes=("", "_cov"))
    df = df[(df["site"] == site_name) & (df["disease"] == 1)].copy()

    base_covs = [c for c in ["Age","Sex","BMI","Smoking","Antibiotics_3m",
                             "PPI_use","Steroids_ongoing","Immuno_ongoing"] if c in df.columns]
    crohn_covs = [c for c in ["Calprotectin_baseline_raw","HBI_baseline_raw","Disease_duration_years",
                              "Perianal_disease","Any_resection","Anti_TNF_current","5ASA_current"]
                  if c in df.columns]

    use_covs = [c for c in (base_covs + crohn_covs)
                if df[c].notna().sum() >= 6 and df[c].nunique(dropna=True) >= 2]
    # Coercions
    for c in use_covs:
        if c == "Sex":
            df[c] = df[c].apply(_norm_sex)
        elif c in {"Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing",
                   "Perianal_disease","Any_resection","Anti_TNF_current","5ASA_current"}:
            df[c] = df[c].apply(_to_bin01)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for m in METRICS:
        if m not in df.columns:
            continue
        cols = list(use_covs)
        dd = df[[m] + cols].dropna()
        # نیاز به تنوع در کوواریت‌ها
        if len(dd) < 8 or any(dd[c].nunique() < 2 for c in cols if c != "Sex"):
            rows.append({"metric": m, "term": "(skipped_small_n)", "estimate": np.nan,
                         "std.error": np.nan, "statistic": np.nan, "p.value": np.nan})
            continue
        y, X = _prep_design(dd, m, cols)
        if len(y) < 8 or X.shape[1] < 2:
            rows.append({"metric": m, "term": "(skipped_small_n)", "estimate": np.nan,
                         "std.error": np.nan, "statistic": np.nan, "p.value": np.nan})
            continue
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
    ensure_dir(outdir)
    out.to_csv(outdir / f"alpha_crohn_only_models_{site_name}.csv", index=False)
    return out

def site_disease_interaction(alpha_all: pd.DataFrame,
                             covars_df: pd.DataFrame,
                             outdir: Path) -> pd.DataFrame:
    """Global disease×site interaction model on matched set."""
    df = alpha_all.merge(covars_df, on="Sample_ID", how="inner", suffixes=("", "_cov")).copy()
    if "site_bin" in df.columns:
        df["site_bin"] = pd.to_numeric(df["site_bin"], errors="coerce")
    else:
        df["site_bin"] = df["site"].map({"fecal": 0, "oral": 1})

    covar_pool = [c for c in WISH_COVARS if c in df.columns]
    for c in covar_pool + ["disease", "site_bin"]:
        if c == "Sex":
            df[c] = df[c].apply(_norm_sex)
        elif c in {"Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing","Responder"}:
            df[c] = df[c].apply(_to_bin01)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for m in METRICS:
        if m not in df.columns:
            continue
        cols = ["disease", "site_bin"] + covar_pool
        dd = df[[m] + cols].dropna()
        if len(dd) < 8 or dd["disease"].nunique() < 2 or dd["site_bin"].nunique() < 2:
            rows.append({"metric": m, "term": "(skipped_small_n)", "estimate": np.nan,
                         "std.error": np.nan, "statistic": np.nan, "p.value": np.nan})
            continue
        X = dd[["disease", "site_bin"]].copy()
        X["disease:site_bin"] = dd["disease"] * dd["site_bin"]
        for c in covar_pool:
            X[c] = dd[c].values
        y, X = _prep_design(pd.concat([dd[[m]], X], axis=1), m, list(X.columns))
        if len(y) < 8 or X.shape[1] < 2:
            rows.append({"metric": m, "term": "(skipped_small_n)", "estimate": np.nan,
                         "std.error": np.nan, "statistic": np.nan, "p.value": np.nan})
            continue
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
    for status, lab in [(1, "Crohn"), (0, "Healthy")]:
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
        nH = int((dd["disease"] == 0).sum())
        nC = int((dd["disease"] == 1).sum())
        if dd["disease"].nunique() < 2:
            tmp_store[m] = (nH, nC, np.nan, np.nan)
            continue
        x = dd.loc[dd["disease"] == 1, m].values
        y = dd.loc[dd["disease"] == 0, m].values
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

def compute_site_mwu_pvals(df_site: pd.DataFrame) -> Dict[str, float]:
    """
    Input: df_site with columns ['disease'] + METRICS (0=Healthy, 1=Crohn).
    Returns: {metric -> raw p-value from Mann–Whitney (two-sided)}.
    If one of the groups is missing or empty → np.nan.
    """
    pvals = {}
    for m in METRICS:
        if m not in df_site.columns: 
            pvals[m] = np.nan; continue
        dd = df_site[["disease", m]].dropna()
        if dd.empty or dd["disease"].nunique() < 2:
            pvals[m] = np.nan; continue
        x = dd.loc[dd["disease"] == 1, m].values  # Crohn
        y = dd.loc[dd["disease"] == 0, m].values  # Healthy
        if len(x) == 0 or len(y) == 0:
            pvals[m] = np.nan; continue
        try:
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        pvals[m] = float(p) if p is not None else np.nan
    return pvals


def _draw_sig_bracket_two_groups(ax, x1, x2, y, pval, alpha_sig=0.05, h=0.03, lw=1.0):
    """
    Draws a horizontal bracket between positions x1 and x2 at height y, with p text above it,
    ONLY if pval < alpha_sig. Otherwise does nothing.
    x1=0 (Healthy), x2=1 (Crohn) for our two box positions.
    h: vertical offset as fraction of current y-range.
    """
    if not (isinstance(pval, (int, float)) and np.isfinite(pval) and pval < alpha_sig):
        return  # not significant → no annotation
    yl = ax.get_ylim()
    y_abs = y + (yl[1]-yl[0]) * h
    ax.plot([x1, x1, x2, x2], [y, y_abs, y_abs, y], lw=lw, c="black")
    txt = f"p={pval:.2e}"
    ax.text((x1 + x2)/2.0, y_abs, txt, ha="center", va="bottom")

def compute_site_mwu_stats(df_site: pd.DataFrame):
    """
    برای هر متریک داخل یک سایت:
      p: Mann–Whitney U (two-sided)
      q: FDR BH بین متریک‌هایی که p معتبر دارند
      δ: Cliff's delta
      caption: 'n(Healthy)=.. | n(Crohn)=.. | q=.. | δ=..'  (توجه: p بالای باکس می‌رود)
    خروجی: (pvals:dict, qvals:dict, deltas:dict, captions:dict)
    """
    pvals, deltas = {}, {}
    counts = {}
    val_mets, val_ps = [], []

    for m in METRICS:
        pvals[m] = np.nan
        deltas[m] = np.nan
        counts[m] = (0, 0)
        if m not in df_site.columns:
            continue

        dd = df_site[["disease", m]].dropna()
        nH = int((dd["disease"] == 0).sum())
        nC = int((dd["disease"] == 1).sum())
        counts[m] = (nH, nC)
        if dd.empty or dd["disease"].nunique() < 2 or nH == 0 or nC == 0:
            continue

        x = dd.loc[dd["disease"] == 1, m].values
        y = dd.loc[dd["disease"] == 0, m].values
        try:
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        pvals[m] = float(p) if p is not None else np.nan
        deltas[m] = float(cliffs_delta(x, y))
        if np.isfinite(pvals[m]): val_mets.append(m); val_ps.append(pvals[m])

    # FDR فقط روی متریک‌هایی که p معتبر دارند
    qvals = {m: np.nan for m in METRICS}
    if len(val_ps):
        q_vec = fdr(val_ps)
        for i, m in enumerate(val_mets):
            qvals[m] = float(q_vec[i])

    captions = {}
    for m in METRICS:
        nH, nC = counts[m]
        q = qvals[m]; d = deltas[m]
        parts = [f"n(Healthy)={nH}", f"n(Crohn)={nC}"]
        if np.isfinite(q): parts.append(f"q={q:.2e}")
        if np.isfinite(d): parts.append(f"δ={round(d, 2)}")
        captions[m] = " | ".join(parts)
    return pvals, qvals, deltas, captions

def _draw_bracket_two_groups_always(ax, x1, x2, y_data_max, pval,
                                    pad_frac=0.04, lw=1.0, txt_offset=0.6):
    """
    همیشه براکت و p را می‌نویسد (حتی اگر معنی‌دار نباشد).
    x1=0 (Healthy), x2=1 (Crohn)
    y_data_max: بیشینه‌ی مقدار داده برای تعیین ارتفاع پایه
    """
    if not (isinstance(pval, (int, float)) and np.isfinite(pval)):
        return
    yl = ax.get_ylim()
    span = (yl[1] - yl[0])
    base = max(y_data_max, yl[1] - 0.85 * span)  # کمی پایین‌تر اگر محور خیلی بالا باشد
    lift = span * pad_frac
    y0 = base + lift
    ax.plot([x1, x1, x2, x2], [y0, y0 + lift, y0 + lift, y0], lw=lw, c="black")
    ax.text((x1 + x2) / 2.0, y0 + lift * (1.0 + txt_offset), f"p={pval:.2e}",
            ha="center", va="bottom", fontsize=9, color="black")

def box_and_scatter_panel_always_p(site_name: str,
                                   df: pd.DataFrame,
                                   out_png: Path,
                                   crohn_color: str, healthy_color: str,
                                   title_suffix: str):
    """
    پنل 4تایی: بالای هر باکس براکت + p خام؛
    زیر هر باکس کپشن شامل nH/nC + q + δ.
    """
    if df.empty:
        placeholder_plot(out_png, f"{site_name.capitalize()} — Alpha (4-up {title_suffix})\n(No data)")
        return

    disease_labels = {0: "Healthy", 1: "Crohn"}
    pal = {"Crohn": crohn_color, "Healthy": healthy_color}

    # p/q/δ و کپشن‌ها
    pvals, qvals, deltas, captions = compute_site_mwu_stats(df)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))

    # سه باکس‌پلات
    for ax, met in zip(axes.flat[:3], METRICS):
        dd = df[["disease", met]].dropna()
        if dd.empty or dd["disease"].nunique() < 2:
            ax.axis("off"); ax.text(0.5, 0.5, f"No data: {met}", ha="center"); continue

        dd = dd.rename(columns={met: "value"})
        dd["group"] = dd["disease"].map(disease_labels)

        sns.boxplot(data=dd, x="group", y="value", order=["Healthy","Crohn"],
                    hue="group", palette=pal, dodge=False, legend=False, ax=ax)
        sns.stripplot(data=dd, x="group", y="value", order=["Healthy","Crohn"],
                      color="black", alpha=0.6, jitter=0.15, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel(met); ax.set_title(met, fontsize=11)

        # براکت + p (همیشه)
        y_max = float(dd["value"].max())
        _draw_bracket_two_groups_always(ax, x1=0, x2=1,
                                        y_data_max=y_max,
                                        pval=pvals.get(met, np.nan))

        # کپشن پایین (nH/nC + q + δ)
        ax.text(0.5, -0.25, captions.get(met, ""),
                transform=ax.transAxes, ha="center", va="top", fontsize=9)

    # اسکتر
    ax = axes.flat[3]
    sub = df[["Shannon", "Richness", "disease"]].dropna()
    if sub.empty:
        ax.axis("off"); ax.text(0.5, 0.5, "No data: scatter", ha="center")
    else:
        sub["group"] = sub["disease"].map(disease_labels)
        for g, ddg in sub.groupby("group"):
            ax.scatter(ddg["Shannon"], ddg["Richness"], label=g, s=28, alpha=0.85, c=pal[g])
        sns.regplot(x="Shannon", y="Richness", data=sub, scatter=False, ci=95, ax=ax)
        ax.set_xlabel("Shannon"); ax.set_ylabel("Richness"); ax.set_title("Shannon vs Richness")
        ax.legend(frameon=False, loc="best")

    handles = [mpatches.Patch(facecolor=pal["Healthy"], label="Healthy"),
               mpatches.Patch(facecolor=pal["Crohn"],   label="Crohn")]
    fig.legend(handles=handles, labels=["Healthy","Crohn"], frameon=False, loc="upper right")

    plt.suptitle(f"{site_name.capitalize()} — Alpha (4-up {title_suffix})", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300); plt.close()

def write_alpha_test_audit(df_oral_raw, df_fecal_raw, outdir: Path):
    for site_name, df_site in [("oral", df_oral_raw), ("fecal", df_fecal_raw)]:
        p, q, d, caps = compute_site_mwu_stats(df_site)
        rows = []
        for m in METRICS:
            dd = df_site[["disease", m]].dropna()
            nH = int((dd["disease"]==0).sum())
            nC = int((dd["disease"]==1).sum())
            rows.append({
                "site": site_name, "metric": m, "n_healthy": nH, "n_crohn": nC,
                "p_raw": p.get(m, np.nan), "q_fdr": q.get(m, np.nan),
                "cliffs_delta": d.get(m, np.nan)
            })
        pd.DataFrame(rows).to_csv(outdir / f"alpha_raw_mwu_audit_{site_name}.csv", index=False)

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
    alpha_oral, idmap_oo   = build_alpha_table(args.oral_crohn,   "oral",  1)
    alpha_fecal, idmap_of  = build_alpha_table(args.fecal_crohn,  "fecal", 1)
    alpha_h_oral, idmap_ho = build_alpha_table(args.oral_healthy, "oral",  0)
    alpha_h_fecal, idmap_hf= build_alpha_table(args.fecal_healthy,"fecal", 0)

    iddir = outdir / "idmaps"
    ensure_dir(iddir)
    idmap_oo.to_csv(iddir / "idmap_oral_crohn.tsv",   sep="\t", index=False)
    idmap_of.to_csv(iddir / "idmap_fecal_crohn.tsv",  sep="\t", index=False)
    idmap_ho.to_csv(iddir / "idmap_oral_healthy.tsv", sep="\t", index=False)
    idmap_hf.to_csv(iddir / "idmap_fecal_healthy.tsv",sep="\t", index=False)

    alpha_all = (
        pd.concat([alpha_oral, alpha_fecal, alpha_h_oral, alpha_h_fecal],
                  axis=0, ignore_index=True)
        .drop_duplicates(subset=["Sample_ID", "site"], keep="first")
    )
    alpha_all["disease"] = pd.to_numeric(alpha_all["disease"], errors="coerce").fillna(0).astype(int)

    # 2) Covariates pool — must contain Sample_ID; merge ONLY by Sample_ID
    covars = pd.read_csv(args.covars_pool, dtype=str)

    bin_cols = ["Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing",
                "Perianal_disease","Any_resection","Anti_TNF_current","5ASA_current","Responder"]
    for c in bin_cols:
        if c in covars.columns:
            covars[c] = covars[c].map(
                lambda x: 1 if str(x).strip().lower() in {"1","true","t","yes","y"}
                else 0 if str(x).strip().lower() in {"0","false","f","no","n"}
                else pd.NA
            ).astype("Int64")

    if "BMI" in covars.columns:
        covars["BMI"] = pd.to_numeric(covars["BMI"], errors="coerce")

    if "Sample_ID" not in covars.columns:
        raise RuntimeError("Covariates file must contain 'Sample_ID'.")
    covars["Sample_ID"] = covars["Sample_ID"].astype(str).map(normalize_sample_id)
    if "Sex" in covars.columns:
        covars["Sex"] = covars["Sex"].apply(_norm_sex)

    # Diagnostics: who matches / who doesn’t
    j = alpha_all.merge(covars[["Sample_ID"]].drop_duplicates(),
                        on="Sample_ID", how="left", indicator=True)
    with open(outdir / "join_report.txt", "w") as fh:
        fh.write(f"Alpha rows (raw): {len(alpha_all)}\n")
        fh.write(f"Matched to covariates by Sample_ID: {int((j['_merge']=='both').sum())}\n")
        fh.write(f"Missing in covariates: {int((j['_merge']=='left_only').sum())}\n")

    j.loc[j["_merge"] == "left_only", ["Sample_ID", "site", "disease"]].to_csv(
        outdir / "alpha_unmatched_in_covariates.csv", index=False
    )
    ccheck = covars.merge(alpha_all[["Sample_ID"]].drop_duplicates(),
                          on="Sample_ID", how="left", indicator=True)
    ccheck.loc[ccheck["_merge"] == "left_only", ["Sample_ID"]].to_csv(
        outdir / "covariates_unmatched_in_alpha.csv", index=False
    )

    # Save consolidated matched table (for other stages)
    alpha_with_covariates = alpha_all.merge(covars, on="Sample_ID", how="inner", suffixes=("", "_cov"))
    alpha_with_covariates.to_csv(outdir / "alpha_with_covariates.csv", index=False)

    # 3) Per-site — WITH covariates (matched)  → (توجه: داخل run_site_with_cov هم باید از پنل جدید استفاده شود)
    matched_oral, _  = run_site_with_cov("oral",  alpha_all, covars, outdir, palettes)
    matched_fecal, _ = run_site_with_cov("fecal", alpha_all, covars, outdir, palettes)
    matched_all = pd.concat([matched_oral, matched_fecal], axis=0, ignore_index=True)

    # 4) Per-site — RAW (no covariates)  → براکت + p بالا؛ q و δ پایین
    raw_oral  = alpha_all[alpha_all["site"]  == "oral"][["disease"] + METRICS].copy()
    raw_fecal = alpha_all[alpha_all["site"] == "fecal"][["disease"] + METRICS].copy()

    box_and_scatter_panel_always_p(
        site_name="oral",
        df=raw_oral,
        out_png=outdir / "panel_oral_4up_raw.png",
        crohn_color=palettes["oral"][0],
        healthy_color=palettes["oral"][1],
        title_suffix="RAW (no covariates)"
    )
    box_and_scatter_panel_always_p(
        site_name="fecal",
        df=raw_fecal,
        out_png=outdir / "panel_fecal_4up_raw.png",
        crohn_color=palettes["fecal"][0],
        healthy_color=palettes["fecal"][1],
        title_suffix="RAW (no covariates)"
    )

    # 5) PPI (Crohn-only matched), Interaction, و cross-site
    ppi_models("oral",  alpha_all, covars, outdir)
    ppi_models("fecal", alpha_all, covars, outdir)
    crohn_only_models("oral",  alpha_all, covars, outdir)
    crohn_only_models("fecal", alpha_all, covars, outdir)

    site_disease_interaction(alpha_all, covars, outdir)
    oral_vs_fecal_unpaired(alpha_all, outdir)

    # 6) Paired Wilcoxon (Crohn)
    try:
        matched_df_file = pd.read_csv(args.pairs)
    except Exception:
        matched_df_file = None

    crohn = alpha_all[alpha_all["disease"] == 1].copy()
    oral  = crohn[crohn["site"]  == "oral"].set_index("Sample_ID")
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
            vo = oral.get(m).get(o_id)  if (m in oral.columns and  o_id in oral.index) else np.nan
            vf = fecal.get(m).get(f_id) if (m in fecal.columns and f_id in fecal.index) else np.nan
            if pd.notna(vo) and pd.notna(vf):
                xo.append(float(vo)); xf.append(float(vf))
        if len(xo) == 0:
            rows.append({"metric": m, "n_pairs": 0, "wilcoxon_stat": np.nan, "p_value": np.nan})
            continue
        stat, p = wilcoxon(xo, xf, zero_method="wilcox", alternative="two-sided",
                           correction=False, mode="auto")
        rows.append({"metric": m, "n_pairs": len(xo), "wilcoxon_stat": float(stat), "p_value": float(p)})
    pd.DataFrame(rows).assign(q_value=lambda d: fdr(d["p_value"].tolist())) \
        .to_csv(outdir / "paired_wilcoxon_summary.csv", index=False)

    # 7) Four-group panels (matched + raw)
    four_group_panel(matched_all,
                     outdir / "panel_4groups_with_covariates.png",
                     palettes, "with covariates")

    four_group_panel(alpha_all[["Sample_ID","site","disease"]+METRICS].copy(),
                     outdir / "panel_4groups_raw.png",
                     palettes, "RAW (no covariates)")

    # 7b) Paired lines panel
    crohn_paired_lines_panel(alpha_all, pairs,
                             outdir / "panel_paired_crohn_oral_vs_fecal.png",
                             palettes)

    print(f"[INFO] Alpha analysis complete (level={args.level or 'n/a'}) → {outdir}")

if __name__ == "__main__":
    main()
