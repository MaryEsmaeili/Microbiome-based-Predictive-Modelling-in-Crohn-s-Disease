#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha diversity (FULL)
======================
- Robust ID harmonization (handles leading zeros in numeric-only IDs)
- Covariate-matched models (merge by Sample_ID **only**; site/disease taken from alpha side)
- Site-wise disease models (HC3 OLS) + FDR (within-metric) + across-metrics q for "disease"
- Crohn-only PPI models (HC3 OLS) per site
- Disease×Site interaction model
- Oral vs Fecal (unpaired) in Crohn and Healthy
- Paired Wilcoxon (Crohn) using given pairs file if available, else intersection
- Clear plots with legends + n, plus 4-up panels
- Diagnostics: ID maps, join report, unmatched lists
- Consolidated dataset for downstream (beta/ML)

CLI (compatible with Snakefile `rule alpha`):
  --oral-crohn PATH
  --fecal-crohn PATH
  --oral-healthy PATH
  --fecal-healthy PATH
  --covars-pool PATH
  --pairs PATH
  --colors PATH
  --outdir PATH

Dependencies: pandas, numpy, scipy, statsmodels, matplotlib, seaborn, pyyaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
METRICS = ["Shannon", "Richness", "Evenness"]
WISH_COVARS = [
    "Age", "Sex", "BMI", "Smoking", "Antibiotics_3m",
    "PPI_use", "Steroids_ongoing", "Immuno_ongoing", "Responder",
]

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    for c in candidates:
        for k, v in low.items():
            if c.lower() in k:
                return v
    return None


def _norm_site(x: str) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"oral", "mouth", "buccal"}:
        return "oral"
    if s in {"fecal", "faecal", "stool"}:
        return "fecal"
    return s


def _norm_disease(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"1", "crohn", "cd", "case", "patient"}:
        return 1.0
    if s in {"0", "healthy", "control"}:
        return 0.0
    try:
        val = float(x)
        return val if val in {0.0, 1.0} else None
    except Exception:
        return None


def _norm_sex(x) -> Optional[float]:
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


def normalize_sample_id(s: str) -> str:
    """Conservative normalization; drops leading zeros only for numeric-only IDs."""
    s = str(s).strip().replace(" ", "_")
    if s.isdigit():
        try:
            return str(int(s))  # "010054" -> "10054"
        except Exception:
            return s
    return s


import yaml
from pathlib import Path

def load_colors(path: str | None) -> dict[str, list[str]]:
    """
    Supports two schemas:
    A) legacy:
       colors:
         Crohn-Oral:    "#..."
         Healthy-Oral:  "#..."
         Crohn-Fecal:   "#..."
         Healthy-Fecal: "#..."
    B) your file:
       group:
         Oral_Crohn:    "#..."
         Oral_Healthy:  "#..."
         Fecal_Crohn:   "#..."
         Fecal_Healthy: "#..."
       synonyms:
         Oral_Crohn:    ["Crohn-Oral", "Oral_Crohn", ...]
         ...
    Returns:
      {
        "oral":  [crohn_color, healthy_color],
        "fecal": [crohn_color, healthy_color],
      }
    """
    # sane defaults in case file missing/incomplete
    fallback = {
        "oral":  ["#516D99", "#83AAAC"],  # [Crohn, Healthy]
        "fecal": ["#E76F51", "#3A455B"],
    }
    if not path or not Path(path).exists():
        return fallback

    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    # --- Schema A: legacy
    if "colors" in cfg and isinstance(cfg["colors"], dict):
        cols = cfg["colors"]
        try:
            return {
                "oral":  [cols["Crohn-Oral"],  cols["Healthy-Oral"]],
                "fecal": [cols["Crohn-Fecal"], cols["Healthy-Fecal"]],
            }
        except KeyError:
            # fallthrough to robust lookup below
            pass

    # --- Schema B: group + synonyms
    group = cfg.get("group", {}) or {}
    syn   = cfg.get("synonyms", {}) or {}

    def get_color(canon_key: str, candidates: list[str]) -> str | None:
        # exact canonical
        if canon_key in group and group[canon_key]:
            return group[canon_key]
        # synonyms list from YAML
        for a in syn.get(canon_key, []):
            if a in group and group[a]:
                return group[a]
        # try some auto aliases
        auto = [
            canon_key,
            canon_key.replace("-", "_"),
            canon_key.replace("_", "-"),
            canon_key.replace("Crohn", "Case"),
            canon_key.replace("Healthy", "Control"),
        ]
        for a in auto:
            if a in group and group[a]:
                return group[a]
        return None

    oral_crohn   = get_color("Oral_Crohn",   ["Crohn-Oral", "Oral_Crohn"])
    oral_healthy = get_color("Oral_Healthy", ["Healthy-Oral","Oral_Healthy"])
    fec_crohn    = get_color("Fecal_Crohn",  ["Crohn-Fecal","Fecal_Crohn"])
    fec_healthy  = get_color("Fecal_Healthy",["Healthy-Fecal","Fecal_Healthy"])

    # apply fallbacks if something missing
    return {
        "oral":  [oral_crohn   or fallback["oral"][0],  oral_healthy or fallback["oral"][1]],
        "fecal": [fec_crohn    or fallback["fecal"][0], fec_healthy  or fallback["fecal"][1]],
    }

def placeholder_plot(path: Path, title_txt: str) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(5.8, 4.2))
    plt.axis('off')
    plt.text(0.5, 0.5, title_txt, ha='center', va='center')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum(); lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (n1 * n2)


def _fdr(pvals: List[float]) -> np.ndarray:
    pvals = np.array([np.nan if p is None else p for p in pvals], dtype=float)
    mask = ~np.isnan(pvals)
    q = np.full_like(pvals, np.nan, dtype=float)
    if mask.sum():
        q[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
    return q


# --------------------------------------------------------------------------------------
# Alpha from abundance + ID maps
# --------------------------------------------------------------------------------------

def to_sample_by_taxa(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cols_lower = [c.lower() for c in df.columns]
    sid_col = None
    for cand in ("sample_id", "sample", "id"):
        if cand in cols_lower:
            sid_col = df.columns[cols_lower.index(cand)]
            break
    if sid_col is not None:
        df = df.copy()
        orig = df[sid_col].astype(str).tolist()
        df = df.set_index(sid_col)
        num = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
        return num, orig
    # otherwise first column is taxon; columns are samples
    df = df.copy().set_index(df.columns[0])
    num = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    return num.T, list(num.columns)


def compute_alpha_from_abundance(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    mat, orig_ids = to_sample_by_taxa(df)
    # ID map
    norm_ids = [normalize_sample_id(x) for x in mat.index.astype(str)]
    idmap = pd.DataFrame({"original_id": mat.index.astype(str), "Sample_ID": norm_ids})
    mat.index = norm_ids
    # Proportions
    row_sums = mat.sum(axis=1).replace(0.0, np.nan)
    props = mat.div(row_sums, axis=0).fillna(0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        sh = -(props.replace(0, np.nan) * np.log(props.replace(0, np.nan))).sum(axis=1).fillna(0.0)
    rich = (mat > 0).sum(axis=1)
    even = [float(s / np.log(r)) if r > 1 else float('nan') for s, r in zip(sh.values, rich.values)]
    alpha = pd.DataFrame({
        "Sample_ID": mat.index.astype(str),
        "Shannon": sh.values.astype(float),
        "Richness": rich.values.astype(int),
        "Evenness": np.array(even, dtype=float),
    })
    return alpha, idmap


def build_alpha_table(abund_path: str, site: str, disease: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a, idmap = compute_alpha_from_abundance(abund_path)
    a["site"], a["disease"] = site, float(disease)
    a = a[["Sample_ID", "site", "disease", "Shannon", "Richness", "Evenness"]]
    return a, idmap.assign(site=site, disease=int(disease))


# --------------------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------------------

def _group_counts_text(df: pd.DataFrame, group_col="group") -> str:
    counts = df[group_col].value_counts(dropna=True).to_dict()
    h = counts.get("Healthy", 0)
    c = counts.get("Crohn", 0)
    return f"n(Healthy)={h} | n(Crohn)={c}"


def _disease_legend_handles(palette_dict: Dict[str, str]):
    return [
        mpatches.Patch(facecolor=palette_dict["Healthy"], label="Healthy"),
        mpatches.Patch(facecolor=palette_dict["Crohn"],   label="Crohn"),
    ]


# --------------------------------------------------------------------------------------
# Stats per site
# --------------------------------------------------------------------------------------

def summarize_groups(d_raw: pd.DataFrame, site_name: str) -> pd.DataFrame:
    if d_raw.empty:
        return pd.DataFrame(columns=["site", "group", "metric", "n", "mean", "sd", "median", "iqr"])
    rows = []
    for m in METRICS:
        if m not in d_raw.columns:
            continue
        for g in [0.0, 1.0]:
            vals = d_raw.loc[d_raw["disease"] == g, m].dropna().values
            if vals.size == 0:
                rows.append({"site": site_name, "group": int(g), "metric": m,
                             "n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "iqr": np.nan})
                continue
            q1, q3 = np.quantile(vals, [0.25, 0.75])
            rows.append({
                "site": site_name, "group": int(g), "metric": m,
                "n": int(vals.size),
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "median": float(np.median(vals)),
                "iqr": float(q3 - q1),
            })
    return pd.DataFrame(rows)


def run_site(site_name: str, alpha_df: pd.DataFrame, covars_df: pd.DataFrame,
             outdir: Path, palettes: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    site_mask = alpha_df["site"] == site_name
    d_raw = alpha_df.loc[site_mask].copy()
    d_cov = d_raw.merge(covars_df, on="Sample_ID", how="inner", suffixes=("", "_cov"))

    # MWU (Crohn vs Healthy)
    wilx_rows = []
    for m in METRICS:
        if m not in d_raw.columns:
            wilx_rows.append({"metric": m, "p_wilcox": np.nan, "cliffs_delta": np.nan}); continue
        dd = d_raw[["disease", m]].dropna()
        if dd["disease"].nunique() < 2:
            wilx_rows.append({"metric": m, "p_wilcox": np.nan, "cliffs_delta": np.nan}); continue
        x = dd.loc[dd["disease"] == 1.0, m].values
        y = dd.loc[dd["disease"] == 0.0, m].values
        try:
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        cd = cliffs_delta(x, y)
        wilx_rows.append({"metric": m, "p_wilcox": p, "cliffs_delta": cd})
    wilx_df = pd.DataFrame(wilx_rows)
    wilx_df["q_wilcox"] = _fdr(wilx_df["p_wilcox"].tolist())

    # OLS with HC3
    covar_pool = [c for c in WISH_COVARS if c in d_cov.columns]
    for c in covar_pool + ["disease"]:
        if c == "Sex":
            d_cov[c] = d_cov[c].apply(_norm_sex)
        else:
            d_cov[c] = pd.to_numeric(d_cov[c], errors="coerce")

    model_rows = []
    for m in METRICS:
        if m not in d_cov.columns:
            model_rows.append({"metric": m, "term": "(missing_metric)",
                               "estimate": np.nan, "std.error": np.nan, "statistic": np.nan, "p.value": np.nan});
            continue
        dd = d_cov[[m, "disease"] + covar_pool].dropna()
        if len(dd) < 6 or dd["disease"].nunique() < 2:
            model_rows.append({"metric": m, "term": "(skipped_small_n_or_one_class)",
                               "estimate": np.nan, "std.error": np.nan, "statistic": np.nan, "p.value": np.nan});
            continue
        X = add_constant(dd[["disease"] + covar_pool], has_constant="add"); y = dd[m].values
        fit = OLS(y, X).fit(cov_type="HC3")
        for term, est, se, tval, pval in zip(fit.params.index, fit.params.values, fit.bse.values, fit.tvalues.values, fit.pvalues.values):
            model_rows.append({"metric": m, "term": term, "estimate": est, "std.error": se, "statistic": tval, "p.value": pval})

    model_df = pd.DataFrame(model_rows)
    if not model_df.empty:
        model_df["q.value"] = np.nan
        for m in METRICS:
            idx = model_df["metric"] == m
            if idx.any():
                model_df.loc[idx, "q.value"] = _fdr(model_df.loc[idx, "p.value"].tolist())
        # FDR across metrics for the disease term
        dis_ps = [
            model_df[(model_df["metric"] == m) & (model_df["term"] == "disease")]["p.value"].iloc[0]
            if ((model_df["metric"] == m) & (model_df["term"] == "disease")).any() else np.nan
            for m in METRICS
        ]
        dq = _fdr(dis_ps); dq_map = {m: q for m, q in zip(METRICS, dq)}
        model_df["q_disease_across_metrics"] = model_df["metric"].map(dq_map)

    # Save models
    out_csv = outdir / f"alpha_models_{site_name}.csv"; ensure_dir(outdir); model_df.to_csv(out_csv, index=False)

    # Summaries
    desc = summarize_groups(d_raw, site_name); desc["group_label"] = desc["group"].map({0: "Healthy", 1: "Crohn"})
    desc.to_csv(outdir / f"group_summary_{site_name}.csv", index=False)

    # Plots (single)
    disease_labels = {0.0: "Healthy", 1.0: "Crohn"}
    fill_map = palettes[site_name]  # [Crohn, Healthy]
    for m in METRICS:
        dd = d_raw[["disease", m]].dropna(); out_png = outdir / f"boxplot_{site_name}_{m}_with_covariates.png"
        if dd.empty or dd["disease"].nunique() < 2:
            placeholder_plot(out_png, f"{site_name.capitalize()} — {m}\n(No data for both classes)"); continue
        dd = dd.rename(columns={m: "value"}); dd["group"] = dd["disease"].map(disease_labels)
        q_row = wilx_df[wilx_df["metric"] == m]
        if len(q_row):
            qtxt = q_row["q_wilcox"].iloc[0]; cdt = q_row["cliffs_delta"].iloc[0]
            annot = f"FDR q = {qtxt:.2e} | Cliff’s δ = {np.nan if pd.isna(cdt) else round(float(cdt), 2)}"
        else:
            annot = "q = NA | δ = NA"
        plt.figure(figsize=(6.2, 4.6))
        palette = {"Crohn": fill_map[0], "Healthy": fill_map[1]}
        ax = sns.boxplot(
            data=dd, x="group", y="value", order=["Healthy", "Crohn"],
            hue="group", palette=palette, dodge=False, legend=False
        )
        sns.stripplot(data=dd, x="group", y="value", order=["Healthy", "Crohn"], color="black", alpha=0.6, jitter=0.15)
        ymax = np.nanmax(dd["value"].values)
        ax.text(0.5, ymax, annot, ha="center", va="bottom", fontsize=9)
        ax.set_xlabel(""); ax.set_ylabel(m)
        ax.set_title(f"{site_name.capitalize()} — {m}  ({_group_counts_text(dd)})", fontsize=12)
        ax.legend(handles=_disease_legend_handles(palette), frameon=False, loc="upper left")
        plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

    # 4-panel figure per site
    panel_png = outdir / f"panel_{site_name}_4up.png"
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8))
        palette = {"Crohn": fill_map[0], "Healthy": fill_map[1]}
        for ax, met in zip(axes.flat[:3], METRICS):
            dd = d_raw[["disease", met]].dropna()
            if dd.empty or dd["disease"].nunique() < 2:
                ax.axis('off'); ax.text(0.5, 0.5, f"No data: {met}", ha='center'); continue
            dd = dd.rename(columns={met: "value"}); dd["group"] = dd["disease"].map(disease_labels)
            sns.boxplot(data=dd, x="group", y="value", order=["Healthy", "Crohn"], hue="group",
                        palette=palette, dodge=False, legend=False, ax=ax)
            sns.stripplot(data=dd, x="group", y="value", order=["Healthy", "Crohn"], color="black", alpha=0.6, jitter=0.15, ax=ax)
            ax.set_xlabel(""); ax.set_ylabel(met)
            ax.set_title(f"{met}  ({_group_counts_text(dd)})", fontsize=11)
        # 4th panel: Shannon vs Richness scatter
        ax = axes.flat[3]
        sub = d_raw[["Shannon", "Richness", "disease"]].dropna()
        if sub.empty:
            ax.axis('off'); ax.text(0.5,0.5,"No data: scatter", ha='center')
        else:
            sub["group"] = sub["disease"].map(disease_labels)
            for g, ddg in sub.groupby("group"):
                ax.scatter(ddg["Shannon"], ddg["Richness"], label=g, s=28, alpha=0.8, c=palette[g])
            ax.set_xlabel("Shannon"); ax.set_ylabel("Richness"); ax.set_title("Shannon vs Richness")
            ax.legend(frameon=False, loc="best")
        fig.legend(handles=_disease_legend_handles(palette), frameon=False, loc="upper right")
        plt.suptitle(f"{site_name.capitalize()} — Alpha (4-up)", y=0.98)
        plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(panel_png, dpi=300); plt.close()
    except Exception:
        placeholder_plot(panel_png, f"{site_name.capitalize()} 4-up (error)")

    return wilx_df, model_df


# --------------------------------------------------------------------------------------
# PPI models (Crohn-only)
# --------------------------------------------------------------------------------------

def ppi_models(site_name: str, alpha_df: pd.DataFrame, covars_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    df = alpha_df.merge(covars_df, on="Sample_ID", how="inner", suffixes=("", "_cov"))
    df = df[(df["site"] == site_name) & (df["disease"] == 1.0)].copy()
    if df.empty or "PPI_use" not in df.columns:
        pd.DataFrame(columns=["metric", "term", "estimate", "std.error", "statistic", "p.value", "q.value"]).to_csv(outdir / f"alpha_ppi_{site_name}.csv", index=False)
        return pd.DataFrame()
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
        rhs_vars = [v for v in covars if v != "PPI_use"]
        use_vars = ["PPI_use"] + rhs_vars
        dd = df[[m] + use_vars].dropna()
        if len(dd) < 6 or dd["PPI_use"].nunique() < 2:
            rows.append({"metric": m, "term": "(skipped_small_n_or_one_class)",
                         "estimate": np.nan, "std.error": np.nan, "statistic": np.nan, "p.value": np.nan})
            continue
        X = add_constant(dd[use_vars], has_constant="add"); y = dd[m].values
        fit = OLS(y, X).fit(cov_type="HC3")
        for term, est, se, tval, pval in zip(fit.params.index, fit.params.values, fit.bse.values, fit.tvalues.values, fit.pvalues.values):
            rows.append({"metric": m, "term": term, "estimate": est, "std.error": se, "statistic": tval, "p.value": pval})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q.value"] = np.nan
        for m in METRICS:
            idx = out["metric"] == m
            if idx.any():
                out.loc[idx, "q.value"] = _fdr(out.loc[idx, "p.value"].tolist())
    out.to_csv(outdir / f"alpha_ppi_{site_name}.csv", index=False)
    return out


# --------------------------------------------------------------------------------------
# Cross-site comparisons & interaction
# --------------------------------------------------------------------------------------

def site_disease_interaction(alpha_df: pd.DataFrame, covars_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    # Merge ONLY by Sample_ID; use site/disease from alpha side
    df = alpha_df.merge(covars_df, on="Sample_ID", how="inner", suffixes=("", "_cov")).copy()
    if df.empty:
        pd.DataFrame(columns=["metric","term","estimate","std.error","statistic","p.value","q.value"]).to_csv(outdir/"alpha_site_disease_interaction.csv", index=False)
        return pd.DataFrame()
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
            rows.append({"metric": m, "term": "(skipped_small_n)", "estimate": np.nan, "std.error": np.nan, "statistic": np.nan, "p.value": np.nan})
            continue
        X = dd[["disease", "site_bin"]].copy()
        X["disease:site_bin"] = dd["disease"] * dd["site_bin"]  # Series, not ndarray
        for c in covar_pool:
            X[c] = dd[c].values
        X = add_constant(X, has_constant="add")
        y = dd[m].values
        fit = OLS(y, X).fit(cov_type="HC3")
        for term, est, se, tval, pval in zip(fit.params.index, fit.params.values, fit.bse.values, fit.tvalues.values, fit.pvalues.values):
            rows.append({"metric": m, "term": term, "estimate": est, "std.error": se, "statistic": tval, "p.value": pval})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q.value"] = np.nan
        for m in METRICS:
            idx = out["metric"] == m
            if idx.any():
                out.loc[idx, "q.value"] = _fdr(out.loc[idx, "p.value"].tolist())
    out.to_csv(outdir/"alpha_site_disease_interaction.csv", index=False)
    return out


def oral_vs_fecal_unpaired(alpha_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for status, lab in [(1.0, "Crohn"), (0.0, "Healthy")]:
        sub = alpha_df[alpha_df["disease"] == status]
        for m in METRICS:
            if m not in sub.columns:
                rows.append({"group": lab, "metric": m, "p_mwu": np.nan}); continue
            dd = sub[["site", m]].dropna()
            if dd["site"].nunique() < 2:
                rows.append({"group": lab, "metric": m, "p_mwu": np.nan}); continue
            x = dd.loc[dd["site"] == "oral", m].values
            y = dd.loc[dd["site"] == "fecal", m].values
            try:
                p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            except Exception:
                p = np.nan
            rows.append({"group": lab, "metric": m, "p_mwu": p})
    out = pd.DataFrame(rows); out["q_mwu"] = _fdr(out["p_mwu"].tolist())
    out.to_csv(outdir/"oral_fecal_unpaired_by_group.csv", index=False)
    return out


def disease_overall_boxplots(alpha_df: pd.DataFrame, outdir: Path, palettes: Dict[str, List[str]]):
    """Crohn vs Healthy across all sites (extra figures)."""
    df = alpha_df.copy()
    if df.empty:
        for m in METRICS:
            placeholder_plot(outdir / f"boxplot_all_{m}_by_disease.png", f"All sites — {m}\n(No data)")
        placeholder_plot(outdir / "panel_all_sites_disease_4up.png", "All sites 4-up\n(No data)")
        return

    df["group"] = df["disease"].map({0.0: "Healthy", 1.0: "Crohn"})
    palette = {"Crohn": palettes["fecal"][0], "Healthy": palettes["fecal"][1]}

    for m in METRICS:
        dd = df[["group", m]].dropna()
        out_png = outdir / f"boxplot_all_{m}_by_disease.png"
        if dd.empty or dd["group"].nunique() < 2:
            placeholder_plot(out_png, f"All sites — {m}\n(No data for both classes)"); continue
        plt.figure(figsize=(6.2, 4.6))
        ax = sns.boxplot(data=dd, x="group", y=m, order=["Healthy", "Crohn"], hue="group", palette=palette, dodge=False, legend=False)
        sns.stripplot(data=dd, x="group", y=m, order=["Healthy", "Crohn"], color="black", alpha=0.5, jitter=0.2)
        ax.set_xlabel(""); ax.set_ylabel(m)
        ax.set_title(f"All sites — {m}  ({_group_counts_text(dd)})", fontsize=12)
        ax.legend(handles=_disease_legend_handles(palette), frameon=False, loc="upper left")
        plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

    panel = outdir / "panel_all_sites_disease_4up.png"
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8))
        for ax, met in zip(axes.flat[:3], METRICS):
            dd = df[["group", met]].dropna()
            if dd.empty or dd["group"].nunique() < 2:
                ax.axis('off'); ax.text(0.5, 0.5, f"No data: {met}", ha='center'); continue
            sns.boxplot(data=dd, x="group", y=met, order=["Healthy", "Crohn"], hue="group", palette=palette, dodge=False, legend=False, ax=ax)
            sns.stripplot(data=dd, x="group", y=met, order=["Healthy", "Crohn"], color="black", alpha=0.5, jitter=0.2, ax=ax)
            ax.set_xlabel(""); ax.set_ylabel(met); ax.set_title(f"{met}  ({_group_counts_text(dd)})", fontsize=11)
        ax = axes.flat[3]
        sub = df[["Shannon", "Richness", "group"]].dropna()
        if sub.empty:
            ax.axis('off'); ax.text(0.5,0.5,"No data: scatter", ha='center')
        else:
            for g, ddg in sub.groupby("group"):
                ax.scatter(ddg["Shannon"], ddg["Richness"], label=g, s=28, alpha=0.8, c=palette[g])
            ax.set_xlabel("Shannon"); ax.set_ylabel("Richness"); ax.set_title("Shannon vs Richness")
            ax.legend(frameon=False, loc="best")
        fig.legend(handles=_disease_legend_handles(palette), frameon=False, loc="upper right")
        plt.suptitle("All sites — Alpha (4-up)", y=0.98)
        plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(panel, dpi=300); plt.close()
    except Exception:
        placeholder_plot(panel, "All sites 4-up (error)")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

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
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    palettes = load_colors(args.colors)
    import json
    with open(Path(args.outdir)/"colors_used.json", "w") as f:
        json.dump(palettes, f, indent=2)
    # 1) Build alpha and ID maps
    alpha_oral, idmap_oo    = build_alpha_table(args.oral_crohn,   "oral",  1)
    alpha_fecal, idmap_of   = build_alpha_table(args.fecal_crohn,  "fecal", 1)
    alpha_h_oral, idmap_ho  = build_alpha_table(args.oral_healthy, "oral",  0)
    alpha_h_fecal, idmap_hf = build_alpha_table(args.fecal_healthy,"fecal", 0)

    # Write ID maps for transparency
    iddir = outdir / "idmaps"; ensure_dir(iddir)
    idmap_oo.to_csv(iddir/"idmap_oral_crohn.tsv", sep='\t', index=False)
    idmap_of.to_csv(iddir/"idmap_fecal_crohn.tsv", sep='\t', index=False)
    idmap_ho.to_csv(iddir/"idmap_oral_healthy.tsv", sep='\t', index=False)
    idmap_hf.to_csv(iddir/"idmap_fecal_healthy.tsv", sep='\t', index=False)

    alpha_all = (
        pd.concat([alpha_oral, alpha_fecal, alpha_h_oral, alpha_h_fecal], axis=0, ignore_index=True)
          .drop_duplicates(subset=["Sample_ID", "site"], keep="first")
    )

    # 2) Covariates (canonicalize minimal columns)
    covars = pd.read_csv(args.covars_pool, dtype=str)
    if "Sample_ID" not in covars.columns:
        raise RuntimeError("Covariates file must contain 'Sample_ID'.")
    covars["Sample_ID"] = covars["Sample_ID"].astype(str).map(normalize_sample_id)
    if "Sex" in covars.columns:
        covars["Sex"] = covars["Sex"].apply(_norm_sex)
    # Keep other covariates as-is (cast numeric later)

    # 2a) Join diagnostics (by Sample_ID only)
    j = alpha_all.merge(covars[["Sample_ID"]].drop_duplicates(), on="Sample_ID", how="left", indicator=True)
    with open(outdir/"join_report.txt", "w") as fh:
        fh.write(f"Alpha rows: {len(alpha_all)}\n")
        fh.write(f"Joined rows (by Sample_ID): {int((j['_merge'] == 'both').sum())}\n")
        fh.write(f"Missing in covariates (by Sample_ID): {int((j['_merge'] == 'left_only').sum())}\n")
    j.loc[j["_merge"]=="left_only", ["Sample_ID", "site", "disease"]].to_csv(outdir/"alpha_unmatched_in_covariates.csv", index=False)

    ccheck = covars.merge(alpha_all[["Sample_ID"]].drop_duplicates(), on="Sample_ID", how="left", indicator=True)
    ccheck.loc[ccheck["_merge"]=="left_only", ["Sample_ID"]].to_csv(outdir/"covariates_unmatched_in_alpha.csv", index=False)

    # 3) Save consolidated dataset for downstream (beta/ML)
    alpha_with_covars = alpha_all.merge(covars, on="Sample_ID", how="inner", suffixes=("", "_cov"))
    alpha_with_covars.to_csv(outdir/"alpha_with_covariates.csv", index=False)

    # 4) Per-site analyses
    wilx_fec, mdl_fec = run_site("fecal", alpha_all, covars, outdir, palettes)
    wilx_orl, mdl_orl = run_site("oral",  alpha_all, covars, outdir, palettes)

    # 5) PPI models (Crohn-only)
    ppi_models("fecal", alpha_all, covars, outdir)
    ppi_models("oral",  alpha_all, covars, outdir)

    # 6) Interaction & cross-site
    site_disease_interaction(alpha_all, covars, outdir)
    oral_vs_fecal_unpaired(alpha_all, outdir)

    # 7) Paired Wilcoxon (Crohn)
    try:
        matched_df = pd.read_csv(args.pairs)
    except Exception:
        matched_df = None
    crohn = alpha_all[alpha_all["disease"]==1.0].copy()
    oral  = crohn[crohn["site"]=="oral"].set_index("Sample_ID")
    fecal = crohn[crohn["site"]=="fecal"].set_index("Sample_ID")
    pairs: List[Tuple[str,str]] = []
    if matched_df is not None and not matched_df.empty:
        oc = fc = None
        for c in matched_df.columns:
            lc = c.lower()
            if oc is None and ("oral" in lc or lc in {"oral_id","sample_id_oral"}): oc = c
            if fc is None and ("fecal" in lc or "faecal" in lc or lc in {"fecal_id","stool","sample_id_fecal"}): fc = c
        if oc and fc:
            for _, r in matched_df[[oc, fc]].dropna().iterrows():
                pairs.append((normalize_sample_id(str(r[oc])), normalize_sample_id(str(r[fc]))))
    if not pairs:
        inter = oral.index.intersection(fecal.index)
        pairs = [(sid, sid) for sid in inter]
    rows = []
    for m in METRICS:
        xo = []; xf = []
        for o_id, f_id in pairs:
            vo = oral.get(m).get(o_id) if m in oral.columns else np.nan
            vf = fecal.get(m).get(f_id) if m in fecal.columns else np.nan
            if pd.notna(vo) and pd.notna(vf):
                xo.append(float(vo)); xf.append(float(vf))
        if len(xo) == 0:
            rows.append({"metric": m, "n_pairs": 0, "wilcoxon_stat": np.nan, "p_value": np.nan}); continue
        stat, p = wilcoxon(xo, xf, zero_method="wilcox", alternative="two-sided", correction=False, mode="auto")
        rows.append({"metric": m, "n_pairs": len(xo), "wilcoxon_stat": float(stat), "p_value": float(p)})
    out = pd.DataFrame(rows); out["q_value"] = _fdr(out["p_value"].tolist())
    out.to_csv(outdir/"paired_wilcoxon_summary.csv", index=False)

    # 8) Extra: disease-overall figures (Crohn vs Healthy across all sites)
    disease_overall_boxplots(alpha_all, outdir, palettes)

    print(f"[INFO] Alpha analysis complete → {outdir}")


if __name__ == "__main__":
    main()
