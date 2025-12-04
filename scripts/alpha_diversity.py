# scripts/alpha_diversity.py
"""
Alpha-diversity analysis for the microbiome pipeline.

This script reads genus/species abundance tables and covariate metadata,
computes basic alpha-diversity metrics (Shannon, Richness, Evenness),
runs simple group comparisons and OLS models, and writes all summary
tables and plots for oral and fecal samples in Crohn’s disease and
healthy controls.
"""

from __future__ import annotations

import argparse, json, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests

# ------------------------------ Globals ---------------------------------
METRICS = ["Shannon", "Richness", "Evenness"]
BIN_TRUE  = {"1","true","t","yes","y"}
BIN_FALSE = {"0","false","f","no","n"}
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Fixed covariates for the whole pipeline
FIXED_COVARS = ["Age","Sex","BMI","Smoking","Antibiotics_3m",
                "Immuno_ongoing","Steroids_ongoing","PPI_use"]

# ------------------------------- Utils ---------------------------------
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def normalize_sample_id(s: str) -> str:
    s = str(s).strip()
    s = pd.Series([s]).str.replace(r"\.\d+$", "", regex=True).iloc[0]
    if s.isdigit():
        try: s = str(int(s))
        except Exception: pass
    return s.upper()

def _norm_sex(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    s = str(x).strip().lower()
    if s in {"f","female"}: return 0.0
    if s in {"m","male"}:   return 1.0
    try: return float(x)
    except Exception: return None

def _to_bin01(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    s = str(x).strip().lower()
    if s in BIN_TRUE:  return 1.0
    if s in BIN_FALSE: return 0.0
    try:
        v = float(s)
        if v in (0.0,1.0): return v
    except Exception: pass
    return None

def safe_numeric_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").astype(float)
    out[~np.isfinite(out)] = np.nan
    return out

def safe_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    X[~np.isfinite(X)] = np.nan
    return X

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0: return float("nan")
    gt = (x[:,None] > y[None,:]).sum(); lt = (x[:,None] < y[None,:]).sum()
    return (gt - lt) / (n1 * n2)

def fdr(p: List[float]) -> np.ndarray:
    p = np.array([np.nan if v is None else v for v in p], dtype=float)
    mask = ~np.isnan(p); q = np.full_like(p, np.nan, dtype=float)
    if mask.sum(): q[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return q

# ------------------------------ Colors ---------------------------------
def load_colors(path: str | None) -> dict:
    fallback = {"oral": ["#edae49","#00798c"], "fecal": ["#30638e","#d1495b"]}  # [Crohn, Healthy]
    if not path or not Path(path).exists(): return fallback
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "colors" in cfg and isinstance(cfg["colors"], dict):
        c = cfg["colors"]
        return {
            "oral":  [c.get("Crohn-Oral",  fallback["oral"][0]),  c.get("Healthy-Oral",  fallback["oral"][1])],
            "fecal": [c.get("Crohn-Fecal", fallback["fecal"][0]), c.get("Healthy-Fecal", fallback["fecal"][1])],
        }
    group = cfg.get("group", {}) or {}
    syn   = cfg.get("synonyms", {}) or {}
    def get(key: str, default: str) -> str:
        if key in group and group[key]: return group[key]
        for alias in syn.get(key, []):
            if alias in group and group[alias]: return group[alias]
        return default
    return {
        "oral":  [get("Oral_Crohn",  fallback["oral"][0]),  get("Oral_Healthy",  fallback["oral"][1])],
        "fecal": [get("Fecal_Crohn", fallback["fecal"][0]), get("Fecal_Healthy", fallback["fecal"][1])],
    }

# ----------------------- Alpha from abundance --------------------------
def to_sample_by_taxa(df: pd.DataFrame):
    cols_lower = [c.lower() for c in df.columns]
    sid_col = None
    for cand in ("sample_id","sample","id"):
        if cand in cols_lower:
            sid_col = df.columns[cols_lower.index(cand)]; break
    if sid_col is not None:
        df = df.copy().set_index(sid_col)
        num = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
        return num, list(num.index.astype(str))
    df = df.copy().set_index(df.columns[0])
    num = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    return num.T, list(num.columns.astype(str))

def compute_alpha_from_abundance(path: str):
    df = pd.read_csv(path)
    mat, _ = to_sample_by_taxa(df)
    norm_ids = [normalize_sample_id(x) for x in mat.index.astype(str)]
    idmap = pd.DataFrame({"original_id": mat.index.astype(str), "Sample_ID": norm_ids})
    mat.index = norm_ids
    row_sums = mat.sum(axis=1).replace(0.0, np.nan)
    props = mat.div(row_sums, axis=0).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = -(props.replace(0, np.nan) * np.log(props.replace(0, np.nan))).sum(axis=1).fillna(0.0)
    rich = (mat > 0).sum(axis=1)
    even = [float(s/np.log(r)) if r > 1 else float("nan") for s, r in zip(sh.values, rich.values)]
    alpha = pd.DataFrame({
        "Sample_ID": mat.index.astype(str),
        "Shannon": sh.values.astype(float),
        "Richness": rich.values.astype(int),
        "Evenness": np.array(even, dtype=float),
    })
    return alpha, idmap

def build_alpha_table(abund_path: str, site: str, disease: int):
    a, idmap = compute_alpha_from_abundance(abund_path)
    a["site"], a["disease"] = site, int(disease)
    a = a[["Sample_ID","site","disease"] + METRICS]
    return a, idmap.assign(site=site, disease=int(disease))

# ----------------------------- Stats helpers ---------------------------
def format_annot_2x(p=None, q=None) -> str:
    """
    Build a compact annotation string for a 2-group comparison.

    We deliberately do NOT include group counts or effect sizes here,
    because those are handled in the global legend for each panel.
    """
    parts = []
    if isinstance(p, (int, float)) and np.isfinite(p):
        parts.append(f"p={p:.2e}")
    if isinstance(q, (int, float)) and np.isfinite(q):
        parts.append(f"q={q:.2e}")
    return " | ".join(parts)

def compute_site_mwu_stats(df_site: pd.DataFrame):
    """
    Compute MWU p-values, FDR q-values, Cliff's delta (for export/use),
    and compact captions (ONLY p and q) for each metric.

    Also return group counts per metric so that the caller can
    build a global legend with n(Crohn) and n(Healthy).
    """
    pvals, deltas, counts = {}, {}, {}
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

        # Cliff's delta (still computed, but NOT shown in captions)
        gt = (x[:, None] > y[None, :]).sum()
        lt = (x[:, None] < y[None, :]).sum()
        deltas[m] = float((gt - lt) / (len(x) * len(y)))

        if np.isfinite(pvals[m]):
            val_mets.append(m)
            val_ps.append(pvals[m])

    # FDR over metrics
    qvals = {m: np.nan for m in METRICS}
    if len(val_ps):
        q_vec = fdr(val_ps)
        for i, m in enumerate(val_mets):
            qvals[m] = float(q_vec[i])

    # Captions: ONLY p and q (no n, no delta)
    captions = {}
    for m in METRICS:
        captions[m] = format_annot_2x(
            p=pvals.get(m, np.nan),
            q=qvals.get(m, np.nan),
        )

    # Return counts as well so caller can put them in a global legend
    return pvals, qvals, deltas, captions, counts

def _draw_bracket(ax, x1, x2, y_data_max, pval,
                  pad_frac=0.04, lw=1.0, txt_offset=0.6):
    if not (isinstance(pval, (int,float)) and np.isfinite(pval)): return
    yl = ax.get_ylim(); span = (yl[1]-yl[0])
    base = max(y_data_max, yl[1] - 0.85*span)
    lift = span * pad_frac
    y0 = base + lift
    ax.plot([x1,x1,x2,x2],[y0,y0+lift,y0+lift,y0], lw=lw, c="black")
    ax.text((x1+x2)/2.0, y0 + lift*(1.0+txt_offset), f"p={pval:.2e}",
            ha="center", va="bottom", fontsize=9, color="black")

# ------------------------------- Plots ---------------------------------
def placeholder_plot(path: Path, title_txt: str) -> None:
    ensure_dir(path.parent); plt.figure(figsize=(5.8,4.2)); plt.axis("off")
    plt.text(0.5,0.5,title_txt,ha="center",va="center"); plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()

def box4_panel(site_name: str, df: pd.DataFrame, out_png: Path,
               crohn_color: str, healthy_color: str, title_suffix: str):
    if df.empty:
        placeholder_plot(out_png, f"{site_name.capitalize()} — Alpha (4-up {title_suffix})\n(No data)")
        return

    disease_labels = {0: "Healthy", 1: "Crohn"}
    pal = {"Crohn": crohn_color, "Healthy": healthy_color}

    # Stats + captions (only p and q) + per-metric counts
    pvals, qvals, deltas, captions, counts = compute_site_mwu_stats(df)

    # Global counts for this site (for legend only)
    nH_global = int((df["disease"] == 0).sum())
    nC_global = int((df["disease"] == 1).sum())

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))

    # 3 boxplots
    for ax, met in zip(axes.flat[:3], METRICS):
        dd = df[["disease", met]].dropna()
        if dd.empty or dd["disease"].nunique() < 2:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No data: {met}", ha="center")
            continue

        dd = dd.rename(columns={met: "value"})
        dd["group"] = dd["disease"].map(disease_labels)

        sns.boxplot(
            data=dd, x="group", y="value",
            order=["Healthy", "Crohn"],
            hue="group", palette=pal, dodge=False,
            legend=False, ax=ax,
        )
        sns.stripplot(
            data=dd, x="group", y="value",
            order=["Healthy", "Crohn"],
            color="black", alpha=0.6, jitter=0.15, ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(met)
        ax.set_title(met, fontsize=11)

        # Bracket with p-value only
        _draw_bracket(
            ax, 0, 1,
            float(dd["value"].max()),
            pvals.get(met, np.nan),
        )

        # Caption under each subplot: ONLY p and q
        ax.text(
            0.5, -0.25,
            captions.get(met, ""),
            transform=ax.transAxes,
            ha="center", va="top", fontsize=9,
        )

    # Scatter Shannon vs Richness
    ax = axes.flat[3]
    sub = df[["Shannon", "Richness", "disease"]].dropna()
    if sub.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data: scatter", ha="center")
    else:
        sub["group"] = sub["disease"].map(disease_labels)
        for g, gg in sub.groupby("group"):
            ax.scatter(
                gg["Shannon"], gg["Richness"],
                s=28, alpha=0.85, c=pal[g], label=g,
            )
        sns.regplot(
            x="Shannon", y="Richness",
            data=sub, scatter=False, ci=95, ax=ax,
        )
        ax.set_xlabel("Shannon")
        ax.set_ylabel("Richness")
        ax.set_title("Shannon vs Richness")

    # Global legend with group counts
    handles = [
        mpatches.Patch(facecolor=pal["Healthy"], label="Healthy"),
        mpatches.Patch(facecolor=pal["Crohn"],   label="Crohn"),
    ]
    label_H = f"Healthy (n={nH_global})"
    label_C = f"Crohn (n={nC_global})"
    fig.legend(
        handles=handles,
        labels=[label_H, label_C],
        frameon=False, loc="upper right",
    )

    plt.suptitle(f"{site_name.capitalize()} — Alpha (4-up {title_suffix})", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300)
    plt.close()

def four_group_panel(df_all: pd.DataFrame, out_png: Path,
                     palettes: Dict[str, List[str]], title_suffix: str):
    if df_all.empty:
        placeholder_plot(out_png, f"Alpha (4 groups — {title_suffix})\n(No data)")
        return

    lab_map = {0: "Healthy", 1: "Crohn"}
    df = df_all.copy()
    df["disease_label"] = df["disease"].map(lab_map)
    df["group4"] = df["disease_label"] + "-" + df["site"].str.capitalize()

    order4 = ["Healthy-Oral", "Crohn-Oral", "Healthy-Fecal", "Crohn-Fecal"]
    pal4 = {
        "Healthy-Oral": palettes["oral"][1],
        "Crohn-Oral":   palettes["oral"][0],
        "Healthy-Fecal": palettes["fecal"][1],
        "Crohn-Fecal":   palettes["fecal"][0],
    }

    # Counts per group for legend
    counts = df.groupby("group4")["group4"].count()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))

    # 3 boxplots
    for ax, met in zip(axes.flat[:3], METRICS):
        dd = df[["group4", "site", "disease", met]].dropna()
        if dd.empty or dd["group4"].nunique() < 2:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No data: {met}", ha="center")
            continue

        sns.boxplot(
            data=dd, x="group4", y=met,
            hue="group4", order=order4,
            palette=pal4, dodge=False,
            legend=False, ax=ax,
        )
        sns.stripplot(
            data=dd, x="group4", y=met,
            order=order4, color="black",
            alpha=0.5, jitter=0.15, ax=ax,
        )
        ax.set_xlabel("")
        ax.set_title(met)
        ax.tick_params(axis="x", rotation=20)

        # Stats per site (Oral / Fecal): ONLY p and q in caption
        dd_site = dd.copy()
        text_chunks = []
        p_list = []
        stat_map = {}

        for site_name in ["oral", "fecal"]:
            sub = dd_site[dd_site["site"] == site_name]
            if sub["disease"].nunique() < 2:
                stat_map[site_name] = {"p": np.nan}
                continue

            x = sub.loc[sub["disease"] == 1, met].values
            y = sub.loc[sub["disease"] == 0, met].values
            try:
                p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            except Exception:
                p = np.nan
            stat_map[site_name] = {"p": p}
            if np.isfinite(p):
                p_list.append(p)

        # FDR across the sites that have a valid p
        q_list = list(fdr(p_list)) if p_list else []
        qi = 0

        for site_name in ["oral", "fecal"]:
            p = stat_map.get(site_name, {}).get("p", np.nan)
            if not (isinstance(p, (int, float)) and np.isfinite(p)):
                text_chunks.append(f"{site_name.capitalize()}(p=NA, q=NA)")
            else:
                qv = q_list[qi] if qi < len(q_list) else np.nan
                qi += 1
                text_chunks.append(f"{site_name.capitalize()}(p={p:.2e}, q={qv:.2e})")

        ax.text(
            0.5, -0.28,
            " | ".join(text_chunks),
            transform=ax.transAxes,
            ha="center", va="top", fontsize=9,
        )

    # Scatter Shannon vs Richness (4 groups)
    ax = axes.flat[3]
    sub = df[["Shannon", "Richness", "group4"]].dropna()
    if sub.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data: scatter", ha="center")
    else:
        for g, gg in sub.groupby("group4"):
            ax.scatter(
                gg["Shannon"], gg["Richness"],
                s=26, alpha=0.85,
                c=pal4.get(g, "#777777"),
                label=g,
            )
        sns.regplot(
            x="Shannon", y="Richness",
            data=sub, scatter=False, ci=95, ax=ax,
        )
        ax.set_xlabel("Shannon")
        ax.set_ylabel("Richness")
        ax.set_title("Shannon vs Richness (4 groups)")
        # We rely on a global legend, so no per-axis legend here

    # Global legend with counts for the 4 groups
    handles = []
    labels = []
    for g in order4:
        if g in pal4:
            handles.append(mpatches.Patch(facecolor=pal4[g], label=g))
            labels.append(f"{g} (n={int(counts.get(g, 0))})")

    fig.legend(
        handles=handles,
        labels=labels,
        frameon=False,
        loc="upper right",
    )

    plt.suptitle(f"Alpha (4 groups — {title_suffix})", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300)
    plt.close()

def crohn_paired_lines_panel(alpha_all: pd.DataFrame, covars_df: pd.DataFrame,
                             pairs: List[Tuple[str,str]], out_png: Path, palettes: Dict[str, List[str]]):
    """
    Plot paired Crohn oral vs fecal alpha metrics.

    - Restrict to Crohn samples; optionally restrict to covariate cohort.
    - Draw paired lines for each metric.
    - Legend shows group colors + n (number of pairs).
    - Subplot captions ONLY show p and q (no n, no effect size).
    """
    # Restrict to Crohn with covariates when covars exist
    base = alpha_all.copy()
    if covars_df is not None and not covars_df.empty:
        have = set(
            covars_df["Sample_ID"]
            .astype(str)
            .map(normalize_sample_id)
            .unique()
        )
        base = base[base["Sample_ID"].isin(have)]

    crohn = base[base["disease"] == 1].copy()
    oral  = crohn[crohn["site"] == "oral"].set_index("Sample_ID")
    fecal = crohn[crohn["site"] == "fecal"].set_index("Sample_ID")

    # If no explicit pairs file, use intersection of IDs
    if not pairs:
        inter = oral.index.intersection(fecal.index)
        pairs = [(sid, sid) for sid in inter]

    if not pairs:
        placeholder_plot(out_png, "Crohn Oral↔Fecal Paired (No pairs)")
        return

    wilco_p = []
    series_by_metric = {}
    n_pairs_per_metric = {}

    # Build paired series per metric and Wilcoxon p-values
    for m in METRICS:
        xo, xf = [], []
        for o_id, f_id in pairs:
            vo = oral[m].get(o_id)  if (m in oral.columns  and o_id in oral.index)  else np.nan
            vf = fecal[m].get(f_id) if (m in fecal.columns and f_id in fecal.index) else np.nan
            if pd.notna(vo) and pd.notna(vf):
                xo.append(float(vo))
                xf.append(float(vf))

        series_by_metric[m] = (xo, xf)
        n_pairs_per_metric[m] = len(xo)

        if len(xo) >= 1:
            try:
                stat, p = wilcoxon(
                    xo, xf,
                    zero_method="wilcox",
                    alternative="two-sided",
                    correction=False,
                    mode="auto",
                )
            except Exception:
                p = np.nan
        else:
            p = np.nan

        wilco_p.append(p)

    # Global n for legend: max usable pairs across metrics
    n_pairs_global = max(n_pairs_per_metric.values()) if n_pairs_per_metric else 0

    # FDR across metrics
    wilco_q = fdr(wilco_p) if any([not pd.isna(p) for p in wilco_p]) else np.array([np.nan] * len(METRICS))

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=False)

    for i, m in enumerate(METRICS):
        ax = axes[i]
        xo, xf = series_by_metric[m]

        if len(xo) == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No pairs: {m}", ha="center")
            continue

        # Paired lines
        for a, b in zip(xo, xf):
            ax.plot([0, 1], [a, b], alpha=0.5, lw=1.0, color="#7f7f7f")

        # Points
        ax.scatter(
            [0] * len(xo), xo,
            s=24, alpha=0.9,
            c=palettes["oral"][0], label="Crohn-Oral",
        )
        ax.scatter(
            [1] * len(xf), xf,
            s=24, alpha=0.9,
            c=palettes["fecal"][0], label="Crohn-Fecal",
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Oral", "Fecal"])
        ax.set_title(m)
        ax.set_xlim(-0.3, 1.3)

        # Caption: ONLY p و q
        p = wilco_p[i]
        q = wilco_q[i] if i < len(wilco_q) else np.nan
        caption = format_annot_2x(p=p, q=q)

        ax.text(
            0.5, -0.22,
            caption,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9,
        )

    # Global legend: رنگ + n
    label_oral  = f"Crohn-Oral (n={n_pairs_global})"
    label_fecal = f"Crohn-Fecal (n={n_pairs_global})"
    handles = [
        mpatches.Patch(color=palettes["oral"][0],  label=label_oral),
        mpatches.Patch(color=palettes["fecal"][0], label=label_fecal),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper right")

    plt.suptitle("Crohn — Oral vs Fecal (Paired lines; WITH covariates cohort)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=300)
    plt.close()

# ------------------------------ Modeling --------------------------------
def _prep_design(dd: pd.DataFrame, ycol: str, xcols: List[str]):
    y = safe_numeric_series(dd[ycol])
    X = add_constant(dd[xcols], has_constant="add")
    X = safe_numeric_df(X)
    mask = np.isfinite(y.values) & np.all(np.isfinite(X.values), axis=1)
    y = y.loc[mask]; X = X.loc[mask]
    return y, X

def summarize_groups(df_site: pd.DataFrame, site_name: str) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        if m not in df_site.columns: continue
        for g in [0,1]:
            vals = df_site.loc[df_site["disease"]==g, m].dropna().values
            if vals.size == 0:
                rows.append({"site":site_name, "group":int(g), "metric":m, "n":0,
                             "mean":np.nan, "sd":np.nan, "median":np.nan, "iqr":np.nan})
                continue
            q1,q3 = np.quantile(vals,[0.25,0.75])
            rows.append({"site":site_name, "group":int(g), "metric":m, "n":int(vals.size),
                         "mean":float(np.mean(vals)),
                         "sd":float(np.std(vals, ddof=1)) if vals.size>1 else np.nan,
                         "median":float(np.median(vals)), "iqr":float(q3-q1)})
    return pd.DataFrame(rows)

def run_site_with_cov(site_name: str, alpha_all: pd.DataFrame, covars_df: pd.DataFrame,
                      outdir: Path, palettes: Dict[str,List[str]], title_tag: str):
    site_mask = alpha_all["site"] == site_name
    d_raw = alpha_all.loc[site_mask].copy()
    d_cov = d_raw.merge(covars_df, on="Sample_ID", how="inner", suffixes=("", "_cov"))

    # select fixed covariates only
    covar_pool = [c for c in FIXED_COVARS if c in d_cov.columns]
    for c in covar_pool + ["disease"]:
        if c == "Sex": d_cov[c] = d_cov[c].apply(_norm_sex)
        elif c in {"Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing"}:
            d_cov[c] = d_cov[c].apply(_to_bin01)
        else:
            d_cov[c] = pd.to_numeric(d_cov[c], errors="coerce")

    # Wilcoxon/MWU on matched
    rows_w = []
    for m in METRICS:
        dd = d_cov[["disease", m]].dropna()
        if dd.empty or dd["disease"].nunique() < 2:
            rows_w.append({"metric": m, "p_wilcox": np.nan, "cliffs_delta": np.nan}); continue
        x = dd.loc[dd["disease"]==1, m].values
        y = dd.loc[dd["disease"]==0, m].values
        try: p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        except Exception: p = np.nan
        rows_w.append({"metric": m, "p_wilcox": p, "cliffs_delta": cliffs_delta(x,y)})
    wilx_df = pd.DataFrame(rows_w); wilx_df["q_wilcox"] = fdr(wilx_df["p_wilcox"].tolist())

    # OLS per metric
    rows = []
    for m in METRICS:
        if m not in d_cov.columns:
            rows.append({"metric": m, "term": "(missing_metric)", "p.value":np.nan}); continue
        cols = ["disease"] + covar_pool
        dd = d_cov[[m] + cols].copy().dropna(subset=[m])
        if dd["disease"].nunique() < 2 or len(dd) < 6:
            rows.append({"metric": m, "term": "(skipped_small_n)", "p.value":np.nan}); continue
        y, X = _prep_design(dd, m, cols)
        if len(y) < 6 or X.shape[1] < 2:
            rows.append({"metric": m, "term": "(skipped_small_n)", "p.value":np.nan}); continue
        fit = OLS(y, X).fit(cov_type="HC3")
        for term, est, se, tval, pval in zip(fit.params.index, fit.params.values,
                                             fit.bse.values, fit.tvalues.values, fit.pvalues.values):
            rows.append({"metric": m, "term": term, "estimate": est,
                         "std.error": se, "statistic": tval, "p.value": pval})
    model_df = pd.DataFrame(rows)
    if not model_df.empty:
        model_df["q.value"] = np.nan
        for m in METRICS:
            idx = model_df["metric"] == m
            if idx.any(): model_df.loc[idx, "q.value"] = fdr(model_df.loc[idx, "p.value"].tolist())

    # save
    ensure_dir(outdir)
    model_df.to_csv(outdir / f"alpha_models_{site_name}.csv", index=False)
    desc = summarize_groups(d_cov[["disease"] + METRICS], site_name)
    desc["group_label"] = desc["group"].map({0:"Healthy",1:"Crohn"})
    desc.to_csv(outdir / f"group_summary_{site_name}_with_covariates.csv", index=False)

    # panel
    colors = palettes[site_name]
    panel_png = outdir / f"panel_{site_name}_4up_with_covariates.png"
    box4_panel(site_name, d_cov[["disease"] + METRICS].copy(), panel_png,
               crohn_color=colors[0], healthy_color=colors[1],
               title_suffix=f"WITH covariates — {title_tag}")

    return d_cov[["Sample_ID","site","disease"]+METRICS].copy(), model_df

# ------------------------------ Scenario runner ------------------------
def run_alpha(*, level: str, oral_crohn: str, fecal_crohn: str,
              oral_healthy_merged: str, oral_healthy_prev: Optional[str],
              fecal_healthy: str, covars_pool: str, pairs_path: Optional[str],
              colors_yaml: str, outdir: Path):

    palettes = load_colors(colors_yaml)
    outdir = Path(outdir); ensure_dir(outdir)

    with open(outdir / "colors_used.json", "w") as f:
        json.dump(palettes, f, indent=2)

    # Build alpha tables
    alpha_oral_C,  _  = build_alpha_table(oral_crohn,   "oral",  1)
    alpha_fecal_C, _  = build_alpha_table(fecal_crohn,  "fecal", 1)
    alpha_fecal_H, _  = build_alpha_table(fecal_healthy,"fecal", 0)

    # RAW uses merged Healthy-Oral
    alpha_oral_H_RAW, _ = build_alpha_table(oral_healthy_merged, "oral", 0)

    # WITH-covariates (if prev provided) uses previous-only Healthy-Oral
    alpha_oral_H_MAT = pd.DataFrame()
    if oral_healthy_prev:
        alpha_oral_H_MAT, _ = build_alpha_table(oral_healthy_prev, "oral", 0)

    # RAW pool
    alpha_raw = pd.concat([alpha_oral_C, alpha_fecal_C, alpha_oral_H_RAW, alpha_fecal_H],
                          axis=0, ignore_index=True).drop_duplicates(subset=["Sample_ID","site"])
    alpha_raw["disease"] = pd.to_numeric(alpha_raw["disease"], errors="coerce").fillna(0).astype(int)

    # Covariates
    covars = pd.read_csv(covars_pool, dtype=str)
    covars["Sample_ID"] = covars["Sample_ID"].astype(str).map(normalize_sample_id)
    if "Sex" in covars.columns: covars["Sex"] = covars["Sex"].apply(_norm_sex)
    for c in ["Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing"]:
        if c in covars.columns:
            covars[c] = covars[c].map(lambda x: 1 if str(x).strip().lower() in BIN_TRUE
                                      else 0 if str(x).strip().lower() in BIN_FALSE else pd.NA).astype("Int64")
    if "BMI" in covars.columns:
        covars["BMI"] = pd.to_numeric(covars["BMI"], errors="coerce")

    # Matched pool (previous-only for oral healthy)
    alpha_matched = alpha_raw.copy()
    if not alpha_oral_H_MAT.empty:
        alpha_matched = pd.concat([alpha_oral_C, alpha_fecal_C, alpha_oral_H_MAT, alpha_fecal_H],
                                  axis=0, ignore_index=True).drop_duplicates(subset=["Sample_ID","site"])
    alpha_matched = alpha_matched.merge(covars, on="Sample_ID", how="inner", suffixes=("", "_cov"))

    # Audits
    j = alpha_raw.merge(covars[["Sample_ID"]].drop_duplicates(), on="Sample_ID", how="left", indicator=True)
    with open(outdir / "join_report.txt", "w") as fh:
        fh.write(f"[Alpha]\nLevel: {level}\n")
        fh.write(f"Alpha rows (RAW): {len(alpha_raw)}\n")
        fh.write(f"Matched rows (WITH covariates): {int((j['_merge']=='both').sum())}\n")
        fh.write(f"Missing in covariates: {int((j['_merge']=='left_only').sum())}\n")
    j.loc[j["_merge"]=="left_only", ["Sample_ID","site","disease"]].to_csv(
        outdir / "alpha_unmatched_in_covariates.csv", index=False
    )
    ccheck = covars.merge(alpha_raw[["Sample_ID"]].drop_duplicates(), on="Sample_ID", how="left", indicator=True)
    ccheck.loc[ccheck["_merge"]=="left_only", ["Sample_ID"]].to_csv(
        outdir / "covariates_unmatched_in_alpha.csv", index=False
    )
    alpha_matched[["Sample_ID"] + [c for c in alpha_matched.columns if c in {"site","disease"}|set(METRICS)]].to_csv(
        outdir / "alpha_with_covariates.csv", index=False
    )

    # Panels — RAW (merged Healthy-Oral)
    box4_panel("oral",  alpha_raw[alpha_raw["site"]=="oral"][["disease"]+METRICS],   outdir / "panel_oral_4up_raw.png",
               palettes["oral"][0],  palettes["oral"][1],  f"RAW (no covariates) — level={level}")
    box4_panel("fecal", alpha_raw[alpha_raw["site"]=="fecal"][["disease"]+METRICS],  outdir / "panel_fecal_4up_raw.png",
               palettes["fecal"][0], palettes["fecal"][1], f"RAW (no covariates) — level={level}")

    # Panels — WITH covariates (previous-only Healthy-Oral, matched)
    matched_oral_tbl,  _ = run_site_with_cov("oral",  alpha_raw, covars, outdir, palettes,
                                             title_tag=f"level={level}")
    matched_fecal_tbl, _ = run_site_with_cov("fecal", alpha_raw, covars, outdir, palettes,
                                             title_tag=f"level={level}")
    matched_all = pd.concat([matched_oral_tbl, matched_fecal_tbl], axis=0, ignore_index=True)

    # 4-groups
    four_group_panel(alpha_raw.copy(),               outdir / "panel_4groups_raw.png",
                     palettes, title_suffix=f"RAW (no covariates) — level={level}")
    four_group_panel(matched_all.copy(),             outdir / "panel_4groups_with_covariates.png",
                     palettes, title_suffix=f"WITH covariates — level={level}")

    # Crohn paired (restrict to covariates cohort)
    pairs: List[Tuple[str,str]] = []
    if pairs_path and Path(pairs_path).exists():
        try:
            pdf = pd.read_csv(pairs_path)
            oc = fc = None
            for c in pdf.columns:
                lc = c.lower()
                if oc is None and ("oral" in lc or lc in {"oral_id","sample_id_oral"}): oc = c
                if fc is None and ("fecal" in lc or "faecal" in lc or lc in {"fecal_id","stool","sample_id_fecal"}): fc = c
            if oc and fc:
                for _, r in pdf[[oc,fc]].dropna().iterrows():
                    pairs.append((normalize_sample_id(str(r[oc])), normalize_sample_id(str(r[fc]))))
        except Exception:
            pairs = []
    crohn_paired_lines_panel(alpha_raw, covars, pairs, outdir / "panel_paired_crohn_oral_vs_fecal.png", palettes)

    # Crohn-only and PPI models (WITH covariates)
    def _ppi(site):
        df = alpha_raw.merge(covars, on="Sample_ID", how="inner", suffixes=("", "_cov"))
        df = df[(df["site"]==site) & (df["disease"]==1)].copy()
        covars_here = [c for c in FIXED_COVARS if c in df.columns]
        for c in covars_here:
            if c == "Sex": df[c] = df[c].apply(_norm_sex)
            elif c in {"Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing"}:
                df[c] = df[c].apply(_to_bin01)
            else: df[c] = pd.to_numeric(df[c], errors="coerce")
        rows = []
        for m in METRICS:
            if m not in df.columns: continue
            rhs = [v for v in covars_here if v != "PPI_use"]
            use = ["PPI_use"] + rhs
            dd = df[[m] + use].dropna()
            y, X = _prep_design(dd, m, use)
            if len(y) < 6 or X.shape[1] < 2 or dd["PPI_use"].nunique() < 2:
                rows.append({"metric":m, "term":"(skipped_small_n)", "p.value":np.nan}); continue
            fit = OLS(y, X).fit(cov_type="HC3")
            for term, est, se, tval, pval in zip(fit.params.index, fit.params.values,
                                                 fit.bse.values, fit.tvalues.values, fit.pvalues.values):
                rows.append({"metric":m, "term":term, "estimate":est,
                             "std.error":se, "statistic":tval, "p.value":pval})
        out = pd.DataFrame(rows)
        if not out.empty:
            out["q.value"] = np.nan
            for m in METRICS:
                idx = out["metric"]==m
                if idx.any(): out.loc[idx, "q.value"] = fdr(out.loc[idx, "p.value"].tolist())
        out.to_csv(outdir / f"alpha_PPI_{site}_WITH_covariates.csv", index=False)

    _ppi("oral"); _ppi("fecal")

    # Crohn-only multi-covariate model (WITH covariates) — keep to Crohn cohort, fixed covars
    def _crohn_only(site):
        df = alpha_raw.merge(covars, on="Sample_ID", how="inner", suffixes=("", "_cov"))
        df = df[(df["site"]==site) & (df["disease"]==1)].copy()
        covs = [c for c in FIXED_COVARS if c in df.columns]
        for c in covs:
            if c == "Sex": df[c] = df[c].apply(_norm_sex)
            elif c in {"Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing"}:
                df[c] = df[c].apply(_to_bin01)
            else: df[c] = pd.to_numeric(df[c], errors="coerce")
        rows = []
        for m in METRICS:
            if m not in df.columns: continue
            dd = df[[m] + covs].dropna()
            if len(dd) < 8 or any(dd[c].nunique() < 2 for c in covs if c != "Sex"):
                rows.append({"metric":m, "term":"(skipped_small_n)", "p.value":np.nan}); continue
            y, X = _prep_design(dd, m, covs)
            if len(y) < 8 or X.shape[1] < 2:
                rows.append({"metric":m, "term":"(skipped_small_n)", "p.value":np.nan}); continue
            fit = OLS(y, X).fit(cov_type="HC3")
            for term, est, se, tval, pval in zip(fit.params.index, fit.params.values,
                                                 fit.bse.values, fit.tvalues.values, fit.pvalues.values):
                rows.append({"metric":m, "term":term, "estimate":est,
                             "std.error":se, "statistic":tval, "p.value":pval})
        out = pd.DataFrame(rows)
        if not out.empty:
            out["q.value"] = np.nan
            for m in METRICS:
                idx = out["metric"]==m
                if idx.any(): out.loc[idx, "q.value"] = fdr(out.loc[idx, "p.value"].tolist())
        out.to_csv(outdir / f"alpha_Crohn_only_{site}_WITH_covariates.csv", index=False)

    _crohn_only("oral"); _crohn_only("fecal")

    # oral vs fecal (unpaired) by disease
    rows = []
    for status, lab in [(1,"Crohn"), (0,"Healthy")]:
        sub = alpha_raw[alpha_raw["disease"]==status]
        for m in METRICS:
            dd = sub[["site", m]].dropna()
            if dd["site"].nunique() < 2:
                rows.append({"group":lab, "metric":m, "p_mwu":np.nan}); continue
            x = dd.loc[dd["site"]=="oral",  m].values
            y = dd.loc[dd["site"]=="fecal", m].values
            try: p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            except Exception: p = np.nan
            rows.append({"group":lab, "metric":m, "p_mwu":p})
    pd.DataFrame(rows).assign(q_mwu=lambda d: fdr(d["p_mwu"].tolist())) \
        .to_csv(outdir / "oral_fecal_unpaired_by_group.csv", index=False)

        # paired wilcoxon summary
    # Use the SAME pairs as for the paired plot if available;
    # otherwise fall back to ID intersection.
    crohn = alpha_raw[alpha_raw["disease"] == 1].copy()
    co_oral  = crohn[crohn["site"] == "oral"].set_index("Sample_ID")
    co_fecal = crohn[crohn["site"] == "fecal"].set_index("Sample_ID")

    if not pairs:  # no pairs file → try automatic matching
        inter = co_oral.index.intersection(co_fecal.index)
        pairs_for_stats = [(sid, sid) for sid in inter]
    else:
        pairs_for_stats = pairs

    rows = []
    for m in METRICS:
        xo, xf = [], []
        for o_id, f_id in pairs_for_stats:
            if m in co_oral.columns and o_id in co_oral.index:
                vo = co_oral.loc[o_id, m]
            else:
                vo = np.nan
            if m in co_fecal.columns and f_id in co_fecal.index:
                vf = co_fecal.loc[f_id, m]
            else:
                vf = np.nan
            if pd.notna(vo) and pd.notna(vf):
                xo.append(float(vo))
                xf.append(float(vf))

        if len(xo) == 0:
            rows.append({
                "metric": m,
                "n_pairs": 0,
                "wilcoxon_stat": np.nan,
                "p_value": np.nan
            })
            continue

        stat, p = wilcoxon(
            xo, xf,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            mode="auto",
        )
        rows.append({
            "metric": m,
            "n_pairs": len(xo),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
        })

    pd.DataFrame(rows).assign(
        q_value=lambda d: fdr(d["p_value"].tolist())
    ).to_csv(outdir / "paired_wilcoxon_summary.csv", index=False)
# ----------------------------------- Main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oral-crohn", required=True)
    ap.add_argument("--fecal-crohn", required=True)
    ap.add_argument("--oral-healthy-merged", required=True)
    ap.add_argument("--oral-healthy-prev", required=False)
    ap.add_argument("--fecal-healthy", required=True)
    ap.add_argument("--covars-pool", required=True)
    ap.add_argument("--pairs", required=False)
    ap.add_argument("--colors", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--level", required=True, choices=["genus","species"])
    args = ap.parse_args()

    outdir_root = Path(args.outdir); ensure_dir(outdir_root)
    run_alpha(
        level=args.level,
        oral_crohn=args.oral_crohn,
        fecal_crohn=args.fecal_crohn,
        oral_healthy_merged=args.oral_healthy_merged,
        oral_healthy_prev=args.oral_healthy_prev,
        fecal_healthy=args.fecal_healthy,
        covars_pool=args.covars_pool,
        pairs_path=args.pairs,
        colors_yaml=args.colors,
        outdir=outdir_root
    )
    print(f"[INFO] Alpha analysis complete. Outdir → {outdir_root}")

if __name__ == "__main__":
    main()