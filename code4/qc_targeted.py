#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_targeted.py — Targeted QC visuals + optional PCA/PERMANOVA (robust)

This script is a drop-in replacement compatible with your current Snakemake rule.
Key robustness features:
  - Safe metadata normalization & alignment with feature matrix
  - Graceful handling of missing/invalid columns
  - Optional permutation *blocks* (strata) derived from pairs or subject_id
  - PERMANOVA via scikit-bio if available; otherwise a constrained fallback
  - Always writes all requested output files (or 'No data' PNGs)

CLI (unchanged):
  --pct-all           path to taxa×samples percentage table (0..100) [index=taxa]
  --meta              pooled metadata CSV (flexibly parsed)
  --ppi-effects-all   optional prioritization table to select informative taxa
  --pairs             optional pairs CSV (for Crohn spaghetti & strata)
  --rank              genus|species
  --outdir            base output directory (results/qc_targeted)
  --topk              number of "top taxa" to visualize (default: 12)
  --colors            optional YAML color map (config/colors.yml)
  --with-pca-permanova  if set, also emit PCA scatter + PERMANOVA TSV
"""

import os
import argparse
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Try scikit-bio PERMANOVA; fall back to custom if missing
_HAS_SKBIO = True
try:
    from skbio import DistanceMatrix as SKB_DM
    from skbio.stats.distance import permanova as skb_permanova
except Exception:
    _HAS_SKBIO = False

# YAML is optional; fall back to defaults if missing
try:
    import yaml
except Exception:
    yaml = None


# =============================================================================
# Small utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def safe_write_empty_png(path: str, msg: str = "No data") -> None:
    """Emit a simple PNG (so Snakemake tests pass) with a friendly message."""
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_to_true = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_true:
            return lower_to_true[c.lower()]
    return None


# =============================================================================
# Metadata normalization (to: sample_id, site, disease, ppi_use)
# =============================================================================

def norm_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize metadata to standardized columns:
      sample_id (str), site in {Oral,Fecal}, disease in {0,1}, ppi_use in {0,1}
    Missing columns become NaN. Values are coerced when possible.
    """
    m = meta.copy()
    m.columns = [str(c).strip() for c in m.columns]

    def get_col(*cands, required=False):
        name = _first_existing_column(m, list(cands))
        if name is None:
            s = pd.Series([np.nan] * len(m), index=m.index)
            if required:
                warnings.warn(f"Required metadata column missing among {cands}; filling with NaN.")
            return s
        return m[name]

    out = pd.DataFrame({
        "sample_id": get_col("sample_id","id","sample","sampleid", required=True).astype(str),
        "site":      get_col("site","body_site","location"),
        "disease":   get_col("disease","status","group"),
        "ppi_use":   get_col("ppi_use","ppi","ppi_current","ppi3m"),
    })

    # Site normalization (Oral/Fecal)
    out["site"] = out["site"].astype(str).str.strip().str.capitalize().replace({"Faecal": "Fecal"})
    bad = ~out["site"].isin(["Oral", "Fecal"])
    out.loc[bad, "site"] = np.nan

    def to01(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        pos = {"1","yes","true","y","crohn","cd","case","ibd"}
        neg = {"0","no","false","n","healthy","control","hc","non-ibd","nonibd"}
        if s in pos: return 1
        if s in neg: return 0
        try:
            v = int(float(s))
            return v if v in (0,1) else np.nan
        except Exception:
            return np.nan

    out["disease"] = out["disease"].map(to01)
    out["ppi_use"] = out["ppi_use"].map(to01)
    return out


# =============================================================================
# Colors
# =============================================================================

def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = [max(0, min(1, x)) for x in rgb]
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

def _mix(c1, c2, w=0.5):
    a = np.array(_hex_to_rgb(c1)); b = np.array(_hex_to_rgb(c2))
    return _rgb_to_hex(tuple(a*(1-w) + b*w))

def _lighten(hexcolor, amt=0.25): return _mix(hexcolor, "#ffffff", amt)
def _darken(hexcolor, amt=0.20):  return _mix(hexcolor, "#000000", amt)

def load_colors(yaml_path=None):
    defaults = {
        "Fecal_Cr":   "#30638e",
        "Oral_Cr":    "#edae49",
        "Fecal_Hc":   "#d1495b",
        "Oral_Hc":    "#00798c",
    }
    cfg = None
    if yaml_path and os.path.exists(yaml_path) and yaml is not None:
        try:
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = None

    g = ((cfg or {}).get("group") or {}) if cfg else {}
    group_colors = {
        "Fecal_Crohn":   g.get("Fecal_Crohn",   defaults["Fecal_Cr"]),
        "Oral_Crohn":    g.get("Oral_Crohn",    defaults["Oral_Cr"]),
        "Fecal_Healthy": g.get("Fecal_Healthy", defaults["Fecal_Hc"]),
        "Oral_Healthy":  g.get("Oral_Healthy",  defaults["Oral_Hc"]),
    }
    site_base = {"Oral": group_colors["Oral_Healthy"], "Fecal": group_colors["Fecal_Healthy"]}

    def ppi_palette(site):
        base = site_base.get(site, "#777777")
        return (_lighten(base, 0.35), _darken(base, 0.15))  # (PPI=0, PPI=1)

    bases = [group_colors["Oral_Healthy"], group_colors["Oral_Crohn"],
             group_colors["Fecal_Healthy"], group_colors["Fecal_Crohn"]]
    qual = []
    for b in bases:
        qual += [_lighten(b, 0.10), b, _darken(b, 0.18)]
    return group_colors, site_base, ppi_palette, qual[:12]


# =============================================================================
# Data transforms & selection
# =============================================================================

def clr_transform(pct_df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """Centered log-ratio (CLR) on a taxa×samples percentage matrix (0..100)."""
    X = (pct_df.astype(float) / 100.0) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)
    return logX.sub(gm, axis=1)

def prettify_taxon(t: str, rank: str) -> str:
    if t is None:
        return ""
    r = (rank or "").lower()
    if r == "genus" and t.startswith("g__"):
        name = t.replace("g__", "").replace("_", " ")
        return " ".join(w.capitalize() for w in name.split())
    if r == "species" and t.startswith("s__"):
        name = t.replace("s__", "").replace("_", " ")
        toks = name.split()
        if not toks:
            return t
        toks[0] = toks[0].capitalize()
        toks[1:] = [x.lower() for x in toks[1:]]
        return " ".join(toks)
    return t

def pick_top_taxa(ppi_all_csv: str, n: int, fallback_by_var: pd.Series) -> List[str]:
    """Prefer taxa prioritized in PPI table by q/p; fallback to highest-variance taxa."""
    if ppi_all_csv and os.path.exists(ppi_all_csv):
        try:
            df = pd.read_csv(ppi_all_csv)
            if {"taxon", "term"}.issubset(df.columns):
                prio = []
                for term in ["ppi_use", "disease:site"]:
                    sub = df[df["term"].astype(str).str.startswith(term, na=False)].copy()
                    if sub.empty:
                        continue
                    if "q" in sub and sub["q"].notna().any():
                        sub = sub.sort_values("q")
                    elif "p" in sub:
                        sub = sub.sort_values("p")
                    prio.append(sub.head(n*2))
                if prio:
                    cand = pd.concat(prio, ignore_index=True)
                    if "p" in cand:
                        top = cand.groupby("taxon")["p"].min().sort_values().head(n).index.tolist()
                        if top:
                            return top
                    return cand["taxon"].value_counts().index[:n].tolist()
        except Exception as e:
            warnings.warn(f"Failed to read {ppi_all_csv}: {e}")
    return fallback_by_var.sort_values(ascending=False).head(n).index.tolist()


# =============================================================================
# PERMANOVA (scikit-bio if available; else custom with optional strata)
# =============================================================================

def _build_strata_from_pairs(pairs_df: pd.DataFrame, sample_ids: pd.Index) -> Optional[pd.Series]:
    """Make block IDs from a pairs CSV; returns None if invalid."""
    p = pairs_df.copy()
    p.columns = [str(c).lower() for c in p.columns]
    cand_oral  = ["oral_sample_id","oral_id","oral","oc","oral_sample"]
    cand_fecal = ["fecal_sample_id","fecal_id","fecal","fc","fecal_sample"]
    col_o = next((c for c in cand_oral  if c in p.columns), None)
    col_f = next((c for c in cand_fecal if c in p.columns), None)
    if not col_o or not col_f:
        return None

    long = pd.DataFrame({
        "sample_id": pd.concat([p[col_o].astype(str), p[col_f].astype(str)], ignore_index=True),
        "subj": np.repeat(np.arange(len(p)), 2)
    })
    m = pd.Series(index=sample_ids.astype(str), dtype="float")
    mm = long.set_index("sample_id")["subj"]
    idx = mm.index.intersection(m.index)
    if len(idx) == 0:
        return None
    m.loc[idx] = mm.loc[idx].values
    tb = m.value_counts(dropna=True)
    if len(tb)==0 or tb.min() < 2:
        return None
    return m.astype(int).astype(str)

def _permanova_skbio(X: np.ndarray, g: np.ndarray, strata: Optional[pd.Series], n_perm: int):
    """PERMANOVA via scikit-bio on Euclidean distances of centered CLR."""
    D = squareform(pdist(X, metric="euclidean"))
    dm = SKB_DM(D, ids=[str(i) for i in range(X.shape[0])])
    grouping = pd.Series(g, index=dm.ids).astype(str)
    strata_use = None
    if strata is not None:
        # Map strata to the DistanceMatrix rows by sample order in X; here X rows already masked/aligned
        # We assume X rows align with some ordering of sample IDs; since we lost IDs here, we just pass None
        # in scikit-bio path unless we keep IDs. To keep IDs properly, we require caller to pass X with aligned index.
        # For simplicity here, we'll not use strata in skbio path unless caller provides aligned series.
        strata_use = None  # see custom path for proper strata permutation
    res = skb_permanova(dm, grouping=grouping.values, permutations=n_perm, strata=strata_use)
    return {
        "F": float(res['test statistic']),
        "R2": float(res['R^2']),
        "p": float(res['p-value']),
        "n_perm": int(n_perm)
    }

def _permute_within_blocks(labels: np.ndarray, blocks: np.ndarray, rng: np.random.Generator):
    """Yield a permutation of labels shuffling *within* each block."""
    perm = labels.copy()
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        if len(idx) > 1:
            perm[idx] = rng.permutation(perm[idx])
    return perm

def _permanova_custom(X: np.ndarray, g: np.ndarray, n_perm: int, rng: np.random.Generator,
                      blocks: Optional[np.ndarray] = None):
    """
    One-factor PERMANOVA with Euclidean distance on centered CLR.
    Supports optional within-block permutation when `blocks` is provided.
    Returns F, R2, p, and the effective number of distinct permutations used.
    """
    n = X.shape[0]
    lev = pd.unique(g)
    if n < 4 or len(lev) < 2:
        return {"F": np.nan, "R2": np.nan, "p": np.nan, "n_perm": n_perm, "n_eff": 0}

    D = squareform(pdist(X, metric="euclidean"))
    TSS = (D**2).sum() / n

    def _anova_terms(labels):
        SSW, dfw = 0.0, 0
        for lv in lev:
            idx = np.where(labels == lv)[0]
            if len(idx) <= 1:
                continue
            Dg = D[np.ix_(idx, idx)]
            SSW += (Dg**2).sum() / len(idx)
            dfw += (len(idx) - 1)
        dfb = len(lev) - 1
        return SSW, dfw, dfb

    SSW, dfw, dfb = _anova_terms(g)
    if dfw <= 0 or dfb <= 0:
        return {"F": np.nan, "R2": np.nan, "p": np.nan, "n_perm": n_perm, "n_eff": 0}

    SSB = TSS - SSW
    MSW = SSW / dfw
    MSB = SSB / dfb
    Fobs = MSB / MSW if MSW > 0 else np.nan
    R2 = SSB / TSS if TSS > 0 else np.nan

    # Permutations (skip identities; count distinct)
    perm_F = []
    n_eff = 0
    for _ in range(n_perm):
        if blocks is None:
            p = rng.permutation(g)
        else:
            p = _permute_within_blocks(g, blocks, rng)
        if np.array_equal(p, g):
            # identical assignment → skip
            continue
        SSWp, dfwp, _ = _anova_terms(p)
        if dfwp <= 0:
            continue
        SSBp = TSS - SSWp
        Fp = (SSBp / (len(lev)-1)) / (SSWp / dfwp) if (SSWp > 0) else np.nan
        if np.isfinite(Fp):
            perm_F.append(Fp)
            n_eff += 1

    if n_eff == 0 or not np.isfinite(Fobs):
        pval = np.nan
    else:
        perm_F = np.asarray(perm_F)
        pval = (1 + np.sum(perm_F >= Fobs)) / (1 + n_eff)

    return {"F": float(Fobs), "R2": float(R2), "p": float(pval), "n_perm": int(n_perm), "n_eff": int(n_eff)}

def run_permanova(sample_df: pd.DataFrame,
                  meta_df: pd.DataFrame,
                  factor: str,
                  pairs_df: Optional[pd.DataFrame],
                  n_perm: int = 999,
                  rng_seed: int = 1):
    """
    Wrapper to run PERMANOVA for a single factor with sensible fallbacks.

    Args:
      sample_df : samples×features CLR matrix (rows aligned to meta_df)
      meta_df   : metadata aligned to sample_df rows; must contain `factor`
      factor    : 'site' or 'disease'
      pairs_df  : optional pairs table to derive permutation blocks
      n_perm    : number of permutations requested
      rng_seed  : RNG seed for reproducibility

    Returns:
      dict with F, R2, p, n_perm, n_eff, n_samples, n_taxa, n_groups
    """
    rng = np.random.default_rng(rng_seed)

    # Mask rows with missing factor
    g = meta_df[factor].values
    mask = ~pd.isna(g)
    X = sample_df.values[mask, :]
    g = g[mask]

    # Not enough data?
    levels = pd.unique(g)
    if X.shape[0] < 4 or len(levels) < 2:
        return {"factor": factor, "n_groups": int(len(levels)),
                "F": np.nan, "R2": np.nan, "p": np.nan,
                "n_perm": n_perm, "n_eff": 0,
                "n_samples": int(X.shape[0]), "n_taxa": int(sample_df.shape[1])}

    # Column-center (CLR is already centered per sample, but we mean-center features too)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    Xc = np.nan_to_num(Xc, nan=0.0)

    # Build strata if possible (pairs → subject-level blocks; else try subject_id)
    blocks = None
    if pairs_df is not None:
        try:
            strata = _build_strata_from_pairs(pairs_df, meta_df.index[mask].astype(str))
        except Exception as e:
            warnings.warn(f"Failed building strata from pairs: {e}")
            strata = None
    else:
        strata = None

    # If no valid pairs-based strata, try subject_id
    if strata is None and "subject_id" in meta_df.columns:
        s = meta_df["subject_id"].astype(str).values[mask]
        vc = pd.Series(s).value_counts()
        if len(vc) > 0 and vc.min() >= 2:
            strata = pd.Series(s, index=np.arange(len(s))).astype(str)

    # If strata exist but min block < 2, disable
    if strata is not None:
        tb = pd.Series(strata).value_counts()
        if tb.empty or tb.min() < 2:
            strata = None

    # IMPORTANT: Never block on the same factor you test
    # (e.g., when testing 'site', 'strata' must NOT be 'site')
    if strata is not None:
        blocks = np.asarray(strata.values if isinstance(strata, pd.Series) else strata).astype(str)
        # But if blocks are identical to g (1-1 mapping), disable
        if len(pd.Series(blocks).value_counts()) == len(blocks):
            # every sample its own block → invalid
            blocks = None

    # Try scikit-bio if installed (NOTE: we don't pass blocks here because we lost stable IDs;
    # our custom path below correctly handles within-block permutations)
    if _HAS_SKBIO and blocks is None:
        try:
            res = _permanova_skbio(Xc, g, strata=None, n_perm=n_perm)
            return {"factor": factor, "n_groups": int(len(levels)),
                    **res, "n_eff": n_perm, "n_samples": int(Xc.shape[0]), "n_taxa": int(Xc.shape[1])}
        except Exception as e:
            warnings.warn(f"scikit-bio PERMANOVA failed ({e}); falling back to custom.")

    # Custom path (supports blocks)
    res = _permanova_custom(Xc, g.astype(str), n_perm=n_perm, rng=rng, blocks=blocks)
    return {"factor": factor, "n_groups": int(len(levels)),
            **res, "n_samples": int(Xc.shape[0]), "n_taxa": int(Xc.shape[1])}


# =============================================================================
# Plots (unchanged logic, minor cleanups)
# =============================================================================

def pca_scatter_top_taxa(long_df: pd.DataFrame, top_taxa: List[str], out_png: str) -> None:
    """2D PCA on samples×top_taxa (CLR). Color by site; marker by disease."""
    if not top_taxa:
        safe_write_empty_png(out_png, "No taxa selected for PCA")
        return

    d = long_df[long_df["taxon"].isin(top_taxa)].copy()
    if d.empty:
        safe_write_empty_png(out_png, "No rows for selected taxa")
        return

    M = d.pivot_table(index="Sample_ID", columns="taxon", values="CLR", aggfunc="mean")
    M = M[[t for t in top_taxa if t in M.columns]]  # keep order
    if M.shape[0] < 2:
        safe_write_empty_png(out_png, "Not enough samples for PCA")
        return

    meta_small = d[["Sample_ID", "site", "disease"]].drop_duplicates().set_index("Sample_ID")
    meta_small = meta_small.reindex(M.index)

    X = M.values.astype(float)
    X = X - np.nanmean(X, axis=0, keepdims=True)
    X = np.nan_to_num(X, nan=0.0)

    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pcs = U[:, :2] * S[:2]
    ss = (S**2) / max(1, (X.shape[0]-1))
    var_ratio = ss / ss.sum() if ss.sum() > 0 else np.array([np.nan, np.nan])
    v1, v2 = float(var_ratio[0]*100), float(var_ratio[1]*100)

    site_color = {"Oral": "#00798c", "Fecal": "#d1495b"}
    disease_marker = {0: "o", 1: "s"}

    ensure_dir(os.path.dirname(out_png))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    gb = meta_small.reset_index().groupby(["site", "disease"])["Sample_ID"].apply(list)
    for (site, dis), indices in gb.items():
        if site not in site_color or dis not in disease_marker:
            continue
        idx = [M.index.get_loc(i) for i in indices if i in M.index]
        if not idx:
            continue
        pts = pcs[idx, :]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=site_color[site], marker=disease_marker[dis],
                   s=52, alpha=0.9,
                   label=f"{site}, {'Healthy' if dis==0 else 'Crohn'}")

    ax.set_xlabel(f"PC1 ({v1:.1f}% var)" if np.isfinite(v1) else "PC1")
    ax.set_ylabel(f"PC2 ({v2:.1f}% var)" if np.isfinite(v2) else "PC2")
    ax.set_title("PCA (top taxa, CLR)")
    ax.axhline(0, color="#bbbbbb", lw=0.8); ax.axvline(0, color="#bbbbbb", lw=0.8)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _cohen_d(x0, x1) -> float:
    x0 = np.asarray(pd.to_numeric(pd.Series(x0), errors="coerce")); x0 = x0[np.isfinite(x0)]
    x1 = np.asarray(pd.to_numeric(pd.Series(x1), errors="coerce")); x1 = x1[np.isfinite(x1)]
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    m0, m1 = x0.mean(), x1.mean()
    s0, s1 = x0.std(ddof=1), x1.std(ddof=1)
    sp = np.sqrt(((len(x0)-1)*s0**2 + (len(x1)-1)*s1**2) / (len(x0)+len(x1)-2))
    if sp <= 0 or not np.isfinite(sp):
        return np.nan
    return (m1 - m0) / sp

def bootstrap_mean_ci(a, n_boot=1000, alpha=0.05, rng=None):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) == 0:
        return (np.nan, np.nan, np.nan)
    if rng is None:
        rng = np.random.default_rng(13)
    m = float(np.mean(a))
    if len(a) < 2:
        return (m, np.nan, np.nan)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boots = np.mean(a[idx], axis=1)
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return (m, float(lo), float(hi))

def box_by_ppi(long_df: pd.DataFrame, taxa: List[str], site: str, out_png: str, ppi_palette_fn) -> None:
    ensure_dir(os.path.dirname(out_png))
    d = long_df[(long_df["site"] == site) & (long_df["taxon"].isin(taxa))].copy()
    if d.empty or "ppi_use" not in d or d["ppi_use"].isna().all():
        safe_write_empty_png(out_png, f"No PPI data for {site}")
        return

    order = list(taxa)
    pos = np.arange(len(order), dtype=float)
    c0, c1 = ppi_palette_fn(site)  # PPI 0 / 1
    data0 = [d[(d["taxon"]==t) & (d["ppi_use"]==0)]["CLR"].astype(float).dropna().values for t in order]
    data1 = [d[(d["taxon"]==t) & (d["ppi_use"]==1)]["CLR"].astype(float).dropna().values for t in order]

    fig_w = min(14, 1.2 * max(1, len(taxa)) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    bp0 = ax.boxplot(data0, positions=pos-0.18, widths=0.32, patch_artist=True)
    bp1 = ax.boxplot(data1, positions=pos+0.18, widths=0.32, patch_artist=True)

    for b in bp0['boxes']: b.set(facecolor=c0, edgecolor=_darken(c0,0.35), alpha=0.95)
    for b in bp1['boxes']: b.set(facecolor=c1, edgecolor=_darken(c1,0.35), alpha=0.95)
    for part in ["whiskers","caps","medians","fliers"]:
        for l in bp0.get(part, []): l.set(color=_darken(c0,0.45))
        for l in bp1.get(part, []): l.set(color=_darken(c1,0.45))

    ax.set_xticks(pos)
    ax.set_xticklabels(order, rotation=60, ha="right")
    ax.set_title(f"CLR by PPI — {site}")
    ax.set_ylabel("CLR")

    all_vals = [v for vv in (data0 + data1) for v in (vv if len(vv)>0 else [np.nan])]
    if np.isfinite(all_vals).any():
        ymax = np.nanmax([np.nanmax(v) if len(v)>0 else np.nan for v in (data0+data1)])
        ymin = np.nanmin([np.nanmin(v) if len(v)>0 else np.nan for v in (data0+data1)])
        yr  = (ymax - ymin) if np.isfinite(ymax - ymin) else 1.0
        yoff = 0.06 * yr
        for i, (x0, x1) in enumerate(zip(data0, data1)):
            if len(x0)==0 and len(x1)==0:
                continue
            xl, xr = pos[i]-0.18, pos[i]+0.18
            top_i = np.nanmax([np.nanmax(x0) if len(x0)>0 else np.nan,
                               np.nanmax(x1) if len(x1)>0 else np.nan])
            yline = (top_i if np.isfinite(top_i) else ymax) + 0.8*yoff
            ax.plot([xl, xr], [yline, yline], color="#6e6e6e", lw=1.4, clip_on=False)

            dmed = (np.nanmedian(x1) - np.nanmedian(x0)) if (len(x0)>0 and len(x1)>0) else np.nan
            d_eff = _cohen_d(x0, x1)
            if np.isfinite(dmed) and np.isfinite(d_eff):
                ax.text((xl+xr)/2, yline + 0.2*yoff, f"Δ median = {dmed:.2f}\n d = {d_eff:.2f}",
                        ha="center", va="bottom", fontsize=9, color="#333", clip_on=False)

    n0 = sum(len(v) for v in data0)
    n1 = sum(len(v) for v in data1)
    legend_handles = [Patch(facecolor=c0, edgecolor=_darken(c0,0.35)),
                      Patch(facecolor=c1, edgecolor=_darken(c1,0.35))]
    ax.legend(legend_handles, [f"PPI=0 (n={n0})", f"PPI=1 (n={n1})"], loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def interaction_means(long, taxa, out_png, rank, qual_palette, outdir_plotdata):
    d = long[long["taxon"].isin(taxa)].copy()
    ensure_dir(os.path.dirname(out_png))
    if d.empty:
        safe_write_empty_png(out_png, "No data for interaction means")
        return

    rows = []
    rng = np.random.default_rng(101)
    for t in taxa:
        for disease in [0,1]:
            for site in ["Oral","Fecal"]:
                vals = d[(d["taxon"]==t) & (d["disease"]==disease) & (d["site"]==site)]["CLR"].values
                m, lo, hi = bootstrap_mean_ci(vals, n_boot=1000, rng=rng)
                rows.append({"taxon": t, "pretty_taxon": prettify_taxon(t, rank),
                             "disease": disease, "site": site, "mean": m, "lo": lo, "hi": hi,
                             "n": int(len(vals))})
    G = pd.DataFrame(rows)
    ensure_dir(outdir_plotdata)
    G.to_csv(os.path.join(outdir_plotdata, "plotdata_interaction_means.csv"), index=False)

    colors = (qual_palette * ((len(taxa)+len(qual_palette)-1)//len(qual_palette)))[:len(taxa)]
    disease_style = {0: dict(ls="-",  marker="o", label="Healthy"),
                     1: dict(ls="--", marker="s", label="Crohn")}

    fig, ax = plt.subplots(figsize=(max(12, 1.0*len(taxa)+6), 6))
    x_positions = {t: i for i,t in enumerate(taxa)}  # base per taxon
    for t,c in zip(taxa, colors):
        for dis in [0,1]:
            row_o = G[(G["taxon"]==t) & (G["disease"]==dis) & (G["site"]=="Oral")]
            row_f = G[(G["taxon"]==t) & (G["disease"]==dis) & (G["site"]=="Fecal")]
            if row_o.empty or row_f.empty:
                continue
            xs = [x_positions[t]+0.0, x_positions[t]+0.6]  # Oral→Fecal horizontally
            ys = [row_o["mean"].iloc[0], row_f["mean"].iloc[0]]
            lo = [row_o["lo"].iloc[0],   row_f["lo"].iloc[0]]
            hi = [row_o["hi"].iloc[0],   row_f["hi"].iloc[0]]
            style = disease_style[dis]
            ax.plot(xs, ys, color=c, lw=2.2, ls=style["ls"], marker=style["marker"])
            ax.vlines(xs, lo, hi, color=c, lw=1.5, alpha=0.9)
    ax.axhline(0, ls=":", lw=1, color="#9aa0a6")

    ax.set_xticks([x_positions[t]+0.3 for t in taxa])
    ax.set_xticklabels([prettify_taxon(t, rank) for t in taxa], rotation=55, ha="right")
    ax.set_ylabel("Mean CLR")
    ax.set_title("Interaction means (disease × site) with 95% CI\nColor = taxon; style = disease")

    leg1 = [Line2D([0],[0], color="black", lw=2.2, ls="-",  marker="o", label="Healthy"),
            Line2D([0],[0], color="black", lw=2.2, ls="--", marker="s", label="Crohn")]
    ax.legend(handles=leg1, loc="upper left", frameon=False, title="Disease")

    tax_handles = [Line2D([0],[0], color=c, lw=3, label=prettify_taxon(t, rank)) for t,c in zip(taxa, colors)]
    ax2 = ax.inset_axes([0.9, 0.5, 0.28, 0.35])
    ax2.axis("off"); ax2.legend(handles=tax_handles, ncol=1, fontsize=8, frameon=False, title="Taxa")

    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)

def spaghetti_crohn(long_df: pd.DataFrame, taxa: List[str], pairs_csv: str, out_png: str, rank: str, out_plotdata_dir: str) -> None:
    ensure_dir(os.path.dirname(out_png))
    if not pairs_csv or not os.path.exists(pairs_csv):
        safe_write_empty_png(out_png, "No pairs CSV found")
        return
    try:
        pairs = pd.read_csv(pairs_csv)
    except Exception as e:
        safe_write_empty_png(out_png, f"Failed to read pairs CSV: {e}")
        return

    cols_lower = [c.lower() for c in pairs.columns]
    def pick(*opts):
        for o in opts:
            if o.lower() in cols_lower:
                return pairs.columns[cols_lower.index(o.lower())]
        return None
    oc = pick("Oral_ID","Oral","OC","oral_id","oral")
    fc = pick("Fecal_ID","Fecal","FC","fecal_id","fecal")
    if oc is None or fc is None:
        safe_write_empty_png(out_png, "Pairs columns not found")
        return

    crohn = long_df[long_df["disease"] == 1].copy()
    if crohn.empty:
        safe_write_empty_png(out_png, "No Crohn samples")
        return

    crohn_ids = set(crohn["Sample_ID"].astype(str))
    usable = pairs[[oc, fc]].dropna().astype(str).values.tolist()
    usable = [(o, f) for (o, f) in usable if (o in crohn_ids and f in crohn_ids)]
    if len(usable) < 2:
        safe_write_empty_png(out_png, "Not enough Crohn pairs")
        return

    del_rows = []
    for t in taxa:
        dd_t = crohn[crohn["taxon"]==t].set_index(["Sample_ID","site"])["CLR"].unstack("site")
        if dd_t.empty or not {"Oral","Fecal"}.issubset(set(dd_t.columns)):
            continue
        for (o, f) in usable:
            if o in dd_t.index and f in dd_t.index:
                xo = dd_t.at[o, "Oral"]
                xf = dd_t.at[f, "Fecal"]
                if pd.notna(xo) and pd.notna(xf):
                    del_rows.append({"taxon": t, "pretty_taxon": prettify_taxon(t, rank),
                                     "oral": float(xo), "fecal": float(xf), "delta": float(xf - xo)})

    D = pd.DataFrame(del_rows)
    if D.empty:
        safe_write_empty_png(out_png, "No paired CLR found")
        return

    ensure_dir(out_plotdata_dir)
    D.to_csv(os.path.join(out_plotdata_dir, "plotdata_spaghetti_crohn.csv"), index=False)

    fig, axes = plt.subplots(nrows=max(1, len(taxa)), ncols=1, figsize=(8.2, 2.2*max(1, len(taxa))), sharex=True)
    if len(taxa) == 1:
        axes = [axes]
    for ax, t in zip(axes, taxa):
        dd = D[D["taxon"]==t]
        if dd.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, prettify_taxon(t, rank)+" (no pairs)", ha="center")
            continue
        inc = (dd["delta"]>0).mean() if len(dd)>0 else np.nan
        for _, r in dd.iterrows():
            color = "#2ca02c" if r["delta"]>0 else "#d62728"
            ax.plot([0, 1], [r["oral"], r["fecal"]], marker="o", alpha=0.35, color=color, lw=1.2)
        ax.axhline(0, ls=":", lw=1, color="#9aa0a6")
        ax.set_title(f"{prettify_taxon(t, rank)} — % increasing: {100*inc:.1f}%")
        ax.set_ylabel("CLR")
    axes[-1].set_xticks([0, 1]); axes[-1].set_xticklabels(["Oral","Fecal"])
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def kde_by_site(long_df: pd.DataFrame, taxa: List[str], site: str, out_png: str, qual_palette: List[str]) -> None:
    ensure_dir(os.path.dirname(out_png))
    d = long_df[(long_df["site"]==site) & (long_df["taxon"].isin(taxa))].copy()
    if d.empty:
        safe_write_empty_png(out_png, f"No data for KDE — {site}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = (qual_palette * ((len(taxa)+len(qual_palette)-1)//len(qual_palette)))[:len(taxa)]
    any_drawn = False
    for t, c in zip(taxa, colors):
        v = d.loc[d["taxon"]==t, "CLR"].astype(float).values
        v = v[np.isfinite(v)]
        if len(v) > 1:
            xs = np.linspace(np.nanmin(v)-1, np.nanmax(v)+1, 200)
            kde = gaussian_kde(v)
            ax.plot(xs, kde(xs), alpha=0.95, lw=2.0, label=t, color=c)
            any_drawn = True
    if not any_drawn:
        safe_write_empty_png(out_png, f"No density to plot — {site}")
        plt.close()
        return

    ax.set_title(f"CLR density — {site} (top taxa)")
    ax.set_xlabel("CLR"); ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=2, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct-all", required=True, help="taxa×samples percentage table CSV (index=taxa)")
    ap.add_argument("--meta", required=True, help="pooled metadata CSV (flexibly parsed)")
    ap.add_argument("--ppi-effects-all", dest="ppi_effects_all", default=None,
                    help="results/ppi_interactions/<rank>/ppi_effects_all.csv (optional)")
    ap.add_argument("--pairs", default=None, help="matched oral–fecal pairs CSV (optional)")
    ap.add_argument("--rank", required=True, choices=["genus","species"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--colors", default="config/colors.yml", help="optional color YAML")
    ap.add_argument("--with-pca-permanova", action="store_true",
                    help="if set, also produce PCA scatter + PERMANOVA TSV")
    args = ap.parse_args()

    # Colors
    _, _, ppi_palette_fn, qual_palette = load_colors(args.colors)

    # Output dirs
    out_rank_dir = os.path.join(args.outdir, args.rank)
    ensure_dir(out_rank_dir)
    out_plotdata = os.path.join(out_rank_dir, "plotdata")
    ensure_dir(out_plotdata)

    # Load percentage table (taxa×samples)
    pct = pd.read_csv(args.pct_all, index_col=0)
    pct.columns = pct.columns.astype(str)

    # CLR
    clr = clr_transform(pct, pseudocount=1e-6)
    clr_T = clr.T  # samples×taxa
    clr_T.index.name = "Sample_ID"

    # Metadata
    meta_raw = pd.read_csv(args.meta)
    meta_std = norm_meta(meta_raw)
    meta_std["sample_id"] = meta_std["sample_id"].astype(str)

    # Keep only samples present in pct table
    meta_std = meta_std[meta_std["sample_id"].isin(clr.columns.astype(str))].copy()

    # Long table (for plotting)
    long = clr_T.stack().reset_index()
    long.columns = ["Sample_ID", "taxon", "CLR"]
    long = long.merge(meta_std.rename(columns={"sample_id": "Sample_ID"}), on="Sample_ID", how="left")

    # Choose top taxa
    var_series = clr_T.var(axis=0)
    top_taxa = pick_top_taxa(args.ppi_effects_all, n=args.topk, fallback_by_var=var_series)

    # Persist selection
    sel_df = pd.DataFrame({
        "taxon": top_taxa,
        "pretty_taxon": [prettify_taxon(t, args.rank) for t in top_taxa]
    })
    sel_df.to_csv(os.path.join(out_rank_dir, "top_taxa_selected.csv"), index=False)

    # Core plots (always write something)
    box_by_ppi(long, top_taxa, "Oral",
               os.path.join(out_rank_dir, "box_by_ppi_oral.png"),
               ppi_palette_fn)
    box_by_ppi(long, top_taxa, "Fecal",
               os.path.join(out_rank_dir, "box_by_ppi_fecal.png"),
               ppi_palette_fn)

    interaction_means(long, top_taxa,
                      os.path.join(out_rank_dir, "interaction_means.png"),
                      args.rank, qual_palette, out_plotdata)

    spaghetti_crohn(long, top_taxa, args.pairs,
                    os.path.join(out_rank_dir, "paired_spaghetti_crohn.png"),
                    args.rank, out_plotdata)

    kde_by_site(long, top_taxa, "Oral",
                os.path.join(out_rank_dir, "density_top_taxa_oral.png"),
                qual_palette)
    kde_by_site(long, top_taxa, "Fecal",
                os.path.join(out_rank_dir, "density_top_taxa_fecal.png"),
                qual_palette)

    # Optional PCA + PERMANOVA
    if args.with_pca_permanova:
        pp_dir = os.path.join(out_rank_dir, "pca_permanova")
        ensure_dir(pp_dir)

        pca_scatter_top_taxa(long, top_taxa, os.path.join(pp_dir, "pca_scatter.png"))

        # Build aligned samples×features matrix for PERMANOVA on top taxa
        d = long[long["taxon"].isin(top_taxa)].copy()
        M = d.pivot_table(index="Sample_ID", columns="taxon", values="CLR", aggfunc="mean")
        M = M[[t for t in top_taxa if t in M.columns]]
        # Align metadata
        meta_small = d[["Sample_ID","site","disease"]].drop_duplicates().set_index("Sample_ID").reindex(M.index)

        # Try reading pairs (if provided) for block construction
        pairs_df = None
        if args.pairs and os.path.exists(args.pairs):
            try:
                pairs_df = pd.read_csv(args.pairs)
            except Exception as e:
                warnings.warn(f"Failed to read pairs CSV for strata: {e}")

        rows = []
        for factor in ["site", "disease"]:
            res = run_permanova(sample_df=M, meta_df=meta_small, factor=factor,
                                pairs_df=pairs_df, n_perm=999, rng_seed=1)
            rows.append(res)

        cols = ["factor","n_groups","F","R2","p","n_perm","n_eff","n_samples","n_taxa"]
        pd.DataFrame(rows)[cols].to_csv(os.path.join(pp_dir, "permanova.tsv"), sep="\t", index=False)

    print(f"[INFO] QC-targeted saved → {out_rank_dir}")

if __name__ == "__main__":
    main()
