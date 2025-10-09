#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified genus/species comparison pipeline (Crohn vs Healthy; Oral & Fecal)

Outputs per-rank (genus, species):
- Tables: group_means_*.csv, group_deltas_*.csv
- Differential abundance (MWU; Crohn vs Healthy per site): da_oral_CH_*.csv, da_fecal_CH_*.csv
- Paired OC↔FC (Wilcoxon) if matched IDs provided for Crohn only: da_paired_OC_FC_*.csv (empty if insufficient)
- Plots: heatmap_group_means_*.png, heatmap_group_deltas_*.png,
         heatmap_pairs_crohn_*.png, heatmap_pairs_healthy_*.png  (Healthy shown as unpaired)
         heatmap_oral_CH_unpaired_*.png, heatmap_fecal_CH_unpaired_*.png,
         volcano_oral_CH_*.png, volcano_fecal_CH_*.png,
         barplot_oral_*_top15.png, barplot_fecal_*_top15.png
- ML-friendly matrices:
    * ml_*_tidy.csv         (long: Sample, Taxon, AbundancePct, Group, Site)
    * ml_*_wide_all.csv     (wide: samples × taxa + Group/Site)
    * ml_*_wide_topK.csv    (wide topK taxa globally)
    * ml_*_wide_clr.csv     (wide CLR on percent table; samples × taxa + Group/Site)

CLI
---
python taxa_compare_unified.py \
  --oral-crohn <csv> --oral-healthy <csv> \
  --fecal-crohn <csv> --fecal-healthy <csv> \
  [--matched-ids <csv>] \
  --outdir results/taxa_compare \
  [--topk 15] [--presence-pct 0.1] [--prevalence 0.10]
"""

from __future__ import annotations
import os, re, argparse, warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib.patheffects as pe

sns.set_context("talk")

# ---------- Colors: fixed palette (from user) ----------
try:
    import yaml
except Exception:
    yaml = None

DEFAULT_PALETTE = {
    "Fecal_Crohn":   "#30638e",
    "Oral_Crohn":    "#edae49",
    "Fecal_Healthy": "#d1495b",
    "Oral_Healthy":  "#00798c",
    "Other":         "#999999",
    # aliases used in code:
    "Crohn-Oral":    "#edae49",
    "Crohn-Fecal":   "#30638e",
    "Healthy-Oral":  "#00798c",
    "Healthy-Fecal": "#d1495b",
    "Oral":          "#edae49",  # for borders in barplots if label contains 'Oral'
    "Fecal":         "#30638e",
}

def _try_read_yaml(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def load_palette(cfg_paths=("config/colors.yml","config/config.yml","config.yaml","config.yml")) -> dict:
    """
    Load palette from YAML if present; otherwise use DEFAULT_PALETTE.
    Accepts either a top-level dict or nested under key 'colors'.
    Missing keys fallback to defaults.
    """
    if yaml is None:
        return dict(DEFAULT_PALETTE)
    for p in cfg_paths:
        if os.path.exists(p):
            cfg = _try_read_yaml(p)
            pal = None
            if isinstance(cfg, dict) and "colors" in cfg and isinstance(cfg["colors"], dict):
                pal = cfg["colors"]
            elif isinstance(cfg, dict):
                pal = cfg
            if isinstance(pal, dict):
                merged = dict(DEFAULT_PALETTE); merged.update(pal)
                return merged
    return dict(DEFAULT_PALETTE)

PALETTE = load_palette()

def apply_global_theme():
    """Consistent axes and legend styling."""
    sns.set_context("talk")
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False

apply_global_theme()

# ---------- ID normalization helpers ----------
def _normalize_id(x: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(x).strip().lower())

def _build_id_map(cols) -> dict:
    m = {}
    for c in cols:
        key = _normalize_id(str(c))
        if key not in m: m[key] = c
    return m

# ---------- Taxonomy helpers ----------
def extract_genus(rowname: str):
    parts = str(rowname).split("|")
    g = [p for p in parts if p.startswith("g__")]
    return g[0] if g else None

def extract_species(rowname: str):
    parts = str(rowname).split("|")
    s = [p for p in parts if p.startswith("s__")]
    return s[0] if s else None

def prettify_taxon(label: str, rank: str) -> str:
    if label is None: return None
    r = (rank or "").lower()
    if r == "genus":
        name = label.replace("g__", "").replace("_", " ")
        return " ".join(w.capitalize() for w in name.split())
    if r == "species":
        name = label.replace("s__", "").replace("_", " ")
        toks = name.split()
        if not toks: return name
        toks[0] = toks[0].capitalize()
        toks[1:] = [t.lower() for t in toks[1:]]
        return " ".join(toks)
    return label

# ---------- Table orientation & transforms ----------
def ensure_taxa_by_samples(df_like: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the matrix is taxa × samples.
    Heuristic: if >50% columns look like taxa (contain 's__' or 'g__'), it's samples×taxa → transpose.
    """
    cols = pd.Index(df_like.columns.astype(str))
    looks_like_taxa = cols.str.contains("s__|g__", regex=True).mean()
    return df_like.T if looks_like_taxa > 0.5 else df_like

def percent_table(df: pd.DataFrame, grouper) -> pd.DataFrame:
    """
    Collapse at rank via `grouper(rowname)` and convert to percent per-sample.
    Returns taxa_rank × samples with percent values.
    """
    g = df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    return g.div(g.sum(axis=0), axis=1) * 100.0

def group_mean_percent(df: pd.DataFrame, grouper) -> pd.Series:
    """Mean percent abundance per taxon rank across samples."""
    g = df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    pct = g.div(g.sum(axis=0), axis=1) * 100
    return pct.mean(axis=1)

def clr_transform(df_taxa_by_samples: pd.DataFrame, pseudocount=1e-6) -> pd.DataFrame:
    X = df_taxa_by_samples.astype(float) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)
    return logX.sub(gm, axis=1)

# ---------- Z-score ----------
def row_zscore_log1p(mat: pd.DataFrame, use_log1p=True) -> pd.DataFrame:
    X = np.log1p(mat.astype(float)) if use_log1p else mat.astype(float)
    mu = X.mean(axis=1)
    sd = X.std(axis=1).replace(0, np.nan)
    return X.sub(mu, axis=0).div(sd, axis=0).fillna(0)

# ---------- Heatmaps ----------
def _subsample_xticks(ax, max_labels=25):
    ticks = ax.get_xticks()
    if len(ticks) <= max_labels or max_labels <= 0:
        return
    step = max(1, int(round(len(ticks) / max_labels)))
    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_visible((i % step) == 0)

def _apply_smart_yticklabels(ax, labels, rank, base_fs=12.0, max_chars=28):
    if labels is None: return
    wrapped = []
    for lab in labels:
        lab = str(lab)
        lines = textwrap.wrap(lab, width=max_chars, break_long_words=False) or [lab]
        wrapped.append("\n".join(lines))
    n = len(wrapped)
    ax.yaxis.set_major_locator(FixedLocator(np.arange(n)))
    ax.set_yticklabels(wrapped, rotation=0, ha="right", va="center", fontsize=base_fs)
    for txt in ax.get_yticklabels():
        if txt.get_text().strip().lower() != "other":
            try: txt.set_style("italic")
            except Exception: pass

def _coolwarm():
    return plt.get_cmap("coolwarm")

def pairs_heatmap(A_pct, B_pct, rows, label_A, label_B, rank, out_png,
                  use_log1p=True, clip_quantile=0.98, col_cluster=False, title_suffix=""):
    """
    Generic two-group heatmap (columns are samples). Unpaired by nature; name may say 'pairs' historically.
    """
    data = pd.concat([A_pct, B_pct], axis=1).reindex(rows).fillna(0)
    pretty_idx = [prettify_taxon(t, rank) for t in data.index]
    Z = row_zscore_log1p(data, use_log1p=use_log1p); Z.index = pretty_idx

    V = np.nanquantile(np.abs(Z.values), clip_quantile)
    if not np.isfinite(V) or V == 0:
        V = max(1.0, float(np.nanmax(np.abs(Z.values)) or 1.0))

    height = max(6.0, 0.35*len(Z.index))
    width  = max(8.0, 0.18*len(Z.columns)+2.0)

    g = sns.clustermap(
        Z, cmap=_coolwarm(), center=0, vmin=-V, vmax=+V,
        col_cluster=col_cluster, row_cluster=True,
        metric="correlation", method="average",
        figsize=(width, height),
        cbar_kws={"label": "Row z-score of % abundance (log1p)", "orientation": "horizontal"},
        cbar_pos=(.25, .97, .50, .02)
    )

    # Y labels on right, italic
    g.ax_heatmap.yaxis.set_ticks_position("right")
    g.ax_heatmap.yaxis.set_label_position("right")
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_ymajorticklabels(),
        rotation=0, ha="left", va="center", fontsize=10
    )
    g.ax_heatmap.tick_params(axis="y", pad=10)
    for t in g.ax_heatmap.get_yticklabels():
        try: t.set_fontstyle("italic")
        except Exception: pass

    # X labels compact
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xmajorticklabels(),
        rotation=90, ha="center", va="top", fontsize=8.5
    )
    _subsample_xticks(g.ax_heatmap, max_labels=25)

    g.ax_heatmap.set_xlabel(f"{label_A} ↔ {label_B}")
    g.ax_heatmap.set_ylabel(rank.capitalize())
    g.ax_heatmap.yaxis.labelpad = 18
    g.ax_heatmap.set_title(f"{label_A} vs {label_B} {title_suffix}".strip())

    maxlen = max((len(s) for s in Z.index), default=12)
    right = min(0.995, 0.88 + 0.016 * max(0, maxlen - 12))
    g.fig.subplots_adjust(left=0.07, right=right, bottom=0.20, top=0.92)

    g.fig.savefig(out_png, dpi=380, bbox_inches="tight", pad_inches=0.70)
    plt.close(g.fig)

def heatmap_two_groups(A_pct, B_pct, rank, out_png, title, n=15,
                       presence_threshold_percent=0.1, prevalence_thresh=0.10):
    """
    Site-fixed, disease-contrast heatmap: select top taxa across the union, unpaired.
    """
    prev_A = (A_pct > presence_threshold_percent).mean(axis=1)
    prev_B = (B_pct > presence_threshold_percent).mean(axis=1)
    union  = A_pct.add(B_pct, fill_value=0)
    keep = (prev_A >= prevalence_thresh) & (prev_B >= prevalence_thresh)
    rows = union.loc[keep].sum(axis=1).sort_values(ascending=False).head(n).index
    if len(rows) < max(5, int(0.4*n)):
        rows = union.sum(axis=1).sort_values(ascending=False).head(n).index
    pairs_heatmap(A_pct, B_pct, rows,
                  label_A=title.split(" vs ")[0], label_B=title.split(" vs ")[1],
                  rank=rank, out_png=out_png, title_suffix="(unpaired)")

# ---------- DA / stats ----------
def bh_qvalues(pvals):
    p = np.asarray(pvals, dtype=float); n = len(p)
    order = np.argsort(p); q = np.empty(n, dtype=float); prev = 1.0
    for i, idx in enumerate(order[::-1], start=1):
        rank = n - i + 1
        val = min(prev, p[idx] * n / rank)
        q[idx] = val; prev = val
    return q.tolist()

def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if x.size==0 or y.size==0: return 0.0
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    m, n = len(x), len(y)
    return (gt - lt) / (m * n)

def robust_log2fc(a, b, eps=1e-9):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if np.nanmin(a) < 0 or np.nanmin(b) < 0:
        return (np.nanmean(a) - np.nanmean(b)) / np.log(2.0)
    return np.log2(np.nanmean(a) + eps) - np.log2(np.nanmean(b) + eps)

def volcano(df, x="log2FC", q="q", title="", out_png="volcano.png",
            q_sig=0.05, k_onplot_each=8, k_side_each=10, figsize=(11.5, 6.8), dpi=340,
            point_size=22, colors=dict(non="#999999", up="#1B4F72", down="#7FB3D5"),
            xlabel=None):
    X  = df[x].astype(float).values
    qv = df[q].astype(float).clip(lower=1e-300).values
    Y  = -np.log10(qv)
    names = (
        df["pretty_taxon"] if "pretty_taxon" in df.columns
        else (df["taxon"] if "taxon" in df.columns else df.index)
    ).astype(str).values

    is_sig = qv <= q_sig
    score = np.abs(X) * (Y + 1)

    up_idx = np.where(is_sig & (X > 0))[0]
    dn_idx = np.where(is_sig & (X < 0))[0]
    up_sel = up_idx[np.argsort(-score[up_idx])[:k_onplot_each]]
    dn_sel = dn_idx[np.argsort(-score[dn_idx])[:k_onplot_each]]
    label_idxs = np.concatenate([up_sel, dn_sel])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[5.2, 2.8], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1]); ax_side.axis("off")

    c_non = colors["non"]; c_up = colors["up"]; c_dn = colors["down"]
    ax.scatter(X[~is_sig], Y[~is_sig], s=point_size, alpha=0.35, color=c_non, zorder=1)
    ax.scatter(X[is_sig & (X > 0)], Y[is_sig & (X > 0)], s=point_size, alpha=0.9, color=c_up, zorder=2)
    ax.scatter(X[is_sig & (X < 0)], Y[is_sig & (X < 0)], s=point_size, alpha=0.9, color=c_dn, zorder=2)
    ax.axhline(-np.log10(q_sig), ls="--", lw=1, color="gray", alpha=0.8)
    ax.axvline(0, ls="--", lw=1, color="gray", alpha=0.8)
    ax.grid(True, ls=":", lw=0.6, alpha=0.4)

    # Light label repulsion
    def repel(ax, xs, ys, texts, fs=9, max_iter=200, pad=0.015):
        xy = np.vstack([xs, ys]).T.astype(float)
        for _ in range(max_iter):
            bumped = False
            for i in range(len(texts)):
                for j in range(i+1, len(texts)):
                    dx = xy[i,0]-xy[j,0]; dy = xy[i,1]-xy[j,1]
                    if abs(dx) < pad and abs(dy) < pad:
                        sgn = 1 if (dy == 0 and (i%2)==0) else (np.sign(dy) or 1)
                        xy[i,1] += sgn*pad; xy[j,1] -= sgn*pad
                        bumped = True
            if not bumped: break
        for (x1,y1,txt) in zip(xy[:,0], xy[:,1], texts):
            ax.text(x1, y1, txt, fontsize=fs, ha="left", va="bottom",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=3)
        for (x0,y0),(x1,y1) in zip(np.vstack([xs, ys]).T, xy):
            ax.annotate("", xy=(x0,y0), xytext=(x1,y1),
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.25", alpha=0.7), zorder=3)

    if len(label_idxs):
        repel(ax, X[label_idxs], Y[label_idxs], [names[i] for i in label_idxs])

    ax.set_xlabel(xlabel or "log2FC (Crohn − Healthy)", fontsize=12)
    ax.set_ylabel("-log10(q)", fontsize=12)
    ax.set_title(title, fontsize=18, pad=10)

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def run_da_group_CH(mat_taxa_samples: pd.DataFrame, labels_crohn, labels_healthy, tax_labels,
                    out_csv, do_volcano_png=None, rank="genus",
                    x_col_name="log2FC", volcano_title=None, volcano_xlabel=None):
    rows, pvals = [], []
    for tax in tax_labels:
        if tax not in mat_taxa_samples.index:
            continue
        a = mat_taxa_samples.loc[tax, labels_crohn].values
        b = mat_taxa_samples.loc[tax, labels_healthy].values
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            u, p = np.nan, 1.0
        eff  = cliffs_delta(a, b)
        l2fc = robust_log2fc(a, b)
        rows.append({
            "taxon": tax,
            "pretty_taxon": prettify_taxon(tax, rank),
            "test": "MWU",
            "stat": float(u) if np.isfinite(u) else np.nan,
            "p": float(p),
            "effect_name": "Cliffs_delta",
            "effect_size": float(eff),
            x_col_name: float(l2fc),
        })
        pvals.append(p)

    if pvals:
        qvals = bh_qvalues(pvals)
        for r, qv in zip(rows, qvals):
            r["q"] = float(qv)

    base_cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size"]
    cols = base_cols + [x_col_name]
    df = pd.DataFrame(rows)
    (pd.DataFrame(columns=cols) if df.empty else df.reindex(columns=cols)).to_csv(out_csv, index=False)

    if do_volcano_png and not df.empty:
        volcano(df, x=x_col_name, q="q",
                title=(volcano_title or "Volcano (Crohn vs Healthy)"),
                out_png=do_volcano_png, xlabel=(volcano_xlabel or "log2FC"))
    return df

def run_da_oral_CH(*, mat_taxa_samples, labels_A, labels_B, tax_labels, out_csv,
                   do_volcano_png=None, rank="genus"):
    return run_da_group_CH(
        mat_taxa_samples=mat_taxa_samples,
        labels_crohn=labels_A, labels_healthy=labels_B, tax_labels=tax_labels,
        out_csv=out_csv, do_volcano_png=do_volcano_png, rank=rank,
        x_col_name="log2FC",
        volcano_title=f"Volcano Oral Crohn vs Healthy ({rank})",
        volcano_xlabel="log2FC (Oral: Crohn − Healthy)",
    )

def run_da_fecal_CH(*, mat_taxa_samples, labels_A, labels_B, tax_labels, out_csv,
                    do_volcano_png=None, rank="genus"):
    return run_da_group_CH(
        mat_taxa_samples=mat_taxa_samples,
        labels_crohn=labels_A, labels_healthy=labels_B, tax_labels=tax_labels,
        out_csv=out_csv, do_volcano_png=do_volcano_png, rank=rank,
        x_col_name="log2FC",
        volcano_title=f"Volcano Fecal Crohn vs Healthy ({rank})",
        volcano_xlabel="log2FC (Fecal: Crohn − Healthy)",
    )

def _guess_pair_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = [str(c) for c in df.columns]
    if len(cols) == 2:
        return cols[0], cols[1]
    low = [c.lower() for c in cols]
    oral_keys  = ("oral", "oc", "mouth")
    fecal_keys = ("fecal", "fc", "stool")
    oral_idx = next((i for i,c in enumerate(low) if any(k in c for k in oral_keys)), None)
    fecal_idx = next((i for i,c in enumerate(low) if any(k in c for k in fecal_keys)), None)
    if oral_idx is not None and fecal_idx is not None and oral_idx != fecal_idx:
        return cols[oral_idx], cols[fecal_idx]
    return cols[0], cols[1]

def _map_ids_to_columns(candidates: list[str], df_cols: pd.Index) -> list[str|None]:
    col_raw = [str(c) for c in df_cols]
    col_norm_map = { _normalize_id(c): c for c in col_raw }
    mapped = []
    for xr in candidates:
        xn = _normalize_id(str(xr))
        col = col_norm_map.get(xn, None)
        if col is None and xr in col_raw:
            col = xr
        mapped.append(col)
    return mapped

def run_da_paired_OC_FC(mat_taxa_samples_OC: pd.DataFrame,
                        mat_taxa_samples_FC: pd.DataFrame,
                        matched_df: pd.DataFrame, tax_labels, out_csv, rank="genus",
                        min_pairs:int=8):
    """
    Paired OC↔FC Wilcoxon (Crohn only). Healthy is never paired here.
    """
    # Ensure taxa × samples
    def _maybe_t(df):
        if df.shape[0] < df.shape[1] and \
           df.columns.to_series().astype(str).str.contains("s__|g__", regex=True).any():
            return df.T
        return df
    mat_taxa_samples_OC = _maybe_t(mat_taxa_samples_OC)
    mat_taxa_samples_FC = _maybe_t(mat_taxa_samples_FC)

    oc_col, fc_col = _guess_pair_columns(matched_df)
    oc_raw = matched_df[oc_col].dropna().astype(str).tolist()
    fc_raw = matched_df[fc_col].dropna().astype(str).tolist()

    pairs = [(o,f) for o,f in zip(oc_raw, fc_raw) if str(o).strip()!="" and str(f).strip()!=""]
    if not pairs:
        empty_cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median","n_pairs"]
        pd.DataFrame(columns=empty_cols).to_csv(out_csv, index=False)
        warnings.warn("[paired] match file has no valid rows.")
        return

    oc_list, fc_list = zip(*pairs)
    oc_map = _map_ids_to_columns(list(oc_list), mat_taxa_samples_OC.columns)
    fc_map = _map_ids_to_columns(list(fc_list), mat_taxa_samples_FC.columns)

    good = [(o,f) for o,f in zip(oc_map, fc_map) if (o is not None and f is not None)]
    n_good = len(good)
    if n_good < min_pairs:
        empty_cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median","n_pairs"]
        pd.DataFrame(columns=empty_cols).to_csv(out_csv, index=False)
        warnings.warn(f"[paired] not enough matched pairs (got {n_good}, need ≥{min_pairs}).")
        return

    oc_cols = [o for o,_ in good]
    fc_cols = [f for _,f in good]

    rows, pvals = [], []
    for tax in tax_labels:
        if tax not in mat_taxa_samples_OC.index or tax not in mat_taxa_samples_FC.index:
            continue
        x = mat_taxa_samples_OC.loc[tax, oc_cols].astype(float).values
        y = mat_taxa_samples_FC.loc[tax, fc_cols].astype(float).values

        dif = y - x
        dif = dif[~np.isnan(dif)]
        if dif.size == 0 or np.allclose(dif, 0.0):
            w, p = np.nan, 1.0
            r = np.nan
        else:
            try:
                w, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                w, p = np.nan, 1.0
            n = len(dif)
            if np.isfinite(w):
                denom = n*(n+1)/2.0
                r = 1.0 - 2.0*(w/denom)
            else:
                r = np.nan

        rows.append({
            "taxon": tax,
            "pretty_taxon": prettify_taxon(tax, rank),
            "test": "Wilcoxon",
            "stat": float(w) if np.isfinite(w) else np.nan,
            "p": float(p),
            "effect_name": "rank_biserial_r",
            "effect_size": float(r) if np.isfinite(r) else np.nan,
            "delta_median": float(np.nanmedian(y) - np.nanmedian(x)) if dif.size>0 else np.nan,
            "n_pairs": n_good,
        })
        pvals.append(p)

    if pvals:
        qvals = bh_qvalues(pvals)
        for r,q in zip(rows, qvals):
            r["q"] = float(q)

    cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median","n_pairs"]
    pd.DataFrame(rows)[cols].to_csv(out_csv, index=False)

# ---------- Barplots with side legend-table ----------
def build_taxa_color_cycle(n_needed=20):
    bases = [
        PALETTE.get("Crohn-Oral", "#edae49"),
        PALETTE.get("Crohn-Fecal", "#30638e"),
        PALETTE.get("Healthy-Oral", "#00798c"),
        PALETTE.get("Healthy-Fecal", "#d1495b"),
    ]
    def _hex_to_rgb01(h):
        h = h.lstrip("#"); return tuple(int(h[i:i+2], 16)/255.0 for i in (0,2,4))
    def _mix(c1, c2, t): return tuple((1-t)*a + t*b for a,b in zip(c1,c2))
    bases = [_hex_to_rgb01(b) for b in bases]
    white = _hex_to_rgb01("#FFFFFF")
    cycle = []
    for b in bases:
        cycle.append(_mix(white, b, 0.35))
        cycle.append(b)
        cycle.append(tuple(max(x-0.20,0) for x in b))
    i = 0
    while len(cycle) < n_needed:
        b = bases[i % len(bases)]; t = 0.15 + 0.15*((i//len(bases))%3)
        cycle.append(_mix(white, b, t)); i += 1
    def rgb01_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    return [rgb01_to_hex(c) for c in cycle[:n_needed]]

def two_bar_stacked_with_table(a_mean: pd.Series, b_mean: pd.Series, rank: str, out_png: str,
                               label_a: str, label_b: str, top_k: int = 15):
    """
    Two stacked bars (A vs B) + a side legend-table (color square, taxon, percent).
    Also draws value labels on bars; if a segment too thin, the side table ensures readability.
    """
    combined = a_mean.add(b_mean, fill_value=0).sort_values(ascending=False)
    top = combined.index[:min(top_k, len(combined))]
    s_a = a_mean.reindex(top).fillna(0)
    s_b = b_mean.reindex(top).fillna(0)
    s_a = pd.concat([s_a, pd.Series({"Other": a_mean.drop(top, errors='ignore').sum()})])
    s_b = pd.concat([s_b, pd.Series({"Other": b_mean.drop(top, errors='ignore').sum()})])

    order = s_a.sort_values(ascending=False).index
    taxa_colors = build_taxa_color_cycle(max(3*len(order), 12))
    color_map = {tax: taxa_colors[i % len(taxa_colors)] for i, tax in enumerate(order)}
    color_map["Other"] = PALETTE.get("Other", "#999999")

    edge_a = PALETTE.get("Oral", PALETTE.get("Crohn-Oral", "#edae49")) if "Oral" in label_a else \
             PALETTE.get("Fecal", PALETTE.get("Crohn-Fecal", "#30638e"))
    edge_b = PALETTE.get("Oral", PALETTE.get("Healthy-Oral", "#00798c")) if "Oral" in label_b else \
             PALETTE.get("Fecal", PALETTE.get("Healthy-Fecal", "#d1495b"))

    fig = plt.figure(figsize=(16.5, 9.0))
    gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[2.4, 1.6], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1]); ax_tbl.axis("off")

    x = np.array([0, 1]); b0 = b1 = 0.0
    labels_rows = []
    for tax in order:
        h0 = float(s_a.loc[tax]); h1 = float(s_b.loc[tax])
        ax.bar(x[0], h0, bottom=b0, color=color_map[tax], width=0.6, edgecolor=edge_a, linewidth=0.7)
        ax.bar(x[1], h1, bottom=b1, color=color_map[tax], width=0.6, edgecolor=edge_b, linewidth=0.7)

        # inline labels on bars (may be clipped if too small)
        if h0 >= 3.0:
            ax.text(x[0], b0 + h0/2, f"{h0:.1f}%", ha="center", va="center", fontsize=9, color="black")
        if h1 >= 3.0:
            ax.text(x[1], b1 + h1/2, f"{h1:.1f}%", ha="center", va="center", fontsize=9, color="black")

        labels_rows.append((tax, color_map[tax], h0, h1))
        b0 += h0; b1 += h1

    ax.set_xticks(x); ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel("Mean relative abundance (%)")
    ax.set_title(f"{label_a} vs {label_b} — Top {len(top)} {rank.capitalize()} (others → Other)")

    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_ylim(0, max(b0, b1) * 1.08)

    # Side legend-table
    y0 = 0.96; dy = 0.04
    ax_tbl.text(0.00, y0, "Color", fontweight="bold", transform=ax_tbl.transAxes)
    ax_tbl.text(0.12, y0, "Taxon", fontweight="bold", transform=ax_tbl.transAxes)
    ax_tbl.text(0.66, y0, f"{label_a} (%)", fontweight="bold", transform=ax_tbl.transAxes, ha="right")
    ax_tbl.text(0.98, y0, f"{label_b} (%)", fontweight="bold", transform=ax_tbl.transAxes, ha="right")
    y = y0 - dy
    for tax, col, v0, v1 in labels_rows:
        # color square
        ax_tbl.add_patch(plt.Rectangle((0.00, y-0.015), 0.06, 0.02, transform=ax_tbl.transAxes,
                                       color=col, clip_on=False))
        name = prettify_taxon(tax, rank) if tax != "Other" else "Other"
        ax_tbl.text(0.12, y, name, transform=ax_tbl.transAxes, va="center")
        ax_tbl.text(0.66, y, f"{v0:.1f}", transform=ax_tbl.transAxes, va="center", ha="right")
        ax_tbl.text(0.98, y, f"{v1:.1f}", transform=ax_tbl.transAxes, va="center", ha="right")
        y -= dy
        if y < 0.02: break

    plt.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

# ---------- Tidy/Wide/CLR for ML ----------
def tidy_from_percent(pct: pd.DataFrame, group_label: str, site_label: str, rank: str) -> pd.DataFrame:
    df = pct.copy()
    tidy = df.stack().reset_index()
    tidy.columns = [rank, "Sample", "AbundancePct"]
    tidy["Taxon"] = tidy[rank].map(lambda t: prettify_taxon(t, rank))
    tidy["Group"] = group_label
    tidy["Site"]  = site_label
    return tidy[["Sample","Taxon","AbundancePct","Group","Site"]]

def wide_from_percent(pct: pd.DataFrame, group: str, site: str, rank: str) -> pd.DataFrame:
    df = pct.T.copy()  # samples × taxa
    df["Group"] = group
    df["Site"]  = site
    return df.reset_index().rename(columns={"index":"Sample"})

def save_topk_wide(wide_df, out_csv, topk=15, meta_cols=("Sample","Group","Site")):
    meta_cols_present = [c for c in meta_cols if c in wide_df.columns]
    feat_df = wide_df.drop(columns=meta_cols_present, errors="ignore")
    top_taxa = (feat_df.sum(axis=0)
                .sort_values(ascending=False)
                .head(int(topk)).index)
    out_df = pd.concat([wide_df.loc[:, top_taxa], wide_df.loc[:, meta_cols_present]], axis=1)
    out_df.to_csv(out_csv, index=False)

def ml_clr_from_percent(pct_all_sites: pd.DataFrame, group_map: dict, site_map: dict,
                        rank: str, pseudocount=1e-6) -> pd.DataFrame:
    clr_mat = clr_transform(pct_all_sites, pseudocount=pseudocount)  # taxa × samples (CLR)
    wide = clr_mat.T.copy()
    meta = pd.DataFrame({
        "Sample": wide.index,
        "Group": [group_map.get(s, None) for s in wide.index],
        "Site":  [site_map.get(s, None) for s in wide.index],
    }).set_index("Sample")
    wide = pd.concat([wide, meta], axis=1).reset_index().rename(columns={"index":"Sample"})
    return wide

# ---------- Rank processing ----------
def group_contrasts_and_csv(oc_pct, fc_pct, oh_pct, fh_pct, rank, outdir,
                            n=15, prevalence_thresh=0.10, presence_threshold_percent=0.1,
                            use_log1p=True, clip_quantile=0.98, col_cluster_pairs=False):
    """
    Build:
      - heatmap_pairs_crohn_{rank}.png
      - heatmap_pairs_healthy_{rank}.png (explicitly labeled 'unpaired')
      - heatmap_group_means_{rank}.png + group_means_{rank}.csv
      - heatmap_group_deltas_{rank}.png + group_deltas_{rank}.csv
      - NEW: heatmap_oral_CH_unpaired_{rank}.png, heatmap_fecal_CH_unpaired_{rank}.png
    """
    os.makedirs(outdir, exist_ok=True)

    # Row selection for Crohn pairs (union OC/FC)
    prev_oc = (oc_pct > presence_threshold_percent).mean(axis=1)
    prev_fc = (fc_pct > presence_threshold_percent).mean(axis=1)
    crohn_union = oc_pct.add(fc_pct, fill_value=0)
    keep = (prev_oc >= prevalence_thresh) & (prev_fc >= prevalence_thresh)
    rows = crohn_union.loc[keep].sum(axis=1).sort_values(ascending=False).head(n).index
    if len(rows) < max(5, int(0.4*n)):
        rows = crohn_union.sum(axis=1).sort_values(ascending=False).head(n).index

    # (1) Crohn Oral vs Fecal
    pairs_heatmap(
        oc_pct, fc_pct, rows,
        "Crohn-Oral", "Crohn-Fecal", rank,
        os.path.join(outdir, f"heatmap_pairs_crohn_{rank}.png"),
        use_log1p=use_log1p, clip_quantile=clip_quantile, col_cluster=col_cluster_pairs,
        title_suffix=""  # Crohn could be paired elsewhere; here it's just columns shown together
    )

    # (2) Healthy Oral vs Fecal (always unpaired)
    prev_oh = (oh_pct > presence_threshold_percent).mean(axis=1)
    prev_fh = (fh_pct > presence_threshold_percent).mean(axis=1)
    healthy_union = oh_pct.add(fh_pct, fill_value=0)
    keep_h = (prev_oh >= prevalence_thresh) & (prev_fh >= prevalence_thresh)
    rows_h = healthy_union.loc[keep_h].sum(axis=1).sort_values(ascending=False).head(n).index
    if len(rows_h) < max(5, int(0.4*n)):
        rows_h = healthy_union.sum(axis=1).sort_values(ascending=False).head(n).index

    pairs_heatmap(
        oh_pct, fh_pct, rows_h,
        "Healthy-Oral", "Healthy-Fecal", rank,
        os.path.join(outdir, f"heatmap_pairs_healthy_{rank}.png"),
        use_log1p=use_log1p, clip_quantile=clip_quantile, col_cluster=col_cluster_pairs,
        title_suffix="(unpaired)"
    )

    # (3) Group means/deltas (+ CSV) — both drawn with coolwarm
    means_df = pd.DataFrame({
        "O.C.": oc_pct.reindex(crohn_union.index).fillna(0).mean(axis=1),
        "F.C.": fc_pct.reindex(crohn_union.index).fillna(0).mean(axis=1),
        "O.H.": oh_pct.reindex(crohn_union.index).fillna(0).mean(axis=1),
        "F.H.": fh_pct.reindex(crohn_union.index).fillna(0).mean(axis=1),
    }).fillna(0.0)
    means_df = means_df.loc[(means_df.sum(axis=1) > 0)]
    means_df = means_df.sort_values(by=["O.C.","F.C.","O.H.","F.H."], ascending=False).head(n)
    means_df.index.name = rank
    means_df.to_csv(os.path.join(outdir, f"group_means_{rank}.csv"))

    deltas_df = pd.DataFrame({
        "O.C.–O.H.": means_df["O.C."] - means_df["O.H."],
        "F.C.–F.H.": means_df["F.C."] - means_df["F.H."],
    })
    deltas_df.index.name = rank
    deltas_df.to_csv(os.path.join(outdir, f"group_deltas_{rank}.csv"))

    # Heatmap: means
    Zm = row_zscore_log1p(means_df, use_log1p=False)
    fig_h = max(6, 0.45*len(Zm.index))
    plt.figure(figsize=(10.5, fig_h))
    ax = sns.heatmap(Zm[["O.C.", "F.C.", "O.H.", "F.H."]],
                     cmap=_coolwarm(),
                     cbar_kws={"label": "Row z-score of means"})
    ax.set_xlabel("Groups"); ax.set_ylabel(rank.capitalize()); ax.set_title("Group means (O.C., F.C., O.H., F.H.)")
    _apply_smart_yticklabels(ax, [prettify_taxon(t, rank) for t in Zm.index], rank, base_fs=12.5)
    plt.gcf().subplots_adjust(left=0.32)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"heatmap_group_means_{rank}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Heatmap: deltas
    Zd = row_zscore_log1p(deltas_df, use_log1p=False)
    fig_h = max(6, 0.45*len(Zd.index))
    plt.figure(figsize=(9.2, fig_h))
    vmax = np.nanquantile(np.abs(Zd.values), 0.98) or 1.0
    ax = sns.heatmap(Zd[["O.C.–O.H.", "F.C.–F.H."]],
                     cmap=_coolwarm(), center=0, vmin=-vmax, vmax=+vmax,
                     cbar_kws={"label": "Row z-score of deltas"})
    ax.set_xlabel("Comparisons"); ax.set_ylabel(rank.capitalize()); ax.set_title("Group deltas (O.C.–O.H., F.C.–F.H.)")
    _apply_smart_yticklabels(ax, [prettify_taxon(t, rank) for t in Zd.index], rank, base_fs=12.5)
    plt.gcf().subplots_adjust(left=0.34)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"heatmap_group_deltas_{rank}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (4) NEW: within-site disease contrasts (unpaired)
    heatmap_two_groups(
        A_pct=oh_pct, B_pct=oc_pct, rank=rank,
        out_png=os.path.join(outdir, f"heatmap_oral_CH_unpaired_{rank}.png"),
        title="Oral_Healthy vs Oral_Crohn", n=n
    )
    heatmap_two_groups(
        A_pct=fh_pct, B_pct=fc_pct, rank=rank,
        out_png=os.path.join(outdir, f"heatmap_fecal_CH_unpaired_{rank}.png"),
        title="Fecal_Healthy vs Fecal_Crohn", n=n
    )

def make_ml_outputs(rank: str, oc_pct: pd.DataFrame, oh_pct: pd.DataFrame,
                    fc_pct: pd.DataFrame, fh_pct: pd.DataFrame,
                    outdir_rank: str, topk: int = 15):
    # tidy
    tidy_all = pd.concat([
        tidy_from_percent(oc_pct, "Crohn",   "Oral",  rank),
        tidy_from_percent(oh_pct, "Healthy", "Oral",  rank),
        tidy_from_percent(fc_pct, "Crohn",   "Fecal", rank),
        tidy_from_percent(fh_pct, "Healthy", "Fecal", rank),
    ], ignore_index=True)
    tidy_all.to_csv(os.path.join(outdir_rank, f"ml_{rank}_tidy.csv"), index=False)

    # wide per group
    wide_oc = wide_from_percent(oc_pct, "Crohn",   "Oral",  rank)
    wide_oh = wide_from_percent(oh_pct, "Healthy", "Oral",  rank)
    wide_fc = wide_from_percent(fc_pct, "Crohn",   "Fecal", rank)
    wide_fh = wide_from_percent(fh_pct, "Healthy", "Fecal", rank)

    meta_cols = ["Sample", "Group", "Site"]
    feature_cols = sorted(set(wide_oc.columns) | set(wide_oh.columns) |
                          set(wide_fc.columns) | set(wide_fh.columns))
    feature_cols = [c for c in feature_cols if c not in meta_cols]

    def _align(df):
        a = df.reindex(columns=feature_cols, fill_value=0)
        for m in meta_cols:
            a[m] = df[m].values if m in df.columns else None
        return a[feature_cols + meta_cols]

    wide_all = pd.concat([_align(wide_oc), _align(wide_oh), _align(wide_fc), _align(wide_fh)], ignore_index=True)
    wide_all_out = os.path.join(outdir_rank, f"ml_{rank}_wide_all.csv")
    wide_all.to_csv(wide_all_out, index=False)

    save_topk_wide(wide_all.copy(), os.path.join(outdir_rank, f"ml_{rank}_wide_topK.csv"), topk=topk, meta_cols=meta_cols)

    all_taxa = oc_pct.index.union(oh_pct.index).union(fc_pct.index).union(fh_pct.index)
    pct_all = pd.concat([
        oc_pct.reindex(all_taxa).fillna(0),
        oh_pct.reindex(all_taxa).fillna(0),
        fc_pct.reindex(all_taxa).fillna(0),
        fh_pct.reindex(all_taxa).fillna(0)
    ], axis=1)

    group_map = {}
    site_map  = {}
    for s in oc_pct.columns: group_map[s] = "Crohn";   site_map[s] = "Oral"
    for s in oh_pct.columns: group_map[s] = "Healthy"; site_map[s] = "Oral"
    for s in fc_pct.columns: group_map[s] = "Crohn";   site_map[s] = "Fecal"
    for s in fh_pct.columns: group_map[s] = "Healthy"; site_map[s] = "Fecal"

    wide_clr = ml_clr_from_percent(pct_all, group_map, site_map, rank=rank, pseudocount=1e-6)
    wide_clr.to_csv(os.path.join(outdir_rank, f"ml_{rank}_wide_clr.csv"), index=False)

# ---------- Rank driver ----------
def process_rank(rank: str,
                 oc_raw: pd.DataFrame, oh_raw: pd.DataFrame,
                 fc_raw: pd.DataFrame, fh_raw: pd.DataFrame,
                 outdir_rank: str, topk: int,
                 matched_path: str|None,
                 presence_pct: float, prevalence: float):
    os.makedirs(outdir_rank, exist_ok=True)
    rank_fn = extract_genus if rank == "genus" else extract_species

    # Ensure taxa × samples
    oc = ensure_taxa_by_samples(oc_raw)
    oh = ensure_taxa_by_samples(oh_raw)
    fc = ensure_taxa_by_samples(fc_raw)
    fh = ensure_taxa_by_samples(fh_raw)

    # Percent tables at `rank`
    oc_pct = percent_table(oc, rank_fn)
    oh_pct = percent_table(oh, rank_fn)
    fc_pct = percent_table(fc, rank_fn)
    fh_pct = percent_table(fh, rank_fn)

    # Pairs & group contrasts (incl. NEW within-site heatmaps)
    group_contrasts_and_csv(
        oc_pct, fc_pct, oh_pct, fh_pct, rank, outdir_rank,
        n=topk, prevalence_thresh=prevalence, presence_threshold_percent=presence_pct,
        use_log1p=True, clip_quantile=0.98, col_cluster_pairs=False
    )

    # Two-bar stacked with side legend-table
    two_bar_stacked_with_table(
        group_mean_percent(oc, rank_fn), group_mean_percent(oh, rank_fn),
        rank=rank, out_png=os.path.join(outdir_rank, f"barplot_oral_{rank}_top{topk}.png"),
        label_a="Oral_Crohn", label_b="Oral_Healthy", top_k=topk
    )
    two_bar_stacked_with_table(
        group_mean_percent(fc, rank_fn), group_mean_percent(fh, rank_fn),
        rank=rank, out_png=os.path.join(outdir_rank, f"barplot_fecal_{rank}_top{topk}.png"),
        label_a="Fecal_Crohn", label_b="Fecal_Healthy", top_k=topk
    )

    # DA — union of taxa across groups
    all_taxa = pd.Index(oc_pct.index).union(fc_pct.index).union(oh_pct.index).union(fh_pct.index)

    # Oral CH (MWU)
    run_da_oral_CH(
        mat_taxa_samples=oc_pct.reindex(all_taxa).fillna(0).join(
            oh_pct.reindex(all_taxa).fillna(0), how="outer"
        ),
        labels_A=oc_pct.columns, labels_B=oh_pct.columns, tax_labels=all_taxa,
        out_csv=os.path.join(outdir_rank, f"da_oral_CH_{rank}.csv"),
        do_volcano_png=os.path.join(outdir_rank, f"volcano_oral_CH_{rank}.png"),
        rank=rank
    )
    # Fecal CH (MWU)
    run_da_fecal_CH(
        mat_taxa_samples=fc_pct.reindex(all_taxa).fillna(0).join(
            fh_pct.reindex(all_taxa).fillna(0), how="outer"
        ),
        labels_A=fc_pct.columns, labels_B=fh_pct.columns, tax_labels=all_taxa,
        out_csv=os.path.join(outdir_rank, f"da_fecal_CH_{rank}.csv"),
        do_volcano_png=os.path.join(outdir_rank, f"volcano_fecal_CH_{rank}.png"),
        rank=rank
    )

    # Paired OC↔FC (Wilcoxon) — Crohn only
    paired_out = os.path.join(outdir_rank, f"da_paired_OC_FC_{rank}.csv")
    try:
        if matched_path and os.path.exists(matched_path):
            matched = pd.read_csv(matched_path)
            if matched is not None and not matched.empty:
                run_da_paired_OC_FC(
                    mat_taxa_samples_OC=oc_pct, mat_taxa_samples_FC=fc_pct,
                    matched_df=matched, tax_labels=all_taxa,
                    out_csv=paired_out, rank=rank
                )
            else:
                pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median","n_pairs"]).to_csv(paired_out, index=False)
        else:
            pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median","n_pairs"]).to_csv(paired_out, index=False)
    except Exception as e:
        warnings.warn(f"[paired] failed: \"{e}\"")
        pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median","n_pairs"]).to_csv(paired_out, index=False)

    # ML outputs
    make_ml_outputs(rank, oc_pct, oh_pct, fc_pct, fh_pct, outdir_rank, topk=topk)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Unified genus/species comparison pipeline (Crohn vs Healthy; Oral & Fecal)")
    p.add_argument("--oral-crohn",    required=True)
    p.add_argument("--oral-healthy",  required=True)
    p.add_argument("--fecal-crohn",   required=True)
    p.add_argument("--fecal-healthy", required=True)
    p.add_argument("--matched-ids",   default=None)
    p.add_argument("--outdir",        required=True)
    p.add_argument("--topk",          type=int, default=15)
    p.add_argument("--presence-pct",  type=float, default=0.1)
    p.add_argument("--prevalence",    type=float, default=0.10)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    oc_raw = pd.read_csv(args.oral_crohn,    index_col=0)
    oh_raw = pd.read_csv(args.oral_healthy,  index_col=0)
    fc_raw = pd.read_csv(args.fecal_crohn,   index_col=0)
    fh_raw = pd.read_csv(args.fecal_healthy, index_col=0)

    outdir_genus = os.path.join(args.outdir, "genus")
    process_rank("genus", oc_raw, oh_raw, fc_raw, fh_raw, outdir_genus,
                 topk=args.topk, matched_path=args.matched_ids,
                 presence_pct=args.presence_pct, prevalence=args.prevalence)
    print("[INFO] [genus] done →", outdir_genus)

    outdir_species = os.path.join(args.outdir, "species")
    process_rank("species", oc_raw, oh_raw, fc_raw, fh_raw, outdir_species,
                 topk=args.topk, matched_path=args.matched_ids,
                 presence_pct=args.presence_pct, prevalence=args.prevalence)
    print("[INFO] [species] done →", outdir_species)

    print("[INFO] All done.")

if __name__ == "__main__":
    main()
