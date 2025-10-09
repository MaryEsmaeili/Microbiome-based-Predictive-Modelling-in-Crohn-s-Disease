#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Taxa Compare (Unified) — Crohn vs Healthy, Oral & Fecal
-------------------------------------------------------

This script builds the exact same tables/figures as before (heatmaps, bars,
DA on CLR, etc.). **The only change you asked for** is in ML outputs:
 - Columns use `Sample_ID` instead of `Sample`
 - `Group` becomes numeric `disease` (Crohn=1, Healthy=0)

Everything else stays the same.

Inputs (CSV):
  --oral-crohn, --oral-healthy, --fecal-crohn, --fecal-healthy
    Matrices with rows=features (e.g., species or genus paths), columns=samples.

Optional:
  --meta data/meta/model_table_pooled.csv
    Used to map Sample → Sample_ID and to obtain disease (0/1).
    If not found, we fallback to Sample_ID=Sample and disease from group label.

Outputs:
  results/taxa_compare_unified/genus/...
  results/taxa_compare_unified/species/...
  (unchanged layout for figures/tables except ML files)

Author: you ♥
"""

from __future__ import annotations

# ================ Standard / scientific stack ================
import os
import re
import json
import textwrap
import argparse
import warnings
import numpy as np
import pandas as pd

# ================ Plotting ================
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator
import matplotlib.patheffects as pe

# ================ Stats ================
from scipy.stats import mannwhitneyu, wilcoxon

# Styling
sns.set_context("talk")
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


# ---------------- Colors (can be overridden later via YAML if you like) -------------
DEFAULT_PALETTE = {
    "Fecal_Crohn":   "#30638e",
    "Oral_Crohn":    "#edae49",
    "Fecal_Healthy": "#d1495b",
    "Oral_Healthy":  "#00798c",
    "Other":         "#999999",
    # aliases used in code (consistent normalization):
    "Crohn-Oral":    "#edae49",
    "Crohn-Fecal":   "#30638e",
    "Healthy-Oral":  "#00798c",
    "Healthy-Fecal": "#d1495b",
    "Oral":          "#edae49",
    "Fecal":         "#30638e",
}
PALETTE = dict(DEFAULT_PALETTE)


# ====================== Small helpers ======================
def _normalize_group_label(label: str) -> str:
    """
    Normalize free-text labels into keys we use in PALETTE.
    e.g., 'Oral_Crohn' → 'Crohn-Oral'
    """
    s = str(label).replace("_", "-").strip().lower()
    site = "oral" if "oral" in s else ("fecal" if ("fecal" in s or "faecal" in s or "stool" in s) else None)
    dis  = "crohn" if ("crohn" in s or "cd" in s) else ("healthy" if ("healthy" in s or "hc" in s) else None)
    if site and dis:
        return f"{dis.capitalize()}-{site.capitalize()}"
    return label


def group_color(label: str, default="#999999") -> str:
    """Map a group label to a hex color using the palette."""
    key = _normalize_group_label(label)
    return PALETTE.get(key, PALETTE.get(label, default))


def build_col_colors(label_left: str, n_left: int, label_right: str, n_right: int):
    """Top color strip for heatmaps."""
    cL = group_color(label_left)
    cR = group_color(label_right)
    return [cL]*int(n_left) + [cR]*int(n_right)


def _normalize_id(x: str) -> str:
    """Normalize IDs for robust matching (lowercase alphanumerics)."""
    return re.sub(r'[^a-z0-9]+', '', str(x).strip().lower())


# ====================== Taxonomy helpers ======================
def extract_genus(rowname: str):
    """Extract 'g__...' from a pipe-delimited path; return None if absent."""
    parts = str(rowname).split("|")
    g = [p for p in parts if p.startswith("g__")]
    return g[0] if g else None


def extract_species(rowname: str):
    """Extract 's__...' from a pipe-delimited path; return None if absent."""
    parts = str(rowname).split("|")
    s = [p for p in parts if p.startswith("s__")]
    return s[0] if s else None


def prettify_taxon(label: str, rank: str) -> str:
    """Human-friendly taxa labels for plotting (Genus Capitalized, species epithet lowercase)."""
    if label is None:
        return None
    r = (rank or "").lower()
    if r == "genus":
        name = label.replace("g__", "").replace("_", " ")
        return " ".join(w.capitalize() for w in name.split())
    if r == "species":
        name = label.replace("s__", "").replace("_", " ")
        toks = name.split()
        if not toks:
            return name
        toks[0] = toks[0].capitalize()
        toks[1:] = [t.lower() for t in toks[1:]]
        return " ".join(toks)
    return label


# ====================== Orientation & transforms ======================
def ensure_taxa_by_samples(df_like: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure matrix is taxa × samples.
    If >50% of columns look like taxa (contain 's__'/'g__'), input is samples×taxa → transpose.
    """
    cols = pd.Index(df_like.columns.astype(str))
    looks_like_taxa = cols.str.contains("s__|g__", regex=True).mean()
    return df_like.T if looks_like_taxa > 0.5 else df_like


def percent_table(df: pd.DataFrame, grouper) -> pd.DataFrame:
    """
    Collapse raw features at the given rank via `grouper(rowname)`,
    then convert to % per sample. Returns taxa × samples (% values).
    """
    g = df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    return g.div(g.sum(axis=0), axis=1) * 100.0


def group_mean_percent(df: pd.DataFrame, grouper) -> pd.Series:
    """Per-taxon mean % abundance across samples of a group (for stacked bars)."""
    g = df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    pct = g.div(g.sum(axis=0), axis=1) * 100
    return pct.mean(axis=1)


def clr_transform(df_taxa_by_samples: pd.DataFrame, pseudocount=1e-6) -> pd.DataFrame:
    """
    CLR transform on taxa × samples (% as proportions first).
    """
    X = df_taxa_by_samples.astype(float) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)
    return logX.sub(gm, axis=1)


# ====================== Row z-score for heatmaps ======================
def row_zscore_log1p(mat: pd.DataFrame, use_log1p=True) -> pd.DataFrame:
    """Row-wise z-score (optionally on log1p of %)."""
    X = np.log1p(mat.astype(float)) if use_log1p else mat.astype(float)
    mu = X.mean(axis=1)
    sd = X.std(axis=1).replace(0, np.nan)
    return X.sub(mu, axis=0).div(sd, axis=0).fillna(0)


# ====================== Heatmap utilities ======================
def _subsample_xticks(ax, max_labels=25):
    """Reduce x tick clutter when too many samples."""
    ticks = ax.get_xticks()
    if len(ticks) <= max_labels or max_labels <= 0:
        return
    step = max(1, int(round(len(ticks) / max_labels)))
    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_visible((i % step) == 0)


def _apply_smart_yticklabels(ax, labels, rank, base_fs=12.0, max_chars=28):
    """Wrap long taxa labels and italicize."""
    if labels is None:
        return
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
            try:
                txt.set_style("italic")
            except Exception:
                pass


def _coolwarm():
    """Diverging colormap for z-scored heatmaps."""
    return plt.get_cmap("coolwarm")


def pairs_heatmap(
    A_pct, B_pct, rows, label_A, label_B, rank, out_png,
    use_log1p=True, clip_quantile=0.98, title_suffix=""
):
    """
    Generic two-block heatmap with fixed order (no clustering).
    A and B are taxa × samples (%).
    """
    data = pd.concat([A_pct, B_pct], axis=1).reindex(rows).fillna(0)
    pretty_idx = [prettify_taxon(t, rank) for t in data.index]

    Z = row_zscore_log1p(data, use_log1p=use_log1p)
    Z.index = pretty_idx

    V = np.nanquantile(np.abs(Z.values), clip_quantile)
    if not np.isfinite(V) or V == 0:
        V = max(1.0, float(np.nanmax(np.abs(Z.values)) or 1.0))

    height = max(6.0, 0.35 * len(Z.index))
    width  = max(8.0, 0.18 * len(Z.columns) + 2.0)

    nA, nB = A_pct.shape[1], B_pct.shape[1]
    col_colors = build_col_colors(label_A, nA, label_B, nB)

    g = sns.clustermap(
        Z, cmap=_coolwarm(), center=0, vmin=-V, vmax=+V,
        col_cluster=False, row_cluster=False,
        figsize=(width, height),
        cbar_kws={"label": "Row z-score of % abundance (log1p)", "orientation": "horizontal"},
        cbar_pos=(.25, .97, .50, .02),
        col_colors=col_colors
    )

    g.ax_heatmap.yaxis.set_ticks_position("right")
    g.ax_heatmap.yaxis.set_label_position("right")
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_ymajorticklabels(),
        rotation=0, ha="left", va="center", fontsize=10
    )
    g.ax_heatmap.tick_params(axis="y", pad=10)

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


def heatmap_two_groups(A_pct, B_pct, rank, out_png, title):
    """Unpaired two-group heatmap for ALL shared taxa with non-zero sum in both groups."""
    shared = A_pct.index.intersection(B_pct.index)
    present_A = (A_pct.loc[shared].sum(axis=1) > 0)
    present_B = (B_pct.loc[shared].sum(axis=1) > 0)
    rows = shared[(present_A & present_B).values]

    union = A_pct.loc[rows].add(B_pct.loc[rows], fill_value=0.0)
    rows = union.sum(axis=1).sort_values(ascending=False).index

    label_A, label_B = title.split(" vs ")
    pairs_heatmap(
        A_pct, B_pct, rows, label_A=label_A, label_B=label_B,
        rank=rank, out_png=out_png, title_suffix="(unpaired)"
    )


# ====================== Multiple testing / effects ======================
def bh_qvalues(pvals):
    """Benjamini–Hochberg correction."""
    p = np.asarray(pvals, dtype=float); n = len(p)
    order = np.argsort(p); q = np.empty(n, dtype=float); prev = 1.0
    for i, idx in enumerate(order[::-1], start=1):
        rank = n - i + 1
        val = min(prev, p[idx] * n / rank)
        q[idx] = val; prev = val
    return q.tolist()


def cliffs_delta(x, y):
    """Cliff's delta (nonparametric effect size)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return 0.0
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    m, n = len(x), len(y)
    return (gt - lt) / (m * n)


def robust_log2fc(a, b, eps=1e-9):
    """log2 fold-change between means (epsilon-stabilized)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if np.nanmin(a) < 0 or np.nanmin(b) < 0:
        return (np.nanmean(a) - np.nanmean(b)) / np.log(2.0)
    return np.log2(np.nanmean(a) + eps) - np.log2(np.nanmean(b) + eps)


# ====================== Volcano ======================
def volcano(
    df, x="log2FC", q="q", title="", out_png="volcano.png",
    q_sig=0.05, k_onplot_each=8, figsize=(11.5, 6.8), dpi=340,
    point_size=22, colors=dict(non="#999999", up="#1B4F72", down="#7FB3D5"),
    xlabel=None
):
    """Volcano plot (labels the top up/down significant taxa)."""
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
    ax = fig.add_subplot(111)
    ax.scatter(X[~is_sig], Y[~is_sig], s=point_size, alpha=0.35, color=colors["non"], zorder=1)
    ax.scatter(X[is_sig & (X > 0)], Y[is_sig & (X > 0)], s=point_size, alpha=0.9, color=colors["up"], zorder=2)
    ax.scatter(X[is_sig & (X < 0)], Y[is_sig & (X < 0)], s=point_size, alpha=0.9, color=colors["down"], zorder=2)
    ax.axhline(-np.log10(q_sig), ls="--", lw=1, color="gray", alpha=0.8)
    ax.axvline(0, ls="--", lw=1, color="gray", alpha=0.8)
    ax.grid(True, ls=":", lw=0.6, alpha=0.4)

    # quick & light label repulsion
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
            if not bumped:
                break
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


# ====================== DA drivers (MWU on CLR) ======================
def run_da_group_CH(
    mat_taxa_samples: pd.DataFrame, labels_crohn, labels_healthy, tax_labels,
    out_csv, do_volcano_png=None, rank="genus",
    mode="clr", x_col_name=None, volcano_title=None, volcano_xlabel=None
):
    """
    Differential abundance: Crohn vs Healthy.
    - mode='clr' (default): CLR transform then MWU; x-axis = ΔCLR(mean).
    """
    M = mat_taxa_samples.loc[tax_labels].fillna(0).astype(float)

    if mode == "clr":
        M_prop = (M / 100.0).clip(lower=0)
        M_test = clr_transform(M_prop, pseudocount=1e-6)
        xname = "delta_clr" if x_col_name is None else x_col_name
        xlab  = "ΔCLR (Crohn − Healthy)" if volcano_xlabel is None else volcano_xlabel
    else:
        raise ValueError("Only CLR mode is supported here.")

    rows, pvals = [], []
    for tax in tax_labels:
        if tax not in M_test.index:
            continue
        a = M_test.loc[tax, labels_crohn].values
        b = M_test.loc[tax, labels_healthy].values
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            u, p = np.nan, 1.0
        eff  = cliffs_delta(a, b)
        xval = float(np.nanmean(a) - np.nanmean(b))  # ΔCLR

        rows.append({
            "taxon": tax,
            "pretty_taxon": prettify_taxon(tax, rank),
            "test": "MWU",
            "stat": float(u) if np.isfinite(u) else np.nan,
            "p": float(p),
            "effect_name": "Cliffs_delta",
            "effect_size": float(eff),
            xname: xval,
        })
        pvals.append(p)

    df = pd.DataFrame(rows)
    if len(df):
        df["q"] = bh_qvalues(df["p"].values)

    base_cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size", xname]
    (pd.DataFrame(columns=base_cols) if df.empty else df.reindex(columns=base_cols)).to_csv(out_csv, index=False)

    if do_volcano_png and not df.empty:
        volcano(df, x=xname, q="q",
                title=(volcano_title or "Volcano (Crohn vs Healthy)"),
                out_png=do_volcano_png, xlabel=xlab)
    return df


def run_da_oral_CH(*, mat_taxa_samples, labels_A, labels_B, tax_labels, out_csv,
                   do_volcano_png=None, rank="genus"):
    """DA for Oral site."""
    return run_da_group_CH(
        mat_taxa_samples=mat_taxa_samples,
        labels_crohn=labels_A, labels_healthy=labels_B, tax_labels=tax_labels,
        out_csv=out_csv, do_volcano_png=do_volcano_png, rank=rank,
        mode="clr", x_col_name="delta_clr",
        volcano_title=f"Volcano Oral Crohn vs Healthy ({rank})",
        volcano_xlabel="ΔCLR (Oral: Crohn − Healthy)",
    )


def run_da_fecal_CH(*, mat_taxa_samples, labels_A, labels_B, tax_labels, out_csv,
                    do_volcano_png=None, rank="genus"):
    """DA for Fecal site."""
    return run_da_group_CH(
        mat_taxa_samples=mat_taxa_samples,
        labels_crohn=labels_A, labels_healthy=labels_B, tax_labels=tax_labels,
        out_csv=out_csv, do_volcano_png=do_volcano_png, rank=rank,
        mode="clr", x_col_name="delta_clr",
        volcano_title=f"Volcano Fecal Crohn vs Healthy ({rank})",
        volcano_xlabel="ΔCLR (Fecal: Crohn − Healthy)",
    )


# ====================== Paired OC ↔ FC (Crohn only) ======================
def _guess_pair_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Heuristic to infer the two columns for (Oral, Fecal) in a matched file."""
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
    """Map user-provided IDs to actual matrix column names via normalized lookup."""
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


def run_da_paired_OC_FC(
    mat_taxa_samples_OC: pd.DataFrame,
    mat_taxa_samples_FC: pd.DataFrame,
    matched_df: pd.DataFrame, tax_labels, out_csv, rank="genus",
    min_pairs:int=8
):
    """Paired Wilcoxon on Crohn Oral vs Fecal using matched IDs."""
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


# ====================== Stacked bars (with small table) ======================
def build_taxa_color_cycle(n_needed=20):
    """Pleasant cycle of colors derived from group palette (for stacked segments)."""
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



def two_bar_stacked_with_table(
    a_mean: pd.Series, b_mean: pd.Series, rank: str, out_png: str,
    label_a: str, label_b: str, top_k: int = 15
):
    """
    Two stacked bars (A vs B) + a compact legend-table on the right.
    Values plotted as proportions (0–1). 'Other' sits on top of the stack.
    """
    combined = a_mean.add(b_mean, fill_value=0).sort_values(ascending=False)
    top = combined.index[:min(top_k, len(combined))]
    s_a_pct = a_mean.reindex(top).fillna(0)
    s_b_pct = b_mean.reindex(top).fillna(0)
    s_a_pct = pd.concat([s_a_pct, pd.Series({"Other": a_mean.drop(top, errors='ignore').sum()})])
    s_b_pct = pd.concat([s_b_pct, pd.Series({"Other": b_mean.drop(top, errors='ignore').sum()})])

    # Convert % → proportion for plotting
    s_a = (s_a_pct / 100.0).clip(lower=0)
    s_b = (s_b_pct / 100.0).clip(lower=0)

    # Order taxa by A’s contribution so the stack reads naturally
    order = s_a.sort_values(ascending=False).index

    # Assign colors
    taxa_colors = build_taxa_color_cycle(max(3*len(order), 12))
    color_map = {tax: taxa_colors[i % len(taxa_colors)] for i, tax in enumerate(order)}
    color_map["Other"] = PALETTE.get("Other", "#999999")

    # Subtle edge colors by site type
    edge_a = PALETTE.get("Oral", PALETTE.get("Crohn-Oral", "#edae49")) if "Oral" in label_a else \
             PALETTE.get("Fecal", PALETTE.get("Crohn-Fecal", "#30638e"))
    edge_b = PALETTE.get("Oral", PALETTE.get("Healthy-Oral", "#00798c")) if "Oral" in label_b else \
             PALETTE.get("Fecal", PALETTE.get("Healthy-Fecal", "#d1495b"))

    # Layout: left = stacked bars; right = mini table
    fig = plt.figure(figsize=(16.5, 9.0))
    gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[2.4, 1.6], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1]); ax_tbl.axis("off")

    # Plot stacks
    x = np.array([0, 1]); b0 = b1 = 0.0
    rows_for_table = []
    for tax in order:
        h0 = float(s_a.loc[tax]); h1 = float(s_b.loc[tax])
        ax.bar(x[0], h0, bottom=b0, color=color_map[tax], width=0.6, edgecolor=edge_a, linewidth=0.7)
        ax.bar(x[1], h1, bottom=b1, color=color_map[tax], width=0.6, edgecolor=edge_b, linewidth=0.7)

        # Inline labels (≥ 3%)
        if h0 >= 0.03:
            ax.text(x[0], b0 + h0/2, f"{h0:.2f}", ha="center", va="center", fontsize=9, color="black")
        if h1 >= 0.03:
            ax.text(x[1], b1 + h1/2, f"{h1:.2f}", ha="center", va="center", fontsize=9, color="black")

        rows_for_table.append((tax, color_map[tax], h0, h1))
        b0 += h0; b1 += h1

    ax.set_xticks(x); ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel("Mean relative abundance (proportion)")
    ax.set_title(f"{label_a} vs {label_b} — Top {min(top_k,len(order))} {rank.capitalize()} (others → Other)")
    for spine in ["top","right"]: ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8); ax.spines["bottom"].set_linewidth(0.8)
    ax.set_ylim(0, 1.05)

    # Legend-table on the right
    rows_sorted = sorted(rows_for_table, key=lambda r: (0 if r[0]=="Other" else 1, -(r[2]+r[3])))
    y0 = 0.975; dy = 0.042
    ax_tbl.text(0.10, y0, f"{label_a}", fontweight="bold", ha="right", transform=ax_tbl.transAxes)
    ax_tbl.text(0.22, y0, f"{label_b}", fontweight="bold", ha="right", transform=ax_tbl.transAxes)
    ax_tbl.text(0.29, y0, "Color",       fontweight="bold", transform=ax_tbl.transAxes)
    ax_tbl.text(0.39, y0, "Taxon",       fontweight="bold", transform=ax_tbl.transAxes)

    y = y0 - dy
    for tax, col, vA, vB in rows_sorted:
        ax_tbl.text(0.10, y, f"{vA:.2f}", ha="right", va="center", transform=ax_tbl.transAxes, fontsize=10)
        ax_tbl.text(0.22, y, f"{vB:.2f}", ha="right", va="center", transform=ax_tbl.transAxes, fontsize=10)
        ax_tbl.add_patch(plt.Rectangle((0.29, y-0.015), 0.06, 0.02, transform=ax_tbl.transAxes, color=col, clip_on=False))
        name = prettify_taxon(tax, rank) if tax != "Other" else "Other"
        ax_tbl.text(0.39, y, name, transform=ax_tbl.transAxes, va="center")
        y -= dy
        if y < 0.02:
            break

    plt.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

def stacked_bars_by_sample(pct: pd.DataFrame, rank: str, out_png: str,
                           topk: int = 10, title: str = ""):
    """
    Per-sample stacked bar chart for one group (taxa × samples, %).
    - Shows top-K taxa by global sum across these samples; the rest go to 'Other'.
    - Y-axis is 0..100 (%). X-axis is samples (many → rotate labels).
    """
    if pct.empty:
        # nothing to draw
        fig = plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    # pick top-K taxa globally within this group
    totals = pct.sum(axis=1).sort_values(ascending=False)
    top = totals.head(int(max(1, topk))).index
    top_mat = pct.loc[top].copy()

    # 'Other' = all remaining taxa
    if len(totals) > len(top):
        other = pct.drop(index=top, errors="ignore").sum(axis=0)
        top_mat.loc["Other"] = other.values

    # order samples by total of the most abundant taxon to make bars readable
    sample_order = top_mat.iloc[0].sort_values(ascending=False).index
    top_mat = top_mat.reindex(columns=sample_order)

    # build colors
    taxa = list(top_mat.index)
    color_cycle = build_taxa_color_cycle(max(3*len(taxa), 12))
    color_map = {t: color_cycle[i % len(color_cycle)] for i, t in enumerate(taxa)}
    if "Other" in color_map:
        color_map["Other"] = PALETTE.get("Other", "#999999")

    # plot
    fig = plt.figure(figsize=(max(12.0, 0.23*top_mat.shape[1] + 6.0), 6.0))
    ax = fig.add_subplot(111)

    bottoms = np.zeros(top_mat.shape[1], dtype=float)
    x = np.arange(top_mat.shape[1])
    for t in taxa:
        vals = top_mat.loc[t].values
        ax.bar(x, vals, bottom=bottoms, color=color_map[t], width=0.85, linewidth=0)
        bottoms += vals

    ax.set_ylim(0, 100)
    ax.set_ylabel("Relative abundance (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in top_mat.columns], rotation=90, fontsize=8)
    ax.set_title(title if title else f"Top {min(topk, len(taxa))} {rank.capitalize()} per sample")

    # legend
    handles = [plt.Rectangle((0,0),1,1,color=color_map[t]) for t in taxa]
    labels  = [prettify_taxon(t, rank) if t!="Other" else "Other" for t in taxa]
    ax.legend(handles, labels, ncol=2, bbox_to_anchor=(1.02, 1.02),
              loc="upper left", frameon=False, title=rank.capitalize())

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------- ML-friendly outputs ----------
def tidy_from_percent(
    pct: pd.DataFrame,
    group_label: str,
    site_label: str,
    rank: str,
    sample_to_id: dict,
    disease_by_id: dict
) -> pd.DataFrame:
    """
    Long format for ML with unified column names:
      Sample_ID, Taxon (pretty), AbundancePct, disease (0/1), Site
    - pct: taxa × samples (%) for a (group, site)
    - group_label: only as fallback when disease is missing in meta
                   (Crohn=1, Healthy=0)
    """
    if pct.empty:
        return pd.DataFrame(columns=["Sample_ID", "Taxon", "AbundancePct", "disease", "Site"])

    # taxa × samples  →  long
    tidy = pct.stack().reset_index()
    tidy.columns = [rank, "Sample", "AbundancePct"]

    # Sample → Sample_ID
    sample_id = tidy["Sample"].map(lambda s: sample_to_id.get(str(s), str(s)))

    # disease from meta; fallback by group label
    gl = str(group_label).strip().lower()
    fallback_dis = 1 if gl in {"crohn", "cd", "case", "ibd"} else 0
    disease = sample_id.map(lambda sid: disease_by_id.get(str(sid), fallback_dis)).astype("int64")

    # pretty taxon for readability (raw taxon column 'rank' لازم نیست برگردونیم)
    pretty = tidy[rank].map(lambda t: prettify_taxon(t, rank))

    out = pd.DataFrame({
        "Sample_ID":   sample_id,
        "Taxon":       pretty,
        "AbundancePct": tidy["AbundancePct"].astype("float64"),
        "disease":     disease,
        "Site":        site_label,
    })
    return out

def wide_from_percent(
    pct: pd.DataFrame,
    group_label: str,
    site_label: str,
    rank: str,
    sample_to_id: dict,
    disease_by_id: dict
) -> pd.DataFrame:

    """
    Return WIDE table with taxa columns + Sample_ID, disease (0/1), Site.
    Columns order: [taxa..., Sample_ID, disease, Site]
    """
    if pct.empty:
        return pd.DataFrame(columns=["Sample_ID", "disease", "Site"])

    df = pct.T.copy()  # samples × taxa
    df["__raw_sample__"] = df.index.astype(str)
    df["Sample_ID"] = df["__raw_sample__"].map(lambda s: sample_to_id.get(s, s))
    df["Site"] = site_label

    gl = group_label.strip().lower()
    fallback_dis = 1 if gl in {"crohn", "cd", "case", "ibd"} else 0
    df["disease"] = df["Sample_ID"].map(lambda sid: disease_by_id.get(str(sid), fallback_dis)).astype("int64")

    df = df.drop(columns=["__raw_sample__"])

    meta_cols = ["Sample_ID", "disease", "Site"]
    taxa_cols = [c for c in df.columns if c not in meta_cols]
    return df[taxa_cols + meta_cols].reset_index(drop=True)


def save_topk_wide(
    wide_df: pd.DataFrame,
    out_csv: str,
    topk: int = 15,
    meta_cols=("Sample_ID", "disease", "Site")
):
    """
    Save a wide matrix with only top-K taxa by global sum (plus meta columns).
    """
    meta_cols_present = [c for c in meta_cols if c in wide_df.columns]
    feat_df = wide_df.drop(columns=meta_cols_present, errors="ignore")
    if feat_df.shape[1] == 0:
        # no features, just write meta
        wide_df.to_csv(out_csv, index=False)
        return
    top_taxa = feat_df.sum(axis=0).sort_values(ascending=False).head(int(topk)).index
    out_df = pd.concat([wide_df.loc[:, top_taxa], wide_df.loc[:, meta_cols_present]], axis=1)
    out_df.to_csv(out_csv, index=False)


def ml_clr_from_percent(
    pct_all_sites: pd.DataFrame,
    site_map: dict,
    sample_to_id: dict,
    disease_by_id: dict,
    pseudocount=1e-6
) -> pd.DataFrame:
    """
    Build CLR-wide matrix for ML with Sample_ID + disease (0/1) + Site.
    - pct_all_sites: taxa × samples (%) for ALL four groups concatenated.
    - site_map: dict raw_sample -> "Oral"/"Fecal" (used to populate Site)
    """
    if pct_all_sites.empty:
        return pd.DataFrame(columns=["Sample_ID", "disease", "Site"])

    # Map columns (raw sample names) → Sample_ID
    sid_cols = [sample_to_id.get(str(c), str(c)) for c in pct_all_sites.columns]
    pct_all = pct_all_sites.copy()
    pct_all.columns = sid_cols

    # CLR on proportions, then transpose → rows = Sample_ID
    clr = clr_transform(pct_all / 100.0, pseudocount=pseudocount).T
    clr.index.name = "Sample_ID"

    # disease from meta; if a Sample_ID is missing from meta, drop or set NA
    dis = pd.Series({sid: disease_by_id.get(str(sid), np.nan) for sid in clr.index}, name="disease")

    # Site from site_map (requires going through raw names; we approximate via inverse mapping)
    # Build inverse mapping: Sample_ID -> one raw sample (first hit)
    inv = {}
    for raw, sid in sample_to_id.items():
        inv.setdefault(sid, raw)
    site = pd.Series({
        sid: site_map.get(inv.get(sid, sid), None) for sid in clr.index
    }, name="Site")

    wide = clr.join(dis).join(site)
    # Place meta at the end
    meta_cols = ["Sample_ID", "disease", "Site"]
    wide = wide.reset_index()
    taxa_cols = [c for c in wide.columns if c not in meta_cols]
    wide = wide[taxa_cols + meta_cols]

    # Ensure disease is integer if possible
    if "disease" in wide.columns:
        wide["disease"] = pd.to_numeric(wide["disease"], errors="coerce").astype("Int64")
    return wide


def make_ml_outputs(
    rank: str,
    oc_pct: pd.DataFrame,
    oh_pct: pd.DataFrame,
    fc_pct: pd.DataFrame,
    fh_pct: pd.DataFrame,
    outdir_rank: str,
    topk: int = 15,
    sample_to_id: dict = None,
    disease_by_id: dict = None
):
    """
    Emit ML-ready files using Sample_ID + disease (0/1):
      - ml_{rank}_tidy.csv
      - ml_{rank}_wide_all.csv
      - ml_{rank}_wide_topK.csv
      - ml_{rank}_wide_clr.csv
    """
    os.makedirs(outdir_rank, exist_ok=True)
    sample_to_id = sample_to_id or {}
    disease_by_id = disease_by_id or {}

    # ---------- TIDY (long) ----------
    tidy_all = pd.concat([
        tidy_from_percent(oc_pct, "Crohn",   "Oral",  rank, sample_to_id, disease_by_id),
        tidy_from_percent(oh_pct, "Healthy", "Oral",  rank, sample_to_id, disease_by_id),
        tidy_from_percent(fc_pct, "Crohn",   "Fecal", rank, sample_to_id, disease_by_id),
        tidy_from_percent(fh_pct, "Healthy", "Fecal", rank, sample_to_id, disease_by_id),
    ], ignore_index=True)
    tidy_all.to_csv(os.path.join(outdir_rank, f"ml_{rank}_tidy.csv"), index=False)

    # ---------- WIDE per-group ----------
    wide_oc = wide_from_percent(oc_pct, "Crohn",   "Oral",  rank, sample_to_id, disease_by_id)
    wide_oh = wide_from_percent(oh_pct, "Healthy", "Oral",  rank, sample_to_id, disease_by_id)
    wide_fc = wide_from_percent(fc_pct, "Crohn",   "Fecal", rank, sample_to_id, disease_by_id)
    wide_fh = wide_from_percent(fh_pct, "Healthy", "Fecal", rank, sample_to_id, disease_by_id)

    # Align feature columns across groups; keep meta at the end
    meta_cols = ["Sample_ID", "disease", "Site"]
    def _features(df): return [c for c in df.columns if c not in meta_cols]
    all_feats = sorted(set(_features(wide_oc)) | set(_features(wide_oh)) |
                       set(_features(wide_fc)) | set(_features(wide_fh)))

    def _align(df):
        a = df.reindex(columns=all_feats + meta_cols, fill_value=0)
        return a

    wide_all = pd.concat([_align(wide_oc), _align(wide_oh), _align(wide_fc), _align(wide_fh)], ignore_index=True)
    wide_all_out = os.path.join(outdir_rank, f"ml_{rank}_wide_all.csv")
    wide_all.to_csv(wide_all_out, index=False)

    # ---------- WIDE topK ----------
    save_topk_wide(
        wide_all.copy(),
        os.path.join(outdir_rank, f"ml_{rank}_wide_topK.csv"),
        topk=topk,
        meta_cols=tuple(meta_cols)
    )

    # ---------- WIDE CLR ----------
    # Build a single taxa × samples table (ALL groups) for CLR
    all_taxa = oc_pct.index.union(oh_pct.index).union(fc_pct.index).union(fh_pct.index)
    pct_all_sites = pd.concat([
        oc_pct.reindex(all_taxa).fillna(0),
        oh_pct.reindex(all_taxa).fillna(0),
        fc_pct.reindex(all_taxa).fillna(0),
        fh_pct.reindex(all_taxa).fillna(0)
    ], axis=1)

    # site_map from raw sample name → site
    site_map = {}
    for s in oc_pct.columns: site_map[str(s)] = "Oral"
    for s in oh_pct.columns: site_map[str(s)] = "Oral"
    for s in fc_pct.columns: site_map[str(s)] = "Fecal"
    for s in fh_pct.columns: site_map[str(s)] = "Fecal"

    wide_clr = ml_clr_from_percent(
        pct_all_sites=pct_all_sites,
        site_map=site_map,
        sample_to_id=sample_to_id,
        disease_by_id=disease_by_id,
        pseudocount=1e-6
    )
    wide_clr.to_csv(os.path.join(outdir_rank, f"ml_{rank}_wide_clr.csv"), index=False)

def select_top_by_delta(A_pct: pd.DataFrame, B_pct: pd.DataFrame, k: int = 20) -> pd.Index:
    shared = A_pct.index.intersection(B_pct.index)
    if len(shared) == 0:
        return pd.Index([])
    A = A_pct.loc[shared].fillna(0.0)
    B = B_pct.loc[shared].fillna(0.0)
    da = (A.mean(axis=1) - B.mean(axis=1)).abs()
    return da.sort_values(ascending=False).head(int(k)).index

def group_contrasts_and_csv(
    oc_pct, fc_pct, oh_pct, fh_pct, rank, outdir,
    n=15, heat_topk=20,
    use_log1p=True, clip_quantile=0.98
):
    """
    Build per-rank outputs:

    (A) Crohn Pairs (Oral vs Fecal):
        - FULL: all real shared taxa (no top-k)
        - TOP-K: top taxa by |Δ mean %| (visual only)

    (B) Group means & deltas (Top-N compact):
        - group_means_{rank}.csv (+ heatmap)
        - group_deltas_{rank}.csv (+ heatmap)

    (C) Within-site Crohn vs Healthy (unpaired):
        - Oral: FULL + TOP-K
        - Fecal: FULL + TOP-K
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

    os.makedirs(outdir, exist_ok=True)

    # ----- (A1) Crohn pairs — FULL (shared & present in both) -----
    shared_cf = oc_pct.index.intersection(fc_pct.index)
    present_oc = (oc_pct.loc[shared_cf].sum(axis=1) > 0)
    present_fc = (fc_pct.loc[shared_cf].sum(axis=1) > 0)
    rows_cf_all = shared_cf[(present_oc & present_fc).values]
    order_cf_all = (
        oc_pct.loc[rows_cf_all].add(fc_pct.loc[rows_cf_all], fill_value=0.0)
        .sum(axis=1).sort_values(ascending=False).index
    )
    pairs_heatmap(
        oc_pct, fc_pct, order_cf_all,
        label_A="Crohn-Oral", label_B="Crohn-Fecal", rank=rank,
        out_png=os.path.join(outdir, f"heatmap_pairs_crohn_{rank}.png"),
        use_log1p=use_log1p, clip_quantile=clip_quantile
    )

    # ----- (A2) Crohn pairs — TOP-K by |Δ mean %| -----
    top_cf = select_top_by_delta(oc_pct, fc_pct, k=heat_topk)
    if len(top_cf) > 0:
        pairs_heatmap(
            oc_pct, fc_pct, top_cf,
            label_A="Crohn-Oral", label_B="Crohn-Fecal", rank=rank,
            out_png=os.path.join(outdir, f"heatmap_pairs_crohn_{rank}_top{heat_topk}.png"),
            use_log1p=use_log1p, clip_quantile=clip_quantile,
            title_suffix=f"(Top {heat_topk} by |Δ mean %|)"
        )

    # ----- (B) Group means & deltas (Top-N compact) -----
    crohn_union_index = oc_pct.index.union(fc_pct.index)
    means_df = pd.DataFrame({
        "O.C.": oc_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
        "F.C.": fc_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
        "O.H.": oh_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
        "F.H.": fh_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
    }, index=crohn_union_index)
    means_df = means_df.loc[(means_df.sum(axis=1) > 0.0)]
    sort_cols = [c for c in ["O.C.", "F.C.", "O.H.", "F.H."] if c in means_df.columns]
    means_df = means_df.sort_values(by=sort_cols, ascending=False).head(n)
    means_df.index.name = rank
    means_df.to_csv(os.path.join(outdir, f"group_means_{rank}.csv"))

    deltas_cols = {}
    if "O.C." in means_df.columns and "O.H." in means_df.columns:
        deltas_cols["O.C.–O.H."] = means_df["O.C."] - means_df["O.H."]
    if "F.C." in means_df.columns and "F.H." in means_df.columns:
        deltas_cols["F.C.–F.H."] = means_df["F.C."] - means_df["F.H."]
    if not deltas_cols:
        deltas_cols["O.C.–F.C."] = means_df["O.C."] - means_df["F.C."]
    deltas_df = pd.DataFrame(deltas_cols)
    deltas_df.index.name = rank
    deltas_df.to_csv(os.path.join(outdir, f"group_deltas_{rank}.csv"))

    # Heatmap: Means (row z-score, no log1p)
    Zm = row_zscore_log1p(means_df, use_log1p=False)
    fig_h = max(6.0, 0.45 * len(Zm.index))
    plt.figure(figsize=(10.5, fig_h))
    ax = sns.heatmap(Zm[sort_cols], cmap=_coolwarm(), cbar_kws={"label": "Row z-score of means"})
    ax.set_xlabel("Groups"); ax.set_ylabel(rank.capitalize()); ax.set_title("Group means (Top-N)")
    _apply_smart_yticklabels(ax, [prettify_taxon(t, rank) for t in Zm.index], rank, base_fs=12.5)
    plt.gcf().subplots_adjust(left=0.32)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"heatmap_group_means_{rank}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Heatmap: Deltas (|Δ|)
    D = deltas_df.abs()
    fig_h = max(6.0, 0.45 * len(D.index))
    plt.figure(figsize=(9.2, fig_h))
    vmax = float(np.nanquantile(D.values, 0.98)) if np.isfinite(np.nanquantile(D.values, 0.98)) else 1.0
    ax = sns.heatmap(D, cmap="Blues", vmin=0.0, vmax=vmax, cbar_kws={"label": "|Delta| (percentage points)"})
    ax.set_xlabel("Comparisons"); ax.set_ylabel(rank.capitalize()); ax.set_title("Group deltas (|Δ|)")
    _apply_smart_yticklabels(ax, [prettify_taxon(t, rank) for t in D.index], rank, base_fs=12.5)
    plt.gcf().subplots_adjust(left=0.34)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"heatmap_group_deltas_{rank}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ----- (C) Within-site: Oral/Fecal, Crohn vs Healthy (unpaired) -----
    heatmap_two_groups(
        A_pct=oh_pct, B_pct=oc_pct, rank=rank,
        out_png=os.path.join(outdir, f"heatmap_oral_CH_unpaired_{rank}.png"),
        title="Oral_Healthy vs Oral_Crohn",
    )
    top_oral = select_top_by_delta(oc_pct, oh_pct, k=heat_topk)
    if len(top_oral) > 0:
        pairs_heatmap(
            oh_pct, oc_pct, top_oral,
            label_A="Healthy-Oral", label_B="Crohn-Oral", rank=rank,
            out_png=os.path.join(outdir, f"heatmap_oral_CH_unpaired_{rank}_top{heat_topk}.png"),
            use_log1p=use_log1p, clip_quantile=clip_quantile,
            title_suffix=f"(Top {heat_topk} by |Δ mean %|; unpaired)"
        )

    heatmap_two_groups(
        A_pct=fh_pct, B_pct=fc_pct, rank=rank,
        out_png=os.path.join(outdir, f"heatmap_fecal_CH_unpaired_{rank}.png"),
        title="Fecal_Healthy vs Fecal_Crohn",
    )
    top_fecal = select_top_by_delta(fc_pct, fh_pct, k=heat_topk)
    if len(top_fecal) > 0:
        pairs_heatmap(
            fh_pct, fc_pct, top_fecal,
            label_A="Healthy-Fecal", label_B="Crohn-Fecal", rank=rank,
            out_png=os.path.join(outdir, f"heatmap_fecal_CH_unpaired_{rank}_top{heat_topk}.png"),
            use_log1p=use_log1p, clip_quantile=clip_quantile,
            title_suffix=f"(Top {heat_topk} by |Δ mean %|; unpaired)"
        )
# ---------- Rank driver (one rank end-to-end) ----------
def process_rank(rank: str,
                 oc_raw: pd.DataFrame, oh_raw: pd.DataFrame,
                 fc_raw: pd.DataFrame, fh_raw: pd.DataFrame,
                 outdir_rank: str, topk: int,
                 matched_path: str|None,
                 heat_topk: int,
                 sample_to_id: dict|None = None,
                 disease_by_id: dict|None = None):
    """
    Run the full pipeline for one rank (genus/species).
    """
    os.makedirs(outdir_rank, exist_ok=True)
    rank_fn = extract_genus if rank == "genus" else extract_species

    # Ensure taxa × samples
    oc = ensure_taxa_by_samples(oc_raw)
    oh = ensure_taxa_by_samples(oh_raw)
    fc = ensure_taxa_by_samples(fc_raw)
    fh = ensure_taxa_by_samples(fh_raw)

    # Build % tables at the chosen rank
    oc_pct = percent_table(oc, rank_fn)
    oh_pct = percent_table(oh, rank_fn)
    fc_pct = percent_table(fc, rank_fn)
    fh_pct = percent_table(fh, rank_fn)

    # Build all heatmaps/tables for this rank
    group_contrasts_and_csv(
        oc_pct, fc_pct, oh_pct, fh_pct, rank, outdir_rank,
        n=topk, heat_topk=heat_topk, use_log1p=True, clip_quantile=0.98
    )

    # Stacked bars (you said these are good; keep as-is)
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

    # Per-sample stacked bars by group (top-K taxa per group)
    stacked_bars_by_sample(
        oc_pct, rank,
        os.path.join(outdir_rank, f"bars_by_sample_oral_crohn_{rank}_top{topk}.png"),
        topk=max(10, topk), title=f"Oral Crohn — Top {max(10, topk)} {rank.capitalize()} per sample"
    )
    stacked_bars_by_sample(
        oh_pct, rank,
        os.path.join(outdir_rank, f"bars_by_sample_oral_healthy_{rank}_top{topk}.png"),
        topk=max(10, topk), title=f"Oral Healthy — Top {max(10, topk)} {rank.capitalize()} per sample"
    )
    stacked_bars_by_sample(
        fc_pct, rank,
        os.path.join(outdir_rank, f"bars_by_sample_fecal_crohn_{rank}_top{topk}.png"),
        topk=max(10, topk), title=f"Fecal Crohn — Top {max(10, topk)} {rank.capitalize()} per sample"
    )
    stacked_bars_by_sample(
        fh_pct, rank,
        os.path.join(outdir_rank, f"bars_by_sample_fecal_healthy_{rank}_top{topk}.png"),
        topk=max(10, topk), title=f"Fecal Healthy — Top {max(10, topk)} {rank.capitalize()} per sample"
    )


    # DA — build a union of taxa across all groups to test consistently
    all_taxa = pd.Index(oc_pct.index).union(fc_pct.index).union(oh_pct.index).union(fh_pct.index)

    # Oral CH (MWU on CLR)
    run_da_oral_CH(
        mat_taxa_samples=oc_pct.reindex(all_taxa).fillna(0).join(
            oh_pct.reindex(all_taxa).fillna(0), how="outer"
        ),
        labels_A=oc_pct.columns, labels_B=oh_pct.columns, tax_labels=all_taxa,
        out_csv=os.path.join(outdir_rank, f"da_oral_CH_{rank}.csv"),
        do_volcano_png=os.path.join(outdir_rank, f"volcano_oral_CH_{rank}.png"),
        rank=rank
    )
    # Fecal CH (MWU on CLR)
    run_da_fecal_CH(
        mat_taxa_samples=fc_pct.reindex(all_taxa).fillna(0).join(
            fh_pct.reindex(all_taxa).fillna(0), how="outer"
        ),
        labels_A=fc_pct.columns, labels_B=fh_pct.columns, tax_labels=all_taxa,
        out_csv=os.path.join(outdir_rank, f"da_fecal_CH_{rank}.csv"),
        do_volcano_png=os.path.join(outdir_rank, f"volcano_fecal_CH_{rank}.png"),
        rank=rank
    )

    # Paired OC↔FC (Crohn only) — write CSV (even if empty) for consistency
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
    make_ml_outputs(rank, oc_pct, oh_pct, fc_pct, fh_pct, outdir_rank,
                    topk=topk,
                    sample_to_id=sample_to_id,
                    disease_by_id=disease_by_id)
# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Unified genus/species comparison pipeline (Crohn vs Healthy; Oral & Fecal)")
    p.add_argument("--oral-crohn",    required=True)
    p.add_argument("--oral-healthy",  required=True)
    p.add_argument("--fecal-crohn",   required=True)
    p.add_argument("--fecal-healthy", required=True)
    p.add_argument("--matched-ids",   default=None)
    p.add_argument("--outdir",        required=True)
    p.add_argument("--topk",          type=int, default=15, help="Top-N rows for compact summary heatmaps (means/deltas) and bar titles")
    p.add_argument("--heat-topk",     type=int, default=20, help="Top-K rows for compact heatmaps (pairs/within-site)")
    p.add_argument("--presence-pct",  type=float, default=0.1, help="(legacy; ignored)")
    p.add_argument("--prevalence",    type=float, default=0.10, help="(legacy; ignored)")
    p.add_argument("--meta",        required=False, default=None)
    return p.parse_args()


def _count_samples(df: pd.DataFrame) -> int:
    """Simple helper for run metadata."""
    try:
        return int(df.shape[1])
    except Exception:
        return 0

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Read input matrices
    oc_raw = pd.read_csv(args.oral_crohn,    index_col=0)
    oh_raw = pd.read_csv(args.oral_healthy,  index_col=0)
    fc_raw = pd.read_csv(args.fecal_crohn,   index_col=0)
    fh_raw = pd.read_csv(args.fecal_healthy, index_col=0)

    # Minimal run metadata (handy for reproducibility)
    run_meta = {
        "params": {
            "topk": args.topk,
            "heat_topk": args.heat_topk,
            "matched_ids": bool(args.matched_ids),
        },
        "n_samples": {
            "Oral_Crohn":    _count_samples(oc_raw),
            "Oral_Healthy":  _count_samples(oh_raw),
            "Fecal_Crohn":   _count_samples(fc_raw),
            "Fecal_Healthy": _count_samples(fh_raw),
        },
        "palette_used": PALETTE
    }
    try:
        with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
            json.dump(run_meta, f, indent=2)
    except Exception:
        pass
    # ---- optional META mapping for Sample_ID & disease ----
    sample_to_id = {}
    disease_by_id = {}

    def _norm01(x):
        s = str(x).strip().lower()
        if s in {"1","crohn","cd","case","ibd"}:   return 1
        if s in {"0","healthy","control","ctr","ctl","hc","non-ibd","nonibd","nibd"}: return 0
        return np.nan

    if args.meta and os.path.exists(args.meta):
        meta = pd.read_csv(args.meta)
        # پیدا کردن ستون شناسه در متا
        id_col = None
        for c in ["Sample_ID","Sample","sample_id","ID"]:
            if c in meta.columns:
                id_col = c; break
        if id_col is None:
            warnings.warn("[meta] no ID column found; skipping meta join.")
        else:
            # اگر هر دو ستون Sample و Sample_ID هست، آن را مپ کن
            if "Sample" in meta.columns and "Sample_ID" in meta.columns:
                sample_to_id = dict(zip(meta["Sample"].astype(str), meta["Sample_ID"].astype(str)))
            else:
                # در غیر این صورت، هویتی
                tmp_ids = meta[id_col].astype(str)
                sample_to_id = {s: s for s in tmp_ids}

            # disease به 0/1
            if "disease" in meta.columns:
                dser = meta["disease"].apply(_norm01)
                disease_by_id = {sid: (int(v) if pd.notna(v) else np.nan)
                                for sid, v in zip(meta.get("Sample_ID", meta[id_col]).astype(str), dser)}

    # GENUS
    outdir_genus = os.path.join(args.outdir, "genus")
    process_rank(
        "genus",
        oc_raw, oh_raw, fc_raw, fh_raw,
        outdir_genus,
        topk=args.topk,
        matched_path=args.matched_ids,
        heat_topk=args.heat_topk,
        sample_to_id=sample_to_id,
        disease_by_id=disease_by_id
    )
    print("[INFO] [genus] done →", outdir_genus)

    # SPECIES
    outdir_species = os.path.join(args.outdir, "species")
    process_rank(
        "species",
        oc_raw, oh_raw, fc_raw, fh_raw,
        outdir_species,
        topk=args.topk,
        matched_path=args.matched_ids,
        heat_topk=args.heat_topk,
        sample_to_id=sample_to_id,
        disease_by_id=disease_by_id
    )
    print("[INFO] [species] done →", outdir_species)

    print("[INFO] All done.")

if __name__ == "__main__":
    main()
