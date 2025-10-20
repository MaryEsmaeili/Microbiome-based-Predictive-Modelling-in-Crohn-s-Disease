#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taxa Compare — CORE (Descriptive only) + Debug & Pairing Helpers
================================================================

What this script produces (per rank = genus/species):
  • Percent matrices per group: pct_oral_crohn.csv, pct_oral_healthy.csv, pct_fecal_crohn.csv, pct_fecal_healthy.csv
  • pct_all.csv (all four groups concatenated; taxa × samples, %)
  • Heatmaps (paired Crohn, unpaired Oral CH, unpaired Fecal CH; FULL and compact Top-K variants)
  • Heatmaps (ALL shared taxa, paginated)  [Option A]
  • Slopegraphs of group means (ALL taxa, paginated)  [Option B]
  • Delta lollipop plots (ALL taxa, paginated)        [Option C]
  • Stacked bar charts (group means; per-sample stacks; y-axis fixed to [0,1])
  • Compact CSVs: group_means_{rank}.csv, group_deltas_{rank}.csv
  • run_meta.json (parameters + sample counts)
  • Rich DEBUG outputs in outdir/debug_{rank}/ to verify row selection and group coverage

Explicitly not included here:
  • Any CLR-wide matrices for ML
  • Any statistical DA tests (MWU / Wilcoxon) or volcano plots

CLI
---
python taxa_compare_core_debug.py \
  --oral-crohn OC.csv --oral-healthy OH.csv \
  --fecal-crohn FC.csv --fecal-healthy FH.csv \
  --outdir results/taxa_compare --topk 15 --heat-topk 20 \
  [--pairs-csv data/processed/matched_sample_ids.csv] \
  [--presence-threshold 0.0]

Pairs CSV schema (if provided):
  subject_id, oral_id, fecal_id

Author: ChatGPT (Maryam's assistant)
"""
from __future__ import annotations

import os
import json
import argparse
import textwrap
import warnings
from typing import Callable, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # headless rendering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator
from matplotlib.colors import Colormap

# =============================
# Global plotting configuration
# =============================
sns.set_context("talk")
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False

PALETTE = {
    "Fecal_Crohn":   "#30638e",
    "Oral_Crohn":    "#edae49",
    "Fecal_Healthy": "#d1495b",
    "Oral_Healthy":  "#00798c",
    "Other":         "#999999",
    "Crohn-Oral":    "#edae49",
    "Crohn-Fecal":   "#30638e",
    "Healthy-Oral":  "#00798c",
    "Healthy-Fecal": "#d1495b",
    "Oral":          "#edae49",
    "Fecal":         "#30638e",
}

# =====================
# Utility / label tools
# =====================

def _normalize_group_label(label: str) -> str:
    """Normalize free-text labels into keys used in PALETTE."""
    s = str(label).replace("_", "-").strip().lower()
    site = "oral" if "oral" in s else ("fecal" if ("fecal" in s or "faecal" in s or "stool" in s) else None)
    dis  = "crohn" if ("crohn" in s or "cd" in s) else ("healthy" if ("healthy" in s or "hc" in s) else None)
    if site and dis:
        return f"{dis.capitalize()}-{site.capitalize()}"
    return label

def group_color(label: str, default: str = "#999999") -> str:
    """Map a group label to a hex color using the palette."""
    key = _normalize_group_label(label)
    return PALETTE.get(key, PALETTE.get(label, default))

# ==================
# Taxonomy utilities
# ==================

def extract_genus(rowname: str) -> Optional[str]:
    """Extract 'g__...' from a pipe-delimited path; return None if absent."""
    parts = str(rowname).split("|")
    g = [p for p in parts if p.startswith("g__")]
    return g[0] if g else None

def extract_species(rowname: str) -> Optional[str]:
    """Extract 's__...' from a pipe-delimited path; return None if absent."""
    parts = str(rowname).split("|")
    s = [p for p in parts if p.startswith("s__")]
    return s[0] if s else None

# ================
# Matrix alignment
# ================

def ensure_taxa_by_samples(df_like: pd.DataFrame) -> pd.DataFrame:
    """Ensure matrix is taxa × samples (safe orientation check)."""
    idx = df_like.index.astype(str)
    cols = df_like.columns.astype(str)
    row_has_tax = (idx.str.contains("s__|g__", regex=True, na=False).mean() > 0.5)
    col_has_tax = (cols.str.contains("s__|g__", regex=True, na=False).mean() > 0.5)
    if row_has_tax and not col_has_tax:
        return df_like
    if col_has_tax and not row_has_tax:
        return df_like.T
    warnings.warn("Ambiguous matrix orientation; leaving as-is. Please verify rows=taxa, cols=samples.")
    return df_like

# ============================
# Percent & summary operations
# ============================

def percent_table(df: pd.DataFrame, grouper: Callable[[str], Optional[str]]) -> pd.DataFrame:
    """Collapse at rank via grouper(rowname), then convert to % per sample."""
    g = df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = g.div(g.sum(axis=0), axis=1) * 100.0
    return pct.fillna(0.0)

def group_mean_percent(df: pd.DataFrame, grouper: Callable[[str], Optional[str]]) -> pd.Series:
    """Per-taxon mean % abundance across samples of a group (for stacked bars)."""
    g = df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    pct = g.div(g.sum(axis=0), axis=1) * 100.0
    return pct.mean(axis=1)

# =====================
# Heatmap helper tools
# =====================
def _rows_after_all_filters(A_pct, B_pct, presence_threshold, min_mean_pct, effect_size_threshold, max_rows):
    shared = A_pct.index.intersection(B_pct.index)
    rows_prev, _, _ = apply_presence_threshold(A_pct, B_pct, shared, presence_threshold)
    if len(rows_prev) == 0:
        return pd.Index([]), pd.Series(dtype=float)

    rows_filt, abs_delta = filter_by_prevalence_and_effect(
        A_pct, B_pct, rows_prev, min_mean_pct, effect_size_threshold
    )
    if len(rows_filt) == 0:
        return pd.Index([]), abs_delta

    if max_rows is not None:
        order = abs_delta.loc[rows_filt].sort_values(ascending=False).index[:int(max_rows)]
        return pd.Index(order), abs_delta
    return pd.Index(rows_filt), abs_delta


def prettify_taxon(label: str, rank: str) -> str:
    """Human-friendly taxa labels for plotting."""
    if label is None:
        return ""
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

def _coolwarm() -> Colormap:
    """Consistent diverging cmap."""
    return plt.get_cmap("coolwarm")

def row_zscore_log1p(mat: pd.DataFrame, use_log1p: bool = True) -> pd.DataFrame:
    """Row-wise z-score (optionally on log1p of %)."""
    X = np.log1p(mat.astype(float)) if use_log1p else mat.astype(float)
    mu = X.mean(axis=1)
    sd = X.std(axis=1).replace(0, np.nan)
    Z = X.sub(mu, axis=0).div(sd, axis=0).fillna(0)
    return Z

def _smart_yticklabels(ax, labels, rank, base_fs: float = 12.0, max_chars: int = 28):
    """Wrap long taxa labels and italicize (except 'Other')."""
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

# =====================
# Debugging primitives
# =====================

def write_debug_lists(debug_dir: str,
                      label: str,
                      shared: pd.Index,
                      used_rows: pd.Index,
                      only_in_A: pd.Index,
                      only_in_B: pd.Index,
                      zero_in_A: pd.Index,
                      zero_in_B: pd.Index) -> None:
    """Write detailed CSVs explaining selection decisions."""
    os.makedirs(debug_dir, exist_ok=True)
    pd.Series(shared, name="taxon").to_csv(os.path.join(debug_dir, f"{label}_shared_raw.csv"), index=False)
    pd.Series(used_rows, name="taxon").to_csv(os.path.join(debug_dir, f"{label}_rows_used.csv"), index=False)
    pd.Series(only_in_A, name="taxon").to_csv(os.path.join(debug_dir, f"{label}_only_in_A.csv"), index=False)
    pd.Series(only_in_B, name="taxon").to_csv(os.path.join(debug_dir, f"{label}_only_in_B.csv"), index=False)
    pd.Series(zero_in_A, name="taxon").to_csv(os.path.join(debug_dir, f"{label}_shared_zero_in_A.csv"), index=False)
    pd.Series(zero_in_B, name="taxon").to_csv(os.path.join(debug_dir, f"{label}_shared_zero_in_B.csv"), index=False)

# =====================
# Plotting primitives
# =====================

def _subsample_xticks(ax, max_labels: int = 25) -> None:
    """Reduce x tick clutter when too many samples are present."""
    ticks = ax.get_xticks()
    if len(ticks) <= max_labels or max_labels <= 0:
        return
    step = max(1, int(round(len(ticks) / max_labels)))
    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_visible((i % step) == 0)

def build_col_colors(label_left: str, n_left: int, label_right: str, n_right: int):
    """Build two-block column color strip."""
    cL = group_color(label_left)
    cR = group_color(label_right)
    return [cL] * int(n_left) + [cR] * int(n_right)

def pairs_heatmap(A_pct: pd.DataFrame,
                  B_pct: pd.DataFrame,
                  rows: pd.Index,
                  label_A: str,
                  label_B: str,
                  rank: str,
                  out_png: str,
                  use_log1p: bool = True,
                  clip_quantile: float = 0.98,
                  title_suffix: str = "") -> None:
    """Two-block heatmap with fixed order (no clustering)."""
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
        col_colors=col_colors,
    )

    g.ax_heatmap.yaxis.set_ticks_position("right")
    g.ax_heatmap.yaxis.set_label_position("right")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), rotation=0, ha="left", va="center", fontsize=10)
    g.ax_heatmap.tick_params(axis="y", pad=10)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), rotation=90, ha="center", va="top", fontsize=8.5)
    _subsample_xticks(g.ax_heatmap, max_labels=25)

    title = f"{label_A} vs {label_B} — {rank.capitalize()} heatmap {title_suffix}".strip()
    g.ax_heatmap.set_title(title)
    g.ax_heatmap.set_xlabel(f"{label_A} ↔ {label_B}")
    g.ax_heatmap.set_ylabel(rank.capitalize())
    g.ax_heatmap.yaxis.labelpad = 18

    maxlen = max((len(s) for s in Z.index), default=12)
    right = min(0.995, 0.88 + 0.016 * max(0, maxlen - 12))
    g.fig.subplots_adjust(left=0.07, right=right, bottom=0.20, top=0.92)

    g.fig.savefig(out_png, dpi=380, bbox_inches="tight", pad_inches=0.70)
    plt.close(g.fig)

# ==============================
# Bars & compact table rendering
# ==============================

def build_taxa_color_cycle(n_needed: int = 20):
    """Generate a gentle color cycle for stacked bars."""
    bases = [
        PALETTE.get("Crohn-Oral", "#edae49"),
        PALETTE.get("Crohn-Fecal", "#30638e"),
        PALETTE.get("Healthy-Oral", "#00798c"),
        PALETTE.get("Healthy-Fecal", "#d1495b"),
    ]
    def _hex_to_rgb01(h):
        h = h.lstrip("#"); return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    def _mix(c1, c2, t):
        return tuple((1 - t) * a + t * b for a, b in zip(c1, c2))
    bases = [_hex_to_rgb01(b) for b in bases]
    white = _hex_to_rgb01("#FFFFFF")
    cycle = []
    for b in bases:
        cycle.append(_mix(white, b, 0.35))
        cycle.append(b)
        cycle.append(tuple(max(x - 0.20, 0) for x in b))
    i = 0
    while len(cycle) < n_needed:
        b = bases[i % len(bases)]; t = 0.15 + 0.15 * ((i // len(bases)) % 3)
        cycle.append(_mix(white, b, t)); i += 1
    def rgb01_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    return [rgb01_to_hex(c) for c in cycle[:n_needed]]

def two_bar_stacked_with_table(a_mean: pd.Series,
                               b_mean: pd.Series,
                               rank: str,
                               out_png: str,
                               label_a: str,
                               label_b: str,
                               top_k: int = 15) -> None:
    """Two stacked bars (group means) with a side legend table (proportions 0–1)."""
    combined = a_mean.add(b_mean, fill_value=0).sort_values(ascending=False)
    top = combined.index[:min(top_k, len(combined))]
    s_a_pct = a_mean.reindex(top).fillna(0)
    s_b_pct = b_mean.reindex(top).fillna(0)
    s_a_pct = pd.concat([s_a_pct, pd.Series({"Other": a_mean.drop(top, errors='ignore').sum()})])
    s_b_pct = pd.concat([s_b_pct, pd.Series({"Other": b_mean.drop(top, errors='ignore').sum()})])

    # Convert % to proportions in [0,1] for plotting
    s_a = (s_a_pct / 100.0).clip(lower=0)
    s_b = (s_b_pct / 100.0).clip(lower=0)

    order = s_a.sort_values(ascending=False).index
    taxa_colors = build_taxa_color_cycle(max(3 * len(order), 12))
    color_map = {tax: taxa_colors[i % len(taxa_colors)] for i, tax in enumerate(order)}
    color_map["Other"] = PALETTE.get("Other", "#999999")

    edge_a = PALETTE.get("Oral", PALETTE.get("Crohn-Oral", "#edae49")) if "Oral" in label_a else PALETTE.get("Fecal", PALETTE.get("Crohn-Fecal", "#30638e"))
    edge_b = PALETTE.get("Oral", PALETTE.get("Healthy-Oral", "#00798c")) if "Oral" in label_b else PALETTE.get("Fecal", PALETTE.get("Healthy-Fecal", "#d1495b"))

    fig = plt.figure(figsize=(16.5, 9.0))
    gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[2.4, 1.6], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1]); ax_tbl.axis("off")

    x = np.array([0, 1]); b0 = b1 = 0.0
    rows_for_table = []
    for tax in order:
        h0 = float(s_a.loc[tax]); h1 = float(s_b.loc[tax])
        ax.bar(x[0], h0, bottom=b0, color=color_map[tax], width=0.6, edgecolor=edge_a, linewidth=0.7)
        ax.bar(x[1], h1, bottom=b1, color=color_map[tax], width=0.6, edgecolor=edge_b, linewidth=0.7)
        if h0 >= 0.03: ax.text(x[0], b0 + h0/2, f"{h0:.2f}", ha="center", va="center", fontsize=9)
        if h1 >= 0.03: ax.text(x[1], b1 + h1/2, f"{h1:.2f}", ha="center", va="center", fontsize=9)
        rows_for_table.append((tax, color_map[tax], h0, h1)); b0 += h0; b1 += h1

    ax.set_xticks(x); ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel("Mean relative abundance (proportion)")
    ax.set_ylim(0, 1.0)  # keep 0–1 on y-axis
    ax.set_title(f"{label_a} vs {label_b} — Top {min(top_k,len(order))} {rank.capitalize()} (others → Other)")

    rows_sorted = sorted(rows_for_table, key=lambda r: (0 if r[0] == "Other" else 1, -(r[2] + r[3])))
    y0 = 0.975; dy = 0.042
    ax_tbl.text(0.10, y0, f"{label_a}", fontweight="bold", ha="right", transform=ax_tbl.transAxes)
    ax_tbl.text(0.22, y0, f"{label_b}", fontweight="bold", ha="right", transform=ax_tbl.transAxes)
    ax_tbl.text(0.29, y0, "Color",       fontweight="bold", transform=ax_tbl.transAxes)
    ax_tbl.text(0.39, y0, "Taxon",       fontweight="bold", transform=ax_tbl.transAxes)

    y = y0 - dy
    for tax, col, vA, vB in rows_sorted:
        ax_tbl.text(0.10, y, f"{vA:.2f}", ha="right", va="center", transform=ax_tbl.transAxes, fontsize=10)
        ax_tbl.text(0.22, y, f"{vB:.2f}", ha="right", va="center", transform=ax_tbl.transAxes, fontsize=10)
        ax_tbl.add_patch(plt.Rectangle((0.29, y - 0.015), 0.06, 0.02, transform=ax_tbl.transAxes, color=col, clip_on=False))
        name = prettify_taxon(tax, rank) if tax != "Other" else "Other"
        ax_tbl.text(0.39, y, name, transform=ax_tbl.transAxes, va="center")
        y -= dy
        if y < 0.02: break

    plt.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

# =====================
# Option A — ALL shared heatmaps (paginated)
# =====================

def heatmap_all_shared_paginated(A_pct, B_pct, rank, out_prefix, label_A, label_B,
                                 page_size=50, use_log1p=True, clip_quantile=0.98):
    """Paginated heatmaps showing ALL shared taxa between two groups."""
    shared = A_pct.index.intersection(B_pct.index)
    rows = shared[(A_pct.loc[shared].sum(axis=1) > 0) | (B_pct.loc[shared].sum(axis=1) > 0)]
    if len(rows) == 0:
        return
    # Stable order by total abundance
    order = A_pct.loc[rows].add(B_pct.loc[rows], fill_value=0.0).sum(axis=1).sort_values(ascending=False).index
    rows = list(order)

    # Fix the color scale across pages
    all_data = pd.concat([A_pct.loc[rows], B_pct.loc[rows]], axis=1).fillna(0.0)
    Z_all   = row_zscore_log1p(all_data, use_log1p=use_log1p)
    V = np.nanquantile(np.abs(Z_all.values), clip_quantile)
    if not np.isfinite(V) or V == 0:
        V = max(1.0, float(np.nanmax(np.abs(Z_all.values)) or 1.0))

    # Render pages
    for i in range(0, len(rows), page_size):
        block = rows[i:i+page_size]
        out_png = f"{out_prefix}_p{i//page_size+1:02d}.png"
        pairs_heatmap(A_pct, B_pct, pd.Index(block), rank=rank,
                      label_A=label_A, label_B=label_B, out_png=out_png,
                      use_log1p=use_log1p, clip_quantile=clip_quantile,
                      title_suffix=f"(ALL shared; page {i//page_size+1})")

# =====================
# Option B — Slopegraph of means (ALL taxa, paginated)
# =====================

def slopegraph_means(A_pct, B_pct, rank, out_prefix, label_A, label_B, page_size=60):
    """Slopegraph: two points per taxon (means in A and B) + connecting line."""
    idx = A_pct.index.union(B_pct.index)
    mA = A_pct.reindex(idx).fillna(0.0).mean(axis=1)
    mB = B_pct.reindex(idx).fillna(0.0).mean(axis=1)
    both = pd.DataFrame({label_A: mA, label_B: mB})
    both = both.loc[(both.sum(axis=1) > 0.0)]
    both["abs_delta"] = (both[label_A] - both[label_B]).abs()
    both = both.sort_values("abs_delta", ascending=False)
    taxa = both.index.tolist()

    for i in range(0, len(taxa), page_size):
        block = taxa[i:i+page_size]
        dfb = both.loc[block]
        fig_h = max(6.5, 0.30 * len(dfb))
        fig = plt.figure(figsize=(10, fig_h))
        ax = fig.add_subplot(111)

        y = np.arange(len(dfb))[::-1]
        ax.hlines(y, dfb[label_B].values, dfb[label_A].values, linewidth=1.0, alpha=0.7)
        ax.plot(dfb[label_A].values, y, 'o', label=label_A, markersize=3)
        ax.plot(dfb[label_B].values, y, 'o', label=label_B, markersize=3)

        ax.set_xlabel("Mean relative abundance (%)")
        ax.set_yticks(y)
        ax.set_yticklabels([prettify_taxon(t, rank) for t in dfb.index], fontsize=9)
        ax.invert_yaxis()
        ax.legend(loc="lower right")
        ax.set_title(f"{label_A} vs {label_B} — {rank.capitalize()} means (ALL taxa; page {i//page_size+1})")

        plt.tight_layout()
        fig.savefig(f"{out_prefix}_p{i//page_size+1:02d}.png", dpi=320, bbox_inches="tight")
        plt.close(fig)

# =====================
# Option C — Delta lollipop (ALL taxa, paginated)
# =====================
def lollipop_filtered(
    A_pct, B_pct, rank, out_png, label_delta,
    presence_threshold: float = 0.0,
    min_mean_pct: float = 0.0,
    effect_size_threshold: float = 0.0,
    max_rows: Optional[int] = None
):
    # دقیقاً همان منطق فیلترها مثل heatmap_two_groups_compact
    rows, _abs_delta = _rows_after_all_filters(
        A_pct, B_pct, presence_threshold, min_mean_pct, effect_size_threshold, max_rows
    )
    if len(rows) == 0:
        fig = plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No taxa after filtering", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    d = A_pct.loc[rows].mean(axis=1) - B_pct.loc[rows].mean(axis=1)  # Δ mean % (A-B)
    df = pd.DataFrame({"delta": d, "abs_delta": d.abs()}).sort_values("abs_delta", ascending=False)

    y = np.arange(len(df))[::-1]
    fig_h = max(7.0, 0.28 * len(df))
    fig = plt.figure(figsize=(10.0, fig_h))
    ax = fig.add_subplot(111)
    ax.hlines(y, 0, df["delta"].values, linewidth=1.0, color="#888888", alpha=0.7)
    ax.plot(df["delta"].values, y, "o", markersize=3, linestyle="None", color="#222222")
    ax.axvline(0, linewidth=1.0, color="#444444")
    ax.set_xlabel("Δ mean % (A − B)")
    ax.set_yticks(y)
    ax.set_yticklabels([prettify_taxon(t, rank) for t in df.index], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"{label_delta} — {rank.capitalize()} (filtered)")
    plt.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

def delta_lollipop(A_pct, B_pct, rank, out_prefix, label_delta, page_size=60):
    """Delta lollipop: Δ = mean(A) − mean(B), signed, sorted by |Δ|."""
    idx = A_pct.index.union(B_pct.index)
    d = A_pct.reindex(idx).fillna(0.0).mean(axis=1) - B_pct.reindex(idx).fillna(0.0).mean(axis=1)
    df = pd.DataFrame({"delta": d, "abs_delta": d.abs()}).sort_values("abs_delta", ascending=False)
    taxa = df.index.tolist()

    for i in range(0, len(taxa), page_size):
        block = taxa[i:i+page_size]
        dsub = df.loc[block]
        y = np.arange(len(dsub))[::-1]
        fig_h = max(6.5, 0.30 * len(dsub))
        fig = plt.figure(figsize=(9.5, fig_h))
        ax = fig.add_subplot(111)

        # Stems and markers
        ax.hlines(y, 0, dsub["delta"].values, linewidth=1.0, color="#888888", alpha=0.7)
        colors = np.where(dsub["delta"].values >= 0, "#377eb8", "#e41a1c")
        ax.plot(dsub["delta"].values, y, "o", markersize=3, linestyle="None", color="#222222")
        for yi, xi, ci in zip(y, dsub["delta"].values, colors):
            ax.plot([0, xi], [yi, yi], "-", color=ci, linewidth=2.0, alpha=0.8)

        ax.axvline(0, linewidth=1.0, color="#444444")
        ax.set_xlabel("Δ mean % (A − B)")
        ax.set_yticks(y)
        ax.set_yticklabels([prettify_taxon(t, rank) for t in dsub.index], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"{label_delta} — {rank.capitalize()} (ALL taxa; page {i//page_size+1})")

        plt.tight_layout()
        fig.savefig(f"{out_prefix}_p{i//page_size+1:02d}.png", dpi=320, bbox_inches="tight")
        plt.close(fig)

# =====================
# Contrast constructors
# =====================

def select_top_by_delta(A_pct: pd.DataFrame, B_pct: pd.DataFrame, k: int = 20) -> pd.Index:
    """Pick Top-K taxa by absolute difference in mean % between two groups."""
    shared = A_pct.index.intersection(B_pct.index)
    if len(shared) == 0:
        return pd.Index([])
    A = A_pct.loc[shared].fillna(0.0)
    B = B_pct.loc[shared].fillna(0.0)
    da = (A.mean(axis=1) - B.mean(axis=1)).abs()
    return da.sort_values(ascending=False).head(int(k)).index

def heatmap_two_groups(A_pct: pd.DataFrame,
                       B_pct: pd.DataFrame,
                       rank: str,
                       out_png: str,
                       title: str) -> None:
    """FULL heatmap with all shared taxa that are non-zero on both sides."""
    shared = A_pct.index.intersection(B_pct.index)
    present_A = (A_pct.loc[shared].sum(axis=1) > 0)
    present_B = (B_pct.loc[shared].sum(axis=1) > 0)
    rows = shared[(present_A & present_B).values]
    union = A_pct.loc[rows].add(B_pct.loc[rows], fill_value=0.0)
    rows = union.sum(axis=1).sort_values(ascending=False).index
    label_A, label_B = title.split(" vs ")
    pairs_heatmap(A_pct, B_pct, rows, label_A=label_A, label_B=label_B, rank=rank, out_png=out_png, title_suffix="(unpaired FULL)")
def process_rank(
    rank: str,
    oc_raw: pd.DataFrame,
    oh_raw: pd.DataFrame,
    fc_raw: pd.DataFrame,
    fh_raw: pd.DataFrame,
    outdir_rank: str,
    topk: int,
    heat_topk: int,
    presence_threshold: float,
    pairs: Optional[pd.DataFrame],
    # NEW: pass all thresholds explicitly; never use `args` inside this function
    min_mean_pct: float,
    effect_size_threshold: float,
    max_rows: Optional[int],
    emit_legacy_pages: bool,
) -> None:
    """Full pipeline for a given rank (no reliance on global `args`)."""

    os.makedirs(outdir_rank, exist_ok=True)
    rank_fn = extract_genus if rank == "genus" else extract_species

    oc, oh, fc, fh = oc_raw, oh_raw, fc_raw, fh_raw 

    # Sanity debug: how many rows contain rank prefixes
    for tag in ["g__", "s__"]:
        def _idx_count_contains(idx, pat):
            s = pd.Series(idx.astype(str), copy=False)
            return int(s.str.contains(pat, regex=True, na=False).sum())
        print(f"[DEBUG:{rank}] rows containing {tag} — OC:{_idx_count_contains(oc.index, tag)}, "
              f"OH:{_idx_count_contains(oh.index, tag)}, FC:{_idx_count_contains(fc.index, tag)}, "
              f"FH:{_idx_count_contains(fh.index, tag)}")

    # Build % tables
    oc_pct = percent_table(oc, rank_fn)
    oh_pct = percent_table(oh, rank_fn)
    fc_pct = percent_table(fc, rank_fn)
    fh_pct = percent_table(fh, rank_fn)

    # Save % matrices
    oc_pct.to_csv(os.path.join(outdir_rank, f"pct_oral_crohn.csv"))
    oh_pct.to_csv(os.path.join(outdir_rank, f"pct_oral_healthy.csv"))
    fc_pct.to_csv(os.path.join(outdir_rank, f"pct_fecal_crohn.csv"))
    fh_pct.to_csv(os.path.join(outdir_rank, f"pct_fecal_healthy.csv"))

    # Concatenate all groups
    all_taxa = oc_pct.index.union(oh_pct.index).union(fc_pct.index).union(fh_pct.index)
    pct_all_sites = pd.concat([
        oc_pct.reindex(all_taxa).fillna(0),
        oh_pct.reindex(all_taxa).fillna(0),
        fc_pct.reindex(all_taxa).fillna(0),
        fh_pct.reindex(all_taxa).fillna(0),
    ], axis=1)
    pct_all_sites.to_csv(os.path.join(outdir_rank, "pct_all.csv"))

    # Figures & summaries
    group_contrasts_and_figs(
        oc_pct, fc_pct, oh_pct, fh_pct, rank, outdir_rank,
        n=topk, heat_topk=heat_topk, use_log1p=True, clip_quantile=0.98,
        presence_threshold=presence_threshold, pairs=pairs,
        # use function parameters, not `args`
        min_mean_pct=min_mean_pct,
        effect_size_threshold=effect_size_threshold,
        max_rows=max_rows,
        emit_legacy_pages=emit_legacy_pages,
    )

    # ===========================
    # Per-sample stacked barplots (y in [0,1])
    # ===========================
    def stacked_bars_by_sample(
        pct: pd.DataFrame,
        rank: str,
        out_png: str,
        topk: int = 10,
        title: str = ""
    ):
        """
        Draw per-sample stacked bars.
        - Input 'pct' is in percentage points [0,100]; we convert to proportions [0,1] for plotting.
        - Y-axis is fixed to [0,1] so panels are directly comparable.
        """
        if pct.empty:
            fig = plt.figure(figsize=(8, 3))
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
            plt.axis("off")
            fig.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return

        totals = pct.sum(axis=1).sort_values(ascending=False)
        top = totals.head(int(max(1, topk))).index
        top_mat = pct.loc[top].copy()

        if len(totals) > len(top):
            other = pct.drop(index=top, errors="ignore").sum(axis=0)
            top_mat.loc["Other"] = other.values

        sample_order = top_mat.iloc[0].sort_values(ascending=False).index
        top_mat = top_mat.reindex(columns=sample_order)

        taxa = list(top_mat.index)
        color_cycle = build_taxa_color_cycle(max(3 * len(taxa), 12))
        color_map = {t: color_cycle[i % len(color_cycle)] for i, t in enumerate(taxa)}
        color_map["Other"] = PALETTE.get("Other", "#999999")

        prop_mat = (top_mat / 100.0).clip(lower=0)

        fig = plt.figure(figsize=(max(12.0, 0.23 * prop_mat.shape[1] + 6.0), 6.0))
        ax = fig.add_subplot(111)

        bottoms = np.zeros(prop_mat.shape[1], dtype=float)
        x = np.arange(prop_mat.shape[1])
        for t in taxa:
            vals = prop_mat.loc[t].values
            ax.bar(x, vals, bottom=bottoms, color=color_map[t], width=0.85, linewidth=0)
            bottoms += vals

        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Relative abundance (proportion)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in prop_mat.columns], rotation=90, fontsize=8)
        ax.set_title(title if title else f"Top {min(topk, len(taxa))} {rank.capitalize()} per sample")

        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[t]) for t in taxa]
        labels = [prettify_taxon(t, rank) if t != "Other" else "Other" for t in taxa]
        ax.legend(handles, labels, ncol=2, bbox_to_anchor=(1.02, 1.02),
                  loc="upper left", frameon=False, title=rank.capitalize())

        plt.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Render per-sample bars for each group (PROPORTIONS 0–1)
    stacked_bars_by_sample(oc_pct, rank, os.path.join(outdir_rank, f"bars_by_sample_oral_crohn_{rank}_top{topk}.png"),
                           topk=max(10, topk), title=f"Oral Crohn — Top {max(10, topk)} {rank.capitalize()} per sample")
    stacked_bars_by_sample(oh_pct, rank, os.path.join(outdir_rank, f"bars_by_sample_oral_healthy_{rank}_top{topk}.png"),
                           topk=max(10, topk), title=f"Oral Healthy — Top {max(10, topk)} {rank.capitalize()} per sample")
    stacked_bars_by_sample(fc_pct, rank, os.path.join(outdir_rank, f"bars_by_sample_fecal_crohn_{rank}_top{topk}.png"),
                           topk=max(10, topk), title=f"Fecal Crohn — Top {max(10, topk)} {rank.capitalize()} per sample")
    stacked_bars_by_sample(fh_pct, rank, os.path.join(outdir_rank, f"bars_by_sample_fecal_healthy_{rank}_top{topk}.png"),
                           topk=max(10, topk), title=f"Fecal Healthy — Top {max(10, topk)} {rank.capitalize()} per sample")

def heatmap_two_groups_compact(A_pct: pd.DataFrame,
                               B_pct: pd.DataFrame,
                               rank: str,
                               out_png: str,
                               title: str,
                               presence_threshold: float = 0.0,
                               min_mean_pct: float = 0.0,
                               effect_size_threshold: float = 0.0,
                               max_rows: Optional[int] = None) -> None:
    """
    Compact heatmap with prevalence + (optional) mean and effect-size filters.
    If max_rows is provided, order by |Δ mean %| and keep the top max_rows.
    Otherwise, show ALL taxa that pass the filters (single page).
    """
    label_A, label_B = title.split(" vs ")
    shared = A_pct.index.intersection(B_pct.index)

    rows_prev, present_A, present_B = apply_presence_threshold(A_pct, B_pct, shared, presence_threshold)
    if len(rows_prev) == 0:
        fig = plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No taxa after prevalence filtering", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig); return

    rows_filt, abs_delta = filter_by_prevalence_and_effect(A_pct, B_pct, rows_prev,
                                                           min_mean_pct, effect_size_threshold)
    if len(rows_filt) == 0:
        fig = plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No taxa after mean/effect-size filtering", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig); return

    # optional cap
    rows = pd.Index(rows_filt)
    if max_rows is not None:
        order = abs_delta.loc[rows].sort_values(ascending=False).index
        rows = pd.Index(order[:int(max_rows)])

    pairs_heatmap(A_pct, B_pct, rows, label_A=label_A, label_B=label_B, rank=rank,
                  out_png=out_png,
                  title_suffix=f"(presence≥{presence_threshold:.0%}"
                               f"{', mean≥'+str(min_mean_pct)+'%' if min_mean_pct>0 else ''}"
                               f"{', |Δ|≥'+str(effect_size_threshold)+'pp' if effect_size_threshold>0 else ''})")

# =====================
# Main per-rank driver
# =====================
def filter_by_prevalence_and_effect(A_pct, B_pct, rows, min_mean_pct: float, effect_size_threshold: float):
    """
    Apply optional mean-abundance and effect-size filters on top of prevalence selection.
    - min_mean_pct: drop taxa with overall mean < threshold (percentage points)
    - effect_size_threshold: keep taxa with |Δ mean %| >= threshold
    Returns filtered Index and a Series of abs-delta (for ordering if needed).
    """
    if len(rows) == 0:
        return rows, pd.Series(dtype=float)

    A = A_pct.loc[rows].fillna(0.0)
    B = B_pct.loc[rows].fillna(0.0)

    mean_overall = (A.add(B, fill_value=0.0).mean(axis=1))  # already in %
    keep = pd.Series(True, index=rows)

    if (min_mean_pct or 0) > 0:
        keep &= (mean_overall >= float(min_mean_pct))

    abs_delta = (A.mean(axis=1) - B.mean(axis=1)).abs()
    if (effect_size_threshold or 0) > 0:
        keep &= (abs_delta >= float(effect_size_threshold))

    rows2 = rows[keep.loc[rows].values]
    return rows2, abs_delta

def apply_presence_threshold(A_pct: pd.DataFrame,
                             B_pct: pd.DataFrame,
                             shared: pd.Index,
                             presence_threshold: float) -> Tuple[pd.Index, pd.Series, pd.Series]:
    """Filter shared taxa by requiring non-zero in ≥ threshold fraction of samples in each side."""
    A = A_pct.loc[shared]
    B = B_pct.loc[shared]
    if presence_threshold <= 0:
        present_A = (A.sum(axis=1) > 0)
        present_B = (B.sum(axis=1) > 0)
    else:
        thr_A = max(1, int(np.ceil(presence_threshold * A.shape[1])))
        thr_B = max(1, int(np.ceil(presence_threshold * B.shape[1])))
        present_A = ((A > 0).sum(axis=1) >= thr_A)
        present_B = ((B > 0).sum(axis=1) >= thr_B)
    rows = shared[(present_A & present_B).values]
    return rows, present_A, present_B

def group_contrasts_and_figs(
    oc_pct: pd.DataFrame,   # Crohn Oral   (taxa x samples, in %)
    fc_pct: pd.DataFrame,   # Crohn Fecal  (taxa x samples, in %)
    oh_pct: pd.DataFrame,   # Healthy Oral (taxa x samples, in %)
    fh_pct: pd.DataFrame,   # Healthy Fecal(taxa x samples, in %)
    rank: str,              # "genus" | "species" (labels/filenames only)
    outdir: str,            # output directory for this rank
    n: int = 15,            # kept only for compatibility (no longer drives removed heatmaps)
    heat_topk: int = 20,    # kept only for compatibility (we removed the _topK pairs fig)
    use_log1p: bool = True,
    clip_quantile: float = 0.98,
    presence_threshold: float = 0.0,
    pairs: Optional[pd.DataFrame] = None,
    min_mean_pct: float = 0.0,          # abundance filter (percentage points)
    effect_size_threshold: float = 0.0, # |Δ mean %| threshold (percentage points)
    max_rows: Optional[int] = None,     # cap rows AFTER filtering (None = no cap)
    emit_legacy_pages: bool = False     # legacy multi-page figures (kept off by default)
) -> None:
    """
    Build figures/tables for a given rank with the NEW policy:
      - KEEP   paired Crohn Oral vs Fecal full heatmap (descriptive)
      - KEEP   filtered unpaired CH heatmaps (Oral CH and Fecal CH) that respect prevalence/abundance filters
      - ADD    lollipop plots (ALL common taxa; single page): ORAL Δ(OC−OH), FECAL Δ(FC−FH)
      - DROP   global/split group-mean & group-delta heatmaps, and paired '_topK' heatmap

    Inputs are taxa x samples (%) matrices.

    Notes on filtering:
      * prevalence filter = presence_threshold (fraction) applied to BOTH sides
      * min_mean_pct      drops very-low-abundance taxa (overall mean %)
      * effect_size_threshold keeps taxa with |Δ mean %| >= threshold
      * max_rows applies after filters (useful to cap figure height)
    """
    os.makedirs(outdir, exist_ok=True)
    debug_dir = os.path.join(outdir, f"debug_{rank}")
    os.makedirs(debug_dir, exist_ok=True)

    # ---------------------------
    # (A) Paired Crohn Oral vs Fecal
    # ---------------------------
    A_cf = oc_pct
    B_cf = fc_pct

    # If a pairs table is provided, hard-align columns to true subject pairs.
    # Expected columns: 'oral_id', 'fecal_id' (robust renaming is done upstream).
    if pairs is not None and {"oral_id", "fecal_id"}.issubset(set(pairs.columns)):
        oc_sub = A_cf.reindex(columns=pairs["oral_id"].dropna().astype(str), fill_value=0)
        fc_sub = B_cf.reindex(columns=pairs["fecal_id"].dropna().astype(str), fill_value=0)
        # Keep only columns where both sides exist
        valid_mask = (~oc_sub.columns.isna()) & (~fc_sub.columns.isna())
        A_cf, B_cf = oc_sub.loc[:, valid_mask], fc_sub.loc[:, valid_mask]

    # Prevalence filter (for a descriptive "FULL" order by total abundance)
    shared_cf = A_cf.index.intersection(B_cf.index)
    rows_cf_all, present_A, present_B = apply_presence_threshold(A_cf, B_cf, shared_cf, presence_threshold)

    # Debug CSVs: what was kept/dropped in the paired selection
    only_in_A = A_cf.index.difference(B_cf.index)
    only_in_B = B_cf.index.difference(A_cf.index)
    zero_in_A = shared_cf[~present_A]
    zero_in_B = shared_cf[~present_B]
    write_debug_lists(
        debug_dir, "crohn_pairs",
        shared_cf, rows_cf_all, only_in_A, only_in_B, zero_in_A, zero_in_B
    )

    # Order by total abundance across both sides
    order_cf_all = (
        A_cf.loc[rows_cf_all].add(B_cf.loc[rows_cf_all], fill_value=0.0)
        .sum(axis=1).sort_values(ascending=False).index
    )

    # Save the paired heatmap (kept)
    pairs_heatmap(
        A_cf, B_cf, order_cf_all,
        label_A="Crohn-Oral", label_B="Crohn-Fecal", rank=rank,
        out_png=os.path.join(outdir, f"heatmap_pairs_crohn_{rank}.png"),
        use_log1p=use_log1p, clip_quantile=clip_quantile,
        title_suffix="(paired subjects; descriptive)"
    )

    # ---------------------------
    # (B) Unpaired CH heatmaps (filtered)
    # ---------------------------
    # These compact heatmaps already apply:
    #   prevalence (presence_threshold), min_mean_pct (abundance), effect_size_threshold, and optional max_rows.
    # ORAL: Healthy vs Crohn
    heatmap_two_groups_compact(
        oh_pct, oc_pct, rank,
        os.path.join(outdir, f"heatmap_oral_CH_unpaired_{rank}.png"),
        title="Healthy-Oral vs Crohn-Oral",
        presence_threshold=presence_threshold,
        min_mean_pct=min_mean_pct,
        effect_size_threshold=effect_size_threshold,
        max_rows=max_rows
    )
    # FECAL: Healthy vs Crohn
    heatmap_two_groups_compact(
        fh_pct, fc_pct, rank,
        os.path.join(outdir, f"heatmap_fecal_CH_unpaired_{rank}.png"),
        title="Healthy-Fecal vs Crohn-Fecal",
        presence_threshold=presence_threshold,
        min_mean_pct=min_mean_pct,
        effect_size_threshold=effect_size_threshold,
        max_rows=max_rows
    )

    # --- Group-mean stacked bars (Crohn vs Healthy) ---
    two_bar_stacked_with_table(
        oc_pct.mean(axis=1),  # میانگین % در نمونه‌های Crohn-Oral
        oh_pct.mean(axis=1),  # میانگین % در نمونه‌های Healthy-Oral
        rank,
        os.path.join(outdir, f"bars_groupmeans_oral_CH_{rank}_top{n}.png"),
        label_a="Crohn-Oral", label_b="Healthy-Oral",
        top_k=n
    )

    two_bar_stacked_with_table(
        fc_pct.mean(axis=1),  # Crohn-Fecal
        fh_pct.mean(axis=1),  # Healthy-Fecal
        rank,
        os.path.join(outdir, f"bars_groupmeans_fecal_CH_{rank}_top{n}.png"),
        label_a="Crohn-Fecal", label_b="Healthy-Fecal",
        top_k=n
    )

    # ---------------------------
    # (C) Lollipop plots — ALL common taxa (single page, no Top-K)
    # ---------------------------
    # ORAL Δ (Crohn − Healthy)
    lollipop_filtered(
        oc_pct, oh_pct, rank,
        os.path.join(outdir, f"lollipop_ORAL_OC_minus_OH_{rank}_ALL.png"),
        label_delta="ORAL Δ (Crohn − Healthy)",
        presence_threshold=presence_threshold,
        min_mean_pct=min_mean_pct,
        effect_size_threshold=effect_size_threshold,
        max_rows=max_rows
    )

    # FECAL Δ (Crohn − Healthy)
    lollipop_filtered(
        fc_pct, fh_pct, rank,
        os.path.join(outdir, f"lollipop_FECAL_FC_minus_FH_{rank}_ALL.png"),
        label_delta="FECAL Δ (Crohn − Healthy)",
        presence_threshold=presence_threshold,
        min_mean_pct=min_mean_pct,
        effect_size_threshold=effect_size_threshold,
        max_rows=max_rows
    )
    # ---------------------------
    # (X) Write compact tables: group_means_{rank}.csv, group_deltas_{rank}.csv
    # ---------------------------
    # Build means over all four groups (per taxon, %); keep rows with any signal.
    crohn_union_index = oc_pct.index.union(fc_pct.index).union(oh_pct.index).union(fh_pct.index)
    means_df = pd.DataFrame({
        "O.C.": oc_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
        "F.C.": fc_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
        "O.H.": oh_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
        "F.H.": fh_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
    }, index=crohn_union_index)
    means_df = means_df.loc[(means_df.sum(axis=1) > 0.0)]
    means_df.index.name = rank
    means_df.to_csv(os.path.join(outdir, f"group_means_{rank}.csv"))

    # Deltas: keep natural site-wise contrasts if both available; else fallback to O.C.–F.C.
    deltas_cols = {}
    if {"O.C.", "O.H."}.issubset(means_df.columns):
        deltas_cols["O.C.–O.H."] = means_df["O.C."] - means_df["O.H."]
    if {"F.C.", "F.H."}.issubset(means_df.columns):
        deltas_cols["F.C.–F.H."] = means_df["F.C."] - means_df["F.H."]
    if not deltas_cols and {"O.C.", "F.C."}.issubset(means_df.columns):
        deltas_cols["O.C.–F.C."] = means_df["O.C."] - means_df["F.C."]

    deltas_df = pd.DataFrame(deltas_cols, index=means_df.index) if deltas_cols else pd.DataFrame(index=means_df.index)
    deltas_df.index.name = rank
    deltas_df.to_csv(os.path.join(outdir, f"group_deltas_{rank}.csv"))

# =============
# CLI & driver
# =============

def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Taxa Compare CORE (descriptive only; genus/species) with debugging and pairing helpers")
    p.add_argument("--oral-crohn",    required=True)
    p.add_argument("--oral-healthy",  required=True)
    p.add_argument("--fecal-crohn",   required=True)
    p.add_argument("--fecal-healthy", required=True)
    p.add_argument("--outdir",        required=True)
    p.add_argument("--topk",          type=int, default=15, help="Top-N rows for compact summary heatmaps and bar titles")
    p.add_argument("--heat-topk",     type=int, default=20, help="Top-K rows for compact heatmaps (pairs/within-site)")
    p.add_argument("--pairs-csv",     default=None, help="Optional CSV with columns: subject_id, oral_id, fecal_id, to enforce true subject pairing")
    p.add_argument("--presence-threshold", type=float, default=0.0, help="Fraction in [0,1]; require non-zero in at least this fraction of samples in EACH group for inclusion in 'FULL' heatmaps. 0.0 = any nonzero")
    p.add_argument("--min-mean-pct", type=float, default=0.0,
                help="Drop taxa with overall mean (percentage points) below this value; 0 disables.")
    p.add_argument("--effect-size-threshold", type=float, default=0.0,
                help="Keep taxa with |Δ mean %| >= this value; 0 disables.")
    p.add_argument("--max-rows", default=None,
                help="Maximum number of taxa to plot after filtering; 'null' or None means no cap.")
    p.add_argument("--emit-legacy-pages", action="store_true",
               help="Also emit old multi-page figures (paginated ALL/shared, slopegraphs, lollipops).")

    return p.parse_args()

def _count_samples(df: pd.DataFrame) -> int:
    """Utility to count columns safely."""
    try:
        return int(df.shape[1])
    except Exception:
        return 0
    
def _parse_max_rows(x):
    """Robust parse for --max-rows: accepts None/'null'/'none'/''/0 as no cap."""
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"", "none", "null"}:
        return None
    try:
        v = int(x)
        return None if v <= 0 else v
    except Exception:
        return None

def main():
    """Entry point."""
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    args.max_rows = _parse_max_rows(args.max_rows)

    # Read input matrices (rows=features, cols=samples; ensure orientation)
    oc_raw = pd.read_csv(args.oral_crohn,    index_col=0)
    oh_raw = pd.read_csv(args.oral_healthy,  index_col=0)
    fc_raw = pd.read_csv(args.fecal_crohn,   index_col=0)
    fh_raw = pd.read_csv(args.fecal_healthy, index_col=0)

    # Ensure matrices are taxa × samples once here; pass these onwards
    oc = ensure_taxa_by_samples(oc_raw)   # <-- added
    oh = ensure_taxa_by_samples(oh_raw)   # <-- added
    fc = ensure_taxa_by_samples(fc_raw)   # <-- added
    fh = ensure_taxa_by_samples(fh_raw)   # <-- added


    # Optional pairs CSV
    pairs = None
    if args.pairs_csv is not None and os.path.exists(args.pairs_csv):
        pairs = pd.read_csv(args.pairs_csv)
        alias_map = {
            "oral_sample_id":  "oral_id",
            "oral_sample":     "oral_id",
            "oral":            "oral_id",
            "fecal_sample_id": "fecal_id",
            "fecal_sample":    "fecal_id",
            "fecal":           "fecal_id",
        }
        for src, dst in alias_map.items():
            if src in pairs.columns and "oral_id" not in pairs.columns and dst == "oral_id":
                pairs = pairs.rename(columns={src: "oral_id"})
            if src in pairs.columns and "fecal_id" not in pairs.columns and dst == "fecal_id":
                pairs = pairs.rename(columns={src: "fecal_id"})
        expected = {"oral_id", "fecal_id"}
        if not expected.issubset(set(pairs.columns)):
            warnings.warn("--pairs-csv provided but columns oral_id,fecal_id not found; ignoring pairs.")
            pairs = None

    # Minimal run metadata
    run_meta = {
        "params": {"topk": args.topk, "heat_topk": args.heat_topk, "presence_threshold": args.presence_threshold},
        "n_samples": {
            "Oral_Crohn":    _count_samples(oc_raw),
            "Oral_Healthy":  _count_samples(oh_raw),
            "Fecal_Crohn":   _count_samples(fc_raw),
            "Fecal_Healthy": _count_samples(fh_raw),
        },
        "palette_used": PALETTE,
        "pairs_csv": args.pairs_csv,
    }
    try:
        with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
            json.dump(run_meta, f, indent=2)
    except Exception:
        warnings.warn("Could not write run_meta.json")

    # GENUS
    outdir_genus = os.path.join(args.outdir, "genus")
    process_rank(
        rank="genus",
        oc_raw=oc, oh_raw=oh, fc_raw=fc, fh_raw=fh,
        outdir_rank=outdir_genus,
        topk=int(args.topk),
        heat_topk=int(args.heat_topk),
        presence_threshold=float(args.presence_threshold),
        pairs=pairs,
        min_mean_pct=float(args.min_mean_pct),
        effect_size_threshold=float(args.effect_size_threshold),
        max_rows=args.max_rows,                         # already parsed to Optional[int]
        emit_legacy_pages=bool(args.emit_legacy_pages),
    )
    print("[INFO] [genus] done →", outdir_genus)

    # SPECIES
    outdir_species = os.path.join(args.outdir, "species")
    
    process_rank(
        rank="species",
        oc_raw=oc, oh_raw=oh, fc_raw=fc, fh_raw=fh,
        outdir_rank=outdir_species,
        topk=int(args.topk),
        heat_topk=int(args.heat_topk),
        presence_threshold=float(args.presence_threshold),
        pairs=pairs,
        min_mean_pct=float(args.min_mean_pct),
        effect_size_threshold=float(args.effect_size_threshold),
        max_rows=args.max_rows,
        emit_legacy_pages=bool(args.emit_legacy_pages),
    )
    print("[INFO] [species] done →", outdir_species)

    print("[INFO] CORE done.")

if __name__ == "__main__":
    main()
