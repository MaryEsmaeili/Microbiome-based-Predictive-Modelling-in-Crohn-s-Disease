#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taxa Compare — CORE (Descriptive only) + Debug & Pairing Helpers
================================================================

What this script produces (per rank = genus/species):
  • Percent matrices per group: pct_oral_crohn.csv, pct_oral_healthy.csv,
    pct_fecal_crohn.csv, pct_fecal_healthy.csv
  • pct_all.csv (all four groups concatenated; taxa × samples, %)
  • Heatmaps (paired Crohn, unpaired Oral CH, unpaired Fecal CH; FULL and compact variants)
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
import yaml
import matplotlib
matplotlib.use("Agg")  # headless rendering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator
from matplotlib.colors import Colormap
from matplotlib import cm, colors as mcolors

# =============================
# Global plotting configuration
# =============================
sns.set_context("talk")
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False

# -------------------------------------------------------------------
# Multicolor palette for many taxa (used ONLY for taxa barplots).
# Order has been reversed, as requested.
# -------------------------------------------------------------------
TAXA_COLOR_CYCLE = [
    "#f781c08f",  # light pink (RGBA)
    "#e377c2",    # pink
    "#984ea3",    # magenta
    "#bcbddc",    # light purple
    "#756bb1",    # purple
    "#6baed6",    # light blue
    "#1f78b4",    # blue
    "#1b9e77",    # dark teal
    "#66c2a5",    # teal
    "#4daf4a",    # green
    "#a6d854",    # yellow-green
    "#ffd92f",    # yellow
    "#fdbf6f",    # light orange
    "#ff7f00",    # orange
    "#e41a1c",    # red
]
# NOTE: PALETTE must be defined elsewhere and loaded from config.yml/colors.yml.
# It should be a dict mapping labels such as "Crohn-Oral", "Healthy-Fecal", etc. to hex colors.

def _load_palette_from_yaml(path: str) -> dict:
    """
    Load group colors from a YAML file.

    Expected structure (example):

      groups:
        Crohn-Oral:    "#516D99"
        Healthy-Oral:  "#83AAAC"
        Crohn-Fecal:   "#E76F51"
        Healthy-Fecal: "#3A455B"
        Other:         "#999999"

    If the YAML does not contain a 'groups' key, the full mapping at the
    top-level will be used as the palette.
    """
    if not os.path.exists(path):
        warnings.warn(
            f"Color palette YAML not found at {path}; using empty palette."
        )
        return {}

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # If there is a 'groups' section, use that; otherwise, use the whole dict.
    if isinstance(cfg, dict) and "groups" in cfg and isinstance(cfg["groups"], dict):
        pal = cfg["groups"]
    else:
        pal = cfg

    # Ensure keys are strings
    return {str(k): v for k, v in pal.items()}


# Path can be overridden by environment variable if needed
COLORS_YAML = os.environ.get("COLORS_YAML", "config/colors.yml")

# Global palette used everywhere in this script
PALETTE = _load_palette_from_yaml(COLORS_YAML)
# =====================
# Utility / label tools
# =====================

def _normalize_group_label(label: str) -> str:
    """Normalize free-text labels into keys used in PALETTE."""
    s = str(label).replace("_", "-").strip().lower()
    site = "oral" if "oral" in s else ("fecal" if ("fecal" in s or "faecal" in s or "stool" in s) else None)
    dis = "crohn" if ("crohn" in s or "cd" in s) else ("healthy" if ("healthy" in s or "hc" in s) else None)
    if site and dis:
        return f"{dis.capitalize()}-{site.capitalize()}"
    return label


def group_color(label: str, default: str = "#999999") -> str:
    """Map a group label to a hex color using PALETTE loaded from config."""
    key = _normalize_group_label(label)
    return PALETTE.get(key, PALETTE.get(label, default))

# ---- Normalize PALETTE keys and ensure required defaults ----
if not isinstance(PALETTE, dict):
    PALETTE = {}
else:
    normed = {}
    for k, v in PALETTE.items():
        # Make sure we also have normalized versions like "Crohn-Oral"
        nk = _normalize_group_label(k)
        normed[nk] = v
    PALETTE.update(normed)

# Hard defaults in case YAML does not define these keys
PALETTE.setdefault("Crohn-Oral",    "#D192D8")  # blue-ish
PALETTE.setdefault("Healthy-Oral",  "#A7BDE7")  # teal-ish
PALETTE.setdefault("Crohn-Fecal",   "#AA31B3")  # orange-ish
PALETTE.setdefault("Healthy-Fecal", "#547DD3")  # dark slate
PALETTE.setdefault("Other",         "#999999")  # grey

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
    """
    Ensure matrix is taxa × samples (rows = taxa, columns = samples) based on
    simple heuristics over row/column names.
    """
    idx = df_like.index.astype(str)
    cols = df_like.columns.astype(str)
    row_has_tax = (idx.str.contains("s__|g__", regex=True, na=False).mean() > 0.5)
    col_has_tax = (cols.str.contains("s__|g__", regex=True, na=False).mean() > 0.5)
    if row_has_tax and not col_has_tax:
        return df_like
    if col_has_tax and not row_has_tax:
        return df_like.T
    warnings.warn(
        "Ambiguous matrix orientation; leaving as-is. "
        "Please verify rows=taxa, cols=samples."
    )
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

def _rows_after_all_filters(A_pct, B_pct, presence_threshold, min_mean_pct,
                            effect_size_threshold, max_rows):
    """
    Shared utility: apply presence, mean-abundance, and effect-size filters,
    then optionally cap to max_rows by |Δ mean %|.
    """
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
    """
    Very light Blue→White colormap.

    - Most of the range is almost white.
    - Only the strongest z-scores می‌روند به آبی پررنگ.
    """
    base = cm.get_cmap("Blues", 256)
    # Take a reasonably light slice of "Blues"
    colors = base(np.linspace(0.15, 0.95, 256))

    # Mix with white so that low/mid values are VERY pale
    white = np.array([1.0, 1.0, 1.0, 1.0])
    # mix[i] = fraction of white; high at low intensities, lower at high
    mix = np.linspace(0.85, 0.25, 256)[:, None]  # 85% white → 25% white
    colors = mix * white + (1.0 - mix) * colors

    return mcolors.LinearSegmentedColormap.from_list("BlueWhiteSoft", colors)

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
    ax.set_yticklabels(
        wrapped,
        rotation=0,
        ha="right",
        va="center",
        fontsize=base_fs,
    )
    for txt in ax.get_yticklabels():
        if txt.get_text().strip().lower() != "other":
            try:
                txt.set_style("italic")
            except Exception:
                pass


# =====================
# Debugging primitives
# =====================

def write_debug_lists(
    debug_dir: str,
    label: str,
    shared: pd.Index,
    used_rows: pd.Index,
    only_in_A: pd.Index,
    only_in_B: pd.Index,
    zero_in_A: pd.Index,
    zero_in_B: pd.Index,
) -> None:
    """Write detailed CSVs explaining selection decisions."""
    os.makedirs(debug_dir, exist_ok=True)
    pd.Series(shared, name="taxon").to_csv(
        os.path.join(debug_dir, f"{label}_shared_raw.csv"), index=False
    )
    pd.Series(used_rows, name="taxon").to_csv(
        os.path.join(debug_dir, f"{label}_rows_used.csv"), index=False
    )
    pd.Series(only_in_A, name="taxon").to_csv(
        os.path.join(debug_dir, f"{label}_only_in_A.csv"), index=False
    )
    pd.Series(only_in_B, name="taxon").to_csv(
        os.path.join(debug_dir, f"{label}_only_in_B.csv"), index=False
    )
    pd.Series(zero_in_A, name="taxon").to_csv(
        os.path.join(debug_dir, f"{label}_shared_zero_in_A.csv"), index=False
    )
    pd.Series(zero_in_B, name="taxon").to_csv(
        os.path.join(debug_dir, f"{label}_shared_zero_in_B.csv"), index=False
    )


# =====================
# Plotting primitives
# =====================

def _subsample_xticks(ax, max_labels: int = 200) -> None:
    """Reduce x tick clutter when too many samples are present."""
    ticks = ax.get_xticks()
    if len(ticks) <= max_labels or max_labels <= 0:
        return
    step = max(1, int(round(len(ticks) / max_labels)))
    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_visible((i % step) == 0)


def build_col_colors(label_left: str, n_left: int, label_right: str, n_right: int):
    """Build two-block column color strip for left/right groups using PALETTE."""
    cL = group_color(label_left)
    cR = group_color(label_right)
    return [cL] * int(n_left) + [cR] * int(n_right)

from matplotlib.ticker import FixedLocator as _FixedLocator, MaxNLocator

def pairs_heatmap(
    A_pct: pd.DataFrame,
    B_pct: pd.DataFrame,
    rows: pd.Index,
    label_A: str,
    label_B: str,
    rank: str,
    out_png: str,
    use_log1p: bool = True,
    clip_quantile: float = 0.98,
    title_suffix: str = "",
) -> None:
    """
    Two-block heatmap with fixed row/column order (no clustering).

    Rows: taxa (subset passed in `rows`)
    Columns: samples from group A, followed by samples from group B.
    Cell values: row-wise z-score of log1p(percent abundance) by default.
    """
    data = pd.concat([A_pct, B_pct], axis=1).reindex(rows).fillna(0)
    pretty_idx = [prettify_taxon(t, rank) for t in data.index]
    Z = row_zscore_log1p(data, use_log1p=use_log1p)
    Z.index = pretty_idx

    # Symmetric clipping based on quantile of absolute z-scores
    V = np.nanquantile(np.abs(Z.values), clip_quantile)
    if not np.isfinite(V) or V == 0:
        V = max(1.0, float(np.nanmax(np.abs(Z.values)) or 1.0))

    # Figure size scales with taxa and samples
    height = max(6.0, 0.35 * len(Z.index))
    width = max(8.0, 0.18 * len(Z.columns) + 2.0)

    nA, nB = A_pct.shape[1], B_pct.shape[1]
    col_colors = build_col_colors(label_A, nA, label_B, nB)

    # Use clustermap only as a layout helper (no clustering)
    g = sns.clustermap(
        Z,
        cmap=_coolwarm(),
        center=0,
        vmin=-V,
        vmax=+V,
        col_cluster=False,
        row_cluster=False,
        figsize=(width, height),
        cbar_kws={
            # label will be set manually to control spacing
            "orientation": "vertical",
        },
        # Small, vertical colorbar on the right
        cbar_pos=(2, 1.5, 1.5, 2),
        col_colors=col_colors,
        dendrogram_ratio=(0.1, 0.1),
    )

    # --- Fix colorbar ticks & label so they do not overlap ---
    cax = g.cax
    # Limit number of ticks
    locator = MaxNLocator(nbins=5)
    cax.yaxis.set_major_locator(locator)
    cax.tick_params(labelsize=8)

    # Put label a bit to the right with small font
    cax.set_ylabel(
        "Row z-score of % abundance (log1p)",
        fontsize=9,
    )
    cax.yaxis.set_label_position("left")

    # Place y-axis ticks on the right and format labels
    g.ax_heatmap.yaxis.set_ticks_position("right")
    g.ax_heatmap.yaxis.set_label_position("right")
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_ymajorticklabels(),
        rotation=0,
        ha="left",
        va="center",
        fontsize=10,
    )
    g.ax_heatmap.tick_params(axis="y", pad=10)

    # Clean x-axis tick labels
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xmajorticklabels(),
        rotation=90,
        ha="center",
        va="top",
        fontsize=8.5,
    )
    _subsample_xticks(g.ax_heatmap, max_labels=25)

    # Title slightly higher so it is clearly separated from the heatmap
    title = f"{label_A} vs {label_B} — {rank.capitalize()} heatmap {title_suffix}".strip()
    g.ax_heatmap.set_title(title, pad=30)

    # Axis labels
    g.ax_heatmap.set_xlabel(f"{label_A} ↔ {label_B}")
    g.ax_heatmap.set_ylabel(rank.capitalize())
    g.ax_heatmap.yaxis.labelpad = 18

    # Annotate group sample counts above the heatmap, using PALETTE colors
    fig = g.fig
    fig.text(
        0.15,
        0.96,
        f"{label_A} (n={nA})",
        color=group_color(label_A),
        ha="left",
        va="center",
        fontsize=10,
    )
    fig.text(
        0.55,
        0.96,
        f"{label_B} (n={nB})",
        color=group_color(label_B),
        ha="left",
        va="center",
        fontsize=10,
    )

    # Adjust layout to keep everything visible
    maxlen = max((len(s) for s in Z.index), default=12)
    right = min(0.96, 0.86 + 0.016 * max(0, maxlen - 12))
    fig.subplots_adjust(left=0.07, right=right, bottom=0.20, top=0.90)

    fig.savefig(out_png, dpi=320, bbox_inches="tight", pad_inches=0.70)
    plt.close(fig)
# ==============================
# Bars & compact table rendering
# ==============================

def build_taxa_color_cycle(n_needed: int = 20):
    """
    Return a list of colors for taxa barplots, based on TAXA_COLOR_CYCLE.

    Colors are taken from TAXA_COLOR_CYCLE (defined above) and repeated if
    more colors are needed. This is completely independent from PALETTE.
    """
    base = list(TAXA_COLOR_CYCLE)
    if n_needed <= len(base):
        return base[:n_needed]

    reps = int(np.ceil(n_needed / len(base)))
    full = (base * reps)[:n_needed]
    return full


def two_bar_stacked_with_table(
    a_mean: pd.Series,
    b_mean: pd.Series,
    rank: str,
    out_png: str,
    label_a: str,
    label_b: str,
    top_k: int = 15,
) -> None:
    """
    Two stacked bars (group means) with a side legend table.

    Inputs:
      - a_mean, b_mean: mean % abundance per taxon in each group
      - rank: 'genus' or 'species' (for labels)
      - out_png: path to save PNG
      - label_a, label_b: group labels (e.g. 'Crohn-Oral', 'Healthy-Oral')
      - top_k: number of taxa to show explicitly; remaining combined into 'Other'

    Bars are drawn in [0,1] (proportions), to keep panels comparable.
    """
    combined = a_mean.add(b_mean, fill_value=0).sort_values(ascending=False)
    top = combined.index[:min(top_k, len(combined))]
    s_a_pct = a_mean.reindex(top).fillna(0)
    s_b_pct = b_mean.reindex(top).fillna(0)

    # Add "Other" bucket
    s_a_pct = pd.concat(
        [s_a_pct, pd.Series({"Other": a_mean.drop(top, errors="ignore").sum()})]
    )
    s_b_pct = pd.concat(
        [s_b_pct, pd.Series({"Other": b_mean.drop(top, errors="ignore").sum()})]
    )

    # Convert % to proportions in [0,1] for plotting
    s_a = (s_a_pct / 100.0).clip(lower=0)
    s_b = (s_b_pct / 100.0).clip(lower=0)

    # Order taxa by abundance in group A
    order = s_a.sort_values(ascending=False).index

    # Build color map from TAXA_COLOR_CYCLE
    taxa_colors = build_taxa_color_cycle(max(3 * len(order), 12))
    color_map = {tax: taxa_colors[i % len(taxa_colors)] for i, tax in enumerate(order)}
    color_map["Other"] = PALETTE.get("Other", "#999999")

    # Edge colors: derived from PALETTE group colors
    edge_a = PALETTE.get(
        _normalize_group_label(label_a),
        PALETTE.get("Crohn-Oral", "#333333"),
    )
    edge_b = PALETTE.get(
        _normalize_group_label(label_b),
        PALETTE.get("Healthy-Oral", "#333333"),
    )

    fig = plt.figure(figsize=(16.5, 9.0))
    gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[2.4, 1.8], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")

    x = np.array([0, 1])
    b0 = b1 = 0.0
    rows_for_table = []

    for tax in order:
        h0 = float(s_a.loc[tax])
        h1 = float(s_b.loc[tax])

        # Main stacked bars
        ax.bar(
            x[0],
            h0,
            bottom=b0,
            color=color_map[tax],
            width=0.6,
            edgecolor=edge_a,
            linewidth=0.7,
        )
        ax.bar(
            x[1],
            h1,
            bottom=b1,
            color=color_map[tax],
            width=0.6,
            edgecolor=edge_b,
            linewidth=0.7,
        )

        # Add text inside segments when they are large enough
        if h0 >= 0.03:
            ax.text(
                x[0],
                b0 + h0 / 2,
                f"{h0:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
        if h1 >= 0.03:
            ax.text(
                x[1],
                b1 + h1 / 2,
                f"{h1:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

        rows_for_table.append((tax, color_map[tax], h0, h1))
        b0 += h0
        b1 += h1

    ax.set_xticks(x)
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel("Mean relative abundance (proportion)")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"{label_a} vs {label_b} — Top {min(top_k, len(order))} "
        f"{rank.capitalize()} (others → Other)"
    )

    # ---------------- Header row in the side table (fixed spacing) ----------------
    y0 = 0.96
    dy = 0.042
    hdr_fs = 9

    # Break labels on '-' so they do not overlap
    hdr_a = label_a.replace("-", "\n")
    hdr_b = label_b.replace("-", "\n")

    ax_tbl.text(
        0.10,
        y0,
        hdr_a,
        fontweight="bold",
        ha="center",
        va="top",
        fontsize=hdr_fs,
        transform=ax_tbl.transAxes,
    )
    ax_tbl.text(
        0.28,
        y0,
        hdr_b,
        fontweight="bold",
        ha="center",
        va="top",
        fontsize=hdr_fs,
        transform=ax_tbl.transAxes,
    )
    ax_tbl.text(
        0.46,
        y0,
        "Color",
        fontweight="bold",
        ha="center",
        va="top",
        fontsize=hdr_fs,
        transform=ax_tbl.transAxes,
    )
    ax_tbl.text(
        0.70,
        y0,
        "Taxon",
        fontweight="bold",
        ha="left",
        va="top",
        fontsize=hdr_fs,
        transform=ax_tbl.transAxes,
    )

    # ---------------- Data rows ----------------
    y = y0 - dy
    rows_sorted = sorted(
        rows_for_table,
        key=lambda r: (0 if r[0] == "Other" else 1, -(r[2] + r[3])),
    )

    for tax, col, vA, vB in rows_sorted:
        if y < 0.03:
            break

        # values
        ax_tbl.text(
            0.10,
            y,
            f"{vA:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax_tbl.transAxes,
        )
        ax_tbl.text(
            0.28,
            y,
            f"{vB:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax_tbl.transAxes,
        )

        # color patch
        ax_tbl.add_patch(
            plt.Rectangle(
                (0.46 - 0.02, y - 0.015),
                0.04,
                0.02,
                transform=ax_tbl.transAxes,
                color=col,
                clip_on=False,
            )
        )

        # taxon label
        name = prettify_taxon(tax, rank) if tax != "Other" else "Other"
        ax_tbl.text(
            0.70,
            y,
            name,
            ha="left",
            va="center",
            fontsize=9,
            transform=ax_tbl.transAxes,
        )

        y -= dy

    plt.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)
# =====================
# Option A — ALL shared heatmaps (paginated)
# =====================

def heatmap_all_shared_paginated(
    A_pct,
    B_pct,
    rank,
    out_prefix,
    label_A,
    label_B,
    page_size=50,
    use_log1p=True,
    clip_quantile=0.98,
):
    """Paginated heatmaps showing ALL shared taxa between two groups."""
    shared = A_pct.index.intersection(B_pct.index)
    rows = shared[
        (A_pct.loc[shared].sum(axis=1) > 0)
        | (B_pct.loc[shared].sum(axis=1) > 0)
    ]
    if len(rows) == 0:
        return

    # Stable order by total abundance
    order = (
        A_pct.loc[rows]
        .add(B_pct.loc[rows], fill_value=0.0)
        .sum(axis=1)
        .sort_values(ascending=False)
        .index
    )
    rows = list(order)

    # Fix the color scale across pages
    all_data = pd.concat(
        [A_pct.loc[rows], B_pct.loc[rows]], axis=1
    ).fillna(0.0)
    Z_all = row_zscore_log1p(all_data, use_log1p=use_log1p)
    V = np.nanquantile(np.abs(Z_all.values), clip_quantile)
    if not np.isfinite(V) or V == 0:
        V = max(1.0, float(np.nanmax(np.abs(Z_all.values)) or 1.0))

    # Render pages
    for i in range(0, len(rows), page_size):
        block = rows[i : i + page_size]
        out_png = f"{out_prefix}_p{i // page_size + 1:02d}.png"
        pairs_heatmap(
            A_pct,
            B_pct,
            pd.Index(block),
            rank=rank,
            label_A=label_A,
            label_B=label_B,
            out_png=out_png,
            use_log1p=use_log1p,
            clip_quantile=clip_quantile,
            title_suffix=f"(ALL shared; page {i // page_size + 1})",
        )


# =====================
# Option B — Slopegraph of means (ALL taxa, paginated)
# =====================

def slopegraph_means(
    A_pct, B_pct, rank, out_prefix, label_A, label_B, page_size=60
):
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
        block = taxa[i : i + page_size]
        dfb = both.loc[block]
        fig_h = max(6.5, 0.30 * len(dfb))
        fig = plt.figure(figsize=(10, fig_h))
        ax = fig.add_subplot(111)

        y = np.arange(len(dfb))[::-1]
        ax.hlines(y, dfb[label_B].values, dfb[label_A].values, linewidth=1.0, alpha=0.7)
        ax.plot(dfb[label_A].values, y, "o", label=label_A, markersize=3)
        ax.plot(dfb[label_B].values, y, "o", label=label_B, markersize=3)

        ax.set_xlabel("Mean relative abundance (%)")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [prettify_taxon(t, rank) for t in dfb.index], fontsize=9
        )
        ax.invert_yaxis()
        ax.legend(loc="lower right")
        ax.set_title(
            f"{label_a} vs {label_b} — Top {min(top_k, len(order))} "
            f"{rank.capitalize()} (others → Other)",
            pad=18,
            fontsize=12,
        )
        plt.tight_layout()
        fig.savefig(
            f"{out_prefix}_p{i // page_size + 1:02d}.png",
            dpi=320,
            bbox_inches="tight",
        )
        plt.close(fig)


# =====================
# Option C — Delta lollipop (filtered and legacy)
# =====================

def lollipop_filtered(
    A_pct,
    B_pct,
    rank,
    out_png,
    label_delta,
    presence_threshold: float = 0.0,
    min_mean_pct: float = 0.0,
    effect_size_threshold: float = 0.0,
    max_rows: Optional[int] = None,
    color_pos: Optional[str] = None,  # Δ > 0
    color_neg: Optional[str] = None,  # Δ < 0
):
    """
    Lollipop plot after applying the *same filtering logic* as in
    `heatmap_two_groups_compact`:
      - presence_threshold
      - min_mean_pct
      - effect_size_threshold
      - optional max_rows cap

    Colors are always taken from PALETTE (optionally overridden via
    color_pos/color_neg).
    """
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

    # Δ mean % (A − B)
    d = A_pct.loc[rows].mean(axis=1) - B_pct.loc[rows].mean(axis=1)
    df = pd.DataFrame({"delta": d, "abs_delta": d.abs()}).sort_values(
        "abs_delta", ascending=False
    )

    # Determine colors from label_delta / PALETTE when not provided
    lbl = str(label_delta).upper()
    if color_pos is None or color_neg is None:
        if "ORAL" in lbl:
            color_pos = PALETTE["Crohn-Oral"]     # Crohn > Healthy
            color_neg = PALETTE["Healthy-Oral"]   # Healthy > Crohn
        elif "FECAL" in lbl:
            color_pos = PALETTE["Crohn-Fecal"]
            color_neg = PALETTE["Healthy-Fecal"]
        else:
            color_pos = PALETTE.get("Crohn-Fecal", "#30638e")
            color_neg = PALETTE.get("Healthy-Fecal", "#d1495b")

    y = np.arange(len(df))[::-1]
    fig_h = max(7.0, 0.28 * len(df))
    fig = plt.figure(figsize=(10.0, fig_h))
    ax = fig.add_subplot(111)

    deltas = df["delta"].values
    for yi, delta in zip(y, deltas):
        c = color_pos if delta >= 0 else color_neg
        ax.hlines(yi, 0, delta, linewidth=1.5, color=c, alpha=0.9)
        ax.plot(delta, yi, "o", markersize=4, color=c)

    ax.axvline(0, linewidth=1.0, color="#444444")
    ax.set_xlabel("Δ mean % (A − B)")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [prettify_taxon(t, rank) for t in df.index], fontsize=9
    )
    ax.invert_yaxis()
    ax.set_title(f"{label_delta} — {rank.capitalize()} (filtered)")
    plt.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def delta_lollipop(
    A_pct, B_pct, rank, out_prefix, label_delta, page_size=60
):
    """
    Legacy paginated lollipop: Δ = mean(A) − mean(B), signed, sorted by |Δ|.

    This is kept for compatibility; colors are now also derived from PALETTE
    based on label_delta instead of hard-coded blue/red.
    """
    idx = A_pct.index.union(B_pct.index)
    d = (
        A_pct.reindex(idx).fillna(0.0).mean(axis=1)
        - B_pct.reindex(idx).fillna(0.0).mean(axis=1)
    )
    df = pd.DataFrame({"delta": d, "abs_delta": d.abs()}).sort_values(
        "abs_delta", ascending=False
    )
    taxa = df.index.tolist()

    # Determine group colors
    lbl = str(label_delta).upper()
    if "ORAL" in lbl:
        color_pos = PALETTE["Crohn-Oral"]
        color_neg = PALETTE["Healthy-Oral"]
    elif "FECAL" in lbl:
        color_pos = PALETTE["Crohn-Fecal"]
        color_neg = PALETTE["Healthy-Fecal"]
    else:
        color_pos = PALETTE.get("Crohn-Fecal", "#30638e")
        color_neg = PALETTE.get("Healthy-Fecal", "#d1495b")

    for i in range(0, len(taxa), page_size):
        block = taxa[i : i + page_size]
        dsub = df.loc[block]

        y = np.arange(len(dsub))[::-1]
        fig_h = max(6.5, 0.30 * len(dsub))
        fig = plt.figure(figsize=(9.5, fig_h))
        ax = fig.add_subplot(111)

        deltas = dsub["delta"].values
        for yi, delta in zip(y, deltas):
            c = color_pos if delta >= 0 else color_neg
            ax.hlines(yi, 0, delta, linewidth=1.5, color=c, alpha=0.9)
            ax.plot(delta, yi, "o", markersize=4, color=c)

        ax.axvline(0, linewidth=1.0, color="#444444")
        ax.set_xlabel("Δ mean % (A − B)")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [prettify_taxon(t, rank) for t in dsub.index], fontsize=9
        )
        ax.invert_yaxis()
        ax.set_title(
            f"{label_delta} — {rank.capitalize()} "
            f"(ALL taxa; page {i // page_size + 1})"
        )

        plt.tight_layout()
        fig.savefig(
            f"{out_prefix}_p{i // page_size + 1:02d}.png",
            dpi=320,
            bbox_inches="tight",
        )
        plt.close(fig)


# =====================
# Contrast constructors
# =====================

def select_top_by_delta(
    A_pct: pd.DataFrame, B_pct: pd.DataFrame, k: int = 20
) -> pd.Index:
    """Pick Top-K taxa by absolute difference in mean % between two groups."""
    shared = A_pct.index.intersection(B_pct.index)
    if len(shared) == 0:
        return pd.Index([])
    A = A_pct.loc[shared].fillna(0.0)
    B = B_pct.loc[shared].fillna(0.0)
    da = (A.mean(axis=1) - B.mean(axis=1)).abs()
    return da.sort_values(ascending=False).head(int(k)).index


def heatmap_two_groups(
    A_pct: pd.DataFrame,
    B_pct: pd.DataFrame,
    rank: str,
    out_png: str,
    title: str,
) -> None:
    """FULL heatmap with all shared taxa that are non-zero on both sides."""
    shared = A_pct.index.intersection(B_pct.index)
    present_A = A_pct.loc[shared].sum(axis=1) > 0
    present_B = B_pct.loc[shared].sum(axis=1) > 0
    rows = shared[(present_A & present_B).values]
    union = A_pct.loc[rows].add(B_pct.loc[rows], fill_value=0.0)
    rows = union.sum(axis=1).sort_values(ascending=False).index
    label_A, label_B = title.split(" vs ")
    pairs_heatmap(
        A_pct,
        B_pct,
        rows,
        label_A=label_A,
        label_B=label_B,
        rank=rank,
        out_png=out_png,
        title_suffix="(unpaired FULL)",
    )


# =====================
# Main per-rank driver
# =====================

def filter_by_prevalence_and_effect(
    A_pct, B_pct, rows, min_mean_pct: float, effect_size_threshold: float
):
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

    mean_overall = A.add(B, fill_value=0.0).mean(axis=1)
    keep = pd.Series(True, index=rows)

    if (min_mean_pct or 0) > 0:
        keep &= mean_overall >= float(min_mean_pct)

    abs_delta = (A.mean(axis=1) - B.mean(axis=1)).abs()
    if (effect_size_threshold or 0) > 0:
        keep &= abs_delta >= float(effect_size_threshold)

    rows2 = rows[keep.loc[rows].values]
    return rows2, abs_delta


def apply_presence_threshold(
    A_pct: pd.DataFrame,
    B_pct: pd.DataFrame,
    shared: pd.Index,
    presence_threshold: float,
) -> Tuple[pd.Index, pd.Series, pd.Series]:
    """
    Filter shared taxa by requiring non-zero presence in each group.

    If presence_threshold == 0:
      - require any non-zero across all samples in each group.
    Otherwise:
      - require non-zero in at least `presence_threshold` fraction of samples
        in EACH group separately.
    """
    A = A_pct.loc[shared]
    B = B_pct.loc[shared]
    if presence_threshold <= 0:
        present_A = A.sum(axis=1) > 0
        present_B = B.sum(axis=1) > 0
    else:
        thr_A = max(1, int(np.ceil(presence_threshold * A.shape[1])))
        thr_B = max(1, int(np.ceil(presence_threshold * B.shape[1])))
        present_A = (A > 0).sum(axis=1) >= thr_A
        present_B = (B > 0).sum(axis=1) >= thr_B
    rows = shared[(present_A & present_B).values]
    return rows, present_A, present_B
def heatmap_two_groups_compact(
    A_pct: pd.DataFrame,
    B_pct: pd.DataFrame,
    rank: str,
    out_png: str,
    title: str,
    presence_threshold: float = 0.0,
    min_mean_pct: float = 0.0,
    effect_size_threshold: float = 0.0,
    max_rows: Optional[int] = None,
    use_log1p: bool = True,
    clip_quantile: float = 0.98,
) -> None:
    """
    Compact two-group heatmap.

    - Applies the same filtering policy as lollipop_filtered:
        * presence_threshold
        * min_mean_pct
        * effect_size_threshold
        * optional max_rows cap
    - Uses pairs_heatmap() for visualization.
    """
    # Recover group labels from the title (e.g. "Healthy-Oral vs Crohn-Oral")
    try:
        label_A, label_B = title.split(" vs ")
    except ValueError:
        # Fallback if title is not in "A vs B" format
        label_A, label_B = "Group A", "Group B"

    # Apply all row-level filters and optional row cap
    rows, _abs_delta = _rows_after_all_filters(
        A_pct,
        B_pct,
        presence_threshold=presence_threshold,
        min_mean_pct=min_mean_pct,
        effect_size_threshold=effect_size_threshold,
        max_rows=max_rows,
    )

    # If nothing survives, still write a small diagnostic figure
    if len(rows) == 0:
        fig = plt.figure(figsize=(6, 3))
        plt.text(
            0.5,
            0.5,
            "No taxa after filtering",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    # Use the compact row set in a two-block heatmap
    pairs_heatmap(
        A_pct=A_pct,
        B_pct=B_pct,
        rows=rows,
        label_A=label_A,
        label_B=label_B,
        rank=rank,
        out_png=out_png,
        use_log1p=use_log1p,
        clip_quantile=clip_quantile,
        title_suffix="(filtered, unpaired)",
    )


def group_contrasts_and_figs(
    oc_pct: pd.DataFrame,   # Crohn Oral   (taxa x samples, in %)
    fc_pct: pd.DataFrame,   # Crohn Fecal  (taxa x samples, in %)
    oh_pct: pd.DataFrame,   # Healthy Oral (taxa x samples, in %)
    fh_pct: pd.DataFrame,   # Healthy Fecal(taxa x samples, in %)
    rank: str,              # "genus" | "species" (labels/filenames only)
    outdir: str,            # output directory for this rank
    n: int = 15,            # top-N taxa used for titles of barplots etc.
    heat_topk: int = 20,    # kept only for compatibility
    use_log1p: bool = True,
    clip_quantile: float = 0.98,
    presence_threshold: float = 0.0,
    pairs: Optional[pd.DataFrame] = None,
    min_mean_pct: float = 0.0,          # abundance filter (percentage points)
    effect_size_threshold: float = 0.0, # |Δ mean %| threshold (percentage points)
    max_rows: Optional[int] = None,     # cap rows AFTER filtering (None = no cap)
    emit_legacy_pages: bool = False,    # legacy multi-page figures (kept off by default)
) -> None:
    """
    Build figures/tables for a given rank with the current policy:

      - Paired Crohn Oral vs Fecal heatmap (descriptive, prevalence-filtered)
      - Filtered unpaired CH heatmaps (Oral CH and Fecal CH), respecting
        prevalence/abundance/effect-size filters
      - Lollipop plots (Crohn − Healthy) for ORAL and FECAL, filtered as above
      - Stacked barplots using TAXA_COLOR_CYCLE
      - Compact CSVs of group means and deltas
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
    if pairs is not None and {"oral_id", "fecal_id"}.issubset(set(pairs.columns)):
        oc_sub = A_cf.reindex(
            columns=pairs["oral_id"].dropna().astype(str), fill_value=0
        )
        fc_sub = B_cf.reindex(
            columns=pairs["fecal_id"].dropna().astype(str), fill_value=0
        )
        valid_mask = (~oc_sub.columns.isna()) & (~fc_sub.columns.isna())
        A_cf, B_cf = oc_sub.loc[:, valid_mask], fc_sub.loc[:, valid_mask]

    # Prevalence filter (for descriptive order)
    shared_cf = A_cf.index.intersection(B_cf.index)
    rows_cf_all, present_A, present_B = apply_presence_threshold(
        A_cf, B_cf, shared_cf, presence_threshold
    )

    # Debug CSVs: what was kept/dropped in the paired selection
    only_in_A = A_cf.index.difference(B_cf.index)
    only_in_B = B_cf.index.difference(A_cf.index)
    zero_in_A = shared_cf[~present_A]
    zero_in_B = shared_cf[~present_B]
    write_debug_lists(
        debug_dir,
        "crohn_pairs",
        shared_cf,
        rows_cf_all,
        only_in_A,
        only_in_B,
        zero_in_A,
        zero_in_B,
    )

    # Order taxa by total abundance across both sites
    order_cf_all = (
        A_cf.loc[rows_cf_all]
        .add(B_cf.loc[rows_cf_all], fill_value=0.0)
        .sum(axis=1)
        .sort_values(ascending=False)
        .index
    )

    # Save paired heatmap
    pairs_heatmap(
        A_cf,
        B_cf,
        order_cf_all,
        label_A="Crohn-Oral",
        label_B="Crohn-Fecal",
        rank=rank,
        out_png=os.path.join(outdir, f"heatmap_pairs_crohn_{rank}.png"),
        use_log1p=use_log1p,
        clip_quantile=clip_quantile,
        title_suffix="(paired subjects; descriptive)",
    )

    # ---------------------------
    # (B) Unpaired CH heatmaps (filtered)
    # ---------------------------
    heatmap_two_groups_compact(
        oh_pct,
        oc_pct,
        rank,
        os.path.join(outdir, f"heatmap_oral_CH_unpaired_{rank}.png"),
        title="Healthy-Oral vs Crohn-Oral",
        presence_threshold=presence_threshold,
        min_mean_pct=min_mean_pct,
        effect_size_threshold=effect_size_threshold,
        max_rows=max_rows,
    )

    heatmap_two_groups_compact(
        fh_pct,
        fc_pct,
        rank,
        os.path.join(outdir, f"heatmap_fecal_CH_unpaired_{rank}.png"),
        title="Healthy-Fecal vs Crohn-Fecal",
        presence_threshold=presence_threshold,
        min_mean_pct=min_mean_pct,
        effect_size_threshold=effect_size_threshold,
        max_rows=max_rows,
    )

    # ---------------------------
    # (B2) Group-mean stacked bars (Crohn vs Healthy)
    # ---------------------------
    two_bar_stacked_with_table(
        oc_pct.mean(axis=1),
        oh_pct.mean(axis=1),
        rank,
        os.path.join(outdir, f"bars_groupmeans_oral_CH_{rank}_top{n}.png"),
        label_a="Crohn-Oral",
        label_b="Healthy-Oral",
        top_k=n,
    )

    two_bar_stacked_with_table(
        fc_pct.mean(axis=1),
        fh_pct.mean(axis=1),
        rank,
        os.path.join(outdir, f"bars_groupmeans_fecal_CH_{rank}_top{n}.png"),
        label_a="Crohn-Fecal",
        label_b="Healthy-Fecal",
        top_k=n,
    )

    # ---------------------------
    # (C) Lollipop plots — filtered taxa
    # ---------------------------
    lollipop_filtered(
        oc_pct,
        oh_pct,
        rank,
        os.path.join(outdir, f"lollipop_ORAL_OC_minus_OH_{rank}_ALL.png"),
        label_delta="ORAL Δ (Crohn − Healthy)",
        presence_threshold=presence_threshold,
        min_mean_pct=min_mean_pct,
        effect_size_threshold=effect_size_threshold,
        max_rows=max_rows,
        color_pos=PALETTE["Crohn-Oral"],
        color_neg=PALETTE["Healthy-Oral"],
    )

    lollipop_filtered(
        fc_pct,
        fh_pct,
        rank,
        os.path.join(outdir, f"lollipop_FECAL_FC_minus_FH_{rank}_ALL.png"),
        label_delta="FECAL Δ (Crohn − Healthy)",
        presence_threshold=presence_threshold,
        min_mean_pct=min_mean_pct,
        effect_size_threshold=effect_size_threshold,
        max_rows=max_rows,
        color_pos=PALETTE["Crohn-Fecal"],
        color_neg=PALETTE["Healthy-Fecal"],
    )

    # (Optional) legacy multi-page outputs
    if emit_legacy_pages:
        heatmap_all_shared_paginated(
            oh_pct,
            oc_pct,
            rank,
            os.path.join(outdir, f"heatmap_oral_CH_unpaired_{rank}_ALL"),
            label_A="Healthy-Oral",
            label_B="Crohn-Oral",
            page_size=heat_topk,
            use_log1p=use_log1p,
            clip_quantile=clip_quantile,
        )
        heatmap_all_shared_paginated(
            fh_pct,
            fc_pct,
            rank,
            os.path.join(outdir, f"heatmap_fecal_CH_unpaired_{rank}_ALL"),
            label_A="Healthy-Fecal",
            label_B="Crohn-Fecal",
            page_size=heat_topk,
            use_log1p=use_log1p,
            clip_quantile=clip_quantile,
        )
        delta_lollipop(
            oc_pct,
            oh_pct,
            rank,
            os.path.join(outdir, f"lollipop_ORAL_OC_minus_OH_{rank}"),
            label_delta="ORAL Δ (Crohn − Healthy)",
            page_size=60,
        )
        delta_lollipop(
            fc_pct,
            fh_pct,
            rank,
            os.path.join(outdir, f"lollipop_FECAL_FC_minus_FH_{rank}"),
            label_delta="FECAL Δ (Crohn − Healthy)",
            page_size=60,
        )

    # ---------------------------
    # (X) Compact CSV tables
    # ---------------------------
    crohn_union_index = (
        oc_pct.index.union(fc_pct.index)
        .union(oh_pct.index)
        .union(fh_pct.index)
    )
    means_df = pd.DataFrame(
        {
            "O.C.": oc_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
            "F.C.": fc_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
            "O.H.": oh_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
            "F.H.": fh_pct.reindex(crohn_union_index).fillna(0.0).mean(axis=1),
        },
        index=crohn_union_index,
    )
    means_df = means_df.loc[(means_df.sum(axis=1) > 0.0)]
    means_df.index.name = rank
    means_df.to_csv(os.path.join(outdir, f"group_means_{rank}.csv"))

    deltas_cols = {}
    if {"O.C.", "O.H."}.issubset(means_df.columns):
        deltas_cols["O.C.–O.H."] = means_df["O.C."] - means_df["O.H."]
    if {"F.C.", "F.H."}.issubset(means_df.columns):
        deltas_cols["F.C.–F.H."] = means_df["F.C."] - means_df["F.H."]
    if not deltas_cols and {"O.C.", "F.C."}.issubset(means_df.columns):
        deltas_cols["O.C.–F.C."] = means_df["O.C."] - means_df["F.C."]

    deltas_df = (
        pd.DataFrame(deltas_cols, index=means_df.index)
        if deltas_cols
        else pd.DataFrame(index=means_df.index)
    )
    deltas_df.index.name = rank
    deltas_df.to_csv(os.path.join(outdir, f"group_deltas_{rank}.csv"))

def process_rank(
    rank: str,
    oc_raw: pd.DataFrame,   # taxa x samples, original clade names
    oh_raw: pd.DataFrame,
    fc_raw: pd.DataFrame,
    fh_raw: pd.DataFrame,
    outdir_rank: str,
    topk: int,
    heat_topk: int,
    presence_threshold: float,
    pairs: Optional[pd.DataFrame],
    min_mean_pct: float,
    effect_size_threshold: float,
    max_rows: Optional[int],
    emit_legacy_pages: bool,
) -> None:
    """
    Rank-specific driver.

    Steps:
      1) Choose the correct grouper for the requested rank (genus/species).
      2) Collapse the full clade matrix to that rank and convert to percent.
      3) Align taxa across groups (union index) and fill missing with 0.
      4) Write per-group percent tables and pct_all.csv.
      5) Call group_contrasts_and_figs() to generate figures + compact CSVs.
    """
    os.makedirs(outdir_rank, exist_ok=True)

    # 1) Choose grouper based on rank
    if rank.lower() == "genus":
        grouper = extract_genus
    elif rank.lower() == "species":
        grouper = extract_species
    else:
        raise ValueError(f"Unsupported rank: {rank!r}; expected 'genus' or 'species'.")

    # 2) Collapse to requested rank and convert to percent per sample
    oc_pct = percent_table(oc_raw, grouper)
    oh_pct = percent_table(oh_raw, grouper)
    fc_pct = percent_table(fc_raw, grouper)
    fh_pct = percent_table(fh_raw, grouper)

    # 3) Align taxa across all four groups (union of indices)
    union_index = (
        oc_pct.index
        .union(oh_pct.index)
        .union(fc_pct.index)
        .union(fh_pct.index)
    )
    oc_pct = oc_pct.reindex(union_index).fillna(0.0)
    oh_pct = oh_pct.reindex(union_index).fillna(0.0)
    fc_pct = fc_pct.reindex(union_index).fillna(0.0)
    fh_pct = fh_pct.reindex(union_index).fillna(0.0)

    # 4) Write per-group percent tables + pct_all.csv
    oc_pct.to_csv(os.path.join(outdir_rank, "pct_oral_crohn.csv"))
    oh_pct.to_csv(os.path.join(outdir_rank, "pct_oral_healthy.csv"))
    fc_pct.to_csv(os.path.join(outdir_rank, "pct_fecal_crohn.csv"))
    fh_pct.to_csv(os.path.join(outdir_rank, "pct_fecal_healthy.csv"))

    pct_all = pd.concat(
        [oc_pct, oh_pct, fc_pct, fh_pct],
        axis=1,
    )
    pct_all.to_csv(os.path.join(outdir_rank, "pct_all.csv"))

    # 5) Run the main contrast/plotting logic for this rank
    group_contrasts_and_figs(
        oc_pct=oc_pct,
        fc_pct=fc_pct,
        oh_pct=oh_pct,
        fh_pct=fh_pct,
        rank=rank,
        outdir=outdir_rank,
        n=topk,
        heat_topk=heat_topk,
        use_log1p=True,
        clip_quantile=0.98,
        presence_threshold=presence_threshold,
        pairs=pairs,
        min_mean_pct=min_mean_pct,
        effect_size_threshold=effect_size_threshold,
        max_rows=max_rows,
        emit_legacy_pages=emit_legacy_pages,
    )

# =============
# CLI & driver
# =============

def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Taxa Compare CORE (descriptive only; genus/species) "
            "with debugging and pairing helpers"
        )
    )
    p.add_argument("--oral-crohn", required=True)
    p.add_argument("--oral-healthy", required=True)
    p.add_argument("--fecal-crohn", required=True)
    p.add_argument("--fecal-healthy", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--topk",
        type=int,
        default=15,
        help="Top-N rows for compact summary figures and titles",
    )
    p.add_argument(
        "--heat-topk",
        type=int,
        default=20,
        help="Top-K rows for legacy paginated heatmaps (if enabled)",
    )
    p.add_argument(
        "--pairs-csv",
        default=None,
        help=(
            "Optional CSV with columns: subject_id, oral_id, fecal_id, "
            "to enforce true subject pairing"
        ),
    )
    p.add_argument(
        "--presence-threshold",
        type=float,
        default=0.0,
        help=(
            "Fraction in [0,1]; require non-zero in at least this fraction "
            "of samples in EACH group for inclusion. 0.0 = any nonzero."
        ),
    )
    p.add_argument(
        "--min-mean-pct",
        type=float,
        default=0.0,
        help=(
            "Drop taxa with overall mean (percentage points) below this "
            "value; 0 disables."
        ),
    )
    p.add_argument(
        "--effect-size-threshold",
        type=float,
        default=0.0,
        help="Keep taxa with |Δ mean %| >= this value; 0 disables.",
    )
    p.add_argument(
        "--max-rows",
        default=None,
        help=(
            "Maximum number of taxa to plot after filtering; "
            "'null' or None means no cap."
        ),
    )
    p.add_argument(
        "--emit-legacy-pages",
        action="store_true",
        help="Also emit legacy multi-page figures (paginated ALL/shared).",
    )

    return p.parse_args()


def _count_samples(df: pd.DataFrame) -> int:
    """Utility to safely count columns."""
    try:
        return int(df.shape[1])
    except Exception:
        return 0


def _parse_max_rows(x):
    """
    Robust parse for --max-rows:
      - accepts None/'null'/'none'/''/0 as 'no cap'
      - otherwise returns positive int
    """
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

    # Read input matrices (rows=features, cols=samples; orientation fixed below)
    oc_raw = pd.read_csv(args.oral_crohn, index_col=0)
    oh_raw = pd.read_csv(args.oral_healthy, index_col=0)
    fc_raw = pd.read_csv(args.fecal_crohn, index_col=0)
    fh_raw = pd.read_csv(args.fecal_healthy, index_col=0)

    # Ensure matrices are taxa × samples once here; pass these onwards
    oc = ensure_taxa_by_samples(oc_raw)
    oh = ensure_taxa_by_samples(oh_raw)
    fc = ensure_taxa_by_samples(fc_raw)
    fh = ensure_taxa_by_samples(fh_raw)

    # Optional pairs CSV
    pairs = None
    if args.pairs_csv is not None and os.path.exists(args.pairs_csv):
        pairs = pd.read_csv(args.pairs_csv)
        alias_map = {
            "oral_sample_id": "oral_id",
            "oral_sample": "oral_id",
            "oral": "oral_id",
            "fecal_sample_id": "fecal_id",
            "fecal_sample": "fecal_id",
            "fecal": "fecal_id",
        }
        for src, dst in alias_map.items():
            if src in pairs.columns and dst not in pairs.columns:
                pairs = pairs.rename(columns={src: dst})
        expected = {"oral_id", "fecal_id"}
        if not expected.issubset(set(pairs.columns)):
            warnings.warn(
                "--pairs-csv provided but columns oral_id,fecal_id not found; "
                "ignoring pairs."
            )
            pairs = None

    # Minimal run metadata
    run_meta = {
        "params": {
            "topk": args.topk,
            "heat_topk": args.heat_topk,
            "presence_threshold": args.presence_threshold,
            "min_mean_pct": args.min_mean_pct,
            "effect_size_threshold": args.effect_size_threshold,
            "max_rows": args.max_rows,
        },
        "n_samples": {
            "Oral_Crohn": _count_samples(oc_raw),
            "Oral_Healthy": _count_samples(oh_raw),
            "Fecal_Crohn": _count_samples(fc_raw),
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
        oc_raw=oc,
        oh_raw=oh,
        fc_raw=fc,
        fh_raw=fh,
        outdir_rank=outdir_genus,
        topk=int(args.topk),
        heat_topk=int(args.heat_topk),
        presence_threshold=float(args.presence_threshold),
        pairs=pairs,
        min_mean_pct=float(args.min_mean_pct),
        effect_size_threshold=float(args.effect_size_threshold),
        max_rows=args.max_rows,
        emit_legacy_pages=bool(args.emit_legacy_pages),
    )

    print("[INFO] [genus] done →", outdir_genus)

    # SPECIES
    outdir_species = os.path.join(args.outdir, "species")
    process_rank(
        rank="species",
        oc_raw=oc,
        oh_raw=oh,
        fc_raw=fc,
        fh_raw=fh,
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
