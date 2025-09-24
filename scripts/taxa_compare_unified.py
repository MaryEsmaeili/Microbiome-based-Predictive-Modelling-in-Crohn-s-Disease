#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified genus/species comparison pipeline (Crohn vs Healthy; Oral & Fecal)

Outputs per-rank (genus, species):
- Tables: group_means_*.csv, group_deltas_*.csv
- Differential abundance (MWU; Crohn vs Healthy per site): da_oral_CH_*.csv, da_fecal_CH_*.csv
- Paired OC↔FC (Wilcoxon) if matched IDs provided: da_paired_OC_FC_*.csv (empty if fail)
- Plots: heatmap_group_means_*.png, heatmap_group_deltas_*.png,
         heatmap_pairs_crohn_*.png, heatmap_pairs_healthy_*.png,
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
import os, re, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib.patheffects as pe
from matplotlib.ticker import FixedLocator
import textwrap

# ---------- Global plotting context ----------
sns.set_context("talk")


# ---------- Robust palette loader ----------
# ---------- Robust palette loader + global theme ----------
try:
    import yaml
except Exception:
    yaml = None

def _try_read_yaml(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def load_palette(cfg_paths=("config/colors.yml", "config/config.yml", "config.yaml", "config.yml")) -> dict:
    """
    Robust palette loader:
    - Accepts either:
        colors:
          Crohn-Oral: "#..."
          ...
      or a top-level dict of the same keys
    - Merges with defaults so essential keys always exist.
    """
    default = {
        "Other": "#999999",
        "Crohn-Oral":    "#B499E5",
        "Crohn-Fecal":   "#78688E",
        "Healthy-Oral":  "#7FB3D5",
        "Healthy-Fecal": "#416388",
        # also provide site-level fallbacks
        "Oral":  "#B499E5",
        "Fecal": "#78688E",
    }
    if yaml is None:
        warnings.warn("[palette] PyYAML not available; using defaults.")
        return default
    for p in cfg_paths:
        if os.path.exists(p):
            cfg = _try_read_yaml(p)
            # accept either colors:{...} or top-level dict
            colors = cfg.get("colors") if isinstance(cfg, dict) else None
            if not isinstance(colors, dict) or not colors:
                if isinstance(cfg, dict):
                    # if it *looks* like a palette at top-level, use it
                    keys = {"Crohn-Oral","Crohn-Fecal","Healthy-Oral","Healthy-Fecal","Other","Oral","Fecal"}
                    if any(k in cfg for k in keys):
                        colors = cfg
            if isinstance(colors, dict) and colors:
                merged = dict(default); merged.update(colors)
                return merged
    warnings.warn(f"[palette] Colors not found in {cfg_paths}; using defaults.")
    return default

PALETTE = load_palette()

def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0,2,4))

def _mix(c1, c2, t):
    return tuple((1-t)*a + t*b for a,b in zip(c1,c2))

def make_diverging_from(key_left, key_right, n=256):
    c1 = _hex_to_rgb01(PALETTE.get(key_left, "#B499E5"))
    c2 = _hex_to_rgb01("#FFFFFF")
    c3 = _hex_to_rgb01(PALETTE.get(key_right, "#78688E"))
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(f"div_{key_left}_to_{key_right}", [c1, c2, c3], N=n)

def make_sequential_from(key, n=256, to_white=True):
    """Light→Base (or Base→Dark) sequential cmap derived from one of your brand colors."""
    base = _hex_to_rgb01(PALETTE.get(key, "#B499E5"))
    white = _hex_to_rgb01("#FFFFFF")
    dark  = tuple(max(x-0.35, 0) for x in base)
    stops = [white, base] if to_white else [base, dark]
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(f"seq_{key}", stops, N=n)

def build_taxa_color_cycle(n_needed=20):
    """
    Generate a cycle of distinct colors *derived from your 4 group colors*,
    instead of using tab20c. Keeps branding consistent even for taxa stacks.
    """
    bases = [
        PALETTE.get("Crohn-Oral", "#B499E5"),
        PALETTE.get("Crohn-Fecal", "#78688E"),
        PALETTE.get("Healthy-Oral", "#7FB3D5"),
        PALETTE.get("Healthy-Fecal", "#416388"),
    ]
    bases = [_hex_to_rgb01(b) for b in bases]
    white = _hex_to_rgb01("#FFFFFF")
    cycle = []
    # For each base, create a few tints/shades
    for b in bases:
        # 3 levels: light tint, base, slightly darker
        cycle.append(_mix(white, b, 0.35))
        cycle.append(b)
        cycle.append(tuple(max(x-0.20,0) for x in b))
    # If more needed, interpolate around bases
    i = 0
    while len(cycle) < n_needed:
        b = bases[i % len(bases)]
        t = 0.15 + 0.15*((i//len(bases))%3)  # 0.15, 0.30, 0.45 ...
        cycle.append(_mix(white, b, t))
        i += 1
    # convert to hex for matplotlib
    def rgb01_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    return [rgb01_to_hex(c) for c in cycle[:n_needed]]

def apply_global_theme():
    """Apply a seaborn/matplotlib theme using your palette for a consistent look."""
    sns.set_context("talk")
    # Set a qualitative cycle using 4 group colors
    sns.set_palette([
        PALETTE.get("Crohn-Oral", "#B499E5"),
        PALETTE.get("Crohn-Fecal", "#78688E"),
        PALETTE.get("Healthy-Oral", "#7FB3D5"),
        PALETTE.get("Healthy-Fecal", "#416388"),
    ])
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False

# Call once at import time
apply_global_theme()

# ---------- ID normalization & mapping ----------
def _normalize_id(x: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(x).strip().lower())

def _build_id_map(cols) -> dict:
    """Map 'normalized' id to original id."""
    m = {}
    for c in cols:
        key = _normalize_id(str(c))
        if key not in m:
            m[key] = c
    return m


# ---------- Taxonomy helpers ----------
# --- helpers for robust matching of paired IDs (English comments) ---

def _guess_pair_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Try to guess which two columns in the match file correspond to Oral/OC and Fecal/FC.
    Strategy:
      - If the file has exactly 2 columns → use them.
      - Else pick one column containing any of ['oral','oc','mouth'] and another containing ['fecal','fc','stool'].
      - Fallback: take the first two columns.
    Returns (oral_col, fecal_col).
    """
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
    # fallback: just take first two
    return cols[0], cols[1]

def _map_ids_to_columns(candidates: list[str], df_cols: pd.Index) -> list[str|None]:
    """
    Map arbitrary ID strings to actual matrix column names.
    - Normalize both sides (strip non-alnum, lowercase).
    - Exact normalized match first; then exact raw match; otherwise None.
    """
    col_raw = [str(c) for c in df_cols]
    col_norm_map = { _normalize_id(c): c for c in col_raw }
    mapped = []
    for x in candidates:
        xr = str(x)
        xn = _normalize_id(xr)
        col = col_norm_map.get(xn, None)
        if col is None and xr in col_raw:
            col = xr
        mapped.append(col)
    return mapped

def _wrap_label(s: str, width: int = 26):
    s = str(s)
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)) or s

def _apply_smart_yticklabels(ax, labels, rank, base_fs=13.0, max_chars=28):
    """
    روی محور y لیبل‌ها را طوری ست می‌کند که با تعداد ردیف‌ها جور باشد.
    - برای clustermap هم کار می‌کند: locator را به همهٔ ردیف‌ها فیکس می‌کنیم.
    - لیبل‌های طولانی wrap می‌شوند؛ نام‌های علمی ایتالیک می‌شوند.
    """
    if labels is None:
        return
    # wrap
    wrapped = []
    for lab in labels:
        lab = str(lab)
        lines = textwrap.wrap(lab, width=max_chars, break_long_words=False) or [lab]
        wrapped.append("\n".join(lines))

    n = len(wrapped)
    # در heatmap‌های seaborn، مراکز سلول‌ها روی 0..n-1 هستند؛ برای clustermap معمولاً 0..n-1 نیز است.
    # FixedLocator را ست می‌کنیم تا دقیقا n تیک داشته باشیم:
    ax.yaxis.set_major_locator(FixedLocator(np.arange(n)))
    ax.set_yticklabels(wrapped, rotation=0, ha="right", va="center", fontsize=base_fs)

    # ایتالیک برای نام‌های علمی (غیر «Other»)
    for txt in ax.get_yticklabels():
        if txt.get_text().strip().lower() != "other":
            try:
                txt.set_style("italic")
            except Exception:
                pass

def extract_genus(rowname: str):
    parts = str(rowname).split("|")
    g = [p for p in parts if p.startswith("g__")]
    return g[0] if g else None

def extract_species(rowname: str):
    parts = str(rowname).split("|")
    s = [p for p in parts if p.startswith("s__")]
    return s[0] if s else None

def prettify_taxon(label: str, rank: str) -> str:
    if label is None:
        return None
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


# ---------- Data orientation & transforms ----------
def ensure_taxa_by_samples(df_like: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the matrix is taxa × samples.
    Heuristic: if >50% of columns look like taxa labels (contain 's__' or 'g__'), it's samples × taxa → transpose.
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
    """
    Mean percent abundance per taxon rank across samples (within the provided df).
    Useful for two-bar stacked plots (Crohn vs Healthy within a site).
    """
    g = df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    pct = g.div(g.sum(axis=0), axis=1) * 100
    return pct.mean(axis=1)

def clr_transform(df_taxa_by_samples: pd.DataFrame, pseudocount=1e-6) -> pd.DataFrame:
    """
    Centered log-ratio transform (natural log).
    Input must be taxa × samples with non-negative values.
    """
    X = df_taxa_by_samples.astype(float) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)
    return logX.sub(gm, axis=1)


# ---------- Z-score helpers ----------
def row_zscore_log1p(mat: pd.DataFrame, use_log1p=True) -> pd.DataFrame:
    X = np.log1p(mat.astype(float)) if use_log1p else mat.astype(float)
    mu = X.mean(axis=1)
    sd = X.std(axis=1).replace(0, np.nan)
    return X.sub(mu, axis=0).div(sd, axis=0).fillna(0)


# ---------- Heatmaps (pairs and group contrasts) ----------
def make_oc_fc_cmap(palette: dict) -> LinearSegmentedColormap:
    # Crohn-Oral (left) → white → Crohn-Fecal (right)
    return make_diverging_from("Crohn-Oral", "Crohn-Fecal", n=256)

def make_oh_fh_cmap(palette: dict) -> LinearSegmentedColormap:
    # Healthy-Oral (left) → white → Healthy-Fecal (right)
    return make_diverging_from("Healthy-Oral", "Healthy-Fecal", n=256)

def _subsample_xticks(ax, max_labels=25):
    ticks = ax.get_xticks()
    if len(ticks) <= max_labels or max_labels <= 0:
        return
    step = max(1, int(round(len(ticks) / max_labels)))
    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_visible((i % step) == 0)

def _shrink_col_colors_axis(g, target_rel_height=0.015):
    """
    Make the col_colors band very thin even on old seaborn versions.
    target_rel_height is relative to figure height.
    """
    if hasattr(g, "ax_col_colors") and g.ax_col_colors is not None:
        fig = g.fig
        fx, fy = fig.get_size_inches()
        # current axes position in figure coordinates
        pos = g.ax_col_colors.get_position()
        # keep the same left/right, shrink height to small constant fraction
        new_h = target_rel_height
        # position this thin band just above the heatmap top
        heat_pos = g.ax_heatmap.get_position()
        new_y = heat_pos.y1 + 0.002   # a tiny gap
        g.ax_col_colors.set_position([heat_pos.x0, new_y, heat_pos.width, new_h])

def pairs_heatmap(A_pct, B_pct, rows, label_A, label_B, rank, out_png,
                  palette=None, use_log1p=True, clip_quantile=0.98, col_cluster=False):
    """Readable Pairs heatmap: thin group band, safe margins, tidy labels."""
    palette = palette or PALETTE

    data = pd.concat([A_pct, B_pct], axis=1).reindex(rows).fillna(0)
    pretty_idx = [prettify_taxon(t, rank) for t in data.index]
    Z = row_zscore_log1p(data, use_log1p=use_log1p); Z.index = pretty_idx

    V = np.nanquantile(np.abs(Z.values), clip_quantile)
    if not np.isfinite(V) or V == 0:
        V = max(1.0, float(np.nanmax(np.abs(Z.values)) or 1.0))

    col_colors = pd.Series(index=Z.columns, dtype=object)
    col_colors.loc[Z.columns.intersection(A_pct.columns)] = palette.get(label_A, "#1f77b4")
    col_colors.loc[Z.columns.intersection(B_pct.columns)] = palette.get(label_B, "#ff7f0e")

    cmap = make_oc_fc_cmap(palette) if ("Crohn" in label_A or "Crohn" in label_B) else make_oh_fh_cmap(palette)

    height = max(6.0, 0.35*len(Z.index))
    width  = max(8.0, 0.18*len(Z.columns)+2.0)

    g = sns.clustermap(
        Z,
        cmap=cmap, center=0, vmin=-V, vmax=+V,
        col_colors=col_colors, col_cluster=col_cluster, row_cluster=True,
        metric="correlation", method="average",
        figsize=(width, height),
        cbar_kws={"label": "Row z-score of % abundance (log1p)", "orientation": "horizontal"},
        # put colorbar on the left so it never eats right margin
        cbar_pos=(10, 10, 0, 0)
    )

    # Make the col_colors band thin even if seaborn ignored colors_ratio
    _shrink_col_colors_axis(g, target_rel_height=0.015)

    # Clean spines to avoid any thick line at the top
    for ax in (g.ax_heatmap, g.ax_row_dendrogram, g.ax_col_dendrogram):
        for sp in getattr(ax, "spines", {}).values():
            sp.set_visible(False)

    # Y labels on the right, italic, readable
    g.ax_heatmap.yaxis.set_ticks_position("right")
    g.ax_heatmap.yaxis.set_label_position("right")
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_ymajorticklabels(),
        rotation=0, ha="left", va="center", fontsize=10
    )
    g.ax_heatmap.tick_params(axis="y", pad=10)  # <-- extra space inside the axes
    for t in g.ax_heatmap.get_yticklabels():
        try: t.set_fontstyle("italic")
        except Exception: pass

    # X labels compact & subsampled
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xmajorticklabels(),
        rotation=90, ha="center", va="top", fontsize=8.5
    )
    _subsample_xticks(g.ax_heatmap, max_labels=25)

    g.ax_heatmap.set_xlabel(f"{label_A} ↔ {label_B}")
    g.ax_heatmap.set_ylabel(rank.capitalize())
    g.ax_heatmap.yaxis.labelpad = 18

    # Make the col_colors band very thin and just above the heatmap
    _shrink_col_colors_axis(g, target_rel_height=0.012)

    # >>> more generous right margin for long names <<<
    maxlen = max((len(s) for s in Z.index), default=12)
    # start at 0.88 and grow ~0.016 per extra char beyond 12
    right = min(0.995, 0.88 + 0.016 * max(0, maxlen - 12))
    g.fig.subplots_adjust(left=0.07, right=right, bottom=0.20, top=0.95)

    # save with a bit more padding so the renderer never clips the far right
    g.fig.savefig(out_png, dpi=380, bbox_inches="tight", pad_inches=0.70)
    plt.close(g.fig)

def group_contrasts_and_csv(oc_pct, fc_pct, oh_pct, fh_pct, rank, outdir, palette=None,
                            n=15, prevalence_thresh=0.10, presence_threshold_percent=0.1,
                            use_log1p=True, clip_quantile=0.98, col_cluster_pairs=False):
    """
    Build:
      - heatmap_pairs_crohn_{rank}.png
      - heatmap_pairs_healthy_{rank}.png
      - heatmap_group_means_{rank}.png + group_means_{rank}.csv
      - heatmap_group_deltas_{rank}.png + group_deltas_{rank}.csv
    """
    palette = palette or PALETTE
    os.makedirs(outdir, exist_ok=True)

    # Row selection: prevalence in both OC and FC; fallback to most abundant overall.
    prev_oc = (oc_pct > presence_threshold_percent).mean(axis=1)
    prev_fc = (fc_pct > presence_threshold_percent).mean(axis=1)
    crohn_union = oc_pct.add(fc_pct, fill_value=0)
    keep = (prev_oc >= prevalence_thresh) & (prev_fc >= prevalence_thresh)
    rows = crohn_union.loc[keep].sum(axis=1).sort_values(ascending=False).head(n).index
    if len(rows) < max(5, int(0.4*n)):
        rows = crohn_union.sum(axis=1).sort_values(ascending=False).head(n).index

    # (1) Crohn pairs
    pairs_heatmap(
        oc_pct, fc_pct, rows,
        "Crohn-Oral", "Crohn-Fecal", rank,
        os.path.join(outdir, f"heatmap_pairs_crohn_{rank}.png"),
        palette=palette, use_log1p=use_log1p, clip_quantile=clip_quantile,
        col_cluster=col_cluster_pairs
    )

    # (2) Healthy pairs
    pairs_heatmap(
        oh_pct, fh_pct, rows,
        "Healthy-Oral", "Healthy-Fecal", rank,
        os.path.join(outdir, f"heatmap_pairs_healthy_{rank}.png"),
        palette=palette, use_log1p=use_log1p, clip_quantile=clip_quantile,
        col_cluster=col_cluster_pairs
    )

    # (3) Group means/deltas (+ CSV)
    means_df = pd.DataFrame({
        "O.C.": oc_pct.reindex(rows).fillna(0).mean(axis=1),
        "F.C.": fc_pct.reindex(rows).fillna(0).mean(axis=1),
        "O.H.": oh_pct.reindex(rows).fillna(0).mean(axis=1),
        "F.H.": fh_pct.reindex(rows).fillna(0).mean(axis=1),
    }).fillna(0.0)
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
    fig_w = 10.5
    plt.figure(figsize=(fig_w, fig_h))
    cmap_means = make_sequential_from("Healthy-Fecal", n=256)
    ax = sns.heatmap(Zm[["O.C.", "F.C.", "O.H.", "F.H."]],
                     cmap=cmap_means,
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
    cmap_div = make_oc_fc_cmap(PALETTE)
    vmax = np.nanquantile(np.abs(Zd.values), 0.98) or 1.0
    ax = sns.heatmap(Zd[["O.C.–O.H.", "F.C.–F.H."]],
                     cmap=cmap_div, center=0, vmin=-vmax, vmax=+vmax,
                     cbar_kws={"label": "Row z-score of deltas"})
    ax.set_xlabel("Comparisons"); ax.set_ylabel(rank.capitalize()); ax.set_title("Group deltas (O.C.–O.H., F.C.–F.H.)")

    _apply_smart_yticklabels(ax, [prettify_taxon(t, rank) for t in Zd.index], rank, base_fs=12.5)
    plt.gcf().subplots_adjust(left=0.34)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"heatmap_group_deltas_{rank}.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ---------- DA statistics ----------
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
    """
    log2(mean(a)+eps) - log2(mean(b)+eps) for non-negative data.
    If negative values present (e.g., CLR), returns (mean(a)-mean(b))/ln(2).
    """
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

    # Side "mini tables" (optional light summary)
    y = 0.98
    ax_side.scatter([0.03],[y], s=45, color=c_up, transform=ax_side.transAxes)
    ax_side.text(0.07, y, f"q ≤ {q_sig} & up", fontsize=10, va="center", transform=ax_side.transAxes)
    ax_side.scatter([0.53],[y], s=45, color=c_dn, transform=ax_side.transAxes)
    ax_side.text(0.57, y, f"q ≤ {q_sig} & down", fontsize=10, va="center", transform=ax_side.transAxes)

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

def run_da_paired_OC_FC(mat_taxa_samples_OC: pd.DataFrame,
                        mat_taxa_samples_FC: pd.DataFrame,
                        matched_df: pd.DataFrame, tax_labels, out_csv, rank="genus",
                        min_pairs:int=8):
    """
    Paired OC↔FC Wilcoxon with robust ID matching.
    - mat_taxa_samples_OC / FC must be taxa × samples (percent table).
    - matched_df: CSV with two columns (or with names like 'oral/oc' vs 'fecal/fc').
    - Drops pairs that cannot be matched to actual columns.
    - Writes an empty table (with header) if pairs < min_pairs.
    """
    # Ensure taxa × samples orientation
    if mat_taxa_samples_OC.shape[0] < mat_taxa_samples_OC.shape[1] and \
       mat_taxa_samples_OC.columns.to_series().astype(str).str.contains("s__|g__", regex=True).any():
        # looks like samples × taxa → transpose
        mat_taxa_samples_OC = mat_taxa_samples_OC.T
    if mat_taxa_samples_FC.shape[0] < mat_taxa_samples_FC.shape[1] and \
       mat_taxa_samples_FC.columns.to_series().astype(str).str.contains("s__|g__", regex=True).any():
        mat_taxa_samples_FC = mat_taxa_samples_FC.T

    # Identify oral/fecal columns in the match file
    oc_col, fc_col = _guess_pair_columns(matched_df)
    oc_raw = matched_df[oc_col].dropna().astype(str).tolist()
    fc_raw = matched_df[fc_col].dropna().astype(str).tolist()

    # Keep only rows with both sides present
    n_before = len(oc_raw)
    pairs = [(o,f) for o,f in zip(oc_raw, fc_raw) if str(o).strip()!="" and str(f).strip()!=""]
    if not pairs:
        empty_cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median","n_pairs"]
        pd.DataFrame(columns=empty_cols).to_csv(out_csv, index=False)
        warnings.warn("[paired] match file has no valid rows.")
        return

    oc_list, fc_list = zip(*pairs)
    # Map to actual matrix column names
    oc_map = _map_ids_to_columns(list(oc_list), mat_taxa_samples_OC.columns)
    fc_map = _map_ids_to_columns(list(fc_list), mat_taxa_samples_FC.columns)

    # Drop pairs with missing mapping
    good = [(o,f) for o,f in zip(oc_map, fc_map) if (o is not None and f is not None)]
    n_drop = len(oc_map) - len(good)
    if n_drop > 0:
        warnings.warn(f"[paired] dropped {n_drop} pairs that didn't match any column.")

    if len(good) < min_pairs:
        empty_cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median","n_pairs"]
        pd.DataFrame(columns=empty_cols).to_csv(out_csv, index=False)
        warnings.warn(f"[paired] not enough matched pairs (got {len(good)}, need ≥{min_pairs}).")
        return

    oc_cols = [o for o,_ in good]
    fc_cols = [f for _,f in good]

    rows, pvals = [], []
    for tax in tax_labels:
        if tax not in mat_taxa_samples_OC.index or tax not in mat_taxa_samples_FC.index:
            continue
        x = mat_taxa_samples_OC.loc[tax, oc_cols].astype(float).values
        y = mat_taxa_samples_FC.loc[tax, fc_cols].astype(float).values

        # If all differences are zero/NaN, Wilcoxon can fail → guard
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
            # rank-biserial r
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
            "n_pairs": len(good),
        })
        pvals.append(p)

    # BH-FDR
    if pvals:
        qvals = bh_qvalues(pvals)
        for r,q in zip(rows, qvals):
            r["q"] = float(q)

    cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median","n_pairs"]
    pd.DataFrame(rows)[cols].to_csv(out_csv, index=False)

# ---------- Stacked bar (two bars) ----------
def two_bar_stacked_generic(a_mean: pd.Series, b_mean: pd.Series, rank: str, out_png: str,
                            label_a: str, label_b: str, top_k: int = 15):
    """
    Two stacked bars (A vs B) with taxa colors derived from your brand palette.
    - Taxa colors come from build_taxa_color_cycle() to keep brand consistency.
    - Bar edges are lightly tinted with the group color.
    """
    combined = a_mean.add(b_mean, fill_value=0).sort_values(ascending=False)
    top = combined.index[:min(top_k, len(combined))]
    s_a = a_mean.reindex(top).fillna(0)
    s_b = b_mean.reindex(top).fillna(0)
    s_a = pd.concat([s_a, pd.Series({"Other": a_mean.drop(top, errors='ignore').sum()})])
    s_b = pd.concat([s_b, pd.Series({"Other": b_mean.drop(top, errors='ignore').sum()})])

    order = s_a.sort_values(ascending=False).index

    # taxa color cycle derived from brand colors
    taxa_colors = build_taxa_color_cycle(max(3*len(order), 12))
    color_map = {tax: taxa_colors[i % len(taxa_colors)] for i, tax in enumerate(order)}
    color_map["Other"] = PALETTE.get("Other", "#999999")

    # choose group edge/border colors
    edge_a = PALETTE.get("Oral", PALETTE.get("Crohn-Oral", "#B499E5")) if "Oral" in label_a else \
             PALETTE.get("Fecal", PALETTE.get("Crohn-Fecal", "#78688E"))
    edge_b = PALETTE.get("Oral", PALETTE.get("Healthy-Oral", "#7FB3D5")) if "Oral" in label_b else \
             PALETTE.get("Fecal", PALETTE.get("Healthy-Fecal", "#416388"))

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.array([0, 1]); b0 = b1 = 0.0
    for tax in order:
        h0 = float(s_a.loc[tax]); h1 = float(s_b.loc[tax])
        ax.bar(x[0], h0, bottom=b0, color=color_map[tax], width=0.6, edgecolor=edge_a, linewidth=0.7)
        ax.bar(x[1], h1, bottom=b1, color=color_map[tax], width=0.6, edgecolor=edge_b, linewidth=0.7)
        b0 += h0; b1 += h1

    ax.set_xticks(x); ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel("Mean relative abundance (%)")
    ax.set_title(f"{label_a} vs {label_b} — Top {len(top)} {rank.capitalize()} (others → Other)")
    handles = [plt.Rectangle((0,0),1,1,color=color_map[t]) for t in order]
    labels = [prettify_taxon(t, rank) if t!="Other" else "Other" for t in order]
    leg = ax.legend(handles, labels, bbox_to_anchor=(1.02,1), loc="upper left", title=rank.capitalize())
    for txt in leg.get_texts():
        if txt.get_text() != "Other":
            try: txt.set_style("italic")
            except Exception: pass

    # axis spines and tick colors consistent with theme
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- ML outputs ----------
def tidy_from_percent(pct: pd.DataFrame, group_label: str, site_label: str, rank: str) -> pd.DataFrame:
    """
    Build tidy long table from percent matrix (taxa × samples).
    Columns: Sample, Taxon, AbundancePct, Group, Site
    """
    df = pct.copy()
    tidy = df.stack().reset_index()
    tidy.columns = [rank, "Sample", "AbundancePct"]
    tidy["Taxon"] = tidy[rank].map(lambda t: prettify_taxon(t, rank))
    tidy["Group"] = group_label
    tidy["Site"]  = site_label
    return tidy[["Sample","Taxon","AbundancePct","Group","Site"]]

def wide_from_percent(pct: pd.DataFrame, group: str, site: str, rank: str) -> pd.DataFrame:
    """
    Build wide table (samples × taxa) and append Group/Site columns.
    """
    df = pct.T.copy()  # samples × taxa
    df["Group"] = group
    df["Site"]  = site
    return df.reset_index().rename(columns={"index":"Sample"})

def save_topk_wide(wide_df, out_csv, topk=15, meta_cols=("Sample","Group","Site")):
    """
    Make a 'topK' wide matrix by selecting the K most abundant taxa across all samples.
    """
    meta_cols_present = [c for c in meta_cols if c in wide_df.columns]
    feat_df = wide_df.drop(columns=meta_cols_present, errors="ignore")
    # Rank taxa by total abundance across samples
    top_taxa = (feat_df.sum(axis=0)
                .sort_values(ascending=False)
                .head(int(topk)).index)
    out_df = pd.concat([wide_df.loc[:, top_taxa], wide_df.loc[:, meta_cols_present]], axis=1)
    out_df.to_csv(out_csv, index=False)

def ml_clr_from_percent(pct_all_sites: pd.DataFrame, group_map: dict, site_map: dict,
                        rank: str, pseudocount=1e-6) -> pd.DataFrame:
    """
    Build CLR-based wide table for ML:
    - Input: percent table taxa × samples across (merged) sites/groups
    - Output: samples × taxa (CLR), with Group/Site columns.
    """
    clr_mat = clr_transform(pct_all_sites, pseudocount=pseudocount)  # taxa × samples (CLR)
    wide = clr_mat.T.copy()
    # add meta
    meta = pd.DataFrame({
        "Sample": wide.index,
        "Group": [group_map.get(s, None) for s in wide.index],
        "Site":  [site_map.get(s, None) for s in wide.index],
    }).set_index("Sample")
    wide = pd.concat([wide, meta], axis=1).reset_index().rename(columns={"index":"Sample"})
    return wide


# ---------- Main rank block ----------
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

    # Pairs & group contrasts
    group_contrasts_and_csv(
        oc_pct, fc_pct, oh_pct, fh_pct, rank, outdir_rank,
        n=topk, prevalence_thresh=prevalence, presence_threshold_percent=presence_pct,
        use_log1p=True, clip_quantile=0.98, col_cluster_pairs=False
    )

    # Two-bar stacked (Crohn vs Healthy within each site)
    two_bar_stacked_generic(
        group_mean_percent(oc, rank_fn), group_mean_percent(oh, rank_fn),
        rank=rank, out_png=os.path.join(outdir_rank, f"barplot_oral_{rank}_top{topk}.png"),
        label_a="Oral_Crohn", label_b="Oral_Healthy", top_k=topk
    )
    two_bar_stacked_generic(
        group_mean_percent(fc, rank_fn), group_mean_percent(fh, rank_fn),
        rank=rank, out_png=os.path.join(outdir_rank, f"barplot_fecal_{rank}_top{topk}.png"),
        label_a="Fecal_Crohn", label_b="Fecal_Healthy", top_k=topk
    )

    # DA — union of taxa across groups
    all_taxa = pd.Index(oc_pct.index).union(fc_pct.index).union(oh_pct.index).union(fh_pct.index)

    # Oral CH
    df_oral = run_da_oral_CH(
        mat_taxa_samples=oc_pct.reindex(all_taxa).fillna(0).join(
            oh_pct.reindex(all_taxa).fillna(0), how="outer"
        ),
        labels_A=oc_pct.columns, labels_B=oh_pct.columns, tax_labels=all_taxa,
        out_csv=os.path.join(outdir_rank, f"da_oral_CH_{rank}.csv"),
        do_volcano_png=os.path.join(outdir_rank, f"volcano_oral_CH_{rank}.png"),
        rank=rank
    )
    # Fecal CH
    df_fecal = run_da_fecal_CH(
        mat_taxa_samples=fc_pct.reindex(all_taxa).fillna(0).join(
            fh_pct.reindex(all_taxa).fillna(0), how="outer"
        ),
        labels_A=fc_pct.columns, labels_B=fh_pct.columns, tax_labels=all_taxa,
        out_csv=os.path.join(outdir_rank, f"da_fecal_CH_{rank}.csv"),
        do_volcano_png=os.path.join(outdir_rank, f"volcano_fecal_CH_{rank}.png"),
        rank=rank
    )

    # Paired OC↔FC (Wilcoxon) — always create output path
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
                pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median"]).to_csv(paired_out, index=False)
        else:
            pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median"]).to_csv(paired_out, index=False)
    except Exception as e:
        warnings.warn(f"[WARNING] [paired] percent failed: \"{e}\"")
        pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median"]).to_csv(paired_out, index=False)

    # ML outputs
    make_ml_outputs(rank, oc_pct, oh_pct, fc_pct, fh_pct, outdir_rank, topk=topk)


def make_ml_outputs(rank: str, oc_pct: pd.DataFrame, oh_pct: pd.DataFrame,
                    fc_pct: pd.DataFrame, fh_pct: pd.DataFrame,
                    outdir_rank: str, topk: int = 15):
    """
    Write ML-friendly outputs:
        - ml_{rank}_tidy.csv
        - ml_{rank}_wide_all.csv
        - ml_{rank}_wide_topK.csv
        - ml_{rank}_wide_clr.csv
    """
    # tidy per group
    tidy_oc = tidy_from_percent(oc_pct, "Crohn",   "Oral",  rank)
    tidy_oh = tidy_from_percent(oh_pct, "Healthy", "Oral",  rank)
    tidy_fc = tidy_from_percent(fc_pct, "Crohn",   "Fecal", rank)
    tidy_fh = tidy_from_percent(fh_pct, "Healthy", "Fecal", rank)
    tidy_all = pd.concat([tidy_oc, tidy_oh, tidy_fc, tidy_fh], ignore_index=True)
    tidy_all.to_csv(os.path.join(outdir_rank, f"ml_{rank}_tidy.csv"), index=False)

    # wide per group (samples × taxa) + Group/Site
    wide_oc = wide_from_percent(oc_pct, "Crohn",   "Oral",  rank)
    wide_oh = wide_from_percent(oh_pct, "Healthy", "Oral",  rank)
    wide_fc = wide_from_percent(fc_pct, "Crohn",   "Fecal", rank)
    wide_fh = wide_from_percent(fh_pct, "Healthy", "Fecal", rank)

    # Align feature columns across groups (outer-join by columns) and fill missing with 0
    # Keep meta columns at the end for consistency.
    meta_cols = ["Sample", "Group", "Site"]
    feature_cols = sorted(set(wide_oc.columns) | set(wide_oh.columns) |
                          set(wide_fc.columns) | set(wide_fh.columns))
    feature_cols = [c for c in feature_cols if c not in meta_cols]

    def _align(df):
        # Reindex feature columns, fill NaNs with 0, then append meta cols
        a = df.reindex(columns=feature_cols, fill_value=0)
        for m in meta_cols:
            if m in df.columns:
                a[m] = df[m].values
            else:
                a[m] = None
        return a[feature_cols + meta_cols]

    wide_all = pd.concat([
        _align(wide_oc), _align(wide_oh), _align(wide_fc), _align(wide_fh)
    ], ignore_index=True)

    # Save "all features" matrix
    wide_all_out = os.path.join(outdir_rank, f"ml_{rank}_wide_all.csv")
    wide_all.to_csv(wide_all_out, index=False)

    # Save "topK" features matrix (by total abundance across all samples)
    wide_topk_out = os.path.join(outdir_rank, f"ml_{rank}_wide_topK.csv")
    save_topk_wide(wide_all.copy(), wide_topk_out, topk=topk, meta_cols=meta_cols)

    # Build CLR matrix across all sites/groups
    # 1) Merge all percent tables on taxa (union) → taxa × samples
    all_taxa = oc_pct.index.union(oh_pct.index).union(fc_pct.index).union(fh_pct.index)
    pct_all = pd.concat([
        oc_pct.reindex(all_taxa).fillna(0),
        oh_pct.reindex(all_taxa).fillna(0),
        fc_pct.reindex(all_taxa).fillna(0),
        fh_pct.reindex(all_taxa).fillna(0)
    ], axis=1)

    # 2) Build sample → group/site maps for meta columns
    group_map = {}
    site_map  = {}
    for s in oc_pct.columns: group_map[s] = "Crohn";   site_map[s] = "Oral"
    for s in oh_pct.columns: group_map[s] = "Healthy"; site_map[s] = "Oral"
    for s in fc_pct.columns: group_map[s] = "Crohn";   site_map[s] = "Fecal"
    for s in fh_pct.columns: group_map[s] = "Healthy"; site_map[s] = "Fecal"

    # 3) CLR transform and attach meta
    wide_clr = ml_clr_from_percent(pct_all, group_map, site_map, rank=rank, pseudocount=1e-6)
    wide_clr_out = os.path.join(outdir_rank, f"ml_{rank}_wide_clr.csv")
    wide_clr.to_csv(wide_clr_out, index=False)


# ---------- CLI / main ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Unified genus/species comparison pipeline (Crohn vs Healthy; Oral & Fecal)"
    )
    p.add_argument("--oral-crohn",    required=True, help="CSV abundance table for Crohn oral (taxa×samples or samples×taxa).")
    p.add_argument("--oral-healthy",  required=True, help="CSV abundance table for Healthy oral.")
    p.add_argument("--fecal-crohn",   required=True, help="CSV abundance table for Crohn fecal.")
    p.add_argument("--fecal-healthy", required=True, help="CSV abundance table for Healthy fecal.")
    p.add_argument("--matched-ids",   default=None,  help="Optional CSV with two columns: <Oral_col>,<Fecal_col> (paired IDs).")
    p.add_argument("--outdir",        required=True, help="Output directory (rank subfolders will be created).")
    p.add_argument("--topk",          type=int, default=15, help="Top-K taxa for plots and ML topK tables (default: 15).")
    p.add_argument("--presence-pct",  type=float, default=0.1, help="Presence threshold in percent for prevalence (default: 0.1).")
    p.add_argument("--prevalence",    type=float, default=0.10, help="Prevalence fraction threshold (default: 0.10).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load input matrices
    oc_raw = pd.read_csv(args.oral_crohn,    index_col=0)
    oh_raw = pd.read_csv(args.oral_healthy,  index_col=0)
    fc_raw = pd.read_csv(args.fecal_crohn,   index_col=0)
    fh_raw = pd.read_csv(args.fecal_healthy, index_col=0)

    # Process genus
    outdir_genus = os.path.join(args.outdir, "genus")
    process_rank(
        "genus", oc_raw, oh_raw, fc_raw, fh_raw, outdir_genus,
        topk=args.topk, matched_path=args.matched_ids,
        presence_pct=args.presence_pct, prevalence=args.prevalence
    )
    print("[INFO] [genus] done →", outdir_genus)

    # Process species
    outdir_species = os.path.join(args.outdir, "species")
    process_rank(
        "species", oc_raw, oh_raw, fc_raw, fh_raw, outdir_species,
        topk=args.topk, matched_path=args.matched_ids,
        presence_pct=args.presence_pct, prevalence=args.prevalence
    )
    print("[INFO] [species] done →", outdir_species)

    print("[INFO] All done.")


if __name__ == "__main__":
    main()

