#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved filtering + QC for microbiome abundance tables (Snakemake-only).

Key improvements:
- Detect input scale (percent vs fraction) and adapt abundance threshold.
- Collapse to a single taxonomic level FIRST (default: genus), then filter.
- Aggregation for duplicate rows at the chosen level = MEAN (not sum).
- Site-wise feature alignment (Crohn vs Healthy) via union, if enabled.
- Smarter QC plots: log-y for prevalence-mean; optional mean-on-positives.
- Optional CLR emission (off by default).

Outputs (same names as before):
  data/filtered/oral_abund_crohn_filtered_normalized.csv
  data/filtered/fecal_abund_crohn_filtered_normalized.csv
  data/filtered/oral_abund_healthy_filtered_normalized.csv
  data/filtered/fecal_abund_healthy_filtered_normalized.csv
  results/filtering/filtering_report.txt
  results/filtering/filtering_barplot_allgroups.png
  + standard QC figs in results/filtering/
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------- utils --------------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

def detect_scale(df: pd.DataFrame) -> str:
    """
    Decide if values are percent (0..100) or fraction (0..1).
    Heuristics:
      - if median of column sums > 10  -> 'percent' (fits MetaPhlAn multi-rank)
      - else                           -> 'fraction'
    """
    if df.shape[1] == 0:
        return "percent"
    colsum_med = float(df.sum(axis=0).median())
    return "percent" if colsum_med > 10 else "fraction"

# -------------- taxonomy helpers --------------
def _has(token_list, prefix): return any(t.startswith(prefix) for t in token_list)
def _last(token_list, prefix):
    vals = [t for t in token_list if t.startswith(prefix)]
    return vals[-1] if vals else None

def collapse_to_level(df0: pd.DataFrame, level: str = "genus", agg: str = "mean") -> pd.DataFrame:
    """
    Keep exactly one taxonomic level from MetaPhlAn lineage strings.
    Rules:
      - genus: rows that include g__ and do NOT include s__/t__ (deepest==genus).
      - species: rows that include s__ and do NOT include t__ (deepest==species).
    Then aggregate duplicate keys with MEAN (or SUM if requested).
    """
    out_rows = {}
    for name, row in df0.iterrows():
        parts = str(name).split("|")
        g = _last(parts, "g__")
        s = _last(parts, "s__")
        t = _last(parts, "t__")
        if level == "genus":
            # only deepest genus-level rows: has g__ and no s__/t__
            if g is not None and (s is None) and (t is None):
                key = g
            else:
                continue
        elif level == "species":
            # only deepest species-level rows: has s__ and no t__
            if s is not None and (t is None):
                key = (g + "|" + s) if g is not None else s
            else:
                continue
        else:
            raise ValueError(f"Unsupported level: {level}")
        accum = out_rows.get(key)
        out_rows[key] = row.values if accum is None else np.vstack([accum, row.values])
    if not out_rows:
        return pd.DataFrame(index=[], columns=df0.columns, dtype=float)

    # aggregate
    keys = []
    mats = []
    for k, v in out_rows.items():
        if v.ndim == 1:
            vec = v.astype(float)
        else:
            if agg == "sum":
                vec = v.astype(float).sum(axis=0)
            else:
                vec = v.astype(float).mean(axis=0)
        keys.append(k); mats.append(vec)
    out = pd.DataFrame(mats, index=keys, columns=df0.columns)
    # drop all-zero taxa
    return out.loc[out.sum(axis=1) > 0]

# -------------- stats helpers --------------
def zeros_by_sample(df):  # frac zeros per sample
    if df.shape[0] == 0: return pd.Series(dtype=float)
    return (df == 0).sum(axis=0) / df.shape[0]

def zeros_by_taxon(df):
    if df.shape[1] == 0: return pd.Series(dtype=float)
    return (df == 0).sum(axis=1) / df.shape[1]

def normalize_pct(df):
    if df.shape[1] == 0: return df
    pct = df.div(df.sum(axis=0), axis=1) * 100.0
    return pct.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all").dropna(axis=1, how="all")

def transform_clr_from_rel(df_rel, pseudocount=1e-6):
    if df_rel.shape[1] == 0:
        return df_rel
    X = df_rel + pseudocount
    logX = np.log(X)
    clr = logX.sub(logX.mean(axis=0), axis=1)
    clr = clr.replace([np.inf, -np.inf], np.nan)
    clr = clr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return clr

# -------------- filtering --------------
def remove_low_abundance(df, thr, scale):
    """thr given in PERCENT if scale=='percent', in FRACTION if 'fraction'."""
    if df.shape[1] == 0: return df
    if scale == "fraction":
        threshold = thr
    else:
        threshold = thr  # already percent
    # Work on the chosen level table; keep taxa whose max >= threshold
    return df[df.max(axis=1) >= threshold]

def remove_low_prevalence(df, prev):
    if df.shape[1] == 0: return df
    p = (df > 0).sum(axis=1) / df.shape[1]
    return df[p >= prev]

# -------------- plotting --------------
def _safe_kde_or_hist(ax, series, label, color):
    s = pd.Series(series).dropna().astype(float)
    n = int(s.size)
    if n >= 2 and s.std(ddof=1) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.plot(kind="kde", label=label, ax=ax, color=color)
    elif n >= 1:
        ax.hist(s, bins=min(10, max(1, n)), alpha=0.35, label=f"{label} (hist)", color=color)
    else:
        ax.plot([], [], label=f"{label} (n=0)", color=color)

def prevalence_mean_scatter(ax, df_before, df_after, label, color, prevalence_thr, abundance_thr, scale, mean_on_positives=False):
    def _prev_mean(df):
        if df.shape[1]==0:
            return pd.DataFrame(columns=["prev","mean"])
        prev = (df>0).sum(axis=1)/max(1,df.shape[1])
        if mean_on_positives:
            means = df.replace(0,np.nan).mean(axis=1).fillna(0.0)
        else:
            means = df.mean(axis=1)
        return pd.DataFrame({"prev":prev, "mean":means})

    pm_b = _prev_mean(df_before); pm_a = _prev_mean(df_after)
    ax.scatter(pm_b["prev"], pm_b["mean"], s=10, alpha=0.25, color=color, label=f"{label} (before)")
    ax.scatter(pm_a["prev"], pm_a["mean"], s=12, alpha=0.85, color=color, label=f"{label} (after)", marker="x")
    ax.axvline(prevalence_thr, ls="--", lw=1, color="#666666")
    ax.axhline(abundance_thr,  ls="--", lw=1, color="#666666")
    ax.set_xlabel("Prevalence (fraction of samples)")
    unit = "%" if scale == "percent" else "fraction"
    ax.set_ylabel(f"Mean abundance ({unit})")
    ax.set_yscale("log")
    ax.legend()

# -------------- per-group processing --------------
def process_group(name, inp_path, prevalence, abundance_thr, level, agg, pseudocount, emit_clr, scale):
    # load
    df_raw = load_data(inp_path)

    # collapse FIRST
    df_lvl = collapse_to_level(df_raw, level=level, agg=agg)

    # keep a copy before filtering for QC
    df_before = df_lvl.copy()

    # filtering order: prevalence -> abundance (safer)
    df1 = remove_low_prevalence(df_lvl, prevalence)
    df2 = remove_low_abundance(df1, abundance_thr, scale)

    # export normalized % and optional CLR
    df_pct = normalize_pct(df2)
    df_rel = df2.div(df2.sum(axis=0), axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if emit_clr:
        df_clr = transform_clr_from_rel(df_rel, pseudocount=pseudocount)
    else:
        df_clr = None

    qc = {
        "df_raw": df_raw,        # multi-level raw
        "df_before": df_before,  # level table before filtering
        "df_after": df2,         # level table after filtering
        "z_s_before": zeros_by_sample(df_before),
        "z_s_after":  zeros_by_sample(df2),
        "z_t_before": zeros_by_taxon(df_before),
        "z_t_after":  zeros_by_taxon(df2),
        "retained_pct": (df2.sum(axis=0) / df_before.sum(axis=0).replace(0,np.nan) * 100.0).fillna(0.0)
    }
    return df_pct, df_clr, qc

# -------------- Snakemake entry --------------
if "snakemake" not in globals():
    raise RuntimeError("This script must be executed via Snakemake (no CLI supported).")

oral_in          = snakemake.input["oral"]           # noqa: F821
fecal_in         = snakemake.input["fecal"]          # noqa: F821
healthy_oral_in  = snakemake.input["healthy_oral"]   # noqa: F821
healthy_fecal_in = snakemake.input["healthy_fecal"]  # noqa: F821

oral_out_pct          = snakemake.output["oral_norm"]           # noqa: F821
fecal_out_pct         = snakemake.output["fecal_norm"]          # noqa: F821
healthy_oral_out_pct  = snakemake.output["healthy_oral_norm"]   # noqa: F821
healthy_fecal_out_pct = snakemake.output["healthy_fecal_norm"]  # noqa: F821

# optional CLR sidecars
oral_out_clr          = oral_out_pct.replace(".csv", "_clr.csv")
fecal_out_clr         = fecal_out_pct.replace(".csv", "_clr.csv")
healthy_oral_out_clr  = healthy_oral_out_pct.replace(".csv", "_clr.csv")
healthy_fecal_out_clr = healthy_fecal_out_pct.replace(".csv", "_clr.csv")

report_path = snakemake.output["report"]    # noqa: F821
barplot_png = snakemake.output["barplot"]   # noqa: F821

# params (with sensible defaults)
prevalence_default = float(snakemake.params.get("prevalence", 0.20))         # fallback
prevalence_oral    = float(snakemake.params.get("prevalence_oral", prevalence_default))
prevalence_fecal   = float(snakemake.params.get("prevalence_fecal", prevalence_default))
abundance_param    = float(snakemake.params.get("abundance", 0.1))           # interpreted by scale
level_mode         = str(snakemake.params.get("level", "genus")).lower()     # "genus"|"species"
agg_mode           = str(snakemake.params.get("agg", "mean")).lower()        # "mean"|"sum"
pseudocount        = float(snakemake.params.get("pseudocount", 1e-6))
qc_level           = str(snakemake.params.get("qc_level", "lite")).lower()   # "lite"|"full"
align_by_site      = bool(snakemake.params.get("align_by_site", True))
emit_clr           = bool(snakemake.params.get("emit_clr", False))
mean_on_positives  = bool(snakemake.params.get("mean_on_positives", False))

# Load once to detect scale on each group (they should match; but detect per file anyway)
_oral_df0  = load_data(oral_in)
_fecal_df0 = load_data(fecal_in)
scale_oral  = detect_scale(_oral_df0)    # expect 'percent'
scale_fecal = detect_scale(_fecal_df0)

# abundance threshold is interpreted in the detected scale
abundance_thr_oral  = abundance_param if scale_oral  == "percent" else abundance_param / 100.0
abundance_thr_fecal = abundance_param if scale_fecal == "percent" else abundance_param / 100.0

# process all four groups
oral_pct, oral_clr, qc_crohn_oral = process_group("Crohn-Oral", oral_in,
    prevalence_oral, abundance_thr_oral, level_mode, agg_mode, pseudocount, emit_clr, scale_oral)

fecal_pct, fecal_clr, qc_crohn_fecal = process_group("Crohn-Fecal", fecal_in,
    prevalence_fecal, abundance_thr_fecal, level_mode, agg_mode, pseudocount, emit_clr, scale_fecal)

h_oral_pct, h_oral_clr, qc_healthy_oral = process_group("Healthy-Oral", healthy_oral_in,
    prevalence_oral, abundance_thr_oral, level_mode, agg_mode, pseudocount, emit_clr, scale_oral)

h_fecal_pct, h_fecal_clr, qc_healthy_fecal = process_group("Healthy-Fecal", healthy_fecal_in,
    prevalence_fecal, abundance_thr_fecal, level_mode, agg_mode, pseudocount, emit_clr, scale_fecal)

# align feature sets by site (union)
if align_by_site:
    # ORAL
    union_oral = sorted(set(oral_pct.index) | set(h_oral_pct.index))
    oral_pct   = oral_pct.reindex(union_oral).fillna(0.0)
    h_oral_pct = h_oral_pct.reindex(union_oral).fillna(0.0)
    # FECAL
    union_fecal = sorted(set(fecal_pct.index) | set(h_fecal_pct.index))
    fecal_pct   = fecal_pct.reindex(union_fecal).fillna(0.0)
    h_fecal_pct = h_fecal_pct.reindex(union_fecal).fillna(0.0)

# write normalized outputs
ensure_dir(oral_out_pct);         oral_pct.to_csv(oral_out_pct)
ensure_dir(fecal_out_pct);        fecal_pct.to_csv(fecal_out_pct)
ensure_dir(healthy_oral_out_pct); h_oral_pct.to_csv(healthy_oral_out_pct)
ensure_dir(healthy_fecal_out_pct);h_fecal_pct.to_csv(healthy_fecal_out_pct)

# (optional) CLR sidecars
if emit_clr:
    oral_clr.to_csv(oral_out_clr)
    fecal_clr.to_csv(fecal_out_clr)
    h_oral_clr.to_csv(healthy_oral_out_clr)
    h_fecal_clr.to_csv(healthy_fecal_out_clr)

# textual report (steps condensed since collapse-first)
ensure_dir(report_path)
with open(report_path, "w") as f:
    def _w(title, qc):
        f.write(f"[{title}]\n")
        f.write(f"scale={detect_scale(qc['df_raw'])}, level={level_mode}, agg={agg_mode}\n")
        f.write(f"rows_before_level={qc['df_before'].shape[0]}, rows_after_filter={qc['df_after'].shape[0]}\n\n")
    _w("Crohn-Oral", qc_crohn_oral)
    _w("Crohn-Fecal", qc_crohn_fecal)
    _w("Healthy-Oral", qc_healthy_oral)
    _w("Healthy-Fecal", qc_healthy_fecal)

# aggregated barplot (counts before/after per group)
ensure_dir(barplot_png)
labels = ["Before(level)", "After"]
width  = 0.20
groups = ["Crohn-Fecal","Crohn-Oral","Healthy-Oral","Healthy-Fecal"]
counts_before = [
    qc_crohn_fecal["df_before"].shape[0],
    qc_crohn_oral["df_before"].shape[0],
    qc_healthy_oral["df_before"].shape[0],
    qc_healthy_fecal["df_before"].shape[0],
]
counts_after = [
    qc_crohn_fecal["df_after"].shape[0],
    qc_crohn_oral["df_after"].shape[0],
    qc_healthy_oral["df_after"].shape[0],
    qc_healthy_fecal["df_after"].shape[0],
]
plt.figure(figsize=(10,5))
x = np.arange(len(groups))
plt.bar(x - width/2, counts_before, width=width, label="Before(level)")
plt.bar(x + width/2, counts_after,  width=width, label="After")
plt.xticks(x, groups); plt.ylabel("Number of taxa")
plt.title("Number of taxa before vs after filtering (level-collapsed)")
plt.legend(); plt.tight_layout(); plt.savefig(barplot_png); plt.close()

# Global QC (after)
outdir = os.path.join("results","filtering")
ensure_dir(os.path.join(outdir,"_dummy"))

# zero fraction by sample & taxon (AFTER)
fig, ax = plt.subplots(figsize=(7,5))
for lbl, qc in [("Crohn-Fecal", qc_crohn_fecal), ("Crohn-Oral", qc_crohn_oral),
                ("Healthy-Oral", qc_healthy_oral), ("Healthy-Fecal", qc_healthy_fecal)]:
    _safe_kde_or_hist(ax, qc["z_s_after"], lbl, None)
ax.set_xlabel("Zero fraction per sample"); ax.set_title("Zero-inflation (by sample)"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(outdir, "zero_fraction_by_sample.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(7,5))
for lbl, qc in [("Crohn-Fecal", qc_crohn_fecal), ("Crohn-Oral", qc_crohn_oral),
                ("Healthy-Oral", qc_healthy_oral), ("Healthy-Fecal", qc_healthy_fecal)]:
    _safe_kde_or_hist(ax, qc["z_t_after"], lbl, None)
ax.set_xlabel("Zero fraction per taxon"); ax.set_title("Zero-inflation (by taxon)"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(outdir, "zero_fraction_by_taxon.png")); plt.close(fig)

# retained mass (AFTER vs BEFORE at chosen level)
fig, ax = plt.subplots(figsize=(7,5))
for lbl, qc in [("Crohn-Fecal", qc_crohn_fecal), ("Crohn-Oral", qc_crohn_oral),
                ("Healthy-Oral", qc_healthy_oral), ("Healthy-Fecal", qc_healthy_fecal)]:
    _safe_kde_or_hist(ax, qc["retained_pct"], lbl, None)
ax.set_xlim(0, 100); ax.set_xlabel("Retained mass after filtering (%)")
ax.set_title("Retained mass distribution (per sample)"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(outdir, "library_size_distribution.png")); plt.close(fig)

# site-wise prevalence-mean (log-y)
def _prevmean_plot(site_name, qc1, qc2, outname):
    plt.figure(figsize=(7,5))
    ax = plt.gca()
    prevalence_mean_scatter(ax, qc1["df_before"], qc1["df_after"], "Crohn",  "#6b5b95",
                            prevalence_oral if site_name=="oral" else prevalence_fecal,
                            abundance_param, "percent", mean_on_positives)
    prevalence_mean_scatter(ax, qc2["df_before"], qc2["df_after"], "Healthy","#6497b1",
                            prevalence_oral if site_name=="oral" else prevalence_fecal,
                            abundance_param, "percent", mean_on_positives)
    ax.set_title(f"{site_name.capitalize()}: prevalence vs mean abundance (log-y)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, outname)); plt.close()

_prevmean_plot("oral",  qc_crohn_oral,  qc_healthy_oral,  "prevalence_mean_scatter_oral_combined.png")
_prevmean_plot("fecal", qc_crohn_fecal, qc_healthy_fecal, "prevalence_mean_scatter_fecal_combined.png")

# rank-abundance (mean across samples, log-y) at chosen level
def _rank_curve(df):
    if df.shape[0] == 0: return np.array([])
    v = df.mean(axis=1).sort_values(ascending=False).values
    return v[v > 0]

def _rank_plot(site_name, qc1, qc2, outname):
    plt.figure(figsize=(7,5))
    rb = _rank_curve(qc1["df_before"]); ra = _rank_curve(qc1["df_after"])
    hb = _rank_curve(qc2["df_before"]); ha = _rank_curve(qc2["df_after"])
    if rb.size: plt.plot(np.arange(1, rb.size+1), rb, lw=1.2, ls="--", label="Crohn (before)")
    if ra.size: plt.plot(np.arange(1, ra.size+1), ra, lw=1.6, ls="-",  label="Crohn (after)")
    if hb.size: plt.plot(np.arange(1, hb.size+1), hb, lw=1.2, ls="--", label="Healthy (before)")
    if ha.size: plt.plot(np.arange(1, ha.size+1), ha, lw=1.6, ls="-",  label="Healthy (after)")
    plt.xlabel("Rank"); plt.ylabel("Mean abundance (%)"); plt.yscale("log")
    plt.title(f"{site_name.capitalize()}: rank-abundance (mean across samples, {level_mode})")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir, outname)); plt.close()

_rank_plot("oral",  qc_crohn_oral,  qc_healthy_oral,  "rank_abundance_oral_combined.png")
_rank_plot("fecal", qc_crohn_fecal, qc_healthy_fecal, "rank_abundance_fecal_combined.png")

print(f"[filtering] Done (level={level_mode}, agg={agg_mode}, align_by_site={align_by_site}, emit_clr={emit_clr}).")
