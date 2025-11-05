#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtering & QC (pooled-mask per site) — includes optional merge of HMP Oral into Healthy Oral.

Canonical groups:
    Oral_Crohn, Fecal_Crohn, Oral_Healthy, Fecal_Healthy
"""

import os, io, tempfile
import warnings
from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# ---------------- paths / io utils ----------------
def write_text_atomic(path: str, text: str) -> None:
    """Atomically write text to path to avoid empty/partial files on NFS/HPC."""
    path = str(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=d, prefix=".tmp_", suffix=".txt") as tf:
        tf.write(text)
        tf.flush()
        os.fsync(tf.fileno())
        tmpname = tf.name
    os.replace(tmpname, path)


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def load_table_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

def detect_scale(df: pd.DataFrame) -> str:
    if df.shape[1] == 0:
        return "percent"
    med_sum = float(df.sum(axis=0).median())
    return "percent" if med_sum > 10 else "fraction"

def normalize_to_percent(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] == 0:
        return df.copy()
    colsum = df.sum(axis=0).replace(0, np.nan)
    pct = df.div(colsum, axis=1) * 100.0
    return pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ---------------- taxonomy collapsing ----------------
def _last_with_prefix(parts, prefix):
    hits = [p for p in parts if p.startswith(prefix)]
    return hits[-1] if hits else None

def collapse_to_level(df0: pd.DataFrame, level: str = "genus", agg: str = "mean") -> pd.DataFrame:
    out: Dict[str, np.ndarray] = {}
    for name, row in df0.iterrows():
        parts = str(name).split("|")
        g = _last_with_prefix(parts, "g__")
        s = _last_with_prefix(parts, "s__")
        t = _last_with_prefix(parts, "t__")

        if level == "genus":
            if (g is not None) and (s is None) and (t is None):
                key = g
            else:
                continue
        elif level == "species":
            if (s is not None) and (t is None):
                key = (g + "|" + s) if g is not None else s
            else:
                continue
        else:
            raise ValueError(f"Unsupported level: {level}")

        arr = row.values.astype(float)
        if key not in out:
            out[key] = arr
        else:
            out[key] = np.vstack([out[key], arr])

    if not out:
        return pd.DataFrame(index=[], columns=df0.columns, dtype=float)

    keys, mats = [], []
    for k, v in out.items():
        vec = v if v.ndim == 1 else (v.mean(axis=0) if agg == "mean" else v.sum(axis=0))
        keys.append(k); mats.append(vec)

    out_df = pd.DataFrame(mats, index=keys, columns=df0.columns)
    return out_df.loc[out_df.sum(axis=1) > 0]

# ---------------- pooled-mask core ----------------
def build_pooled_mask(df_c: pd.DataFrame, df_h: pd.DataFrame,
                      prevalence: float, abundance_param: float,
                      level: str, agg: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    c_lvl = collapse_to_level(df_c, level=level, agg=agg)
    h_lvl = collapse_to_level(df_h, level=level, agg=agg)
    pooled = pd.concat([c_lvl, h_lvl], axis=1).fillna(0.0)

    scale = detect_scale(pooled)
    abundance_thr = abundance_param if scale == "percent" else (abundance_param / 100.0)

    prev = (pooled > 0).sum(axis=1) / max(1, pooled.shape[1])
    max_abund = pooled.max(axis=1)
    mask = (prev >= prevalence) & (max_abund >= abundance_thr)

    c_filt = c_lvl.loc[mask].copy()
    h_filt = h_lvl.loc[mask].copy()

    qc = {
        "scale": scale,
        "rows_before_c": int(c_lvl.shape[0]),
        "rows_before_h": int(h_lvl.shape[0]),
        "rows_after": int(mask.sum()),
        "prevalence": prevalence,
        "abundance_param": abundance_param,
        "abundance_applied": abundance_thr,
    }
    return c_filt, h_filt, qc

# ---------------- colors / labels ----------------
CANONICAL = ["Fecal_Crohn", "Oral_Crohn", "Fecal_Healthy", "Oral_Healthy"]

def load_colors_yaml(path: str) -> Dict[str, str]:
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = {}
    group_colors = (cfg or {}).get("group", {}) or {}
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    palette = {}
    for i, canon in enumerate(CANONICAL):
        palette[canon] = group_colors.get(canon, default_cycle[i % len(default_cycle)] if default_cycle else "#999999")
    return palette

# ---------------- small helper ----------------
def merge_tables_outer_zero(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge by taxa (rows aligned), concat by columns, NaN->0."""
    keep = [d for d in dfs if d is not None and d.shape[1] > 0]
    if not keep:
        return pd.DataFrame()
    return pd.concat(keep, axis=1).fillna(0.0)

# ----------------------------- main (Snakemake) -----------------------------
if "snakemake" not in globals():
    raise RuntimeError("This script must be executed via Snakemake (no CLI supported).")

# Inputs
oral_in          = snakemake.input["oral"]           # noqa: F821
fecal_in         = snakemake.input["fecal"]          # noqa: F821
healthy_oral_in  = snakemake.input["healthy_oral"]   # noqa: F821
healthy_fecal_in = snakemake.input["healthy_fecal"]  # noqa: F821
hmp_oral_in      = snakemake.input.get("hmp_oral", None)  # noqa: F821

include_hmp_oral = bool(snakemake.params.get("include_hmp_oral", True))  # noqa: F821

# Outputs (CSV with clade_name)
oral_out_pct          = snakemake.output["oral_norm"]           # noqa: F821
fecal_out_pct         = snakemake.output["fecal_norm"]          # noqa: F821
healthy_oral_out_pct  = snakemake.output["healthy_oral_norm"]   # noqa: F821
healthy_fecal_out_pct = snakemake.output["healthy_fecal_norm"]  # noqa: F821

# Outputs (figs/reports)
report_path  = snakemake.output["report"]
barplot_png  = snakemake.output["barplot"]
fig_zero_sample = snakemake.output["zsample"]
fig_zero_taxon  = snakemake.output["ztaxon"]
fig_libsize     = snakemake.output["libsize"]
fig_rank_oral   = snakemake.output["rank_oral_comb"]
fig_rank_fecal  = snakemake.output["rank_fecal_comb"]
fig_prev_oral   = snakemake.output["prev_oral_comb"]
fig_prev_fecal  = snakemake.output["prev_fecal_comb"]

# Params
prevalence_oral    = float(snakemake.params.get("prevalence_oral", 0.20))
prevalence_fecal   = float(snakemake.params.get("prevalence_fecal", 0.20))
abundance_param    = float(snakemake.params.get("abundance", 0.1))
level_mode         = str(snakemake.params.get("level", "species")).lower()
agg_mode           = str(snakemake.params.get("agg", "mean")).lower()
mean_on_positives  = bool(snakemake.params.get("mean_on_positives", False))
colors_yaml_path   = str(snakemake.params.get("colors_yaml", "config/colors.yml"))
include_hmp_oral   = bool(snakemake.params.get("include_hmp_oral", True))

# Load color palette
COLORS = load_colors_yaml(colors_yaml_path)

# ---------- Load data ----------
oral_crohn_raw     = load_table_csv(oral_in)
fecal_crohn_raw    = load_table_csv(fecal_in)
oral_healthy_prev  = load_table_csv(healthy_oral_in)
fecal_healthy_raw  = load_table_csv(healthy_fecal_in)
hmp_oral_raw       = load_table_csv(hmp_oral_in) if (hmp_oral_in is not None) else pd.DataFrame()

# ---------- Merge Healthy Oral + HMP Oral (flag-controlled) ----------
if include_hmp_oral and hmp_oral_raw.shape[1] > 0:
    oral_healthy_merged = merge_tables_outer_zero(oral_healthy_prev, hmp_oral_raw)
    merge_note = f"prev={oral_healthy_prev.shape[1]}, HMP={hmp_oral_raw.shape[1]}, merged={oral_healthy_merged.shape[1]}"
else:
    oral_healthy_merged = oral_healthy_prev.copy()
    merge_note = f"prev={oral_healthy_prev.shape[1]}, HMP=0 (skipped), merged={oral_healthy_merged.shape[1]}"

print(f"[filtering] Oral Healthy merge: {merge_note}")

# ---------- Build pooled masks ----------
oral_c_filt,  oral_h_filt,  qc_oral  = build_pooled_mask(
    oral_crohn_raw,  oral_healthy_merged,
    prevalence=prevalence_oral, abundance_param=abundance_param,
    level=level_mode, agg=agg_mode
)
fecal_c_filt, fecal_h_filt, qc_fecal = build_pooled_mask(
    fecal_crohn_raw, fecal_healthy_raw,
    prevalence=prevalence_fecal, abundance_param=abundance_param,
    level=level_mode, agg=agg_mode
)

# ---------- Normalize to percent & SAVE (with clade_name) ----------
oral_c_pct   = normalize_to_percent(oral_c_filt)
oral_h_pct   = normalize_to_percent(oral_h_filt)
fecal_c_pct  = normalize_to_percent(fecal_c_filt)
fecal_h_pct  = normalize_to_percent(fecal_h_filt)

ensure_dir_for_file(oral_out_pct);          oral_c_pct.to_csv(oral_out_pct, index=True, index_label="clade_name")
ensure_dir_for_file(healthy_oral_out_pct);  oral_h_pct.to_csv(healthy_oral_out_pct, index=True, index_label="clade_name")
ensure_dir_for_file(fecal_out_pct);         fecal_c_pct.to_csv(fecal_out_pct, index=True, index_label="clade_name")
ensure_dir_for_file(healthy_fecal_out_pct); fecal_h_pct.to_csv(healthy_fecal_out_pct, index=True, index_label="clade_name")

# ---------------- QC plots ----------------
def zeros_by_sample(df: pd.DataFrame) -> pd.Series:
    if df.shape[0] == 0:
        return pd.Series(dtype=float)
    return (df == 0).sum(axis=0) / df.shape[0]

def zeros_by_taxon(df: pd.DataFrame) -> pd.Series:
    if df.shape[1] == 0:
        return pd.Series(dtype=float)
    return (df == 0).sum(axis=1) / df.shape[1]

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

# colors
COLORS_MAP = {
    "Fecal_Crohn":   COLORS["Fecal_Crohn"],
    "Oral_Crohn":    COLORS["Oral_Crohn"],
    "Fecal_Healthy": COLORS["Fecal_Healthy"],
    "Oral_Healthy":  COLORS["Oral_Healthy"],
}

for p in [barplot_png, fig_zero_sample, fig_zero_taxon, fig_libsize,
          fig_rank_oral, fig_rank_fecal, fig_prev_oral, fig_prev_fecal]:
    ensure_dir_for_file(p)

# 1) Barplot
groups = ["Fecal_Crohn","Oral_Crohn","Oral_Healthy","Fecal_Healthy"]
counts_before = [
    collapse_to_level(fecal_crohn_raw,      level=level_mode, agg=agg_mode).shape[0],
    collapse_to_level(oral_crohn_raw,       level=level_mode, agg=agg_mode).shape[0],
    collapse_to_level(oral_healthy_merged,  level=level_mode, agg=agg_mode).shape[0],
    collapse_to_level(fecal_healthy_raw,    level=level_mode, agg=agg_mode).shape[0],
]
counts_after  = [qc_fecal["rows_after"], qc_oral["rows_after"], qc_oral["rows_after"], qc_fecal["rows_after"]]
x = np.arange(len(groups)); width = 0.35
plt.figure(figsize=(10,5))
plt.bar(x - width/2, counts_before, width=width, label="Before(level)", color="#30638e")
plt.bar(x + width/2, counts_after,  width=width, label="After(mask)",  color="#d1495b")
plt.xticks(x, groups); plt.ylabel("Number of taxa")
plt.title("Number of taxa before (level) vs after (pooled-mask)")
plt.legend(); plt.tight_layout(); plt.savefig(barplot_png); plt.close()

# Precompute BEFORE 
oral_c_lvl = collapse_to_level(oral_crohn_raw,      level=level_mode, agg=agg_mode)
oral_h_lvl = collapse_to_level(oral_healthy_merged, level=level_mode, agg=agg_mode)
fecal_c_lvl= collapse_to_level(fecal_crohn_raw,     level=level_mode, agg=agg_mode)
fecal_h_lvl= collapse_to_level(fecal_healthy_raw,   level=level_mode, agg=agg_mode)

# 2) Zero fractions AFTER
plt.figure(figsize=(7,5)); ax = plt.gca()
_safe_kde_or_hist(ax, zeros_by_sample(fecal_c_filt),  "Fecal_Crohn",   COLORS_MAP["Fecal_Crohn"])
_safe_kde_or_hist(ax, zeros_by_sample(oral_c_filt),   "Oral_Crohn",    COLORS_MAP["Oral_Crohn"])
_safe_kde_or_hist(ax, zeros_by_sample(oral_h_filt),   "Oral_Healthy",  COLORS_MAP["Oral_Healthy"])
_safe_kde_or_hist(ax, zeros_by_sample(fecal_h_filt),  "Fecal_Healthy", COLORS_MAP["Fecal_Healthy"])
ax.set_xlabel("Zero fraction per sample"); ax.set_title("Zero-inflation (by sample) AFTER mask")
ax.legend(); plt.tight_layout(); plt.savefig(fig_zero_sample); plt.close()

plt.figure(figsize=(7,5)); ax = plt.gca()
_safe_kde_or_hist(ax, zeros_by_taxon(fecal_c_filt),  "Fecal_Crohn",   COLORS_MAP["Fecal_Crohn"])
_safe_kde_or_hist(ax, zeros_by_taxon(oral_c_filt),   "Oral_Crohn",    COLORS_MAP["Oral_Crohn"])
_safe_kde_or_hist(ax, zeros_by_taxon(oral_h_filt),   "Oral_Healthy",  COLORS_MAP["Oral_Healthy"])
_safe_kde_or_hist(ax, zeros_by_taxon(fecal_h_filt),  "Fecal_Healthy", COLORS_MAP["Fecal_Healthy"])
ax.set_xlabel("Zero fraction per taxon"); ax.set_title("Zero-inflation (by taxon) AFTER mask")
ax.legend(); plt.tight_layout(); plt.savefig(fig_zero_taxon); plt.close()

# 3) Retained mass AFTER vs BEFORE
def _retained_pct(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.Series:
    b = df_before.sum(axis=0).replace(0, np.nan)
    a = df_after.sum(axis=0)
    return (a.div(b) * 100.0).fillna(0.0)

ret_all = pd.concat([
    _retained_pct(fecal_c_lvl, fecal_c_filt),
    _retained_pct(oral_c_lvl,  oral_c_filt),
    _retained_pct(oral_h_lvl,  oral_h_filt),
    _retained_pct(fecal_h_lvl, fecal_h_filt),
], ignore_index=True)

plt.figure(figsize=(7,5)); ax = plt.gca()
ax.hist(ret_all, bins=20, alpha=0.5, label="All groups")
ax.set_xlim(0, 100)
ax.set_xlabel("Retained mass after filtering (%)")
ax.set_title("Retained mass distribution (per sample)")
ax.legend(); plt.tight_layout(); plt.savefig(fig_libsize); plt.close()

# 4) Rank-abundance (Oral)
def _rank_curve(df):
    if df.shape[0] == 0: return np.array([])
    v = df.mean(axis=1).sort_values(ascending=False).values
    return v[v > 0]

rb = _rank_curve(oral_c_lvl); ra = _rank_curve(oral_c_filt)
hb = _rank_curve(oral_h_lvl); ha = _rank_curve(oral_h_filt)
plt.figure(figsize=(7,5))
if rb.size: plt.plot(np.arange(1, rb.size+1), rb, lw=1.2, ls="--", label="Oral_Crohn (before)",  color=COLORS_MAP["Oral_Crohn"])
if ra.size: plt.plot(np.arange(1, ra.size+1), ra, lw=1.6, ls="-",  label="Oral_Crohn (after)",   color=COLORS_MAP["Oral_Crohn"])
if hb.size: plt.plot(np.arange(1, hb.size+1), hb, lw=1.2, ls="--", label="Oral_Healthy (before)",color=COLORS_MAP["Oral_Healthy"])
if ha.size: plt.plot(np.arange(1, ha.size+1), ha, lw=1.6, ls="-",  label="Oral_Healthy (after)", color=COLORS_MAP["Oral_Healthy"])
plt.xlabel("Rank"); plt.ylabel("Mean abundance (%)"); plt.yscale("log")
plt.title(f"Oral: rank-abundance ({level_mode})")
plt.legend(); plt.tight_layout(); plt.savefig(fig_rank_oral); plt.close()

# 5) Rank-abundance (Fecal)
rb = _rank_curve(fecal_c_lvl); ra = _rank_curve(fecal_c_filt)
hb = _rank_curve(fecal_h_lvl); ha = _rank_curve(fecal_h_filt)
plt.figure(figsize=(7,5))
if rb.size: plt.plot(np.arange(1, rb.size+1), rb, lw=1.2, ls="--", label="Fecal_Crohn (before)",  color=COLORS_MAP["Fecal_Crohn"])
if ra.size: plt.plot(np.arange(1, ra.size+1), ra, lw=1.6, ls="-",  label="Fecal_Crohn (after)",   color=COLORS_MAP["Fecal_Crohn"])
if hb.size: plt.plot(np.arange(1, hb.size+1), hb, lw=1.2, ls="--", label="Fecal_Healthy (before)",color=COLORS_MAP["Fecal_Healthy"])
if ha.size: plt.plot(np.arange(1, ha.size+1), ha, lw=1.6, ls="-",  label="Fecal_Healthy (after)", color=COLORS_MAP["Fecal_Healthy"])
plt.xlabel("Rank"); plt.ylabel("Mean abundance (%)"); plt.yscale("log")
plt.title(f"Fecal: rank-abundance ({level_mode})")
plt.legend(); plt.tight_layout(); plt.savefig(fig_rank_fecal); plt.close()

# 6) Prevalence-mean scatter
def prevalence_mean_scatter(ax, df_before, df_after, label, color,
                            prevalence_thr, abundance_thr, scale, mean_on_positives=False):
    def _prev_mean(df):
        if df.shape[1] == 0:
            return pd.DataFrame(columns=["prev","mean"])
        prev = (df > 0).sum(axis=1) / max(1, df.shape[1])
        mean = df.replace(0, np.nan).mean(axis=1).fillna(0.0) if mean_on_positives else df.mean(axis=1)
        return pd.DataFrame({"prev": prev, "mean": mean})
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

def _prevmean_plot(site_label, c_raw, h_raw, c_after, h_after,
                   prevalence_thr, abundance_thr, scale, outfile):
    plt.figure(figsize=(7,5)); ax = plt.gca()
    prevalence_mean_scatter(
        ax,
        collapse_to_level(c_raw, level=level_mode, agg=agg_mode),
        c_after, "Oral_Crohn" if site_label=="oral" else "Fecal_Crohn",
        COLORS_MAP["Oral_Crohn"] if site_label=="oral" else COLORS_MAP["Fecal_Crohn"],
        prevalence_thr, abundance_thr, scale, mean_on_positives
    )
    prevalence_mean_scatter(
        ax,
        collapse_to_level(h_raw, level=level_mode, agg=agg_mode),
        h_after, "Oral_Healthy" if site_label=="oral" else "Fecal_Healthy",
        COLORS_MAP["Oral_Healthy"] if site_label=="oral" else COLORS_MAP["Fecal_Healthy"],
        prevalence_thr, abundance_thr, scale, mean_on_positives
    )
    ax.set_title(f"{site_label.capitalize()}: prevalence vs mean (log-y)")
    plt.tight_layout(); plt.savefig(outfile); plt.close()

# Prevalence–mean plots (oral uses merged healthy; fecal uses healthy-only)
_prevmean_plot("oral",  oral_crohn_raw,  oral_healthy_merged,  oral_c_filt,  oral_h_filt,
               prevalence_oral,  qc_oral["abundance_applied"],  qc_oral["scale"],  fig_prev_oral)
_prevmean_plot("fecal", fecal_crohn_raw, fecal_healthy_raw,    fecal_c_filt, fecal_h_filt,
               prevalence_fecal, qc_fecal["abundance_applied"], qc_fecal["scale"], fig_prev_fecal)

# Write report (atomically)
try:
    report_lines = []
    report_lines.append("== Filtering report (pooled-mask per site) ==\n\n")
    report_lines.append(f"Level={level_mode}, Aggregation={agg_mode}\n")
    report_lines.append(f"Prevalence oral={prevalence_oral:.3f}, fecal={prevalence_fecal:.3f}\n")
    report_lines.append(f"Abundance parameter={abundance_param} (applied per detected scale)\n\n")
    report_lines.append(
        f"Oral Healthy merged samples: prev={oral_healthy_prev.shape[1]}, "
        f"HMP={hmp_oral_raw.shape[1]}, merged={oral_healthy_merged.shape[1]}\n\n"
    )
    report_lines.append("[ORAL]\n")
    report_lines.append(
        f"scale={qc_oral['scale']}, rows_before_c={qc_oral['rows_before_c']}, "
        f"rows_before_h={qc_oral['rows_before_h']}, rows_after={qc_oral['rows_after']}\n"
    )
    report_lines.append(f"abundance_thr_applied={qc_oral['abundance_applied']}\n\n")
    report_lines.append("[FECAL]\n")
    report_lines.append(
        f"scale={qc_fecal['scale']}, rows_before_c={qc_fecal['rows_before_c']}, "
        f"rows_before_h={qc_fecal['rows_before_h']}, rows_after={qc_fecal['rows_after']}\n"
    )
    report_lines.append(f"abundance_thr_applied={qc_fecal['abundance_applied']}\n")

    write_text_atomic(report_path, "".join(report_lines))
except Exception:
    # fall back: ensure an empty file still exists (to satisfy Snakemake)
    ensure_dir_for_file(report_path)
    Path(report_path).touch()

print(
    f"[filtering] Done. HMP merge flag={include_hmp_oral}. "
    f"Oral Healthy samples -> {oral_healthy_merged.shape[1]}. "
    f"Outputs saved with index_label='clade_name'."
)