# -*- coding: utf-8 -*-
"""
Filtering + QC for microbiome abundance tables.

Pipeline (per group):
  1) Load abundance (rows=taxa, cols=samples).
  2) Filter by prevalence then abundance (order configurable).
  3) Keep g__/s__ taxonomic levels, safely aggregating duplicates.
  4) Export percent-normalized CSV (for plots) and CLR CSV (for compositional stats).
  5) QC tables: zeros by sample/taxon, library sizes (pre-filtered subset), retained mass (%).
  6) QC figures:
      - filtering_barplot_allgroups.png
      - zero_fraction_by_sample.png
      - zero_fraction_by_taxon.png
      - library_size_distribution.png  (now shows retained mass %, 0–100)

Notes
-----
- Abundance threshold is applied on the *current input scale* (your inputs are %).
- CLR is computed from column-wise relative abundances (sum=1) with a pseudocount.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# ---------------- Config (palette) ----------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)
palette = config["colors"]

# ---------------- Snakemake I/O -------------------
oral_in          = snakemake.input.oral
fecal_in         = snakemake.input.fecal
healthy_oral_in  = snakemake.input.healthy_oral
healthy_fecal_in = snakemake.input.healthy_fecal

oral_out_pct          = snakemake.output.oral_norm
fecal_out_pct         = snakemake.output.fecal_norm
healthy_oral_out_pct  = snakemake.output.healthy_oral_norm
healthy_fecal_out_pct = snakemake.output.healthy_fecal_norm

# side-outputs (not declared in Snakefile; created in parallel)
oral_out_clr          = oral_out_pct.replace(".csv", "_clr.csv")
fecal_out_clr         = fecal_out_pct.replace(".csv", "_clr.csv")
healthy_oral_out_clr  = healthy_oral_out_pct.replace(".csv", "_clr.csv")
healthy_fecal_out_clr = healthy_fecal_out_pct.replace(".csv", "_clr.csv")

# Params (with safe defaults if not passed)
prevalence_threshold = float(snakemake.params.get("prevalence", 0.20))
abundance_threshold  = float(snakemake.params.get("abundance", 10.0))  # % scale
filter_order         = snakemake.params.get("order", "prevalence_then_abundance")
pseudocount          = float(snakemake.params.get("pseudocount", 1e-6))
level_mode           = snakemake.params.get("level", "both")  # "genus" | "species" | "both"

# ---------------- Helpers ------------------------
def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def load_data(filepath):
    df = pd.read_csv(filepath, index_col=0)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

def zeros_by_sample(df):
    if df.shape[0] == 0:
        return pd.Series(dtype=float)
    return (df == 0).sum(axis=0) / df.shape[0]

def zeros_by_taxon(df):
    if df.shape[1] == 0:
        return pd.Series(dtype=float)
    return (df == 0).sum(axis=1) / df.shape[1]

def library_sizes(df):
    # sum per sample on the *filtered* table (pre-normalization)
    return df.sum(axis=0)

def remove_low_abundance(df, threshold):
    # Uses max across samples; operates on current scale (your inputs are %)
    if df.shape[1] == 0:
        return df
    return df[df.max(axis=1) >= threshold]

def remove_low_prevalence(df, threshold):
    if df.shape[1] == 0:
        return df
    prev = (df > 0).sum(axis=1) / df.shape[1]
    return df[prev >= threshold]

def keep_taxonomic_levels(df, mode="both"):
    """
    Keep g__/s__ levels; aggregate duplicate keys by sum.
    mode: "genus", "species", "both"
    """
    kept = {}
    for name, row in df.iterrows():
        if "g__" not in name:
            continue
        parts = name.split("|")
        genus = [p for p in parts if p.startswith("g__")]
        species = [p for p in parts if p.startswith("s__")]
        if mode == "genus":
            key = "|".join(genus) if genus else None
        elif mode == "species":
            key = "|".join(genus + species) if genus and species else None
        else:
            key = "|".join(genus + species) if genus else None
        if not key:
            continue
        if key not in kept:
            kept[key] = row.values.copy()
        else:
            kept[key] = kept[key] + row.values
    if not kept:
        return pd.DataFrame(index=[], columns=df.columns, dtype=float)
    out = pd.DataFrame.from_dict(kept, orient="index")
    out.columns = df.columns
    return out

def normalize_pct(df):
    # Relative abundance (%)
    if df.shape[1] == 0:
        return df
    pct = df.div(df.sum(axis=0), axis=1) * 100.0
    return pct.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all").dropna(axis=1, how="all")

def transform_clr_from_rel(df_rel, pseudocount=1e-6):
    """
    df_rel must be relative abundance on [0,1] per column (sum=1).
    """
    if df_rel.shape[1] == 0:
        return df_rel
    X = df_rel + pseudocount
    logX = np.log(X)
    clr = logX.sub(logX.mean(axis=0), axis=1)
    return clr.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all").dropna(axis=1, how="all")

def filter_pipeline(df0, group_name):
    steps = [("Initial", df0.shape[0])]

    if filter_order == "prevalence_then_abundance":
        df1 = remove_low_prevalence(df0, prevalence_threshold)
        steps.append((f"Prevalence≥{prevalence_threshold:.2f}", df1.shape[0]))
        df2 = remove_low_abundance(df1, abundance_threshold)
        steps.append((f"Abundance≥{abundance_threshold:g}", df2.shape[0]))
    else:
        df1 = remove_low_abundance(df0, abundance_threshold)
        steps.append((f"Abundance≥{abundance_threshold:g}", df1.shape[0]))
        df2 = remove_low_prevalence(df1, prevalence_threshold)
        steps.append((f"Prevalence≥{prevalence_threshold:.2f}", df2.shape[0]))

    df3 = keep_taxonomic_levels(df2, mode=level_mode)
    steps.append((f"Taxa level={level_mode}", df3.shape[0]))

    # QC on filtered (pre-normalized)
    z_s = zeros_by_sample(df3)
    z_t = zeros_by_taxon(df3)
    libs = library_sizes(df3)

    # Compute retained mass (%) relative to original table (same samples)
    orig_libs = (df0.sum(axis=0)).replace(0, np.nan)
    retained_pct = (libs / orig_libs) * 100.0
    retained_pct = retained_pct.fillna(0.0)

    return df3, steps, z_s, z_t, libs, retained_pct

def process_group(name, inp, out_pct, out_clr):
    df0 = load_data(inp)
    df_filtered, steps, z_by_sample, z_by_taxon, libs, retained_pct = filter_pipeline(df0, name)

    # Export normalized tables
    df_pct = normalize_pct(df_filtered)
    # CLR needs relative abundances on [0,1]
    df_rel = df_filtered.div(df_filtered.sum(axis=0), axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df_clr = transform_clr_from_rel(df_rel, pseudocount=pseudocount)

    ensure_dir(out_pct); ensure_dir(out_clr)
    df_pct.to_csv(out_pct)
    df_clr.to_csv(out_clr)

    # Save QC tables
    qc_dir = os.path.join("results", "filtering")
    ensure_dir(qc_dir)
    z_by_sample.to_csv(os.path.join(qc_dir, f"{name}_zeros_by_sample.tsv"), sep="\t", header=["zero_fraction"])
    z_by_taxon.to_csv(os.path.join(qc_dir, f"{name}_zeros_by_taxon.tsv"), sep="\t", header=["zero_fraction"])
    libs.to_csv(os.path.join(qc_dir, f"{name}_library_sizes.tsv"), sep="\t", header=["sum"])
    retained_pct.to_csv(os.path.join(qc_dir, f"{name}_retained_mass_percent.tsv"), sep="\t", header=["percent"])

    return steps, z_by_sample, z_by_taxon, retained_pct

# ---------------- Run for all groups ----------------
report = {}
z_s, z_t, retained = {}, {}, {}

report["Crohn-Oral"],   z_s["Crohn-Oral"],   z_t["Crohn-Oral"],   retained["Crohn-Oral"]   = process_group("Crohn-Oral", oral_in,         oral_out_pct,         oral_out_clr)
report["Crohn-Fecal"],  z_s["Crohn-Fecal"],  z_t["Crohn-Fecal"],  retained["Crohn-Fecal"]  = process_group("Crohn-Fecal", fecal_in,       fecal_out_pct,        fecal_out_clr)
report["Healthy-Oral"], z_s["Healthy-Oral"], z_t["Healthy-Oral"], retained["Healthy-Oral"] = process_group("Healthy-Oral", healthy_oral_in, healthy_oral_out_pct, healthy_oral_out_clr)
report["Healthy-Fecal"],z_s["Healthy-Fecal"],z_t["Healthy-Fecal"],retained["Healthy-Fecal"]= process_group("Healthy-Fecal", healthy_fecal_in, healthy_fecal_out_pct, healthy_fecal_out_clr)

# ---------------- Save filtering report ----------------
results_dir = os.path.join("results", "filtering")
ensure_dir(results_dir)
with open(os.path.join(results_dir, "filtering_report.txt"), "w") as f:
    for group, steps in report.items():
        f.write(f"[{group}]\n")
        for step, count in steps:
            f.write(f"{step}: {count}\n")
        f.write("\n")

# ---------------- Plots (safe) ----------------------
def safe_kde_plot(ax, series, label):
    s = pd.Series(series).dropna().astype(float)
    n = int(s.size)
    # KDE only if we have >=2 points AND variance>0
    if n >= 2 and s.std(ddof=1) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.plot(kind="kde", label=label, ax=ax)
    elif n >= 1:
        ax.hist(s, bins=min(10, max(1, n)), alpha=0.35, label=f"{label} (hist)")
    else:
        ax.plot([], [], label=f"{label} (n=0)")

# Filtering barplot
plt.figure(figsize=(10, 5))
width = 0.20
labels = [s[0] for s in report["Crohn-Oral"]]
x = np.arange(len(labels))
group_list = ["Crohn-Fecal", "Crohn-Oral", "Healthy-Oral", "Healthy-Fecal"]
bar_colors = [palette[g] for g in group_list]

for i, (group, color) in enumerate(zip(group_list, bar_colors)):
    counts = [s[1] for s in report[group]]
    plt.bar(x + i*width, counts, width=width, label=group.replace("_", " "), color=color)

plt.xticks(x + width*1.5, labels)
plt.xlabel("Filtering Step")
plt.ylabel("Number of taxa")
plt.title("Number of taxa after each filtering step")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "filtering_barplot_allgroups.png"))
plt.close()

# Zero fraction distribution by sample
fig, ax = plt.subplots(figsize=(7,5))
for g in group_list:
    safe_kde_plot(ax, z_s[g], g)
ax.set_xlabel("Zero fraction per sample")
ax.set_title("Zero-inflation (by sample)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(results_dir, "zero_fraction_by_sample.png"))
plt.close(fig)

# Zero fraction distribution by taxon
fig, ax = plt.subplots(figsize=(7,5))
for g in group_list:
    safe_kde_plot(ax, z_t[g], g)
ax.set_xlabel("Zero fraction per taxon")
ax.set_title("Zero-inflation (by taxon)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(results_dir, "zero_fraction_by_taxon.png"))
plt.close(fig)

# Retained mass distribution (0–100%)
fig, ax = plt.subplots(figsize=(7,5))
for g in group_list:
    safe_kde_plot(ax, retained[g], g)
ax.set_xlim(0, 100)
ax.set_xlabel("Retained mass after filtering (%)")
ax.set_title("Retained mass distribution (per sample)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(results_dir, "library_size_distribution.png"))
plt.close(fig)
