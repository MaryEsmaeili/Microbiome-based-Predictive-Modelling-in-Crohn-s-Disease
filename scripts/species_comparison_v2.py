# scripts/species_comparison_v2.py
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# -------------------- Config / palette --------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)
palette = (config or {}).get('colors', {})

# -------------------- Rank helpers -----------------------
def extract_species(rowname: str):
    parts = rowname.split("|")
    s = [p for p in parts if p.startswith("s__")]
    return s[0] if s else None

def prettify_taxon(label: str, rank: str) -> str:
    """
    Publication-style labels:
      - remove g__/s__
      - underscores -> spaces
      - Genus: Title Case; Species: Genus Capitalized + species lower-case
    """
    if label is None:
        return None
    if rank.lower() == "genus":
        name = label.replace("g__", "").replace("_", " ")
        return " ".join(w.capitalize() for w in name.split())
    if rank.lower() == "species":
        name = label.replace("s__", "").replace("_", " ")
        toks = name.split()
        if not toks:
            return name
        toks[0] = toks[0].capitalize()
        toks[1:] = [t.lower() for t in toks[1:]]
        return " ".join(toks)
    return label

def make_oc_fc_cmap(palette: dict) -> LinearSegmentedColormap:
    """Diverging cmap from OC color -> white -> FC color (from YAML)."""
    return LinearSegmentedColormap.from_list(
        "oc_fc_div", [palette["Crohn-Oral"], "#FFFFFF", palette["Crohn-Fecal"]], N=256
    )

# -------------------- Core table utils -------------------
def _percent_table(df: pd.DataFrame, grouper) -> pd.DataFrame:
    """Aggregate to taxa x samples and convert to % per sample."""
    g = df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    return g.div(g.sum(axis=0), axis=1) * 100

def _row_zscore_log1p(mat: pd.DataFrame, use_log1p: bool = True) -> pd.DataFrame:
    """Row z-score; optional log1p transform to reduce skew."""
    X = np.log1p(mat) if use_log1p else mat
    mu = X.mean(axis=1)
    sd = X.std(axis=1).replace(0, np.nan)
    Z = X.sub(mu, axis=0).div(sd, axis=0).fillna(0)
    return Z

def _prevalence(pct: pd.DataFrame, threshold_percent: float) -> pd.Series:
    """Fraction of samples where %abundance > threshold_percent."""
    return (pct > threshold_percent).sum(axis=1) / pct.shape[1]

def _pair_columns(oc_cols, fc_cols, matched_df: pd.DataFrame | None) -> list:
    """
    Order columns OC1, FC1, OC2, FC2... if matched_df is provided (two columns: oral_id, fecal_id).
    If not provided, returns OC columns followed by FC columns.
    """
    if matched_df is None or matched_df.empty:
        return list(oc_cols) + list(fc_cols)
    ocol = matched_df.iloc[:, 0].astype(str).tolist()
    fcol = matched_df.iloc[:, 1].astype(str).tolist()
    ordered = []
    for o, f in zip(ocol, fcol):
        if o in oc_cols: ordered.append(o)
        if f in fc_cols: ordered.append(f)
    rest = [c for c in oc_cols if c not in ordered] + [c for c in fc_cols if c not in ordered]
    return ordered + rest

# -------------------- Stacked bars per group --------------
def save_top_csv(group_df, n, grouper, out_csv):
    sums = group_df.groupby(grouper).sum(numeric_only=True)
    sums = sums[sums.index.notna()]
    top = sums.sum(axis=1).sort_values(ascending=False).head(n)
    top.to_csv(out_csv, header=['Total_Abundance'])

def plot_group_stacked(group_df, n, title, out_png, grouper, legend_title, rank=None):
    g = group_df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    top = g.sum(axis=1).sort_values(ascending=False).head(n).index
    g.loc["Other"] = g.loc[~g.index.isin(top)].sum()
    abund_plot = g.loc[list(top) + ["Other"]]
    abund_plot = abund_plot.div(abund_plot.sum(axis=0), axis=1) * 100

    # prettify legend labels
    pretty_index = [("Other" if idx == "Other" else prettify_taxon(idx, rank)) for idx in abund_plot.index]
    abund_plot_pretty = abund_plot.copy()
    abund_plot_pretty.index = pretty_index

    colors = plt.get_cmap("tab20c").colors
    color_list = list(colors[:len(abund_plot_pretty)-1]) + [palette.get("Other", "#999999")]
    ax = abund_plot_pretty.T.plot(kind='bar', stacked=True, figsize=(18,7), color=color_list)
    plt.ylabel("Relative Abundance (%)")
    plt.xlabel("Sample")
    plt.title(title)
    plt.ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title=legend_title, prop={'style':'italic'})
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -------------------- Mean barplot (OC,OH,FC) -------------
def plot_barchart_mean_species(oral_crohn, oral_healthy, fecal_crohn, outdir, n=15):
    # to percent (taxa x samples)
    s_oc = oral_crohn.groupby(extract_species).sum(numeric_only=True)
    s_fc = fecal_crohn.groupby(extract_species).sum(numeric_only=True)
    s_oh = oral_healthy.groupby(extract_species).sum(numeric_only=True)
    s_oc = s_oc[s_oc.index.notna()]
    s_fc = s_fc[s_fc.index.notna()]
    s_oh = s_oh[s_oh.index.notna()]

    pct_oc = s_oc.div(s_oc.sum(axis=0), axis=1) * 100
    pct_fc = s_fc.div(s_fc.sum(axis=0), axis=1) * 100
    pct_oh = s_oh.div(s_oh.sum(axis=0), axis=1) * 100

    # x-axis = union topN in OC and FC
    top_oc = pct_oc.sum(axis=1).sort_values(ascending=False).head(n).index
    top_fc = pct_fc.sum(axis=1).sort_values(ascending=False).head(n).index
    union_top = pd.Index(top_oc).union(top_fc)

    # group means
    mean_oc = pct_oc.reindex(union_top).fillna(0).mean(axis=1)
    mean_fc = pct_fc.reindex(union_top).fillna(0).mean(axis=1)
    mean_oh = pct_oh.reindex(union_top).fillna(0).mean(axis=1)

    # order by OC
    order = mean_oc.sort_values(ascending=False).index
    mean_oc = mean_oc.reindex(order)
    mean_fc = mean_fc.reindex(order)
    mean_oh = mean_oh.reindex(order)

    # CSV (means only)
    df_out = pd.DataFrame({
        "Mean_OC_%": mean_oc,
        "Mean_FC_%": mean_fc,
        "Mean_OH_%": mean_oh
    })
    df_out.to_csv(os.path.join(outdir, "means_and_mean_species.csv"))

    # Plot
    x = np.arange(len(order))
    width = 0.25
    c_oc = palette["Crohn-Oral"]
    c_oh = palette["Healthy-Oral"]
    c_fc = palette["Crohn-Fecal"]

    plt.figure(figsize=(16,7))
    ax = plt.gca()
    ax.bar(x - width, mean_oc.values, width, label='Oral_Crohn',   color=c_oc)
    ax.bar(x,         mean_oh.values, width, label='Oral_Healthy', color=c_oh)
    ax.bar(x + width, mean_fc.values, width, label='Fecal_Crohn',  color=c_fc)

    ax.set_ylabel("Mean relative abundance (%)")
    ax.set_xlabel("Species")
    ax.set_title("Mean % (OC, OH, FC) — union of Top15 (OC ∪ FC)")
    ax.set_xticks(x)
    pretty_xticks = [prettify_taxon(t, "species") for t in order]
    ax.set_xticklabels(pretty_xticks, rotation=45, ha='right')
    for lbl in ax.get_xticklabels():
        lbl.set_fontstyle('italic')
    ax.legend(loc='best')
    ax.axhline(0, linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "barplot_mean_species.png"), dpi=200)
    plt.close()

# -------------------- NEW: Two-bar OC vs OH (Top15) -------
def _group_mean_percent(df: pd.DataFrame, grouper) -> pd.Series:
    """Aggregate to rank, convert to % per sample, then mean across samples."""
    g = df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    pct = g.div(g.sum(axis=0), axis=1) * 100
    return pct.mean(axis=1)

def _top_union(s1: pd.Series, s2: pd.Series, n: int) -> pd.Index:
    combined = s1.add(s2, fill_value=0)
    return combined.sort_values(ascending=False).head(n).index

def _stacked_2bar(series_crohn: pd.Series,
                  series_healthy: pd.Series,
                  top_labels: pd.Index,
                  out_png: str,
                  palette: dict,
                  rank: str,
                  order_by: str = "Crohn"):
    # collapse rest to Other
    s_oc = series_crohn.copy()
    s_oh = series_healthy.copy()
    other_oc = s_oc.loc[~s_oc.index.isin(top_labels)].sum()
    other_oh = s_oh.loc[~s_oh.index.isin(top_labels)].sum()
    s_oc = s_oc.reindex(top_labels).fillna(0)
    s_oh = s_oh.reindex(top_labels).fillna(0)
    s_oc = pd.concat([s_oc, pd.Series({"Other": other_oc})])
    s_oh = pd.concat([s_oh, pd.Series({"Other": other_oh})])

    # order stack by chosen group
    key = s_oc if order_by.lower().startswith("crohn") else s_oh
    order = key.sort_values(ascending=False).index
    s_oc = s_oc.reindex(order)
    s_oh = s_oh.reindex(order)

    # colors: unique per taxon; Other from YAML
    cmap = plt.get_cmap("tab20c").colors
    color_map = {tax: cmap[i % len(cmap)] for i, tax in enumerate(order)}
    color_map["Other"] = palette.get("Other", "#999999")

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.array([0, 1])
    bottom_oc = 0
    bottom_oh = 0
    for i, tax in enumerate(order):
        h_oc = float(s_oc.loc[tax])
        h_oh = float(s_oh.loc[tax])
        ax.bar(x[0], h_oc, bottom=bottom_oc, color=color_map[tax], width=0.6)
        ax.bar(x[1], h_oh, bottom=bottom_oh, color=color_map[tax], width=0.6)
        bottom_oc += h_oc
        bottom_oh += h_oh

    ax.set_xticks(x)
    ax.set_xticklabels(["Oral_Crohn", "Oral_Healthy"])
    ax.set_ylabel("Relative abundance (mean %, per group)")
    ax.set_title(f"Oral {rank.capitalize()} — Top {len(top_labels)} (others → Other)")

    # legend (italicize taxa, keep 'Other' normal)
    handles = [plt.Rectangle((0,0),1,1,color=color_map[t]) for t in order]
    labels = [prettify_taxon(t, rank) if t != "Other" else "Other" for t in order]
    leg = ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", title=rank.capitalize())
    for txt in leg.get_texts():
        if txt.get_text() != "Other":
            txt.set_style("italic")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def barplot_oral_species_top15(oral_crohn_df: pd.DataFrame,
                               oral_healthy_df: pd.DataFrame,
                               out_png: str,
                               palette: dict,
                               order_by: str = "Crohn"):
    mean_oc = _group_mean_percent(oral_crohn_df, extract_species)
    mean_oh = _group_mean_percent(oral_healthy_df, extract_species)
    top15 = _top_union(mean_oc, mean_oh, n=15)
    _stacked_2bar(mean_oc, mean_oh, top15, out_png, palette, rank="species", order_by=order_by)

# -------------------- Heatmaps ----------------------------
def plot_heatmaps_for_rank(
    oral_crohn: pd.DataFrame,
    oral_healthy: pd.DataFrame,
    fecal_crohn: pd.DataFrame,
    outdir: str,
    rank: str,                 # "species"
    grouper,                   # extract_species
    palette: dict,
    n: int = 15,
    prevalence_thresh: float = 0.10,
    presence_threshold_percent: float = 0.1,
    matched_pairs: pd.DataFrame | None = None,
    use_log1p: bool = True,
    clip_quantile: float = 0.98,
    col_cluster_pairs: bool = False
):
    os.makedirs(outdir, exist_ok=True)
    # % tables
    pct_oc = _percent_table(oral_crohn, grouper)
    pct_fc = _percent_table(fecal_crohn, grouper)
    pct_oh = _percent_table(oral_healthy, grouper)

    # Row selection (Crohn focus)
    prev_oc = _prevalence(pct_oc, presence_threshold_percent)
    prev_fc = _prevalence(pct_fc, presence_threshold_percent)
    crohn_union = pct_oc.add(pct_fc, fill_value=0)
    keep = (prev_oc >= prevalence_thresh) & (prev_fc >= prevalence_thresh)
    rows = crohn_union.loc[keep].sum(axis=1).sort_values(ascending=False).head(n).index
    if len(rows) < max(5, int(0.4 * n)):
        rows = crohn_union.sum(axis=1).sort_values(ascending=False).head(n).index

    # (1) Crohn pairs clustermap
    data_pct_pairs = pd.concat([pct_oc, pct_fc], axis=1).reindex(rows).fillna(0)
    ordered_cols = _pair_columns(pct_oc.columns, pct_fc.columns, matched_pairs)
    ordered_cols = [c for c in ordered_cols if c in data_pct_pairs.columns]
    data_pct_pairs = data_pct_pairs.reindex(columns=ordered_cols)
    data_pct_pairs.index = [prettify_taxon(t, rank) for t in data_pct_pairs.index]
    Z = _row_zscore_log1p(data_pct_pairs, use_log1p=use_log1p)
    V = np.nanquantile(np.abs(Z.values), clip_quantile)
    if not np.isfinite(V) or V == 0:
        V = max(1.0, float(np.nanmax(np.abs(Z.values))))
    col_colors = pd.Series(index=Z.columns, dtype=object)
    col_colors.loc[pct_oc.columns] = palette["Crohn-Oral"]
    col_colors.loc[pct_fc.columns] = palette["Crohn-Fecal"]
    col_colors = col_colors.reindex(Z.columns)
    cmap = make_oc_fc_cmap(palette)
    height = max(6, 0.35 * len(Z.index))
    width  = max(8,  0.18 * len(Z.columns) + 2)
    g = sns.clustermap(
        Z,
        cmap=cmap, center=0, vmin=-V, vmax=+V,
        col_colors=col_colors,
        col_cluster=col_cluster_pairs,
        metric="correlation", method="average",
        figsize=(width, height),
        cbar_kws={"label": "Row z-score of % abundance (log1p)"}
    )
    for lbl in g.ax_heatmap.get_yticklabels():
        lbl.set_fontstyle("italic")
    g.ax_heatmap.set_xlabel("Crohn patients (Oral ↔ Fecal)")
    g.ax_heatmap.set_ylabel(rank.capitalize())
    g.fig.suptitle(f"{rank.capitalize()} clustermap — Crohn pairs (OC ↔ FC)", y=1.02)
    out_pairs = os.path.join(outdir, f"heatmap_pairs_crohn_{rank}.png")
    g.fig.savefig(out_pairs, dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    # (2) Group means & deltas heatmap (+ CSV)
    mean_oc = pct_oc.reindex(rows).fillna(0).mean(axis=1)
    mean_fc = pct_fc.reindex(rows).fillna(0).mean(axis=1)
    mean_oh = pct_oh.reindex(rows).fillna(0).mean(axis=1)
    delta_oc_fc = mean_oc - mean_fc
    delta_oc_oh = mean_oc - mean_oh
    GC = pd.DataFrame({
        "Mean_OC_%": mean_oc,
        "Mean_OH_%": mean_oh,
        "Mean_FC_%": mean_fc,
        "Δ(OC−FC)":  delta_oc_fc,
        "Δ(OC−OH)":  delta_oc_oh,
    })
    GC = GC.loc[delta_oc_fc.abs().sort_values(ascending=False).index]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * len(GC.index))), gridspec_kw={'width_ratios': [3, 2]})
    sns.heatmap(
        GC[["Mean_OC_%", "Mean_OH_%", "Mean_FC_%"]],
        ax=axes[0], cmap="Greys",
        cbar_kws={"label": "Mean % abundance"}
    )
    axes[0].set_title("Group means")
    axes[0].set_ylabel(rank.capitalize())
    axes[0].set_xlabel("Groups")
    axes[0].set_yticklabels([prettify_taxon(t, rank) for t in GC.index], rotation=0, fontsize=9)
    for lbl in axes[0].get_yticklabels():
        lbl.set_fontstyle("italic")
    Vd = np.nanquantile(np.abs(GC[["Δ(OC−FC)", "Δ(OC−OH)"]].values), 0.98)
    if not np.isfinite(Vd) or Vd == 0:
        Vd = max(1e-6, float(np.nanmax(np.abs(GC[["Δ(OC−FC)", "Δ(OC−OH)"]].values))))
    sns.heatmap(
        GC[["Δ(OC−FC)", "Δ(OC−OH)"]],
        ax=axes[1], cmap=cmap, center=0, vmin=-Vd, vmax=+Vd,
        cbar_kws={"label": "Delta (percentage points)"}
    )
    axes[1].set_title("Deltas")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Comparisons")
    axes[1].set_yticklabels([prettify_taxon(t, rank) for t in GC.index], rotation=0, fontsize=9)
    for lbl in axes[1].get_yticklabels():
        lbl.set_fontstyle("italic")
    fig.suptitle(f"{rank.capitalize()} — group means & deltas", y=1.02)
    out_contrast = os.path.join(outdir, f"heatmap_group_contrast_{rank}.png")
    fig.savefig(out_contrast, dpi=300, bbox_inches="tight")
    plt.close(fig)
    GC.to_csv(os.path.join(outdir, f"group_means_and_deltas_{rank}.csv"))

# -------------------- Main -------------------------------
def main(oral_crohn_file, oral_healthy_file, fecal_crohn_file, outdir):
    os.makedirs(outdir, exist_ok=True)
    oc = pd.read_csv(oral_crohn_file, index_col=0)
    oh = pd.read_csv(oral_healthy_file, index_col=0)
    fc = pd.read_csv(fecal_crohn_file, index_col=0)

    # per-group stackedbars + CSVs
    plot_group_stacked(oc, 15, "Top 15 Species - Oral Crohn Samples",
                       os.path.join(outdir, "stackedbar_top15_oral_crohn_species.png"),
                       extract_species, "Species", rank="species")
    save_top_csv(oc, 15, extract_species, os.path.join(outdir, "top15_oral_crohn_species.csv"))

    plot_group_stacked(oh, 15, "Top 15 Species - Oral Healthy Samples",
                       os.path.join(outdir, "stackedbar_top15_oral_healthy_species.png"),
                       extract_species, "Species", rank="species")
    save_top_csv(oh, 15, extract_species, os.path.join(outdir, "top15_oral_healthy_species.csv"))

    plot_group_stacked(fc, 15, "Top 15 Species - Fecal Crohn Samples",
                       os.path.join(outdir, "stackedbar_top15_fecal_crohn_species.png"),
                       extract_species, "Species", rank="species")
    save_top_csv(fc, 15, extract_species, os.path.join(outdir, "top15_fecal_crohn_species.csv"))

    # Mean barplot (OC,OH,FC) over OC∪FC top-15
    plot_barchart_mean_species(oc, oh, fc, outdir, n=15)

    # Heatmaps (Crohn pairs + group contrast)
    try:
        matched = pd.read_csv("data/processed/matched_sample_ids.csv")
    except Exception:
        matched = None
    plot_heatmaps_for_rank(
        oral_crohn=oc,
        oral_healthy=oh,
        fecal_crohn=fc,
        outdir=outdir,
        rank="species",
        grouper=extract_species,
        palette=palette,
        n=15,
        prevalence_thresh=0.10,
        presence_threshold_percent=0.1,
        matched_pairs=matched,
        use_log1p=True,
        clip_quantile=0.98,
        col_cluster_pairs=False
    )

    # NEW: Two-bar stacked OC vs OH (Top-15 species)
    barplot_oral_species_top15(
        oral_crohn_df=oc,
        oral_healthy_df=oh,
        out_png=os.path.join(outdir, "barplot_oral_species_top15.png"),
        palette=palette,
        order_by="Crohn"
    )

    print("[INFO] Species V2 done.", outdir)

if __name__ == "__main__":
    try:
        oral_crohn_file = snakemake.input.oral_crohn
        oral_healthy_file = snakemake.input.oral_healthy
        fecal_crohn_file = snakemake.input.fecal_crohn
        outdir = snakemake.params.outdir
    except NameError:
        import sys
        if len(sys.argv) < 5:
            raise SystemExit("Usage: python species_comparison_v2.py <oral_crohn.csv> <oral_healthy.csv> <fecal_crohn.csv> <outdir>")
        oral_crohn_file, oral_healthy_file, fecal_crohn_file, outdir = sys.argv[1:5]
    main(oral_crohn_file, oral_healthy_file, fecal_crohn_file, outdir)
