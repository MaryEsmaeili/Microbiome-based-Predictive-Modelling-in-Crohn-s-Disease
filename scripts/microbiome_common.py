# scripts/microbiome_common.py
import os, re, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu, wilcoxon
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
# ---------- Config / palette ----------
def load_palette(cfg_path="config.yaml"):
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("colors") or {})
    except Exception:
        return {}
PALETTE = load_palette()
sns.set_context("talk")

# ---------- ID normalization ----------
def _normalize_id(x: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(x).strip().lower())

def _build_id_map(cols) -> dict:
    m = {}
    for c in cols:
        key = _normalize_id(str(c))
        if key not in m:
            m[key] = c
    return m

# ---------- Rank helpers ----------
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
    r = rank.lower()
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

def make_oc_fc_cmap(palette: dict) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "oc_fc_div",
        [palette.get("Crohn-Oral", "#1f77b4"), "#FFFFFF", palette.get("Crohn-Fecal", "#ff7f0e")],
        N=256
    )

# ---------- Table utilities ----------
def ensure_taxa_by_samples(df_like: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Index(df_like.columns.astype(str))
    looks_like_taxa = cols.str.contains("s__|g__", regex=True).mean()
    return df_like.T if looks_like_taxa > 0.5 else df_like

def percent_table(df: pd.DataFrame, grouper) -> pd.DataFrame:
    g = df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    return g.div(g.sum(axis=0), axis=1) * 100.0

def row_zscore_log1p(mat: pd.DataFrame, use_log1p: bool=True) -> pd.DataFrame:
    X = np.log1p(mat) if use_log1p else mat
    mu = X.mean(axis=1)
    sd = X.std(axis=1).replace(0, np.nan)
    return X.sub(mu, axis=0).div(sd, axis=0).fillna(0)

def prevalence(pct: pd.DataFrame, threshold_percent: float) -> pd.Series:
    return (pct > threshold_percent).sum(axis=1) / pct.shape[1]

def group_mean_percent(df: pd.DataFrame, grouper) -> pd.Series:
    g = df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    pct = g.div(g.sum(axis=0), axis=1) * 100
    return pct.mean(axis=1)

def top_union(s1: pd.Series, s2: pd.Series, n: int) -> pd.Index:
    return s1.add(s2, fill_value=0).sort_values(ascending=False).head(n).index

def clr_transform(df_taxa_by_samples: pd.DataFrame, pseudocount=1e-6) -> pd.DataFrame:
    X = df_taxa_by_samples.astype(float) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)
    return logX.sub(gm, axis=1)

# ---------- Plot helpers ----------
def legend_italicize(legend_obj):
    if legend_obj is None: return
    for txt in legend_obj.get_texts():
        if txt.get_text() != "Other":
            try: txt.set_style("italic")
            except Exception: pass

def save_top_csv(group_df, n, grouper, out_csv):
    sums = group_df.groupby(grouper).sum(numeric_only=True)
    sums = sums[sums.index.notna()]
    top = sums.sum(axis=1).sort_values(ascending=False).head(n)
    top.to_csv(out_csv, header=['Total_Abundance'])

def plot_group_stacked(group_df, n, title, out_png, grouper, legend_title, rank=None, palette=None):
    palette = palette or PALETTE
    g = group_df.groupby(grouper).sum(numeric_only=True)
    g = g[g.index.notna()]
    top = g.sum(axis=1).sort_values(ascending=False).head(n).index
    g.loc["Other"] = g.loc[~g.index.isin(top)].sum()
    abund = g.loc[list(top) + ["Other"]]
    abund = abund.div(abund.sum(axis=0), axis=1) * 100
    pretty = [("Other" if idx=="Other" else prettify_taxon(idx, rank)) for idx in abund.index]
    abund_pretty = abund.copy(); abund_pretty.index = pretty
    colors = plt.get_cmap("tab20c").colors
    color_list = list(colors[:len(abund_pretty)-1]) + [palette.get("Other", "#999999")]
    ax = abund_pretty.T.plot(kind='bar', stacked=True, figsize=(18,7), color=color_list)
    ax.set_ylabel("Relative Abundance (%)"); ax.set_xlabel("Sample"); ax.set_title(title); ax.set_ylim(0,100)
    leg = ax.legend(bbox_to_anchor=(1.01,1), loc='upper left', title=legend_title)
    legend_italicize(leg)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

def barplot_mean_three_groups(oc_pct, oh_pct, fc_pct, rank, out_png, palette=None, n_for_union=10):
    palette = palette or PALETTE
    # union top-N based on OC & FC sums
    top_oc = oc_pct.sum(axis=1).sort_values(ascending=False).head(n_for_union).index
    top_fc = fc_pct.sum(axis=1).sort_values(ascending=False).head(n_for_union).index
    union = pd.Index(top_oc).union(top_fc)
    mean_oc = oc_pct.reindex(union).fillna(0).mean(axis=1)
    mean_fc = fc_pct.reindex(union).fillna(0).mean(axis=1)
    mean_oh = oh_pct.reindex(union).fillna(0).mean(axis=1)
    order = mean_oc.sort_values(ascending=False).index
    x = np.arange(len(order)); width = 0.5
    c_oc = palette.get("Crohn-Oral", "#1f77b4")
    c_oh = palette.get("Healthy-Oral", "#2ca02c")
    c_fc = palette.get("Crohn-Fecal", "#ff7f0e")
    plt.figure(figsize=(16,8)); ax = plt.gca()
    ax.bar(x - width, mean_oc.reindex(order).values, width, label='Oral_Crohn',   color=c_oc)
    ax.bar(x,         mean_oh.reindex(order).values, width, label='Oral_Healthy', color=c_oh)
    ax.bar(x + width, mean_fc.reindex(order).values, width, label='Fecal_Crohn',  color=c_fc)
    ax.set_ylabel("Mean relative abundance (%)")
    ax.set_xlabel(rank.capitalize())
    ax.set_title(f"Mean % (OC, OH, FC) — union of Top{n_for_union} (OC ∪ FC)")
    ax.set_xticks(x)
    pretty = [prettify_taxon(t, rank) for t in order]
    ax.set_xticklabels(pretty, rotation=45, ha='right')
    for lbl in ax.get_xticklabels():
        try: lbl.set_fontstyle('italic')
        except Exception: pass
    ax.legend(loc='center'); ax.axhline(0, linewidth=1)
    plt.tight_layout(rect=[0,0.15,1,1]); plt.savefig(out_png, dpi=220); plt.close()

def two_bar_stacked_oc_oh(oc_mean: pd.Series, oh_mean: pd.Series, rank: str, out_png: str, palette=None):
    palette = palette or PALETTE
    # choose top-N by combined
    combined = oc_mean.add(oh_mean, fill_value=0).sort_values(ascending=False)
    top = combined.index[:max(10, min(20, len(combined)))]
    s_oc = oc_mean.reindex(top).fillna(0); s_oh = oh_mean.reindex(top).fillna(0)
    s_oc = pd.concat([s_oc, pd.Series({"Other": oc_mean.drop(top, errors='ignore').sum()})])
    s_oh = pd.concat([s_oh, pd.Series({"Other": oh_mean.drop(top, errors='ignore').sum()})])
    order = s_oc.sort_values(ascending=False).index
    cmap = plt.get_cmap("tab20c").colors
    color_map = {tax: cmap[i % len(cmap)] for i, tax in enumerate(order)}
    color_map["Other"] = palette.get("Other", "#000000")
    fig, ax = plt.subplots(figsize=(16,8))
    x = np.array([0,1]); b0 = b1 = 0.0
    for tax in order:
        h0 = float(s_oc.loc[tax]); h1 = float(s_oh.loc[tax])
        ax.bar(x[0], h0, bottom=b0, color=color_map[tax], width=0.6)
        ax.bar(x[1], h1, bottom=b1, color=color_map[tax], width=0.6)
        b0 += h0; b1 += h1
    ax.set_xticks(x); ax.set_xticklabels(["Oral_Crohn","Oral_Healthy"])
    ax.set_ylabel("Relative abundance (mean %, per group)")
    ax.set_title(f"Oral {rank.capitalize()} — Top {len(top)-1} (others → Other)")
    handles = [plt.Rectangle((0,0),1,1,color=color_map[t]) for t in order]
    labels = [prettify_taxon(t, rank) if t!="Other" else "Other" for t in order]
    leg = ax.legend(handles, labels, bbox_to_anchor=(1.02,1), loc="upper left", title=rank.capitalize())
    legend_italicize(leg)
    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)

def heatmaps_pairs_and_contrast(oc_pct, oh_pct, fc_pct, rank, outdir, palette=None,
                                n=10, prevalence_thresh=0.10, presence_threshold_percent=0.1,
                                matched_pairs: pd.DataFrame|None=None, use_log1p=True, clip_quantile=0.98,
                                col_cluster_pairs=False):
    palette = palette or PALETTE
    os.makedirs(outdir, exist_ok=True)
    # Row selection (focus Crohn)
    prev_oc = prevalence(oc_pct, presence_threshold_percent)
    prev_fc = prevalence(fc_pct, presence_threshold_percent)
    crohn_union = oc_pct.add(fc_pct, fill_value=0)
    keep = (prev_oc >= prevalence_thresh) & (prev_fc >= prevalence_thresh)
    rows = crohn_union.loc[keep].sum(axis=1).sort_values(ascending=False).head(n).index
    if len(rows) < max(5, int(0.4*n)):
        rows = crohn_union.sum(axis=1).sort_values(ascending=False).head(n).index

    # (1) Pairs clustermap
    data_pairs = pd.concat([oc_pct, fc_pct], axis=1).reindex(rows).fillna(0)
    if matched_pairs is not None and not matched_pairs.empty:
        oc_cols = matched_pairs.iloc[:,0].astype(str).tolist()
        fc_cols = matched_pairs.iloc[:,1].astype(str).tolist()
        inter = []
        for o,f in zip(oc_cols, fc_cols):
            if o in data_pairs.columns: inter.append(o)
            if f in data_pairs.columns: inter.append(f)
        leftovers = [c for c in data_pairs.columns if c not in inter]
        ordered_cols = inter + leftovers
    else:
        ordered_cols = list(data_pairs.columns)
    data_pairs = data_pairs.reindex(columns=ordered_cols)
    pretty_idx = [prettify_taxon(t, rank) for t in data_pairs.index]
    Z = row_zscore_log1p(data_pairs, use_log1p=use_log1p)
    Z.index = pretty_idx
    V = np.nanquantile(np.abs(Z.values), clip_quantile)
    if not np.isfinite(V) or V == 0:
        V = max(1.0, float(np.nanmax(np.abs(Z.values))))
    col_colors = pd.Series(index=Z.columns, dtype=object)
    col_colors.loc[Z.columns.intersection(oc_pct.columns)] = palette.get("Crohn-Oral", "#1f77b4")
    col_colors.loc[Z.columns.intersection(fc_pct.columns)] = palette.get("Crohn-Fecal", "#ff7f0e")
    cmap = make_oc_fc_cmap(palette)
    height = max(6, 0.35*len(Z.index)); width = max(8, 0.18*len(Z.columns)+2)
    g = sns.clustermap(
        Z, cmap=cmap, center=0, vmin=-V, vmax=+V,
        col_colors=col_colors, col_cluster=col_cluster_pairs,
        metric="correlation", method="average",
        figsize=(width, height),
        cbar_kws={"label": "Row z-score of % abundance (log1p)"}
    )
    for lbl in g.ax_heatmap.get_yticklabels():
        try: lbl.set_fontstyle("italic")
        except Exception: pass
    g.ax_heatmap.set_xlabel("Crohn patients (Oral ↔ Fecal)")
    g.ax_heatmap.set_ylabel(rank.capitalize())
    g.fig.suptitle(f"{rank.capitalize()} clustermap — Crohn pairs (OC ↔ FC)", y=1.02)
    g.fig.savefig(os.path.join(outdir, f"heatmap_pairs_crohn_{rank}.png"), dpi=300, bbox_inches="tight")
    plt.close(g.fig)

        # (2) Means & deltas heatmap + CSV
    mean_oc = oc_pct.reindex(rows).fillna(0).mean(axis=1)
    mean_fc = fc_pct.reindex(rows).fillna(0).mean(axis=1)
    mean_oh = oh_pct.reindex(rows).fillna(0).mean(axis=1)
    delta_oc_fc = mean_oc - mean_fc
    delta_oc_oh = mean_oc - mean_oh
    GC = pd.DataFrame({
        "Mean_OC_%": mean_oc, "Mean_OH_%": mean_oh, "Mean_FC_%": mean_fc,
        "Δ(OC−FC)":  delta_oc_fc, "Δ(OC−OH)":  delta_oc_oh,
    })
    GC = GC.loc[delta_oc_fc.abs().sort_values(ascending=False).index]

    # اندازه‌ی پویا با فاصله‌ی بیشتر بین دو هیت‌مپ
    fig_height = max(5, 0.35*len(GC.index))
    fig, axes = plt.subplots(
        1, 2, figsize=(14, fig_height),
        gridspec_kw={'width_ratios':[3,2]}
    )
    fig.subplots_adjust(wspace=0.55)  # << فاصله‌ی بین دو هیت‌مپ

    # Heatmap (gray)
    sns.heatmap(
        GC[["Mean_OC_%","Mean_OH_%","Mean_FC_%"]], ax=axes[0], cmap="Greys",
        cbar_kws={"label":"Mean % abundance"}
    )
    axes[0].set_title("Group means")
    axes[0].set_ylabel(rank.capitalize())
    axes[0].set_xlabel("Groups")

    ylabels = [prettify_taxon(t, rank) for t in GC.index]
    axes[0].set_yticklabels(ylabels, rotation=45, fontsize=9)
    for lbl in axes[0].get_yticklabels():
        try:
            lbl.set_fontstyle("italic")
            lbl.set_ha("right"); lbl.set_va("center")
        except Exception:
            pass

    Vd = np.nanquantile(np.abs(GC[["Δ(OC−FC)", "Δ(OC−OH)"]].values), 0.98)
    if not np.isfinite(Vd) or Vd == 0:
        Vd = max(1e-6, float(np.nanmax(np.abs(GC[["Δ(OC−FC)", "Δ(OC−OH)"]].values))))

   
    sns.heatmap(
        GC[["Δ(OC−FC)","Δ(OC−OH)"]], ax=axes[1], cmap=cmap, center=0,
        vmin=-Vd, vmax=+Vd, cbar_kws={"label":"Delta (percentage points)"}
    )
    axes[1].set_title("Deltas")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Comparisons")

    # rotation labels
    axes[1].set_yticklabels(ylabels, rotation=45, fontsize=9)
    for lbl in axes[1].get_yticklabels():
        try:
            lbl.set_fontstyle("italic")
            lbl.set_ha("right"); lbl.set_va("center")
        except Exception:
            pass

    fig.suptitle(f"{rank.capitalize()} — group means & deltas", y=1.02)
    fig.savefig(os.path.join(outdir, f"heatmap_group_contrast_{rank}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    GC.to_csv(os.path.join(outdir, f"group_means_and_deltas_{rank}.csv"))


# ---------- Differential abundance ----------
def bh_qvalues(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p); order = np.argsort(p); q = np.empty(n, dtype=float); prev = 1.0
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

def _safe_log2_mean(v):
    v = np.asarray(v, dtype=float); v = v[~np.isnan(v)]
    return float(np.log2(np.mean(v) + 1e-9)) if v.size else 0.0

def log2fc(groupA, groupB):
    return _safe_log2_mean(groupA) - _safe_log2_mean(groupB)

def volcano(
    df, x="log2FC_oralCH", q="q", title="", out_png="volcano.png",
    q_sig=0.05, k_onplot_each=8, k_side_each=10, extreme_pct=90,
    figsize=(11.5, 6.8), dpi=340, point_size=22,
    colors=dict(non="#999999", up="#1B4F72", down="#7FB3D5"),
):
    """
    Volcano plot with a clean right panel (tables). Only top-K labels are drawn on-plot.
    """
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.patheffects as pe

    # ---------- data prep
    X  = df[x].astype(float).values
    qv = df[q].astype(float).clip(lower=1e-300).values
    Y  = -np.log10(qv)
    names = (
        df["pretty_taxon"] if "pretty_taxon" in df.columns
        else (df["taxon"] if "taxon" in df.columns else df.index)
    ).astype(str).values

    is_sig = qv <= q_sig
    score = np.abs(X) * (Y + 1)  # importance score for labeling

    up_idx = np.where(is_sig & (X > 0))[0]
    dn_idx = np.where(is_sig & (X < 0))[0]
    up_sel = up_idx[np.argsort(-score[up_idx])[:k_onplot_each]]
    dn_sel = dn_idx[np.argsort(-score[dn_idx])[:k_onplot_each]]
    label_idxs = np.concatenate([up_sel, dn_sel])

    # side tables (sorted by q then |FC|)
    tmp = pd.DataFrame({"name": names, "x": X, "q": qv, "-log10q": Y})
    up_side = (tmp[(tmp["q"] <= q_sig) & (tmp["x"] > 0)]
               .sort_values(["q", "x"], ascending=[True, False]).head(k_side_each))
    dn_side = (tmp[(tmp["q"] <= q_sig) & (tmp["x"] < 0)]
               .sort_values(["q", "x"], ascending=[True, True]).head(k_side_each))

    # ---------- figure layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(ncols=2, nrows=1, width_ratios=[5.2, 2.8], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1]); ax_side.axis("off")

    c_non = colors["non"]; c_up = colors["up"]; c_dn = colors["down"]

    # points
    ax.scatter(X[~is_sig], Y[~is_sig], s=point_size, alpha=0.35, color=c_non, zorder=1)
    ax.scatter(X[is_sig & (X > 0)], Y[is_sig & (X > 0)], s=point_size, alpha=0.9, color=c_up, zorder=2)
    ax.scatter(X[is_sig & (X < 0)], Y[is_sig & (X < 0)], s=point_size, alpha=0.9, color=c_dn, zorder=2)

    # guides
    ax.axhline(-np.log10(q_sig), ls="--", lw=1, color="gray", alpha=0.8)
    ax.axvline(0, ls="--", lw=1, color="gray", alpha=0.8)
    ax.grid(True, ls=":", lw=0.6, alpha=0.4)

    # --- lightweight label-repel (y-only) + white outline
    def repel(ax, xs, ys, texts, fs=9, max_iter=200, pad=0.015):
        import numpy as np
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

    # axes
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel("-log10(q)", fontsize=12)
    ax.set_title(title, fontsize=18, pad=10)

    # ---------- right panel (legend chips + neat tables)
    y = 0.98
    # color chips (act like a legend but off-plot)
    ax_side.scatter([0.03],[y], s=45, color=c_up, transform=ax_side.transAxes)
    ax_side.text(0.07, y, f"q ≤ {q_sig} & up", fontsize=10, va="center",
                 transform=ax_side.transAxes)
    ax_side.scatter([0.53],[y], s=45, color=c_dn, transform=ax_side.transAxes)
    ax_side.text(0.57, y, f"q ≤ {q_sig} & down", fontsize=10, va="center",
                 transform=ax_side.transAxes)
    y -= 0.06
    ax_side.text(0.03, y, f"labeled on-plot: {len(label_idxs)}    total sig: {int(is_sig.sum())}",
                 fontsize=9, color="0.35", transform=ax_side.transAxes)
    y -= 0.02

    # helper to draw a compact table under a separate title
    def mini_table(ax, df_small, x0, y0, title, color,
                   col_widths=(0.58, 0.22, 0.20)):  # wider log2FC and q
        # section title (NOT part of the table)
        ax.text(x0, y0, title, ha="left", va="top", fontsize=11, fontweight="bold",
                color=color, transform=ax.transAxes)
        y_tbl_top = y0 - 0.028  # little gap below the section title

        show = df_small.copy()
        show["log2FC"] = show["x"].map(lambda v: f"{v: .2f}")
        show["q"] = show["q"].map(lambda v: f"{v:.2e}")
        show = show[["name","log2FC","q"]]
        celltxt = show.values.tolist()
        headers = ["taxon","log2FC","q"]

        # bbox=(x, y, w, h) in Axes fraction
        height = min(0.038*max(len(celltxt),1) + 0.06, 0.50)  # autoscale with cap
        table = ax.table(cellText=celltxt, colLabels=headers,
                         colWidths=list(col_widths), loc="upper left",
                         bbox=(x0, y_tbl_top - height, 0.96 - x0, height))
        table.auto_set_font_size(False)
        table.set_fontsize(8.6)
        # zebra rows + thin borders + right-align numbers
        for r in range(1, len(celltxt)+1):
            if r % 2 == 0:
                table[(r,0)].set_facecolor("#F7F7F7")
                table[(r,1)].set_facecolor("#F7F7F7")
                table[(r,2)].set_facecolor("#F7F7F7")
            for c in range(3):
                cell = table[(r,c)]
                cell.set_linewidth(0.4)
                if c > 0: cell._loc = 'right'
        # header style
        for c in range(3):
            h = table[(0,c)]
            h.set_linewidth(0.6); h.set_facecolor("#EDEDED"); h.set_text_props(weight="bold")
        return (y_tbl_top - height) - 0.04  # new y (leave a small gap below)

    y = mini_table(ax_side, up_side, 0.03, y, f"Top {len(up_side)} up (lowest q)", c_up,
                   col_widths=(0.60, 0.22, 0.18))
    _ = mini_table(ax_side, dn_side, 0.03, y, f"Top {len(dn_side)} down (lowest q)", c_dn,
                   col_widths=(0.60, 0.22, 0.18))

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_da_oral_CH(mat_taxa_samples: pd.DataFrame, labels_A, labels_B, tax_labels, out_csv,
                   do_volcano_png=None, rank="genus"):
    rows, pvals = [], []
    for tax in tax_labels:
        if tax not in mat_taxa_samples.index:
            continue
        a = mat_taxa_samples.loc[tax, labels_A].values
        b = mat_taxa_samples.loc[tax, labels_B].values
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            u, p = np.nan, 1.0
        eff  = cliffs_delta(a, b)
        l2fc = log2fc(a, b)
        rows.append({
            "taxon": tax,
            "pretty_taxon": prettify_taxon(tax, rank),
            "test": "MWU",
            "stat": float(u) if np.isfinite(u) else np.nan,
            "p": float(p),
            "effect_name": "Cliffs_delta",
            "effect_size": float(eff),
            "log2FC_oralCH": float(l2fc),
        })
        pvals.append(p)

    # q-values را قبل از ساخت DataFrame به rows اضافه کن
    if pvals:
        qvals = bh_qvalues(pvals)
        for r, qv in zip(rows, qvals):
            r["q"] = float(qv)

    df = pd.DataFrame(rows)
    cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","log2FC_oralCH"]

    if df.empty:
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False)
    else:
        # احتیاط: اگر به هر دلیل q نبود، پرش کن
        if "q" not in df.columns:
            df["q"] = np.nan
        df = df.reindex(columns=cols)
        df.to_csv(out_csv, index=False)

    if do_volcano_png and not df.empty:
        volcano(df, x="log2FC_oralCH", q="q",
                title=f"Volcano Oral Crohn vs Healthy ({rank})",
                out_png=do_volcano_png)

def run_da_paired_OC_FC(mat_taxa_samples_OC: pd.DataFrame, mat_taxa_samples_FC: pd.DataFrame,
                        matched_df: pd.DataFrame, tax_labels, out_csv, rank="genus"):
    oc_cols_raw = matched_df.iloc[:,0].astype(str).tolist()
    fc_cols_raw = matched_df.iloc[:,1].astype(str).tolist()
    # transpose if needed
    if (not set(oc_cols_raw).issubset(mat_taxa_samples_OC.columns)
        and set(oc_cols_raw).issubset(mat_taxa_samples_OC.index)):
        mat_taxa_samples_OC = mat_taxa_samples_OC.T
    if (not set(fc_cols_raw).issubset(mat_taxa_samples_FC.columns)
        and set(fc_cols_raw).issubset(mat_taxa_samples_FC.index)):
        mat_taxa_samples_FC = mat_taxa_samples_FC.T
    # normalize names
    oc_map = _build_id_map(mat_taxa_samples_OC.columns.astype(str))
    fc_map = _build_id_map(mat_taxa_samples_FC.columns.astype(str))
    oc_cols = [oc_map.get(_normalize_id(c)) for c in oc_cols_raw]
    fc_cols = [fc_map.get(_normalize_id(c)) for c in fc_cols_raw]
    miss_oc = [c for c,m in zip(oc_cols_raw, oc_cols) if m is None]
    miss_fc = [c for c,m in zip(fc_cols_raw, fc_cols) if m is None]
    if miss_oc or miss_fc:
        pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median"]).to_csv(out_csv, index=False)
        raise KeyError(f"Missing matched IDs after normalization. OC example: {miss_oc[:5]} | FC example: {miss_fc[:5]}")
    rows, pvals = [], []
    for tax in tax_labels:
        if tax not in mat_taxa_samples_OC.index or tax not in mat_taxa_samples_FC.index: continue
        x = mat_taxa_samples_OC.loc[tax, oc_cols].values.astype(float)
        y = mat_taxa_samples_FC.loc[tax, fc_cols].values.astype(float)
        try: w, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        except ValueError: w, p = np.nan, 1.0
        diff = (y - x); diff = diff[~np.isnan(diff)]; n = diff.size
        r = np.nan
        if n>0 and np.isfinite(w):
            denom = n*(n+1)/2.0
            r = 1.0 - 2.0*(w/denom)
        rows.append({"taxon":tax, "pretty_taxon":prettify_taxon(tax, rank), "test":"Wilcoxon",
                     "stat":float(w) if np.isfinite(w) else np.nan, "p":float(p),
                     "effect_name":"rank_biserial_r", "effect_size":float(r) if np.isfinite(r) else np.nan,
                     "delta_median":float(np.nanmedian(y) - np.nanmedian(x))})
        pvals.append(p)
    if pvals:
        qvals = bh_qvalues(pvals)
        for r,q in zip(rows,qvals): r["q"] = float(q)
    cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median"]
    pd.DataFrame(rows)[cols].to_csv(out_csv, index=False)
