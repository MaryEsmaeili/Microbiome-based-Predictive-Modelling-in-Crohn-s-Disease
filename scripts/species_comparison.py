# scripts/species_comparison.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from microbiome_common import (
    PALETTE,
    ensure_taxa_by_samples,
    extract_species,
    group_mean_percent,
    percent_table,
    plot_group_stacked,
    save_top_csv,
    run_da_oral_CH,       # OC vs OH (x-col: log2FC_oralCH)
    run_da_fecal_CH,      # FC vs FH (x-col: log2FC_fecalCH)
    run_da_paired_OC_FC,  # paired OC vs FC
    clr_transform,
    prettify_taxon,
)

sns.set_context("talk")


# ---------- helpers (لوکال) ----------
def _zscore_rows_log1p(mat: pd.DataFrame) -> pd.DataFrame:
    X = np.log1p(mat.astype(float))
    mu = X.mean(axis=1)
    sd = X.std(axis=1).replace(0, np.nan)
    Z = X.sub(mu, axis=0).div(sd, axis=0).fillna(0)
    return Z


def _heatmap_pairs(A_pct: pd.DataFrame, B_pct: pd.DataFrame,
                   label_a: str, label_b: str, out_png: str, n: int = 15):
    """Heatmap برای جفت‌ها (مثلاً O.C.↔F.C. یا O.H.↔F.H.)"""
    A = A_pct.copy(); B = B_pct.copy()
    taxa = A.index.union(B.index)
    A = A.reindex(taxa).fillna(0); B = B.reindex(taxa).fillna(0)
    P = pd.concat([A, B], axis=1)

    # انتخاب top n بر اساس mean log1p
    top = np.log1p(P).mean(axis=1).sort_values(ascending=False).head(n).index
    P = P.loc[top]
    Z = _zscore_rows_log1p(P)

    # رنگ ستون‌ها
    col_colors = []
    for c in P.columns:
        if c in A_pct.columns:
            col_colors.append(PALETTE.get("Crohn-Oral", "#1f77b4") if label_a.startswith("O")
                              else PALETTE.get("Healthy-Oral", "#2ca02c"))
        else:
            col_colors.append(PALETTE.get("Crohn-Fecal", "#ff7f0e") if label_b.startswith("F")
                              else PALETTE.get("Healthy-Fecal", "#9467bd"))

    # اسم‌های زیبا (ایتالیک)
    pretty_idx = [prettify_taxon(t, "species") for t in Z.index]
    Z.index = pretty_idx

    # سایز پویا
    height = max(6, 0.35 * len(Z.index))
    width = max(8, 0.18 * len(Z.columns) + 2)

    g = sns.clustermap(
        Z, row_cluster=True, col_cluster=False, cmap="Purples",
        col_colors=col_colors, xticklabels=False, yticklabels=True,
        figsize=(width, height)
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("Species (top)")
    g.fig.suptitle(f"Pairs clustermap — {label_a} ↔ {label_b}", y=1.02)
    g.fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def _group_contrast_plots_and_csv(oc_pct, fc_pct, oh_pct, fh_pct,
                                  out_means_png, out_deltas_png,
                                  out_means_csv, out_deltas_csv, n=15):
    """چهارگروه: O.C., F.C., O.H., F.H. — خروجی heatmap + CSV (means & deltas)"""
    means_df = pd.DataFrame({
        "O.C.": oc_pct.mean(axis=1),
        "F.C.": fc_pct.mean(axis=1),
        "O.H.": oh_pct.mean(axis=1),
        "F.H.": fh_pct.mean(axis=1),
    }).fillna(0.0)

    deltas_df = pd.DataFrame({
        "O.C.–O.H.": means_df["O.C."] - means_df["O.H."],
        "F.C.–F.H.": means_df["F.C."] - means_df["F.H."],
    })

    keep = (
        pd.Index(means_df.mean(1).sort_values(ascending=False).head(n).index)
        .union(deltas_df.abs().max(1).sort_values(ascending=False).head(n).index)
    )

    # CSVهایی که Snakefile می‌خواهد
    means_df.loc[keep].to_csv(out_means_csv)
    deltas_df.loc[keep].to_csv(out_deltas_csv)

    # هیت‌مپ میانگین‌ها
    Zm = _zscore_rows_log1p(means_df.loc[keep, ["O.C.", "F.C.", "O.H.", "F.H."]])
    g1 = sns.clustermap(
        Zm, row_cluster=True, col_cluster=False, cmap="Purples",
        xticklabels=True, yticklabels=True, figsize=(12, max(6, 0.45*len(keep)))
    )
    g1.ax_heatmap.set_xlabel("Groups")
    g1.ax_heatmap.set_ylabel("Species (selected)")
    g1.fig.suptitle("Group means (O.C., F.C., O.H., F.H.)", y=1.02)
    g1.fig.savefig(out_means_png, dpi=300, bbox_inches="tight")
    plt.close(g1.fig)

    # هیت‌مپ دلتاها
    Zd = _zscore_rows_log1p(deltas_df.loc[keep, ["O.C.–O.H.", "F.C.–F.H."]])
    g2 = sns.clustermap(
        Zd, row_cluster=True, col_cluster=False, cmap="Purples",
        xticklabels=True, yticklabels=True, figsize=(8, max(6, 0.45*len(keep)))
    )
    g2.ax_heatmap.set_xlabel("Deltas")
    g2.ax_heatmap.set_ylabel("Species (selected)")
    g2.fig.suptitle("Group deltas (O.C.–O.H., F.C.–F.H.)", y=1.02)
    g2.fig.savefig(out_deltas_png, dpi=300, bbox_inches="tight")
    plt.close(g2.fig)


def _two_bar_stacked_generic(a_mean: pd.Series, b_mean: pd.Series, rank: str, out_png: str,
                             label_a: str, label_b: str, top_k: int = 15):
    """دو ستون روی‌هم؛ top_k + Other"""
    combined = a_mean.add(b_mean, fill_value=0).sort_values(ascending=False)
    top = combined.index[:min(top_k, len(combined))]
    s_a = a_mean.reindex(top).fillna(0)
    s_b = b_mean.reindex(top).fillna(0)
    s_a = pd.concat([s_a, pd.Series({"Other": a_mean.drop(top, errors='ignore').sum()})])
    s_b = pd.concat([s_b, pd.Series({"Other": b_mean.drop(top, errors='ignore').sum()})])

    order = s_a.sort_values(ascending=False).index
    cmap = plt.get_cmap("tab20c").colors
    color_map = {tax: cmap[i % len(cmap)] for i, tax in enumerate(order)}
    color_map["Other"] = PALETTE.get("Other", "#999999")

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.array([0, 1]); b0 = b1 = 0.0
    for tax in order:
        h0 = float(s_a.loc[tax]); h1 = float(s_b.loc[tax])
        ax.bar(x[0], h0, bottom=b0, color=color_map[tax], width=0.6)
        ax.bar(x[1], h1, bottom=b1, color=color_map[tax], width=0.6)
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
    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)


# ---------- main ----------
def main(oral_crohn_file, oral_healthy_file, fecal_crohn_file, healthy_fecal_file, outdir):
    os.makedirs(outdir, exist_ok=True)

    # بارگذاری و راستا (taxa x samples)
    oc = ensure_taxa_by_samples(pd.read_csv(oral_crohn_file, index_col=0))
    oh = ensure_taxa_by_samples(pd.read_csv(oral_healthy_file, index_col=0))
    fc = ensure_taxa_by_samples(pd.read_csv(fecal_crohn_file, index_col=0))
    fh = ensure_taxa_by_samples(pd.read_csv(healthy_fecal_file, index_col=0))

    # Stacked bars + CSV (Top15)
    plot_group_stacked(
        oc, 15, "Top 15 Species — Oral Crohn",
        os.path.join(outdir, "stackedbar_top15_oral_crohn_species.png"),
        extract_species, "Species", rank="species", palette=PALETTE
    )
    save_top_csv(oc, 15, extract_species, os.path.join(outdir, "top15_oral_crohn_species.csv"))

    plot_group_stacked(
        oh, 15, "Top 15 Species — Oral Healthy",
        os.path.join(outdir, "stackedbar_top15_oral_healthy_species.png"),
        extract_species, "Species", rank="species", palette=PALETTE
    )
    save_top_csv(oh, 15, extract_species, os.path.join(outdir, "top15_oral_healthy_species.csv"))

    plot_group_stacked(
        fc, 15, "Top 15 Species — Fecal Crohn",
        os.path.join(outdir, "stackedbar_top15_fecal_crohn_species.png"),
        extract_species, "Species", rank="species", palette=PALETTE
    )
    save_top_csv(fc, 15, extract_species, os.path.join(outdir, "top15_fecal_crohn_species.csv"))

    plot_group_stacked(
        fh, 15, "Top 15 Species — Fecal Healthy",
        os.path.join(outdir, "stackedbar_top15_fecal_healthy_species.png"),
        extract_species, "Species", rank="species", palette=PALETTE
    )
    save_top_csv(fh, 15, extract_species, os.path.join(outdir, "top15_fecal_healthy_species.csv"))

    # % tables
    oc_pct = percent_table(oc, extract_species)
    oh_pct = percent_table(oh, extract_species)
    fc_pct = percent_table(fc, extract_species)
    fh_pct = percent_table(fh, extract_species)

    # Heatmap pairs (Crohn & Healthy)
    _heatmap_pairs(oc_pct, fc_pct, "O.C.", "F.C.", os.path.join(outdir, "heatmap_pairs_crohn_species.png"), n=15)
    _heatmap_pairs(oh_pct, fh_pct, "O.H.", "F.H.", os.path.join(outdir, "heatmap_pairs_healthy_species.png"), n=15)

    # Group means/deltas + CSV + Heatmaps (نام‌گذاری مطابق Snakefile)
    _group_contrast_plots_and_csv(
        oc_pct, fc_pct, oh_pct, fh_pct,
        out_means_png=os.path.join(outdir, "heatmap_group_means_species.png"),
        out_deltas_png=os.path.join(outdir, "heatmap_group_deltas_species.png"),
        out_means_csv=os.path.join(outdir, "group_means_species.csv"),
        out_deltas_csv=os.path.join(outdir, "group_deltas_species.csv"),
        n=15
    )

    # خلاصهٔ میانگین‌ها (برای means_and_mean_species.csv)
    top_oc = oc_pct.sum(axis=1).nlargest(15).index
    top_fc = fc_pct.sum(axis=1).nlargest(15).index
    union = top_oc.union(top_fc)
    means_union = pd.DataFrame({
        "Mean_OC_%": oc_pct.reindex(union).fillna(0).mean(axis=1),
        "Mean_OH_%": oh_pct.reindex(union).fillna(0).mean(axis=1),
        "Mean_FC_%": fc_pct.reindex(union).fillna(0).mean(axis=1),
        "Mean_FH_%": fh_pct.reindex(union).fillna(0).mean(axis=1),
    })
    means_union.to_csv(os.path.join(outdir, "means_and_mean_species.csv"))

    # Two-bar stacked: Oral (OC vs OH) و Fecal (FC vs FH)
    _two_bar_stacked_generic(
        group_mean_percent(oc, extract_species),
        group_mean_percent(oh, extract_species),
        rank="species",
        out_png=os.path.join(outdir, "barplot_oral_species_top15.png"),
        label_a="Oral_Crohn", label_b="Oral_Healthy", top_k=15
    )
    _two_bar_stacked_generic(
        group_mean_percent(fc, extract_species),
        group_mean_percent(fh, extract_species),
        rank="species",
        out_png=os.path.join(outdir, "barplot_fecal_species_top15.png"),
        label_a="Fecal_Crohn", label_b="Fecal_Healthy", top_k=15
    )

    # ---------- Differential abundance (Crohn vs Healthy) ----------
    all_taxa = (
        pd.Index(oc_pct.index)
        .union(fc_pct.index)
        .union(oh_pct.index)
        .union(fh_pct.index)
    )

    # ORAL: OC vs OH
    run_da_oral_CH(
        mat_taxa_samples=oc_pct.reindex(all_taxa).fillna(0).join(
            oh_pct.reindex(all_taxa).fillna(0), how="outer"
        ),
        labels_A=oc_pct.columns,
        labels_B=oh_pct.columns,
        tax_labels=all_taxa,
        out_csv=os.path.join(outdir, "da_oral_CH_species.csv"),
        do_volcano_png=os.path.join(outdir, "volcano_oral_CH_species.png"),
        rank="species",
    )

    # FECAL: FC vs FH
    run_da_fecal_CH(
        mat_taxa_samples=fc_pct.reindex(all_taxa).fillna(0).join(
            fh_pct.reindex(all_taxa).fillna(0), how="outer"
        ),
        labels_A=fc_pct.columns,
        labels_B=fh_pct.columns,
        tax_labels=all_taxa,
        out_csv=os.path.join(outdir, "da_fecal_CH_species.csv"),
        do_volcano_png=os.path.join(outdir, "volcano_fecal_CH_species.png"),
        rank="species",
    )

    # Paired OC vs FC
    try:
        matched = pd.read_csv("data/processed/matched_sample_ids.csv")
    except Exception:
        matched = None

    empty_cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median"]
    if matched is not None and not matched.empty:
        try:
            run_da_paired_OC_FC(
                mat_taxa_samples_OC=oc_pct,
                mat_taxa_samples_FC=fc_pct,
                matched_df=matched,
                tax_labels=all_taxa,
                out_csv=os.path.join(outdir, "da_paired_OC_FC_species.csv"),
                rank="species"
            )
        except Exception as e:
            print("[WARN] paired OC-FC:", e)
            pd.DataFrame(columns=empty_cols).to_csv(os.path.join(outdir, "da_paired_OC_FC_species.csv"), index=False)
    else:
        pd.DataFrame(columns=empty_cols).to_csv(os.path.join(outdir, "da_paired_OC_FC_species.csv"), index=False)

    # ---------- CLR variants (independent + paired) ----------
    def _safe_clr(df, rank_fn):
        try:
            agg = df.groupby(rank_fn).sum(numeric_only=True)
            if agg.empty:
                return pd.DataFrame()
            return clr_transform(agg)
        except Exception as e:
            print("[WARN] CLR transform failed:", e)
            return pd.DataFrame()

    # محاسبهٔ CLR (ایمن)
    clr_oc = _safe_clr(oc, extract_species)
    clr_fc = _safe_clr(fc, extract_species)
    clr_oh = _safe_clr(oh, extract_species)
    clr_fh = _safe_clr(fh, extract_species)

    def _idx(df):
        return df.index if isinstance(df, pd.DataFrame) and not df.empty else pd.Index([])

    clr_taxa = _idx(clr_oc).union(_idx(clr_fc)).union(_idx(clr_oh)).union(_idx(clr_fh))

    # --- ORAL CLR: OC vs OH
    cols_oral = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","log2FC_oralCH"]
    if not clr_oc.empty and not clr_oh.empty and len(clr_taxa) > 0:
        run_da_oral_CH(
            mat_taxa_samples=clr_oc.reindex(clr_taxa).fillna(0).join(
                clr_oh.reindex(clr_taxa).fillna(0), how="outer"
            ),
            labels_A=clr_oc.columns,
            labels_B=clr_oh.columns,
            tax_labels=clr_taxa,
            out_csv=os.path.join(outdir, "clr_da_oral_CH_species.csv"),
            do_volcano_png=None,
            rank="species"
        )
    else:
        pd.DataFrame(columns=cols_oral).to_csv(os.path.join(outdir, "clr_da_oral_CH_species.csv"), index=False)

    # --- FECAL CLR: FC vs FH
    cols_fecal = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","log2FC_fecalCH"]
    if not clr_fc.empty and not clr_fh.empty and len(clr_taxa) > 0:
        run_da_fecal_CH(
            mat_taxa_samples=clr_fc.reindex(clr_taxa).fillna(0).join(
                clr_fh.reindex(clr_taxa).fillna(0), how="outer"
            ),
            labels_A=clr_fc.columns,
            labels_B=clr_fh.columns,
            tax_labels=clr_taxa,
            out_csv=os.path.join(outdir, "clr_da_fecal_CH_species.csv"),
            do_volcano_png=None,
            rank="species"
        )
    else:
        pd.DataFrame(columns=cols_fecal).to_csv(os.path.join(outdir, "clr_da_fecal_CH_species.csv"), index=False)

    # --- Paired CLR: OC vs FC
    empty_cols_paired = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median"]
    if (matched is not None and not matched.empty and
        not clr_oc.empty and not clr_fc.empty and len(clr_taxa) > 0):
        try:
            run_da_paired_OC_FC(
                mat_taxa_samples_OC=clr_oc,
                mat_taxa_samples_FC=clr_fc,
                matched_df=matched,
                tax_labels=clr_taxa,
                out_csv=os.path.join(outdir, "clr_da_paired_OC_FC_species.csv"),
                rank="species"
            )
        except Exception as e:
            print("[WARN] paired CLR OC-FC:", e)
            pd.DataFrame(columns=empty_cols_paired).to_csv(os.path.join(outdir, "clr_da_paired_OC_FC_species.csv"), index=False)
    else:
        pd.DataFrame(columns=empty_cols_paired).to_csv(os.path.join(outdir, "clr_da_paired_OC_FC_species.csv"), index=False)

    # Top10 برای اسلاید (از OC vs OH)
    try:
        df_sig = pd.read_csv(os.path.join(outdir, "da_oral_CH_species.csv"))
        if not df_sig.empty and "q" in df_sig.columns:
            df_sig.sort_values("q").head(10)[["taxon", "pretty_taxon", "q"]].to_csv(
                os.path.join(outdir, "top10_significant_species.csv"), index=False
            )
        else:
            pd.DataFrame(columns=["taxon", "pretty_taxon", "q"]).to_csv(
                os.path.join(outdir, "top10_significant_species.csv"), index=False
            )
    except Exception:
        pd.DataFrame(columns=["taxon", "pretty_taxon", "q"]).to_csv(
            os.path.join(outdir, "top10_significant_species.csv"), index=False
        )

    print("[INFO] Species comparison done ->", outdir)


if __name__ == "__main__":
    try:
        oral_crohn_file    = snakemake.input[0]
        oral_healthy_file  = snakemake.input[1]
        fecal_crohn_file   = snakemake.input[2]
        healthy_fecal_file = snakemake.input[3]
        outdir = "results/species_comparison"
    except NameError:
        import sys
        if len(sys.argv) < 6:
            raise SystemExit("Usage: python species_comparison.py <oral_crohn.csv> <oral_healthy.csv> <fecal_crohn.csv> <healthy_fecal.csv> <outdir>")
        oral_crohn_file, oral_healthy_file, fecal_crohn_file, healthy_fecal_file, outdir = sys.argv[1:6]
    main(oral_crohn_file, oral_healthy_file, fecal_crohn_file, healthy_fecal_file, outdir)
