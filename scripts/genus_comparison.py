# scripts/genus_comparison.py
import os, pandas as pd
from microbiome_common import (
    PALETTE, ensure_taxa_by_samples, extract_genus, group_mean_percent,
    percent_table, plot_group_stacked, save_top_csv, barplot_mean_three_groups,
    two_bar_stacked_oc_oh, heatmaps_pairs_and_contrast,
    run_da_oral_CH, run_da_paired_OC_FC, clr_transform
)

def main(oral_crohn_file, oral_healthy_file, fecal_crohn_file, outdir):
    os.makedirs(outdir, exist_ok=True)
    oc_raw = pd.read_csv(oral_crohn_file, index_col=0)
    oh_raw = pd.read_csv(oral_healthy_file, index_col=0)
    fc_raw = pd.read_csv(fecal_crohn_file, index_col=0)

    oc = ensure_taxa_by_samples(oc_raw)
    oh = ensure_taxa_by_samples(oh_raw)
    fc = ensure_taxa_by_samples(fc_raw)

    # Stacked bars + CSV
    plot_group_stacked(oc, 10, "Top 10 Genera - Oral Crohn Samples",
                       os.path.join(outdir, "stackedbar_top10_oral_crohn.png"),
                       extract_genus, "Genus", rank="genus", palette=PALETTE)
    save_top_csv(oc, 10, extract_genus, os.path.join(outdir, "top10_oral_crohn.csv"))

    plot_group_stacked(oh, 10, "Top 10 Genera - Oral Healthy Samples",
                       os.path.join(outdir, "stackedbar_top10_oral_healthy.png"),
                       extract_genus, "Genus", rank="genus", palette=PALETTE)
    save_top_csv(oh, 10, extract_genus, os.path.join(outdir, "top10_oral_healthy.csv"))

    plot_group_stacked(fc, 10, "Top 10 Genera - Fecal Crohn Samples",
                       os.path.join(outdir, "stackedbar_top10_fecal_crohn.png"),
                       extract_genus, "Genus", rank="genus", palette=PALETTE)
    save_top_csv(fc, 10, extract_genus, os.path.join(outdir, "top10_fecal_crohn.csv"))

    # % tables
    oc_pct = percent_table(oc, extract_genus)
    oh_pct = percent_table(oh, extract_genus)
    fc_pct = percent_table(fc, extract_genus)

    # Heatmaps (pairs + contrast)
    try:
        matched = pd.read_csv("data/processed/matched_sample_ids.csv")
    except Exception:
        matched = None
    heatmaps_pairs_and_contrast(oc_pct, oh_pct, fc_pct, rank="genus", outdir=outdir,
                                palette=PALETTE, n=10, matched_pairs=matched,
                                use_log1p=True, clip_quantile=0.98, col_cluster_pairs=False)

    # Mean barplot OC/OH/FC
    barplot_mean_three_groups(
        oc_pct, oh_pct, fc_pct, rank="genus",
        out_png=os.path.join(outdir, "barplot_mean_genus.png"),
        palette=PALETTE, n_for_union=10
    )
    # Save means_and_mean_genus.csv to satisfy Snakemake outputs
    top_oc = oc_pct.sum(axis=1).sort_values(ascending=False).head(10).index
    top_fc = fc_pct.sum(axis=1).sort_values(ascending=False).head(10).index
    union  = top_oc.union(top_fc)
    mean_oc = oc_pct.reindex(union).fillna(0).mean(axis=1)
    mean_fc = fc_pct.reindex(union).fillna(0).mean(axis=1)
    mean_oh = oh_pct.reindex(union).fillna(0).mean(axis=1)
    pd.DataFrame({"Mean_OC_%": mean_oc, "Mean_FC_%": mean_fc, "Mean_OH_%": mean_oh}) \
    .to_csv(os.path.join(outdir, "means_and_mean_genus.csv"))

    # Two-bar OC vs OH
    mean_oc = group_mean_percent(oc, extract_genus)
    mean_oh = group_mean_percent(oh, extract_genus)
    two_bar_stacked_oc_oh(
        oc_mean=mean_oc, oh_mean=mean_oh, rank="genus",
        out_png=os.path.join(outdir, "barplot_oral_genus_top10.png"),
        palette=PALETTE
    )

    # Differential abundance
    all_taxa = pd.Index(oc_pct.index).union(fc_pct.index).union(oh_pct.index)

    run_da_oral_CH(
        mat_taxa_samples=oc_pct.reindex(all_taxa).fillna(0).join(
            oh_pct.reindex(all_taxa).fillna(0), how="outer"
        ),
        labels_A=oc_pct.columns, labels_B=oh_pct.columns, tax_labels=all_taxa,
        out_csv=os.path.join(outdir, "da_oral_CH_genus.csv"),
        do_volcano_png=os.path.join(outdir, "volcano_oral_CH_genus.png"),
        rank="genus"
    )

    if matched is not None and not matched.empty:
        try:
            run_da_paired_OC_FC(
                mat_taxa_samples_OC=oc_pct, mat_taxa_samples_FC=fc_pct,
                matched_df=matched, tax_labels=all_taxa,
                out_csv=os.path.join(outdir, "da_paired_OC_FC_genus.csv"),
                rank="genus"
            )
        except KeyError as e:
            print("[WARN]", e)
            empty_cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median"]
            pd.DataFrame(columns=empty_cols).to_csv(os.path.join(outdir, "da_paired_OC_FC_genus.csv"), index=False)
    else:
        empty_cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median"]
        pd.DataFrame(columns=empty_cols).to_csv(os.path.join(outdir, "da_paired_OC_FC_genus.csv"), index=False)

    # CLR variants
    clr_oc = clr_transform(oc.groupby(extract_genus).sum(numeric_only=True))
    clr_fc = clr_transform(fc.groupby(extract_genus).sum(numeric_only=True))
    clr_oh = clr_transform(oh.groupby(extract_genus).sum(numeric_only=True))
    clr_taxa = pd.Index(clr_oc.index).union(clr_fc.index).union(clr_oh.index)

    run_da_oral_CH(
        mat_taxa_samples=clr_oc.reindex(clr_taxa).fillna(0).join(
            clr_oh.reindex(clr_taxa).fillna(0), how="outer"
        ),
        labels_A=clr_oc.columns, labels_B=clr_oh.columns, tax_labels=clr_taxa,
        out_csv=os.path.join(outdir, "clr_da_oral_CH_genus.csv"),
        do_volcano_png=None, rank="genus"
    )

    if matched is not None and not matched.empty:
        try:
            run_da_paired_OC_FC(
                mat_taxa_samples_OC=clr_oc, mat_taxa_samples_FC=clr_fc,
                matched_df=matched, tax_labels=clr_taxa,
                out_csv=os.path.join(outdir, "clr_da_paired_OC_FC_genus.csv"),
                rank="genus"
            )
        except KeyError as e:
            print("[WARN]", e)
            empty_cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median"]
            pd.DataFrame(columns=empty_cols).to_csv(os.path.join(outdir, "clr_da_paired_OC_FC_genus.csv"), index=False)
    else:
        empty_cols = ["taxon","pretty_taxon","test","stat","p","q","effect_name","effect_size","delta_median"]
        pd.DataFrame(columns=empty_cols).to_csv(os.path.join(outdir, "clr_da_paired_OC_FC_genus.csv"), index=False)

    # Top10 for slides (optional)
    try:
        df_sig = pd.read_csv(os.path.join(outdir, "da_oral_CH_genus.csv"))
        if not df_sig.empty and "q" in df_sig.columns:
            df_sig.sort_values("q").head(10).to_csv(os.path.join(outdir, "top10_significant_genus.csv"), index=False)
        else:
            pd.DataFrame(columns=["taxon","pretty_taxon","q"]).to_csv(os.path.join(outdir, "top10_significant_genus.csv"), index=False)
    except Exception:
        pd.DataFrame(columns=["taxon","pretty_taxon","q"]).to_csv(os.path.join(outdir, "top10_significant_genus.csv"), index=False)

    print("[INFO] Genus comparison done ->", outdir)

if __name__ == "__main__":
    try:
        oral_crohn_file = snakemake.input[0]
        oral_healthy_file = snakemake.input[1]
        fecal_crohn_file = snakemake.input[2]
        outdir = "results/genus_comparison"
    except NameError:
        import sys, os
        if len(sys.argv) < 5:
            raise SystemExit("Usage: python genus_comparison.py <oral_crohn.csv> <oral_healthy.csv> <fecal_crohn.csv> <outdir>")
        oral_crohn_file, oral_healthy_file, fecal_crohn_file, outdir = sys.argv[1:5]
    main(oral_crohn_file, oral_healthy_file, fecal_crohn_file, outdir)
