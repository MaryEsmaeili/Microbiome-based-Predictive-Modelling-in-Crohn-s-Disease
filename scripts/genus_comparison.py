import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------- Utilities -----------
def extract_genus(rowname):
    parts = rowname.split("|")
    genus = [p for p in parts if p.startswith("g__")]
    return genus[0] if genus else None

# ----------- Functions -----------

def per_patient_overlap(oral_crohn, fecal_crohn, matched_ids, outdir):
    """
    For each matched patient, find shared genera, oral-only, and fecal-only genera.
    Write report as CSV.
    """
    report = []
    for idx, row in matched_ids.iterrows():
        patient_id = f"Patient_{idx+1}"
        oc_col, fc_col = row['Oral_col'], row['Fecal_col']
        oc_g = set(map(extract_genus, oral_crohn[oc_col][oral_crohn[oc_col] > 0].index))
        fc_g = set(map(extract_genus, fecal_crohn[fc_col][fecal_crohn[fc_col] > 0].index))
        shared = sorted(oc_g & fc_g)
        only_oral = sorted(oc_g - fc_g)
        only_fecal = sorted(fc_g - oc_g)
        report.append({
            "Patient": patient_id,
            "Shared genera": ", ".join(shared) if shared else "-",
            "Oral-only genera": ", ".join(only_oral) if only_oral else "-",
            "Fecal-only genera": ", ".join(only_fecal) if only_fecal else "-"
        })
    report_df = pd.DataFrame(report)
    report_df.to_csv(os.path.join(outdir, "patient_matched_genus_presence.csv"), index=False)
    return report_df

def oral_crohn_vs_healthy(oral_crohn, oral_healthy, outdir):
    """
    Compare genera between Oral Crohn and Oral Healthy groups.
    Writes a transposed CSV report: shared, healthy-only, crohn-only.
    """
    oral_crohn_genera = set(map(extract_genus, oral_crohn.index))
    oral_healthy_genera = set(map(extract_genus, oral_healthy.index))
    shared = sorted(oral_crohn_genera & oral_healthy_genera)
    only_healthy = sorted(oral_healthy_genera - oral_crohn_genera)
    only_crohn = sorted(oral_crohn_genera - oral_healthy_genera)
    report = {
        "Shared genera": ", ".join(shared) if shared else "-",
        "Oral-Healthy only genera": ", ".join(only_healthy) if only_healthy else "-",
        "Oral-Crohn only genera": ", ".join(only_crohn) if only_crohn else "-"
    }
    report_df = pd.DataFrame([report])
    report_df.T.to_csv(os.path.join(outdir, "oral_crohn_vs_healthy_genera_comparison.csv"), header=False)
    return report_df

def stacked_barplot_matched_oral_genus(oral_crohn, matched_ids, outdir, topN=12):
    """
    For all matched oral Crohn samples: plot stacked barplot of topN genera per patient.
    """
    genus_abund = []
    for idx, row in matched_ids.iterrows():
        oc_col = row['Oral_col']
        abund = oral_crohn[oc_col].copy()
        abund = abund.groupby(extract_genus).sum()
        genus_abund.append(abund)
    abund_df = pd.DataFrame(genus_abund).fillna(0).T
    abund_df.columns = [f"Patient_{i+1}" for i in range(len(matched_ids))]
    top_genera = abund_df.sum(axis=1).sort_values(ascending=False).head(topN).index
    abund_plot = abund_df.copy()
    abund_plot.loc['Other'] = abund_df.loc[~abund_df.index.isin(top_genera)].sum()
    abund_plot = abund_plot.loc[list(top_genera) + ['Other']]
    abund_plot.T.plot(kind='bar', stacked=True, figsize=(22,7), colormap="tab20")
    plt.ylabel("Total Abundance")
    plt.xlabel("Matched Patient")
    plt.title(f"Stacked Barplot (Top {topN} Genus) - Oral Crohn per Patient")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Genus')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "stackedbar_matched_oral_genus.png"))
    plt.close()
    return abund_plot

def genus_barplot(oral_crohn, oral_healthy, outdir, topN=15):
    """
    Plot a barplot of the topN genera by relative abundance (Oral Crohn vs Oral Healthy).
    """
    oral_crohn = oral_crohn.copy()
    oral_healthy = oral_healthy.copy()
    oral_crohn['Genus'] = oral_crohn.index.map(extract_genus)
    oral_healthy['Genus'] = oral_healthy.index.map(extract_genus)
    abund_crohn = oral_crohn.groupby('Genus').sum().drop(columns='Genus', errors='ignore').sum(axis=1)
    abund_healthy = oral_healthy.groupby('Genus').sum().drop(columns='Genus', errors='ignore').sum(axis=1)
    df = pd.DataFrame({
        'Oral_Crohn': abund_crohn,
        'Oral_Healthy': abund_healthy
    }).fillna(0)
    df_norm = df.div(df.sum(axis=0), axis=1) * 100
    df_norm = df_norm.sort_values('Oral_Crohn', ascending=False)
    df_top = df_norm.head(topN)
    df_top.plot(kind='bar', width=0.9, figsize=(18,7), color=["#DC143C", "#4682B4"])
    plt.ylabel("Relative Abundance (%)")
    plt.xlabel("Genus")
    plt.title(f"Genus-level Relative Abundance (Top {topN}) - Oral Crohn vs Oral Healthy")
    plt.legend(["Oral Crohn", "Oral Healthy"], fontsize=12)
    plt.tight_layout()
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    plt.grid(axis='y')
    plt.savefig(os.path.join(outdir, "genus_barplot_oral_crohn_vs_healthy.png"))
    plt.close()
    return df_top




# ----------- Main Entrypoint: Snakemake or CLI -----------
if __name__ == "__main__":
    try:
        # If run by Snakemake
        oral_crohn_file = snakemake.input.oral_crohn
        fecal_crohn_file = snakemake.input.fecal_crohn
        oral_healthy_file = snakemake.input.oral_healthy
        matched_ids_file = snakemake.input.matched_ids
        outdir = snakemake.params.outdir
    except NameError:
        # If run as standalone script
        import sys
        oral_crohn_file = sys.argv[1]
        fecal_crohn_file = sys.argv[2]
        oral_healthy_file = sys.argv[3]
        matched_ids_file = sys.argv[4]
        outdir = sys.argv[5]
    os.makedirs(outdir, exist_ok=True)
    oral_crohn = pd.read_csv(oral_crohn_file, index_col=0)
    fecal_crohn = pd.read_csv(fecal_crohn_file, index_col=0)
    oral_healthy = pd.read_csv(oral_healthy_file, index_col=0)
    matched_ids = pd.read_csv(matched_ids_file)
    per_patient_overlap(oral_crohn, fecal_crohn, matched_ids, outdir)
    stacked_barplot_matched_oral_genus(oral_crohn, matched_ids, outdir, topN=12)
    oral_crohn_vs_healthy(oral_crohn, oral_healthy, outdir)
    genus_barplot(oral_crohn, oral_healthy, outdir, topN=15)
    print("[INFO] All done! Check:", outdir)

