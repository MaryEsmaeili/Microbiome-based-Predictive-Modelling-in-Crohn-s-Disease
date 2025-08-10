import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml

# yaml colors
with open("config.yaml") as f:
    config = yaml.safe_load(f)
palette = config['colors']

def extract_genus(rowname):
    """
    Extract genus name from a Metaphlan taxonomic label string (e.g., 'k__Bacteria|p__...|g__Streptococcus').
    """
    parts = rowname.split("|")
    genus = [p for p in parts if p.startswith("g__")]
    return genus[0] if genus else None

def save_top10_csv(genus_sum, out_path):
    """
    Save the top 10 genera (with total abundance) as a CSV file.
    """
    top10 = genus_sum.sort_values(ascending=False).head(10)
    top10.to_csv(out_path, header=['Total_Abundance'])

def plot_stacked_bar(abund_plot, out_path, title):
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("tab20").colors
    color_list = list(colors[:len(abund_plot)-1]) + ['#999999']
    abund_plot.T.plot(kind='bar', stacked=True, figsize=(18,7), color=color_list)
    plt.ylabel("Relative Abundance (%)")
    plt.xlabel("Sample")
    plt.title(title)
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Genus/Species')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_top10_all_oral(oral_crohn, oral_healthy, outdir):
    """
    Calculate the top 10 most abundant genera across all oral samples (Crohn + Healthy).
    Save a stacked barplot and a CSV with top 10 genera and their total abundances.
    """
    # Combine all oral samples
    all_oral = pd.concat([oral_crohn, oral_healthy], axis=1)
    # Calculate total abundance for each genus
    genus_sum = all_oral.groupby(extract_genus).sum().sum(axis=1)
    top10 = genus_sum.sort_values(ascending=False).head(10).index

    # Aggregate abundances by genus
    abund = all_oral.groupby(extract_genus).sum()
    # Add "Other" category for non-top10 genera
    abund.loc["Other"] = abund.loc[~abund.index.isin(top10)].sum()
    abund_plot = abund.loc[list(top10) + ["Other"]]
    # Normalize abundances to percent per sample
    abund_plot = abund_plot.div(abund_plot.sum(axis=0), axis=1) * 100

    plot_stacked_bar(
        abund_plot,
        os.path.join(outdir, "stackedbar_top10_all_oral.png"),
        "Top 10 Genera Across All Oral Samples (Crohn + Healthy)"
    )
    # Save top10 genera to CSV
    save_top10_csv(genus_sum, os.path.join(outdir, "top10_all_oral.csv"))

def plot_top10_by_group(oral_group, group_name, outdir):
    """
    Find top 10 most abundant genera in one group (Crohn or Healthy oral).
    Save stacked barplot and CSV of top 10 genera and their total abundances.
    """
    genus_sum = oral_group.groupby(extract_genus).sum().sum(axis=1)
    top10 = genus_sum.sort_values(ascending=False).head(10).index

    abund = oral_group.groupby(extract_genus).sum()
    abund.loc["Other"] = abund.loc[~abund.index.isin(top10)].sum()
    abund_plot = abund.loc[list(top10) + ["Other"]]
    abund_plot = abund_plot.div(abund_plot.sum(axis=0), axis=1) * 100

    plot_stacked_bar(
        abund_plot,
        os.path.join(outdir, f"stackedbar_top10_{group_name}.png"),
        f"Top 10 Genera - {group_name.capitalize()} Samples"
    )
    # Save top10 genera to CSV
    save_top10_csv(genus_sum, os.path.join(outdir, f"top10_{group_name}.csv"))

def plot_compare_bar(oral_crohn, oral_healthy, outdir):
    """
    Plot a grouped bar chart comparing mean abundance (%) of top 10 genera between Crohn and Healthy oral samples.
    Save CSV of mean abundances for top 10 genera.
    """
    # Find top 10 genera by combined abundance (Crohn + Healthy)
    genus_sum_crohn = oral_crohn.groupby(extract_genus).sum().sum(axis=1)
    genus_sum_healthy = oral_healthy.groupby(extract_genus).sum().sum(axis=1)
    all_genus = genus_sum_crohn.add(genus_sum_healthy, fill_value=0)
    top10 = all_genus.sort_values(ascending=False).head(10).index

    # Calculate mean abundance per genus (percent, per group)
    mean_crohn = oral_crohn.groupby(extract_genus).sum().loc[top10].mean(axis=1)
    mean_healthy = oral_healthy.groupby(extract_genus).sum().loc[top10].mean(axis=1)
    df = pd.DataFrame({'Oral_Crohn': mean_crohn, 'Oral_Healthy': mean_healthy})
    df = df.div(df.sum(axis=0), axis=1) * 100  # Normalize to percent
    df.plot(
        kind='bar',
        width=0.6,
        figsize=(12,6),
        color=[palette['Crohn-Oral'], palette['Healthy-Oral']]
    )
    plt.ylabel("Relative Abundance (%)")
    plt.xlabel("Genus")
    plt.title("Top 10 Genus: Crohn Oral vs Healthy Oral (Mean Abundance)")
    plt.legend(["Oral Crohn", "Oral Healthy"], fontsize=12)
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(outdir, "barplot_top10_oral_crohn_vs_healthy.png"))
    plt.close()
    # Save to CSV
    df.to_csv(os.path.join(outdir, "top10_compare_oral_crohn_vs_healthy.csv"))

if __name__ == "__main__":
    try:
        # For Snakemake pipeline
        oral_crohn_file = snakemake.input.oral_crohn
        oral_healthy_file = snakemake.input.oral_healthy
        outdir = snakemake.params.outdir
    except NameError:
        # For command-line usage
        import sys
        oral_crohn_file = sys.argv[1]
        oral_healthy_file = sys.argv[2]
        outdir = sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    # Load input abundance tables (indexed by taxon)
    oral_crohn = pd.read_csv(oral_crohn_file, index_col=0)
    oral_healthy = pd.read_csv(oral_healthy_file, index_col=0)

    # Generate all top10 genus plots and CSVs
    plot_top10_all_oral(oral_crohn, oral_healthy, outdir)
    plot_top10_by_group(oral_crohn, "oral_crohn", outdir)
    plot_top10_by_group(oral_healthy, "oral_healthy", outdir)
    plot_compare_bar(oral_crohn, oral_healthy, outdir)

    print("[INFO] All done! Check:", outdir)
