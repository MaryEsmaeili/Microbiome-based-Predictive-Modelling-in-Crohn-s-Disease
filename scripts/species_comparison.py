import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml

# yaml colors
with open("config.yaml") as f:
    config = yaml.safe_load(f)
palette = config['colors']

def extract_species(rowname):
    """
    Extract species name from a Metaphlan taxonomic label string (e.g., 'k__Bacteria|...|s__salivarius').
    """
    parts = rowname.split("|")
    species = [p for p in parts if p.startswith("s__")]
    return species[0] if species else None

def save_top15_csv(species_sum, out_path):
    """
    Save the top 15 species (with total abundance) as a CSV file.
    """
    top15 = species_sum.sort_values(ascending=False).head(15)
    top15.to_csv(out_path, header=['Total_Abundance'])

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


def plot_top15_all_oral(oral_crohn, oral_healthy, outdir):
    """
    Calculate the top 15 most abundant species across all oral samples (Crohn + Healthy).
    Save a stacked barplot and a CSV with top 15 species and their total abundances.
    """
    # Combine all oral samples
    all_oral = pd.concat([oral_crohn, oral_healthy], axis=1)
    # Calculate total abundance for each species
    species_sum = all_oral.groupby(extract_species).sum().sum(axis=1)
    top15 = species_sum.sort_values(ascending=False).head(15).index

    # Aggregate abundances by species
    abund = all_oral.groupby(extract_species).sum()
    # Add "Other" category for non-top15 species
    abund.loc["Other"] = abund.loc[~abund.index.isin(top15)].sum()
    abund_plot = abund.loc[list(top15) + ["Other"]]
    # Normalize abundances to percent per sample
    abund_plot = abund_plot.div(abund_plot.sum(axis=0), axis=1) * 100

    plot_stacked_bar(
        abund_plot,
        os.path.join(outdir, "stackedbar_top15_all_oral_species.png"),
        "Top 15 Species Across All Oral Samples (Crohn + Healthy)"
    )
    # Save top15 species to CSV
    save_top15_csv(species_sum, os.path.join(outdir, "top15_all_oral_species.csv"))

def plot_top15_by_group(oral_group, group_name, outdir):
    """
    Find top 15 most abundant species in one group (Crohn or Healthy oral).
    Save stacked barplot and CSV of top 15 species and their total abundances.
    """
    species_sum = oral_group.groupby(extract_species).sum().sum(axis=1)
    top15 = species_sum.sort_values(ascending=False).head(15).index

    abund = oral_group.groupby(extract_species).sum()
    abund.loc["Other"] = abund.loc[~abund.index.isin(top15)].sum()
    abund_plot = abund.loc[list(top15) + ["Other"]]
    abund_plot = abund_plot.div(abund_plot.sum(axis=0), axis=1) * 100

    plot_stacked_bar(
        abund_plot,
        os.path.join(outdir, f"stackedbar_top15_{group_name}_species.png"),
        f"Top 15 Species - {group_name.capitalize()} Samples"
    )
    # Save top15 species to CSV
    save_top15_csv(species_sum, os.path.join(outdir, f"top15_{group_name}_species.csv"))

def plot_compare_bar(oral_crohn, oral_healthy, outdir):
    """
    Plot a grouped bar chart comparing mean abundance (%) of top 15 species between Crohn and Healthy oral samples.
    Save CSV of mean abundances for top 15 species.
    """
    # Find top 15 species by combined abundance (Crohn + Healthy)
    species_sum_crohn = oral_crohn.groupby(extract_species).sum().sum(axis=1)
    species_sum_healthy = oral_healthy.groupby(extract_species).sum().sum(axis=1)
    all_species = species_sum_crohn.add(species_sum_healthy, fill_value=0)
    top15 = all_species.sort_values(ascending=False).head(15).index

    # Calculate mean abundance per species (percent, per group)
    mean_crohn = oral_crohn.groupby(extract_species).sum().loc[top15].mean(axis=1)
    mean_healthy = oral_healthy.groupby(extract_species).sum().loc[top15].mean(axis=1)
    df = pd.DataFrame({'Oral_Crohn': mean_crohn, 'Oral_Healthy': mean_healthy})
    df = df.div(df.sum(axis=0), axis=1) * 100  # Normalize to percent

    # Use same color scheme as genus compare bar
    df.plot(
        kind='bar',
        width=0.6,
        figsize=(12,6),
        color=[palette['Crohn-Oral'], palette['Healthy-Oral']]
    )    
    plt.ylabel("Relative Abundance (%)")
    plt.xlabel("Species")
    plt.title("Top 15 Species: Crohn Oral vs Healthy Oral (Mean Abundance)")
    plt.legend(["Oral Crohn", "Oral Healthy"], fontsize=12)
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(outdir, "barplot_top15_oral_crohn_vs_healthy_species.png"))
    plt.close()
    # Save to CSV
    df.to_csv(os.path.join(outdir, "top15_compare_oral_crohn_vs_healthy_species.csv"))

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

    # Generate all top15 species plots and CSVs
    plot_top15_all_oral(oral_crohn, oral_healthy, outdir)
    plot_top15_by_group(oral_crohn, "oral_crohn", outdir)
    plot_top15_by_group(oral_healthy, "oral_healthy", outdir)
    plot_compare_bar(oral_crohn, oral_healthy, outdir)

    print("[INFO] All done! Check:", outdir)
