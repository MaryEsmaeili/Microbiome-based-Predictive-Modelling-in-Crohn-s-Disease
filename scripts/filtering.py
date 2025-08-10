import pandas as pd
import os
import matplotlib.pyplot as plt
import yaml

# yaml colors
with open("config.yaml") as f:
    config = yaml.safe_load(f)
palette = config['colors']

# Snakemake Inputs/Outputs
oral_in = snakemake.input.oral
fecal_in = snakemake.input.fecal
healthy_in = snakemake.input.healthy
oral_out = snakemake.output.oral
fecal_out = snakemake.output.fecal
healthy_out = snakemake.output.healthy
prevalence_threshold = float(snakemake.params.prevalence)
abundance_threshold = float(snakemake.params.abundance)

# Results Directory
results_dir = os.path.join("results", "filtering")
os.makedirs(results_dir, exist_ok=True)
def load_data(filepath):
    return pd.read_csv(filepath, index_col=0)

def remove_low_abundance(df, threshold):
    filtered = df[df.max(axis=1) >= threshold]
    return filtered, df.shape[0], df.shape[0] - filtered.shape[0]

def remove_low_prevalence(df, threshold):
    prevalence = (df > 0).sum(axis=1) / df.shape[1]
    filtered = df[prevalence >= threshold]
    return filtered, df.shape[0], df.shape[0] - filtered.shape[0]

def keep_genus_species_only(df):
    filtered_rows = {}
    for name, row in df.iterrows():
        if "g__" in name:
            parts = name.split("|")
            genus_species = [p for p in parts if p.startswith("g__") or p.startswith("s__")]
            new_name = "|".join(genus_species)
            filtered_rows[new_name] = row
    new_df = pd.DataFrame.from_dict(filtered_rows, orient='index')
    new_df.index.name = "clade_name"
    return new_df, df.shape[0], df.shape[0] - len(new_df)

def save_filter_report(report, path):
    with open(path, "w") as f:
        for group, steps in report.items():
            f.write(f"[{group}]\n")
            for stepname, count in steps:
                f.write(f"{stepname}: {count}\n")
            f.write("\n")

report = {}

for group, inp, out in [
    ("Crohn-Oral", oral_in, oral_out),
    ("Crohn-Fecal", fecal_in, fecal_out),
    ("Healthy-Oral", healthy_in, healthy_out)
]:
    df0 = load_data(inp)
    step_counts = [("Initial", df0.shape[0])]
    df1, _, _ = remove_low_abundance(df0, abundance_threshold)
    step_counts.append(("Abundance", df1.shape[0]))
    df2, _, _ = remove_low_prevalence(df1, prevalence_threshold)
    step_counts.append(("Prevalence", df2.shape[0]))
    df3, _, _ = keep_genus_species_only(df2)
    df3 = df3.apply(pd.to_numeric, errors='coerce')
    df3_percent = df3.div(df3.sum(axis=0), axis=1) * 100
    df3_percent = df3_percent.dropna(axis=0, how='all').dropna(axis=1, how='all')
    step_counts.append(("Genus/Species (normalized)", df3_percent.shape[0]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df3_percent.to_csv(out)
    report[group] = step_counts

print(f"[INFO] Filtering done! Files saved in {os.path.dirname(oral_out)}")
save_filter_report(report, os.path.join(results_dir, "filtering_report.txt"))

# --- Plot: Bar chart with custom YAML colors ---
import numpy as np
plt.figure(figsize=(10, 5))
width = 0.25
labels = [s[0] for s in report["Crohn-Oral"]]
x = np.arange(len(labels))

group_list = ["Crohn-Fecal", "Crohn-Oral", "Healthy-Oral"]
bar_colors = [palette[g] for g in group_list]

for i, (group, color) in enumerate(zip(group_list, bar_colors)):
    counts = [s[1] for s in report[group]]
    plt.bar(x + i*width, counts, width=width, label=group.replace("_", " "), color=color)

plt.xticks(x + width, labels)
plt.xlabel("Filtering Step")
plt.ylabel("Number of taxa")
plt.title("Number of taxa after each filtering step")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "filtering_barplot_allgroups.png"))
plt.close()