import pandas as pd
import os
import matplotlib.pyplot as plt
import yaml
import numpy as np

# --- Load colors from config.yaml ---
with open("config.yaml") as f:
    config = yaml.safe_load(f)
palette = config['colors']

# --- Snakemake I/O ---
oral_in = snakemake.input.oral
fecal_in = snakemake.input.fecal
healthy_oral_in = snakemake.input.healthy_oral
healthy_fecal_in = snakemake.input.healthy_fecal

oral_out = snakemake.output.oral_norm
fecal_out = snakemake.output.fecal_norm
healthy_oral_out = snakemake.output.healthy_oral_norm
healthy_fecal_out = snakemake.output.healthy_fecal_norm

prevalence_threshold = float(snakemake.params.prevalence)
abundance_threshold = float(snakemake.params.abundance)

# --- Helpers ---
def load_data(filepath):
    return pd.read_csv(filepath, index_col=0)

def remove_low_abundance(df, threshold):
    filtered = df[df.max(axis=1) >= threshold]
    return filtered

def remove_low_prevalence(df, threshold):
    prevalence = (df > 0).sum(axis=1) / df.shape[1]
    filtered = df[prevalence >= threshold]
    return filtered

def keep_genus_species_only(df):
    filtered_rows = {}
    for name, row in df.iterrows():
        if "g__" in name:
            parts = name.split("|")
            genus_species = [p for p in parts if p.startswith("g__") or p.startswith("s__")]
            new_name = "|".join(genus_species)
            filtered_rows[new_name] = row
    return pd.DataFrame.from_dict(filtered_rows, orient='index')

def process_group(name, inp, out):
    df0 = load_data(inp)
    steps = [("Initial", df0.shape[0])]
    df1 = remove_low_abundance(df0, abundance_threshold)
    steps.append(("Abundance", df1.shape[0]))
    df2 = remove_low_prevalence(df1, prevalence_threshold)
    steps.append(("Prevalence", df2.shape[0]))
    df3 = keep_genus_species_only(df2)
    df3 = df3.apply(pd.to_numeric, errors='coerce')
    df3_percent = df3.div(df3.sum(axis=0), axis=1) * 100
    df3_percent = df3_percent.dropna(axis=0, how='all').dropna(axis=1, how='all')
    steps.append(("Genus/Species (normalized)", df3_percent.shape[0]))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df3_percent.to_csv(out)

    return steps

# --- Run for all groups ---
report = {}
report["Crohn-Oral"] = process_group("Crohn-Oral", oral_in, oral_out)
report["Crohn-Fecal"] = process_group("Crohn-Fecal", fecal_in, fecal_out)
report["Healthy-Oral"] = process_group("Healthy-Oral", healthy_oral_in, healthy_oral_out)
report["Healthy-Fecal"] = process_group("Healthy-Fecal", healthy_fecal_in, healthy_fecal_out)

# --- Save filtering report ---
results_dir = os.path.join("results", "filtering")
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "filtering_report.txt"), "w") as f:
    for group, steps in report.items():
        f.write(f"[{group}]\n")
        for step, count in steps:
            f.write(f"{step}: {count}\n")
        f.write("\n")

# --- Plot ---
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
