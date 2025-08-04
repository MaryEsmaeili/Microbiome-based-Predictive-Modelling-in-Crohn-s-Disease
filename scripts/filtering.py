import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Snakemake Inputs/Outputs ---
oral_in = snakemake.input.oral
fecal_in = snakemake.input.fecal
healthy_in = snakemake.input.healthy
oral_out = snakemake.output.oral
fecal_out = snakemake.output.fecal
healthy_out = snakemake.output.healthy
prevalence_threshold = float(snakemake.params.prevalence)
abundance_threshold = float(snakemake.params.abundance)

# --- Results Directory ---
results_dir = os.path.join("results", "filtering")
os.makedirs(results_dir, exist_ok=True)

def load_data(filepath):
    return pd.read_csv(filepath, index_col=0)

def remove_low_abundance(df, threshold):
    before = df.shape[0]
    filtered = df[df.max(axis=1) >= threshold]
    removed = before - filtered.shape[0]
    return filtered, before, removed

def remove_low_prevalence(df, threshold):
    before = df.shape[0]
    prevalence = (df > 0).sum(axis=1) / df.shape[1]
    filtered = df[prevalence >= threshold]
    removed = before - filtered.shape[0]
    return filtered, before, removed

def keep_genus_species_only(df):
    before = df.shape[0]
    filtered_rows = {}
    removed_rows = []
    for name, row in df.iterrows():
        if "g__" in name:
            parts = name.split("|")
            genus_species = [p for p in parts if p.startswith("g__") or p.startswith("s__")]
            new_name = "|".join(genus_species)
            filtered_rows[new_name] = row
        else:
            removed_rows.append(name)
    new_df = pd.DataFrame.from_dict(filtered_rows, orient='index')
    new_df.index.name = "clade_name"
    removed = before - len(new_df)
    return new_df, before, removed

# --- Helper to save filtering stats as text file ---
def save_filter_report(report, path):
    with open(path, "w") as f:
        for group, steps in report.items():
            f.write(f"[{group}]\n")
            for stepname, count in steps:
                f.write(f"{stepname}: {count}\n")
            f.write("\n")

# --- Main filtering and counting logic ---
report = {}

for group, inp, out in [
    ("Oral_Crohn", oral_in, oral_out),
    ("Fecal_Crohn", fecal_in, fecal_out),
    ("Healthy_Oral", healthy_in, healthy_out)
]:
    df0 = load_data(inp)
    step_counts = [("Initial", df0.shape[0])]
    df1, before1, rm1 = remove_low_abundance(df0, abundance_threshold)
    step_counts.append(("Abundance", df1.shape[0]))
    df2, before2, rm2 = remove_low_prevalence(df1, prevalence_threshold)
    step_counts.append(("Prevalence", df2.shape[0]))
    df3, before3, rm3 = keep_genus_species_only(df2)
    step_counts.append(("Genus/Species", df3.shape[0]))
    # Save output file
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df3.to_csv(out)
    report[group] = step_counts

print(f"[INFO] Filtering done! Files saved in {os.path.dirname(oral_out)}")
save_filter_report(report, os.path.join(results_dir, "filtering_report.txt"))

# --- Plot: Bar chart with all groups next to each other ---
plt.figure(figsize=(10, 5))
width = 0.25
labels = [s[0] for s in report["Oral_Crohn"]]
x = range(len(labels))

for i, group in enumerate(["Oral_Crohn", "Fecal_Crohn", "Healthy_Oral"]):
    counts = [s[1] for s in report[group]]
    plt.bar([p + i * width for p in x], counts, width=width, label=group.replace("_", " "))

plt.xticks([p + width for p in x], labels)
plt.xlabel("Filtering Step")
plt.ylabel("Number of taxa")
plt.title("Number of taxa after each filtering step")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "filtering_barplot_allgroups.png"))
plt.close()
