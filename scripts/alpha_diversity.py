import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, mannwhitneyu, wilcoxon, kruskal

# -----------------------------------
# SNAKEMAKE INPUTS/OUTPUTS
# -----------------------------------
oral_file = snakemake.input.oral
fecal_file = snakemake.input.fecal
healthy_file = snakemake.input.healthy
matched_ids_file = snakemake.input.matched_ids

oral_out = snakemake.output.oral
fecal_out = snakemake.output.fecal
healthy_out = snakemake.output.healthy
results_dir = os.path.dirname(oral_out)
os.makedirs(results_dir, exist_ok=True)

# -----------------------------------
# ALPHA DIVERSITY FUNCTIONS
# -----------------------------------
def shannon(x):
    x = np.array(x)
    x = x[x > 0]
    return entropy(x, base=np.e) if x.size else 0

def simpson(x):
    x = np.array(x)
    x = x[x > 0]
    p = x / x.sum() if x.sum() > 0 else np.zeros_like(x)
    return 1 - np.sum(p ** 2) if p.size else 0

def observed(x):
    return np.sum(np.array(x) > 0)

def alpha_df(abund):
    return pd.DataFrame({
        'Shannon': abund.apply(shannon, axis=0),
        'Simpson': abund.apply(simpson, axis=0),
        'Richness': abund.apply(observed, axis=0)
    })

# -----------------------------------
# LOAD DATA
# -----------------------------------
oral = pd.read_csv(oral_file, index_col=0)
fecal = pd.read_csv(fecal_file, index_col=0)
healthy = pd.read_csv(healthy_file, index_col=0)
matched = pd.read_csv(matched_ids_file)

# -----------------------------------
# UNPAIRED ALPHA DIVERSITY
# -----------------------------------
alpha_oral = alpha_df(oral)
alpha_oral.index.name = 'Sample'
alpha_oral.to_csv(oral_out)

alpha_fecal = alpha_df(fecal)
alpha_fecal.index.name = 'Sample'
alpha_fecal.to_csv(fecal_out)

alpha_healthy = alpha_df(healthy)
alpha_healthy.index.name = 'Sample'
alpha_healthy.to_csv(healthy_out)

# -----------------------------------
# PAIRED (MATCHED) ALPHA DIVERSITY
# -----------------------------------
oral_matched = oral[matched['Oral_col']].copy()
fecal_matched = fecal[matched['Fecal_col']].copy()

matched_alpha = pd.DataFrame({
    'STUDY_ID': matched['STUDY_ID'],
    'Oral_Sample': matched['Oral_col'],
    'Fecal_Sample': matched['Fecal_col'],
    'Oral_Shannon': [shannon(oral_matched[col]) for col in oral_matched.columns],
    'Fecal_Shannon': [shannon(fecal_matched[col]) for col in fecal_matched.columns],
    'Oral_Simpson': [simpson(oral_matched[col]) for col in oral_matched.columns],
    'Fecal_Simpson': [simpson(fecal_matched[col]) for col in fecal_matched.columns],
    'Oral_Richness': [observed(oral_matched[col]) for col in oral_matched.columns],
    'Fecal_Richness': [observed(fecal_matched[col]) for col in fecal_matched.columns]
})
matched_alpha.to_csv(os.path.join(results_dir, "alpha_matched.csv"), index=False)

# -----------------------------------
# STATISTICAL TESTS
# -----------------------------------
with open(f"{results_dir}/alpha_stats.txt", "w") as statout:
    statout.write("== Crohn Oral vs Healthy Oral (independent) ==\n")
    for metric in ['Richness', 'Shannon', 'Simpson']:
        u_stat, pval = mannwhitneyu(
            alpha_oral[metric], alpha_healthy[metric], alternative="two-sided"
        )
        statout.write(f"Mann-Whitney U ({metric}): U={u_stat:.2f}, p={pval:.4g}\n")
    statout.write("\n")

    statout.write("== Crohn Oral vs Crohn Fecal (paired) ==\n")
    for metric in ['Richness', 'Shannon', 'Simpson']:
        w_stat, pval = wilcoxon(
            matched_alpha[f'Oral_{metric}'], matched_alpha[f'Fecal_{metric}']
        )
        statout.write(f"Wilcoxon ({metric}): W={w_stat:.2f}, p={pval:.4g}\n")
    statout.write("\n")

    statout.write("== All groups Kruskal-Wallis (independent) ==\n")
    for metric in ['Richness', 'Shannon', 'Simpson']:
        h_stat, pval = kruskal(
            alpha_oral[metric], alpha_fecal[metric], alpha_healthy[metric]
        )
        statout.write(f"Kruskal-Wallis ({metric}): H={h_stat:.2f}, p={pval:.4g}\n")
    statout.write("\n")
print(f"[INFO] Statistical tests done. Results in {results_dir}/alpha_stats.txt")

# -----------------------------------
# PLOTTING
# -----------------------------------
sns.set(style="whitegrid", font_scale=1.1)

# 1. All three groups, all metrics
for metric in ['Shannon', 'Simpson', 'Richness']:
    df = pd.DataFrame({
        'Value': pd.concat([alpha_oral[metric], alpha_fecal[metric], alpha_healthy[metric]]),
        'Group': (['Crohn-Oral']*len(alpha_oral) +
                  ['Crohn-Fecal']*len(alpha_fecal) +
                  ['Healthy-Oral']*len(alpha_healthy))
    })
    plt.figure(figsize=(7, 5))
    ax = sns.boxplot(data=df, x='Group', y='Value', palette='Set2')
    sns.stripplot(data=df, x='Group', y='Value', color='k', alpha=0.4, ax=ax)
    plt.title(f'Alpha Diversity ({metric}) Across Groups')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/boxplot_{metric.lower()}_allgroups.png")
    plt.close()

# 2. Paired Crohn Oral vs Crohn Fecal (matched pairs)
def plot_paired_boxplot(matched_alpha, metric, outdir):
    data = pd.DataFrame({
        'Oral': matched_alpha[f'Oral_{metric}'],
        'Fecal': matched_alpha[f'Fecal_{metric}']
    })
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=data.melt(var_name='SampleType', value_name=metric), x='SampleType', y=metric, palette='Set2')
    sns.stripplot(data=data.melt(var_name='SampleType', value_name=metric), x='SampleType', y=metric, color='k', alpha=0.4, jitter=0.2)
    # Draw lines for each pair
    for i in range(len(data)):
        plt.plot(['Oral', 'Fecal'], [data.iloc[i, 0], data.iloc[i, 1]], color='gray', alpha=0.4, linewidth=1)
    plt.title(f'Paired Alpha Diversity: Crohn Oral vs Fecal ({metric})')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"{outdir}/paired_boxplot_{metric.lower()}_oral_fecal.png")
    plt.close()

for metric in ['Shannon', 'Simpson', 'Richness']:
    plot_paired_boxplot(matched_alpha, metric, results_dir)

# 3. Crohn Oral vs Healthy Oral (unpaired)
for metric in ['Shannon', 'Simpson', 'Richness']:
    df = pd.DataFrame({
        'Value': pd.concat([alpha_oral[metric], alpha_healthy[metric]]),
        'Group': (['Crohn-Oral']*len(alpha_oral) +
                  ['Healthy-Oral']*len(alpha_healthy))
    })
    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(data=df, x='Group', y='Value', palette='Set1')
    sns.stripplot(data=df, x='Group', y='Value', color='k', alpha=0.4, ax=ax)
    plt.title(f'Alpha Diversity ({metric}) Crohn Oral vs Healthy Oral')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/boxplot_{metric.lower()}_oral_vs_healthy.png")
    plt.close()

print("All alpha diversity plots saved in:", results_dir)
