# Oral–Gut Microbiome Divergence and Predictive Modeling in Crohn’s Disease

This repository provides a fully reproducible Snakemake workflow for processing, analyzing, and modeling paired oral–fecal metagenomic profiles in Crohn’s disease. The pipeline performs preprocessing, filtering, normalization, alpha and beta diversity analyses, unified QC, differential abundance, machine learning (nested CV and transfer learning), and ML summary aggregation, all within a structured and automated environment.

The workflow follows a strict modular structure, ensures reproducibility through conda environments, and produces publication-ready figures and tables directly from raw MetaPhlAn4 outputs.

## Repository Structure

```
.
├── Snakefile
├── config/
│   ├── config.yaml
│   ├── enviroment_r.yml
│   ├── enviroment.yml
│   └── colors.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── filtered/
├── results/
│   ├── preprocessing/
│   ├── filtering/
│   ├── alpha/
│   ├── beta/
│   ├── taxa_compare/
│   ├── qc/
│   ├── da/
│   └── ml/
│       └── ml_summary/
├── scripts/
│   ├── preprocessing.py
│   ├── filtering.py
│   ├── alpha_diversity.py
│   ├── beta_diversity.R
│   ├── taxa_comparison.py
│   ├── da_unified_module.py
│   ├── qc_unified.py
│   └── ml/
│       ├── data.py
│       ├── train.py
│       ├── cv_utils.py
│       └── summary_ml_results.py/
└──
```

## Installation

Create the Python environment:

```
conda env create -f environment.yml
conda activate microbiome_pipeline
```

Create the R environment required for beta diversity:

```
conda env create -f config/environment_r.yml
```

## Configuration

All settings, paths, filtering thresholds, DA and QC parameters, and ML configuration are controlled through:

```
config/config.yaml
```

## Run the Workflow

Run the entire pipeline:

```
snakemake --cores 8
```

Run selected modules:

Preprocessing:
```
snakemake results/_flags/preprocessing.done --cores 4
```

Filtering:
```
snakemake results/filtering/filtering_report.txt --cores 4
```

Alpha diversity:
```
snakemake results/alpha/species/.done --cores 4
```

Beta diversity (R):
```
snakemake results/beta/species/.done --cores 4
```

Differential abundance:
```
snakemake results/da/species/.done --cores 4
```

Machine learning (within-site + transfer):
```
snakemake ml --cores 8
```

ML summary and combined plots:
```
snakemake ml_summary --cores 4
```

## Machine Learning

The ML subsystem performs CLR transformation, feature concatenation, nested cross-validation, learning curves, transfer learning between oral and fecal sites, and consensus feature importance across all four model types (logistic regression, random forest, SVM, XGBoost). All outputs are saved to:

```
results/ml/
results/ml/summary/
```

## Outputs

The pipeline generates:

- Preprocessing: harmonized abundance tables, HMP integration, ID-map logs, matched pairs.
- Filtering: zero-count QC, prevalence vs mean plots, rank abundance curves.
- Alpha diversity: species/genus diversity metrics, matched analyses.
- Beta diversity: Bray and Aitchison PCoA, covariate-adjusted models, PERMANOVA/PERMDISP.
- Taxa comparison: group means, heatmaps, volcano-style effect summaries.
- QC: PCA (no covariates / covariate-adjusted / PPI-focused).
- Differential abundance: unadjusted, adjusted, and paired analyses.
- Machine learning: nested CV predictions, ROC/PR/calibration, learning curves, transfer evaluations.
- ML summary: combined ROC, PR, calibration, learning-curve plots, and consensus feature importance.

## Reproducibility

The workflow follows Hanze reproducibility standards:
- strict separation of raw, processed, filtered, and results data
- deterministic preprocessing steps
- locked conda environments
- no manual editing of intermediate files
- modular Snakemake DAG
- all figures regenerated programmatically

## Citation

Esmaeili, M. (December 2025).  
Oral–Gut Microbiome Divergence and Predictive Modeling in Crohn’s Disease.
Hanze University of Applied Sciences & UMCG.

## Contact

Maryam Esmaeili  
maryam.esmaeili1985@gmail.com
