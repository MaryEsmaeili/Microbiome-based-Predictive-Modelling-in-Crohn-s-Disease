import pandas as pd
import numpy as np
import os

# ---------------------
# Snakemake Inputs
# ---------------------
metadata_file = snakemake.input.metadata
crohn_file = snakemake.input.crohn
healthy_file = snakemake.input.healthy

# ---------------------
# Snakemake Outputs
# ---------------------
oral_abund_crohn_out = snakemake.output[0]
fecal_abund_crohn_out = snakemake.output[1]
oral_abund_healthy_out = snakemake.output[2]
matched_ids_out = snakemake.output[3]

# Ensure output directory exists
os.makedirs(os.path.dirname(oral_abund_crohn_out), exist_ok=True)

# ---------------------
# 1. Load metadata and abundance tables
# ---------------------
metadata = pd.read_excel(metadata_file, header=None, index_col=0)
metadata = metadata.T.reset_index(drop=True)
metadata = metadata[['STUDY_ID', 'Oral_sample_ID', 'Fecal_sample_ID']]
metadata.columns = metadata.columns.astype(str).str.strip()

# Load Crohn Metaphlan merged data
crohn = pd.read_csv(crohn_file, sep="\t")
rowdata = crohn.iloc[:, :2]
abund_data = crohn.iloc[:, 2:]
rowdata.rename(columns={"#clade_name": "clade_name"}, inplace=True)
# Load healthy oral metaphlan data
healthy = pd.read_csv(
    healthy_file, 
    sep="\t", 
    comment="#", 
    header=0, 
    index_col=0, 
    low_memory=False
)

# ---------------------
# 2. Clean sample names for consistent IDs
# ---------------------
abund_data = abund_data.drop(columns=[col for col in abund_data.columns if col.endswith('_y')], errors='ignore')
abund_data.columns = [col[:5] for col in abund_data.columns]

healthy = healthy.drop(['NCBI_tax_id'], axis=1, errors='ignore')
healthy.columns = [col.split('-')[-1].replace('_metaphlan','')[:6] for col in healthy.columns]

# ---------------------
# 3. Convert all abundance data to numeric
# ---------------------
abund_data = abund_data.apply(pd.to_numeric, errors='coerce')
healthy = healthy.apply(pd.to_numeric, errors='coerce')

# ---------------------
# 4. Find matched oral/fecal pairs for Crohn patients
# ---------------------
def match_sample_col(sample_id, columns):
    if pd.isnull(sample_id) or sample_id == '':
        return np.nan
    if sample_id in columns:
        return sample_id
    else:
        print(f"[WARN] No match for sample {sample_id}")
        return np.nan

metadata['Oral_col'] = metadata['Oral_sample_ID'].apply(lambda x: match_sample_col(x, abund_data.columns))
metadata['Fecal_col'] = metadata['Fecal_sample_ID'].apply(lambda x: match_sample_col(x, abund_data.columns))
matched = metadata.dropna(subset=['Oral_col', 'Fecal_col']).reset_index(drop=True)

# ---------------------
# 5. Export processed abundance tables (all Crohn oral/fecal and healthy oral)
# ---------------------
oral_abund_crohn = abund_data[metadata['Oral_col'].dropna().unique()].copy()
oral_abund_crohn.index = rowdata['clade_name']
oral_abund_crohn = oral_abund_crohn.loc[oral_abund_crohn.sum(axis=1) > 0]
oral_abund_crohn.to_csv(oral_abund_crohn_out, index=True)

fecal_abund_crohn = abund_data[metadata['Fecal_col'].dropna().unique()].copy()
fecal_abund_crohn.index = rowdata['clade_name']
fecal_abund_crohn = fecal_abund_crohn.loc[fecal_abund_crohn.sum(axis=1) > 0]
fecal_abund_crohn.to_csv(fecal_abund_crohn_out, index=True)

oral_abund_healthy = healthy.loc[healthy.sum(axis=1) > 0].copy()
oral_abund_healthy.to_csv(oral_abund_healthy_out, index=True)

matched[['STUDY_ID','Oral_col','Fecal_col']].to_csv(matched_ids_out, index=False)

print(f"Saved:\n - {oral_abund_crohn_out}\n - {fecal_abund_crohn_out}\n - {oral_abund_healthy_out}\n - {matched_ids_out}")
