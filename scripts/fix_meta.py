# scripts/fix_meta.py
import pandas as pd
from pathlib import Path

inp  = snakemake.input[0]
outf = snakemake.output["fixed"]
dropped_txt = snakemake.output["dropped"]

REQUIRED = ["Sample_ID","disease","site","Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use"]

df = pd.read_csv(inp)

# یکنواخت‌سازی نام ستون‌ها
rename_map = {}
if "Group" in df.columns: rename_map["Group"] = "disease"
if "Site"  in df.columns: rename_map["Site"]  = "site"
if rename_map:
    df = df.rename(columns=rename_map)

# disease -> 0/1
if "disease" in df.columns:
    if df["disease"].dtype == object:
        df["disease"] = df["disease"].map({"Crohn":1, "Healthy":0})
    df["disease"] = df["disease"].astype("Int64")

# site را عددی کن (oral=0, fecal=1)
if "site" in df.columns:
    df["site"] = (
        df["site"].astype(str).str.lower()
          .map({"oral":0, "fecal":1, "na":pd.NA})
          .astype("Int64")
    )

# اطمینان از وجود ستون‌های لازم
for c in REQUIRED:
    if c not in df.columns:
        df[c] = pd.NA

# تبدیل بقیه متاها به عدد (در صورت وجود)
for c in ["Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ردیف‌های ناقص در REQUIRED
mask_missing = df[REQUIRED].isna().any(axis=1)
dropped = df.loc[mask_missing, "Sample_ID"].astype(str).tolist()

Path(dropped_txt).parent.mkdir(parents=True, exist_ok=True)
with open(dropped_txt, "w") as f:
    for sid in dropped:
        f.write(f"{sid}\n")

df_fixed = df.loc[~mask_missing].copy()
Path(outf).parent.mkdir(parents=True, exist_ok=True)
df_fixed.to_csv(outf, index=False)

print({
    "input": inp,
    "output": outf,
    "n_rows_in": len(df),
    "n_dropped": len(dropped),
    "dropped_samples_preview": dropped[:20] + (["..."] if len(dropped)>20 else [])
})
