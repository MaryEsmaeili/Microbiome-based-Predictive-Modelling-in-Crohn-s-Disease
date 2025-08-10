#!/usr/bin/env python3
# Reads row-oriented Excel (variables in column A, samples across columns)
# and writes a tidy CSV with both Oral_sample_ID and Fecal_sample_ID + key clinicals.

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from dateutil import parser as dtp

# ----- helpers -----
def norm_label(s: str) -> str:
    """Normalize labels for robust matching (case-insensitive, spaces/dots -> underscore)."""
    s = str(s).strip().lower()
    s = re.sub(r"[\s\.]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"__+", "_", s)
    return s

YES = {"yes","y","true","1","ja","oui","evet"}
NO  = {"no","n","false","0","nee","non","hayir"}

def yesno_to_bool(x):
    if pd.isna(x): return np.nan
    s = norm_label(x)
    if s in YES: return True
    if s in NO:  return False
    return np.nan

def to_number(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", ".")
    s = re.sub(r"[^0-9.\-eE]", "", s)
    try: return float(s)
    except: return np.nan

def to_date(x):
    if pd.isna(x): return pd.NaT
    try: return pd.to_datetime(dtp.parse(str(x), dayfirst=True, fuzzy=True))
    except: return pd.NaT

# Canonical rows we want (left = output key, right = possible row-name(s) in Excel col A)
ROW_KEYS = {
    "Oral_sample_ID":      ["oral_sample_id"],
    "Fecal_sample_ID":     ["fecal_sample_id"],
    "Hospital":            ["hospital"],
    "STUDY_ID":            ["study_id", "studyid"],
    "Responder_study_raw": ["responder_study",
                            "clinical_responder_hbi_4_points_or_a_reduction_in_the_hbi_by_3_points",
                            "biochemical_responder_fecal_calprotectin_level_250_or_50_reduction"],
    "V1_AgeatFecalSampling": ["v1_ageatfecalsampling", "v1_age_at_fecalsampling", "anthro_age", "age"],
    "V1_BMI":              ["v1_bmi","anthro_bmi","bmi"],
    "V1_Sex":              ["v1_sex","sex"],
    # Optional useful dates (not required but nice to have)
    "V1_Dateofbirth":      ["v1_dateofbirth","dateofbirth","dob"],
    "V1_DateFecalSample":  ["v1_datefecalsample","v1_datefecalsample_80freezer","v1_datevisit"],
}

def main():
    ap = argparse.ArgumentParser(description="Clean row-oriented Excel metadata into tidy CSV.")
    ap.add_argument("--input", required=True, help="Path to Metadata_metagenomics.xlsx")
    ap.add_argument("--sheet", default=0, help="Sheet index/name (default 0)")
    ap.add_argument("--out", default="data/processed/clinical_clean.csv", help="Output CSV path")
    ap.add_argument("--id-row-oral", default="Oral_sample_ID", help="Row name for Oral sample IDs")
    ap.add_argument("--id-row-fecal", default="Fecal_sample_ID", help="Row name for Fecal sample IDs")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read as row-oriented: NO header; first column = variable names; others = samples
    raw = pd.read_excel(in_path, sheet_name=args.sheet, header=None)
    raw.columns = [f"...{i+1}" for i in range(raw.shape[1])]
    raw = raw.rename(columns={"...1": "var"})
    raw["var_norm"] = raw["var"].map(norm_label)

    # Build a matrix with variables as index and sample columns as columns
    # We'll pick the Oral_sample_ID row to assign column names (sample IDs)
    def row_values_by_name(name_or_list):
        target = [norm_label(name_or_list)] if isinstance(name_or_list, str) else [norm_label(x) for x in name_or_list]
        hit = raw[raw["var_norm"].isin(target)].index.tolist()
        return hit

    # Get Oral and Fecal row indices
    oral_idx  = row_values_by_name([args.id_row_oral])
    fecal_idx = row_values_by_name([args.id_row_fecal])

    if len(oral_idx) != 1:
        raise SystemExit(f"[ERR] Could not uniquely find Oral ID row '{args.id_row_oral}'. Found: {len(oral_idx)}")
    if len(fecal_idx) != 1:
        raise SystemExit(f"[ERR] Could not uniquely find Fecal ID row '{args.id_row_fecal}'. Found: {len(fecal_idx)}")

    # Create a matrix (variables as index, sample columns as columns)
    mat = raw.set_index("var_norm").iloc[:, 1:].copy()

    # Pull Oral IDs row and clean it; drop empty header cells and match mat columns
    oral_row = raw.iloc[oral_idx[0], 1:]
    # keep only non-empty IDs
    mask = oral_row.notna() & (oral_row.astype(str).str.strip() != "")
    # apply the same mask to the matrix columns
    mat = mat.loc[:, mask.values]

    oral_ids = oral_row[mask].astype(str).str.strip().values

    # Optional: make IDs unique if there are duplicates (rare but safer)
    def make_unique(vals):
        seen = {}
        out = []
        for v in vals:
            if v in seen:
                seen[v] += 1
                out.append(f"{v}_{seen[v]}")
            else:
                seen[v] = 0
                out.append(v)
        return out

    # If duplicates exist, de-dup while warning
    if pd.Series(oral_ids).duplicated().any():
        print("[WARN] Duplicate Oral_sample_ID values detected; appending suffixes to make them unique.")
        oral_ids = make_unique(oral_ids)

    # finally set the column names
    mat.columns = oral_ids

    # Helper to pull a variable row into a Series aligned with columns (samples)
    def take_row(keys, converter=None):
        keys_norm = [norm_label(k) for k in (keys if isinstance(keys, (list, tuple)) else [keys])]
        for k in keys_norm:
            if k in mat.index:
                s = mat.loc[k]
                if converter is not None:
                    s = s.apply(converter)
                return s
        return None

    # Build tidy output with both IDs
    out = pd.DataFrame(index=mat.columns)
    out.index.name = "Oral_sample_ID"
    out["Fecal_sample_ID"] = take_row(ROW_KEYS["Fecal_sample_ID"])
    out["Hospital"]        = take_row(ROW_KEYS["Hospital"])
    out["STUDY_ID"]        = take_row(ROW_KEYS["STUDY_ID"])

    # Responder: keep raw + booleanized
    resp_raw = take_row(ROW_KEYS["Responder_study_raw"])
    if resp_raw is not None:
        out["Responder_study_raw"] = resp_raw
        out["Responder_study"] = resp_raw.apply(yesno_to_bool).astype("boolean")
    else:
        out["Responder_study_raw"] = pd.NA
        out["Responder_study"] = pd.Series([pd.NA]*len(out), dtype="boolean")

    # Age / BMI / Sex
    age = take_row(ROW_KEYS["V1_AgeatFecalSampling"], to_number)
    bmi = take_row(ROW_KEYS["V1_BMI"], to_number)
    out["Age_years"] = age.values if age is not None else np.nan
    out["BMI"]       = bmi.values if bmi is not None else np.nan
    sex = take_row(ROW_KEYS["V1_Sex"])
    out["Sex"] = sex.values if sex is not None else np.nan

    # Dates (optional)
    dob = take_row(ROW_KEYS["V1_Dateofbirth"], to_date)
    dfs = take_row(ROW_KEYS["V1_DateFecalSample"], to_date)
    if dob is not None: out["Date_of_birth"] = dob.values
    if dfs is not None: out["Date_fecal_sample"] = dfs.values

    # If age missing but have DOB & sample date, back-calc age
    if ("Age_years" not in out.columns or out["Age_years"].isna().all()) and \
       ("Date_of_birth" in out.columns and "Date_fecal_sample" in out.columns):
        out["Age_years"] = ((out["Date_fecal_sample"] - out["Date_of_birth"]).dt.days / 365.25).round(2)

    # Finalize columns and write CSV
    out = out.reset_index()  # bring Oral_sample_ID as a column
    # Order columns
    order = [
        "Oral_sample_ID", "Fecal_sample_ID", "Hospital", "STUDY_ID",
        "Responder_study", "Responder_study_raw",
        "Age_years", "BMI", "Sex",
        "Date_of_birth", "Date_fecal_sample"
    ]
    # keep order + any extras that made it in
    cols = [c for c in order if c in out.columns] + [c for c in out.columns if c not in order]
    out = out[cols]

    out.to_csv(out_path, index=False)
    print(f"[OK] Saved clean clinical CSV to: {out_path}")
    print(f"[INFO] Rows (samples): {out.shape[0]}, Columns: {out.shape[1]}")

if __name__ == "__main__":
    main()
