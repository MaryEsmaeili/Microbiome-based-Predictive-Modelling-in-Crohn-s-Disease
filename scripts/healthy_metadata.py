#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make ONE clean healthy metadata CSV (minimal outputs, per Maryam's spec).

Outputs (both next to --out):
- healthy_metadata.csv        (final clean metadata)
- summary_counts.csv          (summary metrics incl. dropped id6 list)

Behavior changes:
- REMOVE columns: Immuno_ongoing, Steroids_ongoing, Antibiotics_3m, PPI_use.
- Smoking is taken from DAG3 metadata (if present).
- Convert core columns IN-PLACE to categories:
    * Sex  -> 0/1  (Male=0, Female=1)
    * Age  -> 0/1/2/3 (single column, NOT a *_cat companion)
        Category bins (commented here for clarity):
          0: Age < 18
          1: 18 ≤ Age ≤ 34
          2: 35 ≤ Age ≤ 49
          3: Age ≥ 50
        (Example: Age=60 => 3)
    * BMI  -> 0/1/2/3 (WHO-ish):
          0: BMI < 18.5
          1: 18.5 ≤ BMI < 25
          2: 25 ≤ BMI < 30
          3: BMI ≥ 30
- Keep Healthy_only (0/1) as an informational column. No filtering on it unless --filter-healthy.
- DROP rows whose chosen DAG3_sampleID is NOT present in DAG3 metadata; list dropped id6 in summary_counts.csv.
"""

from __future__ import annotations
import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- DAG3 columns ----------
COL_ID_DAG3      = "DAG3_sampleID"
COL_AGE          = "ANTHRO.AGE"
COL_BMI          = "ANTHRO.BMI"
COL_SEX          = "ANTHRO.Sex"
COL_HEALTHY_FLAG = "MED.DISEASES.None.No.Diseases"
COL_SMOKER_NOW   = "EXP.SMOKING.Smoker.Now"

# Pilot columns
PILOT_COL_STUDY  = "STUDY_ID"
PILOT_COL_SAMPLE = "Sample"
PILOT_COL_DAG3_A = "dag3_sampleid"
PILOT_COL_DAG3_B = "dag3_sampleid_additional"
PILOT_COL_TYPE   = "sample.type"  # fecal.adult / oral.adult

# Final output columns (exact order)
FINAL_COLS = [
    "STUDY_ID",
    "Oral_sample_ID",
    "Fecal_sample_ID",
    "Age",               # 0/1/2/3
    "Sex",               # 0/1
    "BMI",               # 0/1/2/3
    "Smoking",           # from DAG3
    "Antibiotics_3m",    # NEW
    "PPI_use",           # NEW
    "Steroids_ongoing",  # NEW
    "Healthy_only",
]

# ---------- helpers ----------
_id6_re_mpa = re.compile(r"(\d{6})(?=_metaphlan)")
_id6_re_any = re.compile(r"(\d{6})")

def extract_id6_from_header_col(colname: str) -> str | None:
    if not isinstance(colname, str):
        return None
    m = _id6_re_mpa.search(colname)
    if m:
        return m.group(1)
    m2 = _id6_re_any.search(colname)
    return m2.group(1) if m2 else None

def header_id6_list(path: Path) -> list[str]:
    """Return list of 6-digit ids from a MetaPhlAn merged header. Handles space/comma separated."""
    if path is None or not Path(path).exists():
        return []
    header = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.strip():
                continue
            if ln.lstrip().startswith("#"):
                continue
            header = ln.rstrip("\r\n")
            break
    if header is None:
        return []
    parts = re.split(r"[\s,]+", header)
    if parts and parts[0].lower() == "clade_name":
        parts = parts[1:]
    ids = [extract_id6_from_header_col(c) for c in parts]
    return [i for i in ids if i]

def to_6d(x) -> str:
    s = re.sub(r"\D", "", str(x)).zfill(6)
    return s[-6:]

def read_pilot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    cols_norm = {c.lower(): c for c in df.columns}
    sample_col = cols_norm.get(PILOT_COL_SAMPLE.lower(), None)
    if sample_col is None:
        best, best_rate = None, 0
        for c in df.columns:
            rate = df[c].astype(str).str.contains(r"\d{6}$").mean()
            if rate > best_rate:
                best, best_rate = c, rate
        sample_col = best
    df["id6"] = df[sample_col].astype(str).str.extract(r"(\d{6})$", expand=False)
    df["id6"] = df["id6"].dropna().astype(str).apply(to_6d)

    ren = {}
    a = cols_norm.get(PILOT_COL_DAG3_A.lower())
    b = cols_norm.get(PILOT_COL_DAG3_B.lower())
    s = cols_norm.get(PILOT_COL_STUDY.lower())
    t = cols_norm.get(PILOT_COL_TYPE.lower())
    if a and a != PILOT_COL_DAG3_A: ren[a] = PILOT_COL_DAG3_A
    if b and b != PILOT_COL_DAG3_B: ren[b] = PILOT_COL_DAG3_B
    if s and s != PILOT_COL_STUDY:  ren[s] = PILOT_COL_STUDY
    if t and t != PILOT_COL_TYPE:   ren[t] = PILOT_COL_TYPE
    if ren: df = df.rename(columns=ren)
    return df

def read_dag3_metadata(path: Path, sheet=0) -> pd.DataFrame:
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    if COL_ID_DAG3 not in df.columns:
        cols_norm = {c.lower(): c for c in df.columns}
        if COL_ID_DAG3.lower() in cols_norm:
            df = df.rename(columns={cols_norm[COL_ID_DAG3.lower()]: COL_ID_DAG3})
        else:
            raise KeyError("DAG3_sampleID column not found in DAG3 metadata.")
    df[COL_ID_DAG3] = df[COL_ID_DAG3].astype(str).str.strip()
    return df

def normalize_bool(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="int64")
    s = series.astype(str).str.strip().str.lower()
    yes_vals = {"1","true","yes","y","ja","waar","present","pos","positive","on"}
    no_vals  = {"0","false","no","n","nee","absent","neg","negative","off"}
    out = s.map(lambda x: 1 if x in yes_vals else (0 if x in no_vals else None))
    out = out.fillna(pd.to_numeric(s, errors="coerce").clip(lower=0).fillna(0).astype(float))
    return (out > 0).astype(int)

def build_base_from_pilot(pilot: pd.DataFrame,
                          fecal_ids_mpa: set[str],
                          oral_ids_mpa: set[str]) -> pd.DataFrame:
    """Pilot is authoritative for fecal/oral via sample.type; fallback to MetaPhlAn presence."""
    base = pilot.dropna(subset=["id6"])[["id6"]].drop_duplicates().copy()
    if PILOT_COL_TYPE in pilot.columns:
        typemap = (pilot.dropna(subset=["id6", PILOT_COL_TYPE])
                        .groupby("id6")[PILOT_COL_TYPE]
                        .agg(lambda s: "|".join(sorted(set(s))))
                        .reset_index())
        base = base.merge(typemap, on="id6", how="left")
        base["is_fecal"] = base[PILOT_COL_TYPE].fillna("").str.contains("fecal", case=False, regex=True)
        base["is_oral"]  = base[PILOT_COL_TYPE].fillna("").str.contains("oral",  case=False, regex=True)
        base = base.drop(columns=[PILOT_COL_TYPE], errors="ignore")
    else:
        base["is_fecal"] = base["id6"].isin(fecal_ids_mpa)
        base["is_oral"]  = base["id6"].isin(oral_ids_mpa)
    base["is_fecal"] = base["is_fecal"].astype(bool)
    base["is_oral"]  = base["is_oral"].astype(bool)
    return base

def pick_best_dag3(candidates: list[str]) -> str | None:
    """Return first non-empty candidate, preferring A over B implicitly by order of collection."""
    for c in candidates:
        if c is None:
            continue
        c2 = str(c).strip()
        if c2 and c2.lower() not in {"nan", "na"} and c2 not in {"' '", '" "', " "}:
            return c2
    return None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Make a single clean healthy metadata CSV (minimal outputs).")
    ap.add_argument("--pilot", default="data/pilot_with_all_IDs_mici.csv", type=str)
    ap.add_argument("--dag3",  default="data/DAG3_metadata_merged_ready_v27.xlsx", type=str)
    ap.add_argument("--fecal", default="data/fecal_only_results_metaphlan4.txt", type=str)
    ap.add_argument("--oral",  default="data/oral_only_results_metaphlan4.txt", type=str)
    ap.add_argument("--out",   default="data/meta/healthy_metadata.csv", type=str)
    ap.add_argument("--sheet", default="0", type=str)
    ap.add_argument("--filter-healthy", action="store_true",
                    help="If set, DROP rows where Healthy_only != 1")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_dir  = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot = read_pilot(Path(args.pilot))
    dag3  = read_dag3_metadata(Path(args.dag3), sheet=int(args.sheet) if args.sheet.isdigit() else args.sheet)
    fecal_ids_mpa = set(header_id6_list(Path(args.fecal)))
    oral_ids_mpa  = set(header_id6_list(Path(args.oral)))

    # Map id6 -> DAG3 (ignore blanks)
    rows = []
    for _, r in pilot.iterrows():
        i6 = r.get("id6")
        if not i6:
            continue
        cand = []
        for key in [PILOT_COL_DAG3_A, PILOT_COL_DAG3_B]:
            if key in pilot.columns:
                cand.append(r.get(key))
        chosen = pick_best_dag3(cand)
        rows.append({"id6": to_6d(i6), COL_ID_DAG3: chosen})
    map_id = pd.DataFrame(rows).drop_duplicates(["id6"])

    base = build_base_from_pilot(pilot, fecal_ids_mpa, oral_ids_mpa)
    inv = base.merge(map_id, on="id6", how="left")

    # DAG3 index to decide drops
    dag3_sub_cols = [c for c in [
        COL_ID_DAG3, COL_AGE, COL_BMI, COL_SEX, COL_HEALTHY_FLAG, COL_SMOKER_NOW
    ] if c in dag3.columns]
    dag3_sub = dag3[dag3_sub_cols].drop_duplicates(COL_ID_DAG3).copy()
    dag3_ids_set = set(dag3_sub[COL_ID_DAG3].dropna().astype(str).str.strip())

    inv["in_DAG3_meta"] = inv[COL_ID_DAG3].astype(str).str.strip().isin(dag3_ids_set)
    missing_in_dag3_mask = ~inv["in_DAG3_meta"].fillna(False)
    missing_id6_list = sorted(inv.loc[missing_in_dag3_mask, "id6"].astype(str).unique().tolist())

    # Merge DAG3 values AFTER marking missing, then drop missing
    inv = inv.merge(dag3_sub, on=COL_ID_DAG3, how="left")
    # Healthy_only (info only, unless --filter-healthy)
    if COL_HEALTHY_FLAG in inv.columns:
        inv["Healthy_only"] = (inv[COL_HEALTHY_FLAG].astype(str).str.strip().str.lower().isin(
            {"1","true","yes","y","ja","waar","present","pos","positive","on"}
        )).astype(int)
    else:
        inv["Healthy_only"] = 0

    # DROP rows not in DAG3 metadata (explicit request)
    inv = inv.loc[~missing_in_dag3_mask].copy()

    # Build sample IDs from base presence
    inv["Oral_sample_ID"]  = inv.apply(lambda r: r["id6"] if r["is_oral"]  else "NA", axis=1)
    inv["Fecal_sample_ID"] = inv.apply(lambda r: r["id6"] if r["is_fecal"] else "NA", axis=1)

    # STUDY_ID
    if PILOT_COL_STUDY in pilot.columns:
        study_map = pilot.dropna(subset=["id6"])[["id6", PILOT_COL_STUDY]].drop_duplicates("id6")
        inv = inv.merge(study_map, on="id6", how="left")
        inv["STUDY_ID"] = inv[PILOT_COL_STUDY].fillna("DAG3")
    else:
        inv["STUDY_ID"] = "DAG3"

    # ---- recode IN-PLACE ----
    # Sex -> 0/1 (Male=0, Female=1)
    sex_raw = inv.get(COL_SEX).astype(str).str.strip().str.upper()
    inv["Sex"] = sex_raw.map({
        "F":1, "FEMALE":1, "V":1, "W":1, "WOMAN":1,
        "M":0, "MALE":0, "H":0, "MAN":0
    }).astype("Int64")

    # Age (numeric source) -> 0/1/2/3
    age_num = pd.to_numeric(inv.get(COL_AGE), errors="coerce")
    age_code = pd.Series(pd.NA, index=inv.index, dtype="Int64")
    # Bins (inclusive notes in header docstring):
    # 0: Age < 18
    # 1: 18 ≤ Age ≤ 34
    # 2: 35 ≤ Age ≤ 49
    # 3: Age ≥ 50
    age_code.loc[age_num.notna() & (age_num < 18)] = 0
    age_code.loc[age_num.notna() & (age_num >= 18) & (age_num <= 34)] = 1
    age_code.loc[age_num.notna() & (age_num >= 35) & (age_num <= 49)] = 2
    age_code.loc[age_num.notna() & (age_num >= 50)] = 3
    inv["Age"] = age_code

    # BMI (numeric source) -> 0/1/2/3 (WHO-ish)
    bmi_num = pd.to_numeric(inv.get(COL_BMI), errors="coerce")
    bmi_code = pd.Series(pd.NA, index=inv.index, dtype="Int64")
    # 0: BMI < 18.5
    # 1: 18.5 ≤ BMI < 25
    # 2: 25 ≤ BMI < 30
    # 3: BMI ≥ 30
    bmi_code.loc[bmi_num.notna() & (bmi_num < 18.5)] = 0
    bmi_code.loc[bmi_num.notna() & (bmi_num >= 18.5) & (bmi_num < 25)] = 1
    bmi_code.loc[bmi_num.notna() & (bmi_num >= 25)   & (bmi_num < 30)] = 2
    bmi_code.loc[bmi_num.notna() & (bmi_num >= 30)]  = 3
    inv["BMI"] = bmi_code

    # Smoking (0/1) from DAG3
    inv["Smoking"] = normalize_bool(inv.get(COL_SMOKER_NOW))

    # Antibiotics in last 3 months
    if "META.Antibiotics_3m" in inv.columns:
        inv["Antibiotics_3m"] = normalize_bool(inv["META.Antibiotics_3m"])
    else:
        inv["Antibiotics_3m"] = 0

    # PPI use (prefer specific ATC for PPIs; fallback to general stomach-acid meds)
    ppi_cols = ["MED.MEDS.PPIs_ATC_A02BC", "MED.MEDS.Stomach_Acid_ATC_A02"]
    ppi_flag = pd.Series(0, index=inv.index)
    for c in ppi_cols:
        if c in inv.columns:
            ppi_flag = ppi_flag | normalize_bool(inv[c])
    inv["PPI_use"] = ppi_flag.astype(int)

    # Steroids ongoing (OR across oral / inhaled / eyedrops)
    steroid_cols = [
        "MED.MEDS.Corticosteroids_Oral_ATC_H02",
        "MED.MEDS.Glucocorticoids_Inhaled_ATC_R03BA",
        "MED.MEDS.Corticosteroids_Eyedrops_ATC_S03A",
    ]
    st_flag = pd.Series(0, index=inv.index)
    for c in steroid_cols:
        if c in inv.columns:
            st_flag = st_flag | normalize_bool(inv[c])
    inv["Steroids_ongoing"] = st_flag.astype(int)

    # Final select & write
    out = inv[FINAL_COLS].drop_duplicates().reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote: {out_path} (rows={len(out)})")

    # ---- summary_counts.csv (only) ----
    summary = pd.DataFrame([
        {"metric":"final_rows", "value": len(out)},
        {"metric":"final_fecal_ids_nonNA", "value": int((out['Fecal_sample_ID']!='NA').sum())},
        {"metric":"final_oral_ids_nonNA",  "value": int((out['Oral_sample_ID']!='NA').sum())},
        {"metric":"dropped_not_in_DAG3_count", "value": len(missing_id6_list)},
        {"metric":"dropped_not_in_DAG3_list",  "value": ";".join(missing_id6_list)},
        {"metric":"count_antibiotics_3m_1", "value": int(out["Antibiotics_3m"].sum())},
        {"metric":"count_ppi_use_1", "value": int(out["PPI_use"].sum())},
        {"metric":"count_steroids_1", "value": int(out["Steroids_ongoing"].sum())},
    ])
    summary.to_csv(out_dir / "summary_counts.csv", index=False)
    print(f"[OK] wrote: {out_dir / 'summary_counts.csv'}")

if __name__ == "__main__":
    main()
