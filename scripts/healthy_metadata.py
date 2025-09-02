#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------- constants -----------------
COL_ID_DAG3      = "DAG3_sampleID"
COL_AGE          = "ANTHRO.AGE"
COL_BMI          = "ANTHRO.BMI"
COL_SEX          = "ANTHRO.Sex"
COL_HEALTHY_FLAG = "MED.DISEASES.None.No.Diseases"
COL_SMOKER_NOW   = "EXP.SMOKING.Smoker.Now"

PILOT_COL_STUDY  = "STUDY_ID"
PILOT_COL_SAMPLE = "Sample"
PILOT_COL_DAG3_A = "dag3_sampleid"
PILOT_COL_DAG3_B = "dag3_sampleid_additional"
PILOT_COL_TYPE   = "sample.type"

FINAL_COLS = [
    "STUDY_ID","Oral_sample_ID","Fecal_sample_ID",
    "Age","Sex","BMI","Smoking",
    "Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing",
    "Healthy_only",
]

_id6_re_mpa = re.compile(r"(\d{6})(?=_metaphlan)")
_id6_re_any = re.compile(r"(\d{6})")

# ----------------- helpers -----------------
def extract_id6_from_header_col(colname: str) -> str | None:
    if not isinstance(colname, str):
        return None
    m = _id6_re_mpa.search(colname)
    if m:
        return m.group(1)
    m2 = _id6_re_any.search(colname)
    return m2.group(1) if m2 else None

def header_id6_list(path: Path) -> list[str]:
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

    # find the "Sample" column (fallback: best column with 6-digit suffix)
    sample_col = cols_norm.get(PILOT_COL_SAMPLE.lower(), None)
    if sample_col is None:
        best, best_rate = None, 0
        for c in df.columns:
            rate = df[c].astype(str).str.contains(r"\d{6}$").mean()
            if rate > best_rate:
                best, best_rate = c, rate
        sample_col = best

    # id6 extracted from Sample
    df["id6"] = df[sample_col].astype(str).str.extract(r"(\d{6})$", expand=False)
    df["id6"] = df["id6"].astype(str).apply(to_6d)

    # normalize known pilot column names
    ren = {}
    a = cols_norm.get(PILOT_COL_DAG3_A.lower())
    b = cols_norm.get(PILOT_COL_DAG3_B.lower())
    s = cols_norm.get(PILOT_COL_STUDY.lower())
    t = cols_norm.get(PILOT_COL_TYPE.lower())
    if a and a != PILOT_COL_DAG3_A: ren[a] = PILOT_COL_DAG3_A
    if b and b != PILOT_COL_DAG3_B: ren[b] = PILOT_COL_DAG3_B
    if s and s != PILOT_COL_STUDY:  ren[s] = PILOT_COL_STUDY
    if t and t != PILOT_COL_TYPE:   ren[t] = PILOT_COL_TYPE
    if ren:
        df = df.rename(columns=ren)

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

def normalize_bool(series):
    """Map mixed boolean-like values to 0/1."""
    if series is None:
        return pd.Series(dtype="int64")
    s = pd.Series(series).astype(str).str.strip().str.lower()
    yes_vals = {"1","true","yes","y","ja","waar","present","pos","positive","on"}
    no_vals  = {"0","false","no","n","nee","absent","neg","negative","off"}
    out = s.map(lambda x: 1 if x in yes_vals else (0 if x in no_vals else None))
    # numeric fallback (>0 -> 1 else 0)
    out = out.fillna(pd.to_numeric(s, errors="coerce").clip(lower=0).fillna(0).astype(float))
    return (out > 0).astype(int)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Build clean healthy metadata CSV.")
    # NOTE: --in is accepted for Snakefile compatibility but ignored.
    ap.add_argument("--in", dest="ignored_in", type=str, default=None,
                    help="(ignored; kept for Snakefile compatibility)")
    ap.add_argument("--pilot", default="data/pilot_with_all_IDs_mici.csv", type=str)
    ap.add_argument("--dag3",  default="data/DAG3_metadata_merged_ready_v27.xlsx", type=str)
    ap.add_argument("--fecal", default="data/fecal_only_results_metaphlan4.txt", type=str)
    ap.add_argument("--oral",  default="data/oral_only_results_metaphlan4.txt", type=str)
    ap.add_argument("--out",   default="data/meta/healthy_metadata.csv", type=str)
    ap.add_argument("--sheet", default="0", type=str)
    ap.add_argument("--filter-healthy", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pilot = read_pilot(Path(args.pilot))
    dag3  = read_dag3_metadata(Path(args.dag3), sheet=int(args.sheet) if args.sheet.isdigit() else args.sheet)

    fecal_ids_mpa = set(header_id6_list(Path(args.fecal)))
    oral_ids_mpa  = set(header_id6_list(Path(args.oral)))

    # map id6 -> chosen DAG3_sampleID (prefer A then B, first non-empty)
    rows = []
    for _, r in pilot.iterrows():
        i6 = r.get("id6")
        if not i6:
            continue
        candidates = []
        for key in (PILOT_COL_DAG3_A, PILOT_COL_DAG3_B):
            if key in pilot.columns:
                candidates.append(r.get(key))
        chosen = None
        for c in candidates:
            if c is None:
                continue
            c2 = str(c).strip()
            if c2 and c2.lower() not in {"nan","na"} and c2 not in {"' '","", " "}:
                chosen = c2
                break
        rows.append({"id6": to_6d(i6), COL_ID_DAG3: chosen})
    map_id = pd.DataFrame(rows).drop_duplicates(["id6"])

    # id6 with site flags
    base = pilot.dropna(subset=["id6"])[["id6"]].drop_duplicates().copy()
    if PILOT_COL_TYPE in pilot.columns:
        typemap = (
            pilot.dropna(subset=["id6", PILOT_COL_TYPE])
                 .groupby("id6")[PILOT_COL_TYPE]
                 .agg(lambda s: "|".join(sorted(set(s))))
                 .reset_index()
        )
        base = base.merge(typemap, on="id6", how="left")
        base["is_fecal"] = base[PILOT_COL_TYPE].fillna("").str.contains("fecal", case=False, regex=True)
        base["is_oral"]  = base[PILOT_COL_TYPE].fillna("").str.contains("oral",  case=False, regex=True)
        base = base.drop(columns=[PILOT_COL_TYPE], errors="ignore")
    else:
        base["is_fecal"] = base["id6"].isin(fecal_ids_mpa)
        base["is_oral"]  = base["id6"].isin(oral_ids_mpa)
    base["is_fecal"] = base["is_fecal"].astype(bool)
    base["is_oral"]  = base["is_oral"].astype(bool)

    inv = base.merge(map_id, on="id6", how="left")

    # pull limited set of covariates from DAG3
    dag3_sub_cols = [c for c in [
        COL_ID_DAG3, COL_AGE, COL_BMI, COL_SEX, COL_HEALTHY_FLAG, COL_SMOKER_NOW,
        "MED.MEDS.PPIs_ATC_A02BC", "MED.MEDS.Stomach_Acid_ATC_A02",
        "MED.MEDS.Corticosteroids_Oral_ATC_H02", "MED.MEDS.Glucocorticoids_Inhaled_ATC_R03BA",
        "MED.MEDS.Corticosteroids_Eyedrops_ATC_S03A", "META.Antibiotics_3m",
        "MED.MEDS.Antibacterials_ATC_J01"
    ] if c in dag3.columns]
    dag3_sub = dag3[dag3_sub_cols].drop_duplicates(COL_ID_DAG3).copy()
    dag3_ids_set = set(dag3_sub[COL_ID_DAG3].dropna().astype(str).str.strip())

    # keep only rows whose DAG3_sampleID exists in DAG3 metadata
    inv["in_DAG3_meta"] = inv[COL_ID_DAG3].astype(str).str.strip().isin(dag3_ids_set)
    missing_mask = ~inv["in_DAG3_meta"].fillna(False)
    dropped_id6 = sorted(inv.loc[missing_mask, "id6"].astype(str).unique().tolist())
    inv = inv.merge(dag3_sub, on=COL_ID_DAG3, how="left")
    inv = inv.loc[~missing_mask].copy()

    # flags
    inv["Healthy_only"] = inv.get(COL_HEALTHY_FLAG, 0).astype(str).str.strip().str.lower().isin(
        {"1","true","yes","y","ja","waar","present","pos","positive","on"}).astype(int)

    # site-specific sample IDs (use id6)
    inv["Oral_sample_ID"]  = np.where(inv["is_oral"],  inv["id6"], "NA")
    inv["Fecal_sample_ID"] = np.where(inv["is_fecal"], inv["id6"], "NA")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # CRITICAL: STUDY_ID must equal DAG3_sampleID for Healthy
    inv["STUDY_ID"] = inv[COL_ID_DAG3].fillna("DAG3")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # covariates: Sex (F=1, M=0), Age bins, BMI bins, Smoking
    sex_raw = inv.get(COL_SEX).astype(str).str.strip().str.upper()
    inv["Sex"] = sex_raw.map({
        "F":1, "FEMALE":1, "V":1, "W":1, "WOMAN":1,
        "M":0, "MALE":0, "H":0, "MAN":0
    }).astype("Int64")

    age_num = pd.to_numeric(inv.get(COL_AGE), errors="coerce")
    age_code = pd.Series(pd.NA, index=inv.index, dtype="Int64")
    age_code.loc[age_num.notna() & (age_num < 15)] = 0
    age_code.loc[age_num.notna() & (age_num >= 15) & (age_num <= 39)] = 1
    age_code.loc[age_num.notna() & (age_num >= 40) & (age_num <= 59)] = 2
    age_code.loc[age_num.notna() & (age_num >= 60)] = 3
    inv["Age"] = age_code

    bmi_num = pd.to_numeric(inv.get(COL_BMI), errors="coerce")
    bmi_code = pd.Series(pd.NA, index=inv.index, dtype="Int64")
    bmi_code.loc[bmi_num.notna() & (bmi_num < 18.5)] = 0
    bmi_code.loc[bmi_num.notna() & (bmi_num >= 18.5) & (bmi_num < 25)] = 1
    bmi_code.loc[bmi_num.notna() & (bmi_num >= 25) & (bmi_num < 30)] = 2
    bmi_code.loc[bmi_num.notna() & (bmi_num >= 30)] = 3
    inv["BMI"] = bmi_code

    inv["Smoking"] = inv.get(COL_SMOKER_NOW, 0).astype(str).str.strip().str.lower().isin(
        {"1","true","yes","y","ja","waar","present","pos","positive","on"}).astype(int)

    # meds
    abx_flag = pd.Series(0, index=inv.index)
    for c in ["MED.MEDS.Antibacterials_ATC_J01", "META.Antibiotics_3m"]:
        if c in inv.columns:
            abx_flag = abx_flag | normalize_bool(inv[c])
    inv["Antibiotics_3m"] = abx_flag.astype(int)

    ppi_flag = pd.Series(0, index=inv.index)
    for c in ["MED.MEDS.PPIs_ATC_A02BC", "MED.MEDS.Stomach_Acid_ATC_A02"]:
        if c in inv.columns:
            ppi_flag = ppi_flag | normalize_bool(inv[c])
    inv["PPI_use"] = ppi_flag.astype(int)

    st_flag = pd.Series(0, index=inv.index)
    for c in [
        "MED.MEDS.Corticosteroids_Oral_ATC_H02",
        "MED.MEDS.Glucocorticoids_Inhaled_ATC_R03BA",
        "MED.MEDS.Corticosteroids_Eyedrops_ATC_S03A",
    ]:
        if c in inv.columns:
            st_flag = st_flag | normalize_bool(inv[c])
    inv["Steroids_ongoing"] = st_flag.astype(int)

    im_flag = pd.Series(0, index=inv.index)
    for col in inv.columns:
        if re.search(r"^MED\.MEDS\..*immun.*ATC_L", col, flags=re.I):
            im_flag = im_flag | normalize_bool(inv[col])
    inv["Immuno_ongoing"] = im_flag.astype(int)

    # output
    out = inv[FINAL_COLS].drop_duplicates().reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote: {out_path} (rows={len(out)})")

    # small summary sidecar
    summary = pd.DataFrame([
        {"metric":"final_rows", "value": len(out)},
        {"metric":"final_fecal_ids_nonNA", "value": int((out['Fecal_sample_ID']!='NA').sum())},
        {"metric":"final_oral_ids_nonNA",  "value": int((out['Oral_sample_ID']!='NA').sum())},
        {"metric":"dropped_not_in_DAG3_count", "value": len(dropped_id6)},
        {"metric":"dropped_not_in_DAG3_list",  "value": ";".join(dropped_id6)},
        {"metric":"count_antibiotics_3m_1", "value": int(out['Antibiotics_3m'].sum())},
        {"metric":"count_ppi_use_1", "value": int(out['PPI_use'].sum())},
        {"metric":"count_steroids_1", "value": int(out['Steroids_ongoing'].sum())},
        {"metric":"count_immuno_1", "value": int(out["Immuno_ongoing"].sum())},
    ])
    summary.to_csv(out_path.parent / "summary_counts.csv", index=False)
    print(f"[OK] wrote: {out_path.parent / 'summary_counts.csv'}")

if __name__ == "__main__":
    main()
