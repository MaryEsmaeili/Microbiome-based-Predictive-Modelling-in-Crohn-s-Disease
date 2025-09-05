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

def to_6d(x) -> str | None:
    if x is None: return None
    s = re.sub(r"\D", "", str(x)).zfill(6)
    return s[-6:] if s else None

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
            if not ln.strip(): continue
            if ln.lstrip().startswith("#"): continue
            header = ln.rstrip("\r\n")
            break
    if header is None:
        return []
    parts = re.split(r"[\s,]+", header)
    if parts and parts[0].lower() == "clade_name":
        parts = parts[1:]
    ids = [extract_id6_from_header_col(c) for c in parts]
    return [i for i in ids if i]

def read_pilot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    cols_norm = {c.lower(): c for c in df.columns}

    # ---- Sample → id6 (۶ رقم انتهایی) ----
    sample_col = cols_norm.get(PILOT_COL_SAMPLE.lower())
    if sample_col is None:
        best, best_rate = None, 0
        for c in df.columns:
            rate = df[c].astype(str).str.contains(r"\d{6}$").mean()
            if rate > best_rate:
                best, best_rate = c, rate
        sample_col = best
    if sample_col:
        df["id6"] = df[sample_col].astype(str).str.extract(r"(\d{6})$", expand=False)
        df["id6"] = df["id6"].apply(to_6d)
    else:
        df["id6"] = None

    # ---- normalize col names ----
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

    # ---- pick one DAG3_sampleID per row (A then B) ----
    chosen = []
    for _, r in df.iterrows():
        c = None
        for key in (PILOT_COL_DAG3_A, PILOT_COL_DAG3_B):
            v = r.get(key)
            if v and str(v).strip() and str(v).strip().lower() not in {"nan","na"}:
                c = str(v).strip()
                break
        chosen.append(c)
    df[COL_ID_DAG3] = chosen
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
    if series is None:
        return pd.Series(dtype="int64")
    s = pd.Series(series).astype(str).str.strip().str.lower()
    yes_vals = {"1","true","yes","y","ja","waar","present","pos","positive","on"}
    no_vals  = {"0","false","no","n","nee","absent","neg","negative","off"}
    out = s.map(lambda x: 1 if x in yes_vals else (0 if x in no_vals else None))
    out = out.fillna(pd.to_numeric(s, errors="coerce").clip(lower=0).fillna(0).astype(float))
    return (out > 0).astype(int)

def read_links(path: Path | None) -> pd.DataFrame:
    """انتظار: دو ستون DAG3_sampleID و OralID (۶رقمی)."""
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=[COL_ID_DAG3, "OralID"])
    df = pd.read_csv(path, dtype=str)
    cols = {c.lower(): c for c in df.columns}
    dag3_col = cols.get("dag3_sampleid") or cols.get(COL_ID_DAG3.lower())
    oral_col = cols.get("oralid") or cols.get("oral_id")
    if dag3_col is None or oral_col is None:
        raise RuntimeError("links csv باید ستون‌های DAG3_sampleID و OralID داشته باشد.")
    out = pd.DataFrame({
        COL_ID_DAG3: df[dag3_col].astype(str).str.strip(),
        "OralID": df[oral_col].apply(to_6d)
    })
    out = out.dropna(subset=["OralID"]).drop_duplicates()
    return out

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Build clean healthy metadata CSV (با لینک‌های اورال).")
    # پشتیبانی از نام جدید و قدیمی آرگومان لینک
    ap.add_argument("--links", dest="links", type=str, default=None,
                    help="CSV با ستون‌های DAG3_sampleID,OralID")
    ap.add_argument("--in", dest="links_legacy", type=str, default=None,
                    help="(alias قدیمی) همان --links")
    ap.add_argument("--pilot", default="data/pilot_with_all_IDs_mici.csv", type=str)
    ap.add_argument("--dag3",  default="data/DAG3_metadata_merged_ready_v27.xlsx", type=str)
    ap.add_argument("--fecal", default="data/fecal_only_results_metaphlan4.txt", type=str)
    ap.add_argument("--oral",  default="data/oral_only_results_metaphlan4.txt", type=str)
    ap.add_argument("--out",   default="data/meta/healthy_metadata.csv", type=str)
    ap.add_argument("--sheet", default="0", type=str)
    args = ap.parse_args()

    links_path = args.links or args.links_legacy

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pilot = read_pilot(Path(args.pilot))
    dag3  = read_dag3_metadata(Path(args.dag3), sheet=int(args.sheet) if str(args.sheet).isdigit() else args.sheet)
    links = read_links(Path(links_path)) if links_path else pd.DataFrame(columns=[COL_ID_DAG3, "OralID"])

    fecal_ids_mpa = set(header_id6_list(Path(args.fecal)))
    oral_ids_mpa  = set(header_id6_list(Path(args.oral)))

    # --- رکوردهای پایلوت (id6 + نوع نمونه از sample.type یا هدر متافلن) ---
    base = pilot.dropna(subset=[COL_ID_DAG3]).copy()
    base["id6"] = base["id6"].apply(to_6d)
    if PILOT_COL_TYPE in base.columns:
        typ = base[PILOT_COL_TYPE].astype(str).str.lower()
        base["is_oral_src"]  = typ.str.contains("oral",  na=False)
        base["is_fecal_src"] = typ.str.contains("fecal", na=False)
    else:
        base["is_oral_src"]  = base["id6"].isin(oral_ids_mpa)
        base["is_fecal_src"] = base["id6"].isin(fecal_ids_mpa)

    # --- رکوردهای لینک (اورال ۶رقمی به ازای DAG3) ---
    link_rows = links.rename(columns={"OralID":"id6"}).copy()
    link_rows["is_oral_src"]  = True
    link_rows["is_fecal_src"] = link_rows["id6"].isin(fecal_ids_mpa)  # احتمالا False

    # هم‌شکل‌سازی ستون‌ها
    keep_cols = [COL_ID_DAG3, "id6", "is_oral_src", "is_fecal_src"]
    base_rows = base[[COL_ID_DAG3, "id6", "is_oral_src", "is_fecal_src"]].dropna(subset=[COL_ID_DAG3])
    records  = pd.concat([base_rows, link_rows[keep_cols]], ignore_index=True)
    records["id6"] = records["id6"].apply(to_6d)

    # --- الحاق کووریِت‌ها از DAG3 ---
    dag3_sub_cols = [c for c in [
        COL_ID_DAG3, COL_AGE, COL_BMI, COL_SEX, COL_HEALTHY_FLAG, COL_SMOKER_NOW,
        "MED.MEDS.PPIs_ATC_A02BC", "MED.MEDS.Stomach_Acid_ATC_A02",
        "MED.MEDS.Corticosteroids_Oral_ATC_H02", "MED.MEDS.Glucocorticoids_Inhaled_ATC_R03BA",
        "MED.MEDS.Corticosteroids_Eyedrops_ATC_S03A", "META.Antibiotics_3m",
        "MED.MEDS.Antibacterials_ATC_J01"
    ] if c in dag3.columns]
    dag3_sub = dag3[dag3_sub_cols].drop_duplicates(COL_ID_DAG3).copy()

    records = records.merge(dag3_sub, on=COL_ID_DAG3, how="inner")  # فقط آن‌هایی که در DAG3 هستند

    # --- ساخت جدول نهایی به ازای هر DAG3_sampleID ---
    def first_non_na(s):
        x = s.dropna()
        return x.iloc[0] if len(x) else pd.NA

    # اورال: اولویت با لینک؛ اگر نبود، id6هایی که از پایلوت به‌عنوان oral آمده‌اند
    oral_pref = records.assign(
        oral_candidate = np.where(records["is_oral_src"], records["id6"], pd.NA)
    )
    oral_link = links.set_index(COL_ID_DAG3)["OralID"]

    grouped = []
    for dag3_id, grp in records.groupby(COL_ID_DAG3, dropna=True):
        fecal_id = first_non_na(grp.loc[grp["is_fecal_src"]==True, "id6"])
        oral_id  = oral_link.get(dag3_id, pd.NA)
        if pd.isna(oral_id):
            oral_id = first_non_na(grp.loc[grp["is_oral_src"]==True, "id6"])

        row = {
            "STUDY_ID": dag3_id,
            "Oral_sample_ID": oral_id if pd.notna(oral_id) else "NA",
            "Fecal_sample_ID": fecal_id if pd.notna(fecal_id) else "NA",
        }
        # covariates (همه از dag3_sub ثابت‌اند در گروه)
        anyrow = grp.iloc[0]
        # Sex
        sex_raw = str(anyrow.get(COL_SEX, "")).strip().upper()
        sex = 1 if sex_raw in {"F","FEMALE","V","W","WOMAN"} else (0 if sex_raw in {"M","MALE","H","MAN"} else pd.NA)
        # Age
        age_num = pd.to_numeric(anyrow.get(COL_AGE, pd.NA), errors="coerce")
        if pd.isna(age_num): age_code = pd.NA
        elif age_num < 15:   age_code = 0
        elif age_num <= 39:  age_code = 1
        elif age_num <= 59:  age_code = 2
        else:                age_code = 3
        # BMI
        bmi_num = pd.to_numeric(anyrow.get(COL_BMI, pd.NA), errors="coerce")
        if pd.isna(bmi_num): bmi_code = pd.NA
        elif bmi_num < 18.5: bmi_code = 0
        elif bmi_num < 25:   bmi_code = 1
        elif bmi_num < 30:   bmi_code = 2
        else:                bmi_code = 3
        # Flags
        def norm1(x):
            s = str(x).strip().lower()
            if s in {"1","true","yes","y","ja","waar","present","pos","positive","on"}: return 1
            if s in {"0","false","no","n","nee","absent","neg","negative","off"}: return 0
            try:
                return 1 if float(s) > 0 else 0
            except: return 0

        abx  = norm1(anyrow.get("MED.MEDS.Antibacterials_ATC_J01", 0)) | norm1(anyrow.get("META.Antibiotics_3m", 0))
        ppi  = norm1(anyrow.get("MED.MEDS.PPIs_ATC_A02BC", 0)) | norm1(anyrow.get("MED.MEDS.Stomach_Acid_ATC_A02", 0))
        ster = norm1(anyrow.get("MED.MEDS.Corticosteroids_Oral_ATC_H02", 0)) \
             | norm1(anyrow.get("MED.MEDS.Glucocorticoids_Inhaled_ATC_R03BA", 0)) \
             | norm1(anyrow.get("MED.MEDS.Corticosteroids_Eyedrops_ATC_S03A", 0))

        # immuno: هر ستونی با الگوی ATC_L و واژه immun
        immuno = 0
        for c in grp.columns:
            if re.search(r"^MED\.MEDS\..*immun.*ATC_L", c, flags=re.I):
                immuno = immuno | norm1(first_non_na(grp[c]))

        row.update({
            "Age": age_code,
            "Sex": sex,
            "BMI": bmi_code,
            "Smoking": norm1(anyrow.get(COL_SMOKER_NOW, 0)),
            "Antibiotics_3m": int(abx),
            "PPI_use": int(ppi),
            "Steroids_ongoing": int(ster),
            "Immuno_ongoing": int(immuno),
            "Healthy_only": norm1(anyrow.get(COL_HEALTHY_FLAG, 0)),
        })
        grouped.append(row)

    out = pd.DataFrame(grouped, columns=FINAL_COLS).drop_duplicates().reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote: {out_path} (rows={len(out)})")

    # خلاصه QC
    summary = pd.DataFrame([
        {"metric":"final_rows", "value": len(out)},
        {"metric":"final_fecal_ids_nonNA", "value": int((out['Fecal_sample_ID']!='NA').sum())},
        {"metric":"final_oral_ids_nonNA",  "value": int((out['Oral_sample_ID']!='NA').sum())},
    ])
    summary.to_csv(out_path.parent / "summary_counts.csv", index=False)
    print(f"[OK] wrote: {out_path.parent / 'summary_counts.csv'}")

if __name__ == "__main__":
    main()
