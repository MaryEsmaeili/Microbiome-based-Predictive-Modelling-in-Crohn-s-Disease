#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Healthy and Crohn metadata as standalone (no Snakemake needed).

HEALTHY:
- Inputs:
    pilot  = data/pilot_with_all_IDs_mici.csv
    dag3   = data/DAG3_metadata_merged_ready_v27.xlsx
    fecal  = data/fecal_only_results_metaphlan4.txt
    oral   = data/oral_only_results_metaphlan4.txt
    links  = data/meta/healthy_oral_links_clean.csv   (DAG3_sampleID,OralID)
- Output:
    data/meta/healthy_metadata.csv

CROHN:
- Input:
    data/Metadata_metagenomics.xlsx   (row-oriented; variables in column 0)
- Output (single rich file):
    data/meta/crohn_metadata.csv
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------- Paths (change only if needed) ---------------------------

# Healthy
IN_PILOT   = Path("data/pilot_with_all_IDs_mici.csv")
IN_DAG3    = Path("data/DAG3_metadata_merged_ready_v27.xlsx")
IN_FECAL   = Path("data/fecal_only_results_metaphlan4.txt")
IN_ORAL    = Path("data/oral_only_results_metaphlan4.txt")
IN_LINKS   = Path("data/meta/healthy_oral_links_clean.csv")
OUT_HEALTH = Path("data/meta/healthy_metadata.csv")

# Crohn
IN_CROHN_XLSX = Path("data/Metadata_metagenomics.xlsx")
OUT_CR_EXT    = Path("data/meta/crohn_metadata.csv")

# Ensure out dir exists
OUT_HEALTH.parent.mkdir(parents=True, exist_ok=True)
OUT_CR_EXT.parent.mkdir(parents=True, exist_ok=True)

# --------------------------- Small helpers ---------------------------

def to_6d(x) -> str | None:
    """Normalize any id to the last 6 digits (zero-padded)."""
    if x is None:
        return None
    s = re.sub(r"\D", "", str(x))
    if not s:
        return None
    s = s.zfill(6)
    return s[-6:]

def header_id6_list(path: Path) -> list[str]:
    """Read the first non-comment non-empty line of a MetaPhlAn table header and extract 6-digit IDs."""
    if not path.exists():
        return []
    header = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            header = ln.rstrip("\r\n")
            break
    if header is None:
        return []
    parts = re.split(r"[\s,]+", header)
    if parts and parts[0].lower() == "clade_name":
        parts = parts[1:]
    ids = []
    for c in parts:
        m = re.search(r"(\d{6})(?=_metaphlan)", c)
        if m:
            ids.append(m.group(1))
            continue
        m2 = re.search(r"(\d{6})", c)
        if m2:
            ids.append(m2.group(1))
    return [i for i in ids if i]

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Trim/normalize header spaces."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return out

def clean_str_na(s: pd.Series) -> pd.Series:
    """Strip strings and set '', 'nan' to <NA>."""
    return (
        s.astype(str)
         .str.strip()
         .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NAN": pd.NA})
    )

def norm_yesno_scalar(x):
    """Map yes/no style values to {0,1,None}."""
    s = "" if x is None else str(x).strip().lower()
    if s in {"1","true","yes","y","ja","waar","present","pos","positive","on"}:
        return 1
    if s in {"0","false","no","n","nee","absent","neg","negative","off"}:
        return 0
    try:
        f = float(s)
        return 1 if f > 0 else 0
    except Exception:
        return None

def choose_first(series: pd.Series):
    """First non-null value in a series."""
    x = series.dropna()
    return x.iloc[0] if len(x) else pd.NA

def drop_all_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are all-NA (ignoring ID columns)."""
    ignore = [c for c in df.columns if c.endswith("_ID") or c in {"STUDY_ID"}]
    base = df.drop(columns=ignore, errors="ignore")
    keep = ~base.isna().all(axis=1)
    return df.loc[keep].reset_index(drop=True)

# =============================================================================
#                                HEALTHY
# =============================================================================

# Column constants (DAG3)
COL_ID_DAG3      = "DAG3_sampleID"
COL_AGE          = "ANTHRO.AGE"
COL_BMI          = "ANTHRO.BMI"
COL_SEX          = "ANTHRO.Sex"
COL_HEALTHY_FLAG = "MED.DISEASES.None.No.Diseases"
COL_SMOKER_NOW   = "EXP.SMOKING.Smoker.Now"

# Pilot columns (DB linking)
PILOT_COL_STUDY  = "STUDY_ID"
PILOT_COL_SAMPLE = "Sample"
PILOT_COL_DAG3_A = "dag3_sampleid"
PILOT_COL_DAG3_B = "dag3_sampleid_additional"
PILOT_COL_TYPE   = "sample.type"

def read_pilot(path: Path) -> pd.DataFrame:
    """Read pilot CSV and derive 6-digit id (id6), DAG3_sampleID, sample.type (if present)."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    cols_norm = {c.lower(): c for c in df.columns}

    # Find a column that ends with a 6-digit id (fallback best-match)
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

    # Normalize expected column names (DAG3 A/B, STUDY_ID, sample.type)
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

    # Choose one DAG3_sampleID per row (A then B)
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
    df[COL_ID_DAG3] = df[COL_ID_DAG3].astype(str).str.strip()
    return df

def read_dag3(path: Path) -> pd.DataFrame:
    """Read DAG3 xlsx/csv with DAG3_sampleID column present."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    cols_norm = {c.lower(): c for c in df.columns}
    if COL_ID_DAG3 not in df.columns:
        if COL_ID_DAG3.lower() in cols_norm:
            df = df.rename(columns={cols_norm[COL_ID_DAG3.lower()]: COL_ID_DAG3})
        else:
            raise KeyError("DAG3_sampleID column not found in DAG3 metadata.")
    df = normalize_headers(df)
    df[COL_ID_DAG3] = df[COL_ID_DAG3].astype(str).str.strip()
    return df

def read_links(path: Path) -> pd.DataFrame:
    """CSV with DAG3_sampleID,OralID (6 digits)."""
    if not path.exists():
        return pd.DataFrame(columns=[COL_ID_DAG3, "OralID"])
    df = pd.read_csv(path, dtype=str)
    cols = {c.lower(): c for c in df.columns}
    dag3_col = cols.get("dag3_sampleid") or cols.get(COL_ID_DAG3.lower())
    oral_col = cols.get("oralid") or cols.get("oral_id")
    if dag3_col is None or oral_col is None:
        raise RuntimeError("links CSV must have columns: DAG3_sampleID, OralID")
    out = pd.DataFrame({
        COL_ID_DAG3: df[dag3_col].astype(str).str.strip(),
        "OralID": df[oral_col].apply(to_6d)
    })
    out = out.dropna(subset=["OralID"]).drop_duplicates()
    return out

def build_healthy():
    # Read sources
    pilot = read_pilot(IN_PILOT)
    dag3  = read_dag3(IN_DAG3)
    links = read_links(IN_LINKS)

    fecal_ids = set(header_id6_list(IN_FECAL))
    oral_ids  = set(header_id6_list(IN_ORAL))

    # Row-wise records from pilot (possible oral/fecal sources)
    base = pilot.dropna(subset=[COL_ID_DAG3]).copy()
    base["id6"] = base["id6"].apply(to_6d)
    if PILOT_COL_TYPE in base.columns:
        typ = base[PILOT_COL_TYPE].astype(str).str.lower()
        base["is_oral_src"]  = typ.str.contains("oral",  na=False)
        base["is_fecal_src"] = typ.str.contains("fecal", na=False)
    else:
        base["is_oral_src"]  = base["id6"].isin(oral_ids)
        base["is_fecal_src"] = base["id6"].isin(fecal_ids)

    base_rows = base[[COL_ID_DAG3,"id6","is_oral_src","is_fecal_src"]].copy()

    # Add link rows (OralID from links)
    link_rows = links.rename(columns={"OralID":"id6"}).copy()
    link_rows["is_oral_src"]  = True
    link_rows["is_fecal_src"] = link_rows["id6"].isin(fecal_ids)

    rec = pd.concat([base_rows, link_rows[[COL_ID_DAG3,"id6","is_oral_src","is_fecal_src"]]], ignore_index=True)
    rec["id6"] = rec["id6"].apply(to_6d)

    # Keep DAG3 covariates + all disease columns
    base_keep = [c for c in [
        COL_ID_DAG3, COL_AGE, COL_BMI, COL_SEX, COL_SMOKER_NOW, COL_HEALTHY_FLAG,
        "MED.MEDS.PPIs_ATC_A02BC", "MED.MEDS.Stomach_Acid_ATC_A02",
        "MED.MEDS.Corticosteroids_Oral_ATC_H02", "MED.MEDS.Glucocorticoids_Inhaled_ATC_R03BA",
        "MED.MEDS.Corticosteroids_Eyedrops_ATC_S03A",
        "META.Antibiotics_3m", "MED.MEDS.Antibacterials_ATC_J01"
    ] if c in dag3.columns]
    disease_cols_all = [c for c in dag3.columns if str(c).startswith("MED.DISEASES.")]
    dag3_keep = list(dict.fromkeys(base_keep + disease_cols_all))
    dag3_sub = dag3[dag3_keep].copy()

    # Merge rec with DAG3 covariates
    rec = rec.merge(dag3_sub, on=COL_ID_DAG3, how="inner")

    # Choose Oral/Fecal per STUDY_ID
    # Priority Oral: link (if exists) else any record flagged oral
    # Priority Fecal: any record flagged fecal
    def pick_ids(g):
        sid = g[COL_ID_DAG3].iloc[0]
        # oral by link
        oral_link = links.loc[links[COL_ID_DAG3]==sid, "OralID"]
        oral_id = oral_link.iloc[0] if len(oral_link) else pd.NA
        if pd.isna(oral_id):
            oral_id = choose_first(g.loc[g["is_oral_src"]==True, "id6"])
        fecal_id = choose_first(g.loc[g["is_fecal_src"]==True, "id6"])
        return pd.Series({"STUDY_ID": sid,
                          "Oral_sample_ID": str(oral_id).upper() if pd.notna(oral_id) else "NA",
                          "Fecal_sample_ID": str(fecal_id).upper() if pd.notna(fecal_id) else "NA"})

    chosen = rec.groupby(COL_ID_DAG3, group_keys=False).apply(pick_ids).reset_index(drop=True)

    # Covariates per STUDY_ID (single row)
    # Sex 0/1, Age/BMI continuous
    def sex01(x):
        s = str(x).strip().upper()
        if s in {"F","FEMALE","V","W","WOMAN"}: return 1
        if s in {"M","MALE","H","MAN"}: return 0
        return pd.NA

    cov = (
        rec.drop_duplicates(COL_ID_DAG3)
           .rename(columns={COL_ID_DAG3:"STUDY_ID"})
           .loc[:, ["STUDY_ID", COL_AGE, COL_BMI, COL_SEX, COL_SMOKER_NOW,
                    "MED.MEDS.Antibacterials_ATC_J01","META.Antibiotics_3m",
                    "MED.MEDS.PPIs_ATC_A02BC", "MED.MEDS.Stomach_Acid_ATC_A02",
                    "MED.MEDS.Corticosteroids_Oral_ATC_H02",
                    "MED.MEDS.Glucocorticoids_Inhaled_ATC_R03BA",
                    "MED.MEDS.Corticosteroids_Eyedrops_ATC_S03A"]]
           .copy()
    )
    cov["Age"] = pd.to_numeric(cov[COL_AGE], errors="coerce")
    cov["BMI"] = pd.to_numeric(cov[COL_BMI], errors="coerce")
    cov["Sex"] = cov[COL_SEX].map(sex01).astype("Int64")

    def bin01(s):
        return s.map(norm_yesno_scalar).fillna(0).astype(int)

    cov["Smoking"]          = bin01(cov[COL_SMOKER_NOW])
    cov["Antibiotics_3m"]   = (bin01(cov["MED.MEDS.Antibacterials_ATC_J01"]) | bin01(cov["META.Antibiotics_3m"])).astype(int)
    cov["PPI_use"]          = (bin01(cov["MED.MEDS.PPIs_ATC_A02BC"]) | bin01(cov["MED.MEDS.Stomach_Acid_ATC_A02"])).astype(int)
    cov["Steroids_ongoing"] = (bin01(cov["MED.MEDS.Corticosteroids_Oral_ATC_H02"])
                               | bin01(cov["MED.MEDS.Glucocorticoids_Inhaled_ATC_R03BA"])
                               | bin01(cov["MED.MEDS.Corticosteroids_Eyedrops_ATC_S03A"])).astype(int)

    # Immuno_ongoing: OR over any ATC_L (immuno) columns if present
    l_cols = [c for c in rec.columns if re.match(r"MED\.MEDS\..*ATC_L", str(c), flags=re.I)]
    if l_cols:
        tmp = rec.drop_duplicates(COL_ID_DAG3).rename(columns={COL_ID_DAG3:"STUDY_ID"})[["STUDY_ID"]+l_cols].copy()
        for c in l_cols:
            tmp[c] = tmp[c].map(norm_yesno_scalar).fillna(0).astype(int)
        tmp["Immuno_ongoing"] = (tmp[l_cols].sum(axis=1) > 0).astype(int)
        cov = cov.merge(tmp[["STUDY_ID","Immuno_ongoing"]], on="STUDY_ID", how="left")
    else:
        cov["Immuno_ongoing"] = 0

    # Build Disease_labels and strict Healthy_only
    disease_cols = [c for c in dag3_sub.columns if c.startswith("MED.DISEASES.")]
    other_dz_cols = [c for c in disease_cols if c != COL_HEALTHY_FLAG]
    dz_frame = dag3_sub[[COL_ID_DAG3] + disease_cols].drop_duplicates(COL_ID_DAG3).rename(columns={COL_ID_DAG3:"STUDY_ID"})

    def row_to_labels(r):
        labs = []
        for c in other_dz_cols:
            v = norm_yesno_scalar(r.get(c))
            if v == 1:
                labs.append(c.replace("MED.DISEASES.", ""))
        return ";".join(sorted(set(labs))) if labs else "NA"

    dz_labels = dz_frame.copy()
    dz_labels["Disease_labels"] = dz_labels.apply(row_to_labels, axis=1)
    # strict healthy: flag==1 and all other disease columns == 0
    dz_bin = dz_frame[disease_cols].applymap(norm_yesno_scalar).fillna(0).astype(int)
    strict_healthy = (dz_bin[COL_HEALTHY_FLAG].eq(1) & dz_bin[[c for c in disease_cols if c != COL_HEALTHY_FLAG]].sum(axis=1).eq(0)).astype(int)
    dz_labels["Healthy_only"] = strict_healthy.values

    # Assemble final Healthy table
    out = (chosen
           .merge(cov[["STUDY_ID","Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing"]],
                  on="STUDY_ID", how="left")
           .merge(dz_labels[["STUDY_ID","Healthy_only","Disease_labels"]], on="STUDY_ID", how="left")
           )

    # Ensure proper dtypes and NA string for Disease_labels if missing
    out["Disease_labels"] = out["Disease_labels"].fillna("NA")
    out["Healthy_only"]   = out["Healthy_only"].fillna(0).astype(int)

    # Order columns
    final_cols = ["STUDY_ID","Oral_sample_ID","Fecal_sample_ID","Age","Sex","BMI","Smoking",
                  "Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing",
                  "Healthy_only","Disease_labels"]
    out = out[final_cols].drop_duplicates().reset_index(drop=True)

    # Write
    out.to_csv(OUT_HEALTH, index=False)
    print(f"[HEALTHY] wrote: {OUT_HEALTH} (rows={len(out)})")

# =============================================================================
#                                 CROHN
# =============================================================================

def read_row_oriented_excel(meta_path: Path) -> pd.DataFrame:
    """Row-oriented Excel: variable names in column 0, samples across columns."""
    df = pd.read_excel(meta_path, header=None)
    df_t = df.set_index(0).T
    df_t.columns = (
        df_t.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df_t.reset_index(drop=True)

def pick_series(df: pd.DataFrame, names: list[str]) -> pd.Series:
    """Pick first non-null among candidate columns (after header normalization)."""
    d = normalize_headers(df)
    frames = [d[[nm]] for nm in names if nm in d.columns]
    if not frames:
        return pd.Series(pd.NA, index=d.index, dtype="object")
    block = pd.concat(frames, axis=1).bfill(axis=1)
    return block.iloc[:, 0]

def map_yesno(series: pd.Series) -> pd.Series:
    return pd.Series(series.map(norm_yesno_scalar), index=series.index, dtype="Int64")

def or_flags(df: pd.DataFrame, groups) -> pd.Series:
    """OR over groups of yes/no columns (group can be a string or list of names)."""
    norm_groups = [list(g) if isinstance(g, (list,tuple,set)) else [g] for g in (groups or [])]
    if not norm_groups:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")
    s_list = [map_yesno(pick_series(df, nm_list)) for nm_list in norm_groups]
    out = s_list[0].copy()
    for s in s_list[1:]:
        out = (out.fillna(0) | s.fillna(0)).astype("Int64")
    all_na = s_list[0].isna()
    for s in s_list[1:]:
        all_na = all_na & s.isna()
    return out.mask(all_na, pd.NA)

def recode_sex(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    return s.map({"M":0,"MALE":0,"H":0,"F":1,"FEMALE":1,"V":1,"W":1,"WOMAN":1}).astype("Int64")

def recode_age_cont(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def recode_bmi_cont(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def recode_hbi_binary(series: pd.Series, threshold:int=5) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=x.index, dtype="Int64")
    out.loc[x.notna() & (x >= threshold)] = 1
    out.loc[x.notna() & (x < threshold)]  = 0
    return out

def recode_calprotectin_category(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    cat = pd.Series(pd.NA, index=x.index, dtype="Int64")
    cat.loc[x.notna() & (x < 50)]           = 0
    cat.loc[x.notna() & (x >= 50) & (x < 250)]  = 1
    cat.loc[x.notna() & (x >= 250) & (x < 1000)] = 2
    cat.loc[x.notna() & (x >= 1000)]            = 3
    return cat

def recode_montreal_behavior(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    s = s.str.replace(r"\s*\(.*\)\s*$", "", regex=True)
    s = s.replace({"B1":"1","B2":"2","B3":"3"})
    num = s.str.extract(r"^([123])", expand=False)
    return pd.to_numeric(num, errors="coerce").astype("Int64")

def recode_montreal_location(series: pd.Series) -> pd.Series:
    """Normalize Montreal location to 1/2/3 (supports 'L2', words, or first digit in text)."""
    s = series.astype(str).str.strip()
    su = s.str.upper()
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")

    # Direct 1/2/3
    num = pd.to_numeric(s, errors="coerce")
    m = num.isin([1,2,3])
    out.loc[m] = num.loc[m].astype("Int64")

    need = out.isna()
    if need.any():
        dig = su.str.extract(r"([123])", expand=False)
        ok = need & dig.notna()
        out.loc[ok] = pd.to_numeric(dig[ok], errors="coerce").astype("Int64")

    need = out.isna()
    if need.any():
        lcode = su.str.extract(r"L\s*([123])", expand=False)
        ok = need & lcode.notna()
        out.loc[ok] = pd.to_numeric(lcode[ok], errors="coerce").astype("Int64")

    need = out.isna()
    if need.any():
        tmp = pd.Series(pd.NA, index=s.index, dtype="Int64")
        tmp.loc[su.str.contains(r"\bILEO[- ]?COL", regex=True, na=False)] = 3
        tmp.loc[su.str.contains(r"\bCOLONIC\b",    regex=True, na=False)] = 2
        tmp.loc[su.str.contains(r"\bILEAL\b",      regex=True, na=False)] = 1
        out.loc[need] = tmp[need]
    return out

def derive_montreal_location(df: pd.DataFrame) -> pd.Series:
    primary = pick_series(df, ["V1_MontrealLCD","V1 Montreal LCD","V1_MontrealLCD "])
    loc = recode_montreal_location(primary)
    if loc.isna().any():
        l1 = map_yesno(pick_series(df, ["V1_L1","V1_Montreal_L1","Montreal_L1","L1","V1 L1"]))
        l2 = map_yesno(pick_series(df, ["V1_L2","V1_Montreal_L2","Montreal_L2","L2","V1 L2"]))
        l3 = map_yesno(pick_series(df, ["V1_L3","V1_Montreal_L3","Montreal_L3","L3","V1 L3"]))
        out2 = pd.Series(pd.NA, index=df.index, dtype="Int64")
        s1, s2, s3 = l1.fillna(0), l2.fillna(0), l3.fillna(0)
        ssum = s1 + s2 + s3
        out2[(s1==1) & (ssum==1)] = 1
        out2[(s2==1) & (ssum==1)] = 2
        out2[(s3==1) & (ssum==1)] = 3
        out2[ssum >= 2] = 3
        loc = loc.fillna(out2)
    return loc

def build_crohn():
    df = read_row_oriented_excel(IN_CROHN_XLSX)
    df = normalize_headers(df)

    # IDs (expected columns already in your sheet)
    out = pd.DataFrame(index=df.index)
    out["STUDY_ID"]       = pick_series(df, ["STUDY_ID"])
    out["Oral_sample_ID"] = pick_series(df, ["Oral_sample_ID"]).fillna("NA").astype(str).str.upper()
    out["Fecal_sample_ID"]= pick_series(df, ["Fecal_sample_ID"]).fillna("NA").astype(str).str.upper()

    # Continuous Age/BMI + categorical Sex (0/1)
    out["Age"] = recode_age_cont(pick_series(df, ["V1_AgeatFecalSampling"]))
    out["Sex"] = recode_sex(pick_series(df, ["V1_Sex"]))
    out["BMI"] = recode_bmi_cont(pick_series(df, ["V1_BMI"]))

    # Binary flags (0/1)
    out["Smoking"]         = map_yesno(pick_series(df, ["V1_Currentsmoker"])).fillna(0).astype("Int64")
    out["Antibiotics_3m"]  = map_yesno(pick_series(df, ["V1_AntibioticsWithin3months"])).fillna(0).astype("Int64")
    out["PPI_use"]         = map_yesno(pick_series(df, ["V1_PPI_yes_or_no"])).fillna(0).astype("Int64")
    out["Steroids_ongoing"]= or_flags(df, [
        "V1_Prednisone_yes_or_no",
        "V1_Budesonide_yes_or_no",
        "V1_Beclometason_yes_or_no",
        "V1_steroid_inhaler_yes_or_no"
    ]).fillna(0).astype("Int64")

    # Immuno ongoing (biologics + immunosuppressants)
    out["Immuno_ongoing"] = or_flags(df, [
        ["V1_Infliximab_yes_or_no", "V1_Adalimumab_yes_or_no", "V1_Golimumab_yes_or_no", "V1_Certolizumab_yes_or_no"],
        ["V1_Ustekinumab_yes_or_no"],  # add other biologics if present
        ["V1_Methotrexate_yes_or_no", "V1_Azathioprin_yes_or_no", "V1_6_MP_yes_or_no",
         "V1_Tioguanine_yes_or_no", "V1_Tacrolimus_yes_or_no"]
    ]).fillna(0).astype("Int64")

    # Disease duration (years) — keep as ordinal bins or continuous? You asked to keep Age/BMI continuous.
    # We'll keep duration as an ordinal 0/1/2/3 (common in IBD), but you can switch to continuous easily.
    ddur = pd.to_numeric(pick_series(df, [
        "V1_DiseaseDurationYears","V1_DiseaseDuration_Years","V1_DiseaseDuration",
        "V1_DurationYears","V1 Disease Duration Years","DiseaseDurationYears"
    ]), errors="coerce")
    ddur_cat = pd.Series(pd.NA, index=df.index, dtype="Int64")
    ddur_cat.loc[ddur.notna() & (ddur < 1)]  = 0
    ddur_cat.loc[ddur.notna() & (ddur >=1) & (ddur <5)]  = 1
    ddur_cat.loc[ddur.notna() & (ddur >=5) & (ddur <10)] = 2
    ddur_cat.loc[ddur.notna() & (ddur >=10)] = 3
    out["Disease_duration_years"] = ddur_cat

    # Montreal Location (1/2/3), L4, Behavior (1/2/3), Perianal
    out["Montreal_phenotype"] = derive_montreal_location(df)
    out["Montreal_L4"]        = map_yesno(pick_series(df, ["V1_L4inCD"]))
    out["Montreal_behavior"]  = recode_montreal_behavior(pick_series(df, ["V1_MontrealBCD","Montreal_behavior","V1 Montreal BCD"]))
    out["Perianal_disease"]   = map_yesno(pick_series(df, ["V1_PinCD"]))

    # HBI baseline (raw + bin)
    hbi_raw = pd.to_numeric(pick_series(df, ["V1_HarveyBradshawScore","V1_HBI","V1_HarveyBradshawScore "]), errors="coerce")
    out["HBI_baseline_raw"] = hbi_raw
    out["HBI_baseline_bin"] = recode_hbi_binary(hbi_raw, threshold=5)

    # Calprotectin baseline (raw + bin + category)
    # Per your note, use V1_FecalCalprotectine only; assume no NAs in original sheet.
    calp_raw = pd.to_numeric(pick_series(df, ["V1_FecalCalprotectine","V1_Fecalcalprotectine"]), errors="coerce")
    out["Calprotectin_baseline_raw"] = calp_raw
    out["Calprotectin_baseline_bin"] = pd.Series((calp_raw >= 250).astype("Int64"), index=df.index)
    out["Calprotectin_baseline_cat"] = recode_calprotectin_category(calp_raw)

    # Resections, Anti-TNF, 5-ASA, Responder
    out["Any_resection"] = or_flags(df, [
        "V1_Resections_yes_or_no", "V1_ResectionAny_yes_or_no",
        "V1_ResectionIlealAny_yes_or_no", "V1_ResectionColonicAny_yes_or_no"
    ])

    out["Anti_TNF_current"] = or_flags(df, [
        "V1_Infliximab_yes_or_no", "V1_Adalimumab_yes_or_no",
        "V1_Golimumab_yes_or_no", "V1_Certolizumab_yes_or_no"
    ])

    out["5ASA_current"] = map_yesno(pick_series(df, ["V1_5_ASA_yes_or_no"]))

    def map_responder(series: pd.Series) -> pd.Series:
        def _parse(x):
            s = "" if x is None else str(x).strip().lower()
            if s in {"yes","ja","true","present","pos","1"}: return 1
            if s in {"no","nee","false","neg","off","0"}:    return 0
            if re.search(r"\bnon[-\s_]?responder\b", s):     return 0
            if re.search(r"\bresponder\b", s):               return 1
            try:
                f = float(s)
                if f in (0.0,1.0): return int(f)
            except Exception:
                pass
            return pd.NA
        return pd.Series(series.map(_parse), index=series.index, dtype="Int64")

    out["Responder"] = map_responder(pick_series(df, ["Responder_study","Responder","Responder study"]))

    # Clean IDs and remove trailing all-NA row
    for c in ["STUDY_ID","Oral_sample_ID","Fecal_sample_ID"]:
        out[c] = clean_str_na(out[c]).astype(str)
        if c.endswith("_sample_ID"):
            out[c] = out[c].fillna("NA").replace({"nan":"NA"}).str.upper()

    out = drop_all_na_rows(out)

    # Final column order (extended rich)
    extended_cols = [
        "STUDY_ID","Oral_sample_ID","Fecal_sample_ID",
        "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing",
        "Disease_duration_years","Montreal_phenotype","Montreal_L4","Montreal_behavior","Perianal_disease",
        "HBI_baseline_raw","HBI_baseline_bin",
        "Calprotectin_baseline_raw","Calprotectin_baseline_bin","Calprotectin_baseline_cat",
        "Any_resection","Anti_TNF_current","5ASA_current","Responder"
    ]
    out = out.reindex(columns=extended_cols)

    out.to_csv(OUT_CR_EXT, index=False)
    print(f"[CROHN] wrote: {OUT_CR_EXT} (rows={len(out)})")

# =============================================================================
#                                 MAIN
# =============================================================================

if __name__ == "__main__":
    pd.set_option("future.no_silent_downcasting", True)
    build_healthy()
    build_crohn()
