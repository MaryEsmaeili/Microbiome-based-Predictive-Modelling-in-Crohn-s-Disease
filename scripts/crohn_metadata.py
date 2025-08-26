#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build two Crohn metadata files from the row-oriented clinical Excel:

  1) crohn_metadata_core.csv
     Harmonized columns for Crohn vs Healthy comparisons.

  2) crohn_metadata_extended.csv
     Disease-specific extensions for Crohn-only analyses.

Inputs (row-oriented Excel: variables in first column):
  e.g., data/Metadata_metagenomics.xlsx

Usage:
  python scripts/crohn_metadata.py \
      --input data/Metadata_metagenomics.xlsx \
      --outdir data/meta \
      --prefix crohn_metadata
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ---------- helpers: IO for row-oriented Excel ----------
def read_excel_row_oriented(meta_path: str) -> pd.DataFrame:
    """Read row-oriented Excel (variable names in first column) and transpose."""
    df = pd.read_excel(meta_path, header=None)
    df_t = df.set_index(0).T
    # normalize column names (strip/space-normalize)
    df_t.columns = (
        df_t.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df_t.reset_index(drop=True)

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return out

# ---------- robust pickers ----------
def pick_series(df: pd.DataFrame, names: list[str]) -> pd.Series:
    """
    Collapse duplicate/variant columns: take the first non-null per row
    across the provided list of candidate column names. Returns <NA> Series
    if none are present.
    """
    df = normalize_headers(df)
    frames = []
    for nm in names:
        if nm in df.columns:
            frames.append(df[[nm]])
    if not frames:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    block = pd.concat(frames, axis=1)
    # stable against future downcasting changes
    block2 = block.bfill(axis=1)
    collapsed = block2.iloc[:, 0]
    if collapsed.dtype == object:
        collapsed = collapsed.infer_objects(copy=False)
    return collapsed

YES_TOKENS = {"yes","ja","y","true","t","present","pos","positive","on","1"}
NO_TOKENS  = {"no","nee","n","false","f","absent","neg","negative","off","0"}

def parse_yesno_cell(x):
    s = str(x).strip().lower() if x is not None else ""
    if s in YES_TOKENS: return 1
    if s in NO_TOKENS:  return 0
    m = re.search(r"\b(yes|ja|true|present|pos|no|nee|false|neg|off)\b", s)
    if m:
        return 1 if m.group(1) in {"yes","ja","true","present","pos"} else 0
    try:
        f = float(s)
        if f == 1.0: return 1
        if f == 0.0: return 0
    except Exception:
        pass
    return pd.NA

def map_yesno(series: pd.Series) -> pd.Series:
    return pd.Series(series.map(parse_yesno_cell), index=series.index, dtype="Int64")

def pick_yesno(df: pd.DataFrame, names: list[str]) -> pd.Series:
    return map_yesno(pick_series(df, names))

def or_flags(df: pd.DataFrame, groups) -> pd.Series:
    """
    OR across multiple flags.
    Accepts either:
      - list[str]: each is a column name (single-candidate per flag), OR
      - list[list[str]]: each inner list is the candidate names for one flag.
    """
    # normalize to list[list[str]]
    norm_groups = []
    for g in groups:
        if isinstance(g, (list, tuple, set)):
            norm_groups.append(list(g))
        else:
            norm_groups.append([g])
    series_list = [pick_yesno(df, nm_list) for nm_list in norm_groups]
    if not series_list:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")
    out = series_list[0].copy()
    for s in series_list[1:]:
        out = (out.fillna(0) | s.fillna(0)).astype("Int64")
    # set NA where all inputs were NA
    all_na = series_list[0].isna()
    for s in series_list[1:]:
        all_na = all_na & s.isna()
    return out.mask(all_na, pd.NA)

# ---------- recoding ----------
def recode_sex(series: pd.Series) -> pd.Series:
    """Male -> 0, Female -> 1, else NA"""
    s = series.astype(str).str.strip().str.upper()
    return s.map({
        "M":0, "MALE":0, "MAN":0, "H":0,
        "F":1, "FEMALE":1, "WOMAN":1, "V":1, "W":1
    }).astype("Int64")

def recode_age(series: pd.Series) -> pd.Series:
    """
    Age category (harmonized 4-bin):
      0: <15
      1: 15–39
      2: 40–59
      3: ≥60
    """
    a = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=a.index, dtype="Int64")
    out.loc[a.notna() & (a < 15)] = 0
    out.loc[a.notna() & (a >= 15) & (a <= 39)] = 1
    out.loc[a.notna() & (a >= 40) & (a <= 59)] = 2
    out.loc[a.notna() & (a >= 60)] = 3
    return out

def recode_bmi(series: pd.Series) -> pd.Series:
    """
    BMI category:
      0: <18.5
      1: 18.5–24.9
      2: 25–29.9
      3: ≥30
    """
    b = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=b.index, dtype="Int64")
    out.loc[b.notna() & (b < 18.5)] = 0
    out.loc[b.notna() & (b >= 18.5) & (b < 25)] = 1
    out.loc[b.notna() & (b >= 25)   & (b < 30)] = 2
    out.loc[b.notna() & (b >= 30)]  = 3
    return out

def recode_hbi_binary(series: pd.Series, threshold:int = 5) -> pd.Series:
    """
    Binary HBI:
      0 if HBI < threshold
      1 if HBI >= threshold
    Default threshold=5 (any activity). Change to 8 for moderate/severe only.
    """
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=x.index, dtype="Int64")
    out.loc[x.notna() & (x >= threshold)] = 1
    out.loc[x.notna() & (x <  threshold)] = 0
    return out

def map_responder(series: pd.Series) -> pd.Series:
    """
    Map responder-like values to 0/1 with nullable Int64:
      1 for yes/ja/true/present/pos OR strings containing 'responder'
      0 for no/nee/false/neg/off    OR strings containing 'non-responder'
      <NA> otherwise
    """
    def _parse(x):
        s = "" if x is None else str(x).strip()
        if s == "":
            return pd.NA

        y = parse_yesno_cell(s)
        if not pd.isna(y):
            return int(y)

        sl = s.lower()
        if re.search(r"\bnon[-\s_]?responder(s)?\b", sl) or re.search(r"\bno[-\s_]?response\b", sl):
            return 0
        if re.search(r"\bresponder(s)?\b", sl) or re.search(r"\bresponse\b", sl):
            return 1

        try:
            f = float(sl)
            if f == 0.0: return 0
            if f == 1.0: return 1
        except Exception:
            pass
        return pd.NA

    return pd.Series(series.map(_parse), index=series.index, dtype="Int64")

def recode_disease_duration(series: pd.Series) -> pd.Series:
    """
    Disease duration bins (IN-PLACE USE):
      0: <1 year
      1: 1–5 years
      2: 5–10 years
      3: ≥10 years
    """
    d = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=d.index, dtype="Int64")
    out.loc[d.notna() & (d < 1)] = 0
    out.loc[d.notna() & (d >= 1) & (d < 5)] = 1
    out.loc[d.notna() & (d >= 5) & (d < 10)] = 2
    out.loc[d.notna() & (d >= 10)] = 3
    return out

def recode_calprotectin_category(series: pd.Series) -> pd.Series:
    """
    Calprotectin categories (IN-PLACE USE), µg/g:
      0: <50
      1: 50–249
      2: 250–999
      3: ≥1000
    """
    x = pd.to_numeric(series, errors="coerce")
    cat = pd.Series(pd.NA, index=x.index, dtype="Int64")
    cat.loc[x.notna() & (x < 50)] = 0
    cat.loc[x.notna() & (x >= 50) & (x < 250)] = 1
    cat.loc[x.notna() & (x >= 250) & (x < 1000)] = 2
    cat.loc[x.notna() & (x >= 1000)] = 3
    return cat

def recode_montreal_behavior_numeric(series: pd.Series) -> pd.Series:
    """
    Normalize Montreal behavior to numeric 1/2/3 (IN-PLACE USE).
    Accepts: 'B1','B2','B3','1','2','3','B2 (stricturing)', etc.
    """
    s = series.astype(str).str.upper().str.strip()
    # drop any parenthetical note
    s = s.str.replace(r"\s*\(.*\)\s*$", "", regex=True)
    # map numerics or B-codes to digits
    s = s.replace({"B1":"1", "B2":"2", "B3":"3"})
    # keep only leading 1/2/3
    num = s.str.extract(r"^([123])", expand=False)
    out = pd.to_numeric(num, errors="coerce").astype("Int64")
    return out

def recode_montreal_location_numeric(series: pd.Series) -> pd.Series:
    """
    Normalize Montreal location L1/L2/L3 to numeric 1/2/3 (IN-PLACE USE).
    Accepts: 'L1','L 2','l3', also common words.
    """
    s = series.astype(str).str.upper()
    # prioritize explicit L-codes
    m = s.str.extract(r"L\s*([123])", expand=False)
    out = pd.to_numeric(m, errors="coerce").astype("Int64")
    # fallback by keywords if no L-code
    mask_na = out.isna()
    s_na = s[mask_na]
    if not s_na.empty:
        tmp = pd.Series(pd.NA, index=s_na.index, dtype="Int64")
        tmp.loc[s_na.str.contains(r"\bILEAL\b", regex=True, na=False)] = 1
        tmp.loc[s_na.str.contains(r"\bCOLONIC\b", regex=True, na=False)] = 2
        tmp.loc[s_na.str.contains(r"\bILEO[- ]?COL", regex=True, na=False)] = 3
        out.loc[mask_na] = tmp
    return out

# ---------- main transform ----------
def build_core_and_extended(df_in: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Columns expected (variants handled):
      IDs: STUDY_ID, Oral_sample_ID, Fecal_sample_ID
      V1_AgeatFecalSampling, V1_Sex, V1_BMI
      V1_Currentsmoker, V1_AntibioticsWithin3months, V1_PPI_yes_or_no
      V1_BristolStoolChart (or variants)
      Steroids at V1 (Prednisone/Budesonide/Beclometason/[inhaler])

      Extended (all in-place recoded to numeric as requested):
        Disease_duration_years -> categorical (0..3)
        Montreal_phenotype     -> 1/2/3
        Montreal_L4            -> 0/1
        Montreal_behavior      -> 1/2/3
        Perianal_disease       -> 0/1
        HBI_baseline           -> binary 0/1 (threshold=5)
        Calprotectin_baseline  -> categorical (0..3)
        Resections/meds        -> 0/1
        Responder              -> 0/1
    """
    df = normalize_headers(df_in)

    # --- CORE ---
    core = pd.DataFrame(index=df.index)
    # IDs
    for c in ["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID"]:
        core[c] = pick_series(df, [c])

    # Demographics (recoded categories)
    core["Age"] = recode_age(pick_series(df, ["V1_AgeatFecalSampling"]))
    core["Sex"] = recode_sex(pick_series(df, ["V1_Sex"]))
    core["BMI"] = recode_bmi(pick_series(df, ["V1_BMI"]))

    # Exposures
    core["Smoking"]        = pick_yesno(df, ["V1_Currentsmoker"])
    core["Antibiotics_3m"] = pick_yesno(df, ["V1_AntibioticsWithin3months"])
    core["PPI_use"]        = pick_yesno(df, ["V1_PPI_yes_or_no"])

    # Steroids (ongoing at V1): OR across available steroid-related flags
    core["Steroids_ongoing"] = or_flags(df, [
        "V1_Prednisone_yes_or_no",
        "V1_Budesonide_yes_or_no",
        "V1_Beclometason_yes_or_no",
        "V1_steroid_inhaler_yes_or_no"
    ])

    # Bristol stool scale (numeric, not recoded)
    core["Bristol_stool_scale"] = pd.to_numeric(
        pick_series(df, ["V1_BristolStoolChart","V1_Bristolstoolchart","V1_BristolStoolChart "]),
        errors="coerce"
    )

    core = core[
        ["STUDY_ID","Oral_sample_ID","Fecal_sample_ID",
         "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing",
         "Bristol_stool_scale"]
    ]

    # --- EXTENDED (start from CORE and add disease-specifics) ---
    ext = core.copy()

    # Montreal & Perianal (now numeric)
    ext["Montreal_phenotype"] = recode_montreal_location_numeric(
        pick_series(df, ["V1_MontrealLCD"])
    )
    ext["Montreal_L4"] = pick_yesno(df, ["V1_L4inCD"])  # 0/1
    ext["Montreal_behavior"] = recode_montreal_behavior_numeric(
        pick_series(df, ["V1_MontrealBCD", "Montreal_behavior", "V1 Montreal BCD"])
    )
    ext["Perianal_disease"] = pick_yesno(df, ["V1_PinCD"])  # 0/1

    # HBI baseline -> binary 0/1 (threshold=5)
    ext["HBI_baseline"] = recode_hbi_binary(
        pick_series(df, ["V1_HarveyBradshawScore","V1_HBI","V1_HarveyBradshawScore "]),
        threshold=5
    )

    # Disease duration -> categorical (in-place)
    dd_series = pick_series(df, [
        "V1_DiseaseDurationYears","V1_DiseaseDuration_Years","V1_DiseaseDuration",
        "V1_DurationYears","V1 Disease Duration Years","DiseaseDurationYears",
    ])
    ext["Disease_duration_years"] = recode_disease_duration(dd_series)

    # Calprotectin baseline -> categorical (in-place)
    cp_series = pick_series(df, [
        "V1_FecalCalprotectine","V1_Fecalcalprotectine","V1_FecalCalprotectin",
        "V1 Fecal Calprotectine","V1_Fecalcal","V1_FecalCal",
    ])
    ext["Calprotectin_baseline"] = recode_calprotectin_category(cp_series)

    # Resections / meds
    ext["Any_resection"] = or_flags(df, [
        "V1_Resections_yes_or_no","V1_ResectionAny_yes_or_no",
        "V1_ResectionIlealAny_yes_or_no","V1_ResectionColonicAny_yes_or_no"
    ])
    ext["Anti_TNF_current"] = or_flags(df, [
        "V1_Infliximab_yes_or_no","V1_Adalimumab_yes_or_no",
        "V1_Golimumab_yes_or_no","V1_Certolizumab_yes_or_no"
    ])
    ext["5ASA_current"] = pick_yesno(df, ["V1_5_ASA_yes_or_no"])

    # Responder (0/1/NA)
    ext["Responder"] = map_responder(
        pick_series(df, ["Responder_study", "Responder", "Responder study"])
    )

    # Column order (no extra derived columns)
    ext_cols = list(core.columns) + [
        "Disease_duration_years",
        "Montreal_phenotype","Montreal_L4","Montreal_behavior","Perianal_disease",
        "HBI_baseline","Calprotectin_baseline",
        "Any_resection","Anti_TNF_current","5ASA_current",
        "Responder",
    ]
    ext = ext[ext_cols]

    # Clean IDs
    for c in ["STUDY_ID","Oral_sample_ID","Fecal_sample_ID"]:
        if c in ext.columns:
            ext[c] = ext[c].astype(str).str.strip().replace({"": pd.NA})
            if c.endswith("_sample_ID"):
                ext[c] = ext[c].str.upper()

    # Drop trailing fully empty rows (Excel tails)
    def drop_all_na_tail(df0):
        mask = ~df0.isna().all(axis=1)
        if not mask.any():
            return df0.iloc[0:0]
        last = mask[mask].index[-1]
        return df0.loc[:last].reset_index(drop=True)

    core = drop_all_na_tail(core)
    ext  = drop_all_na_tail(ext)

    return core, ext

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Build Crohn core & extended metadata CSVs.")
    ap.add_argument("--input",  required=True, help="Row-oriented clinical Excel (variables in first column)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--prefix", default="crohn_metadata", help="Output filename prefix")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = read_excel_row_oriented(args.input)
    core, ext = build_core_and_extended(df_raw)

    core_path = outdir / f"{args.prefix}_core.csv"
    ext_path  = outdir / f"{args.prefix}_extended.csv"

    core.to_csv(core_path, index=False)
    ext.to_csv(ext_path, index=False)

    print(f"[OK] wrote: {core_path}  (rows={len(core)})")
    print(f"[OK] wrote: {ext_path}   (rows={len(ext)})")

if __name__ == "__main__":
    main()
