#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Crohn core & extended metadata CSVs from a row-oriented Excel.

Key fixes:
- Montreal_phenotype is filled from V1_MontrealLCD (1/2/3); also supports "L2",
  "ileocolonic", and strings like "2 or 7" by taking the first valid digit.
- Bristol_stool_scale takes the first integer found; falls back to 0 if none.
- Oral_sample_ID empty -> "NA" (uppercase).
- Drop trailing garbage/blank rows (no valid STUDY_ID).
- Add Immuno_ongoing to CORE (OR over relevant immuno/biologics).
"""

import argparse
from pathlib import Path
import re
import pandas as pd


# -------------------- small utilities --------------------

def read_excel_row_oriented(meta_path: str) -> pd.DataFrame:
    """Read a row-oriented Excel where variable names are in column 0."""
    df = pd.read_excel(meta_path, header=None)
    df_t = df.set_index(0).T
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


def clean_str_na(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NAN": pd.NA})
    )


def pick_series(df: pd.DataFrame, names: list[str]) -> pd.Series:
    """Pick the first non-null among candidate columns (after header normalization)."""
    df = normalize_headers(df)
    frames = [df[[nm]] for nm in names if nm in df.columns]
    if not frames:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    block = pd.concat(frames, axis=1).bfill(axis=1)
    return block.iloc[:, 0]


YES_TOKENS = {"yes", "ja", "y", "true", "t", "present", "pos", "positive", "on", "1"}
NO_TOKENS  = {"no", "nee", "n", "false", "f", "absent", "neg", "negative", "off", "0"}


def parse_yesno_cell(x):
    s = str(x).strip().lower() if x is not None else ""
    if s in YES_TOKENS:
        return 1
    if s in NO_TOKENS:
        return 0
    m = re.search(r"\b(yes|ja|true|present|pos|no|nee|false|neg|off)\b", s)
    if m:
        return 1 if m.group(1) in {"yes", "ja", "true", "present", "pos"} else 0
    try:
        f = float(s)
        if f == 1.0:
            return 1
        if f == 0.0:
            return 0
    except Exception:
        pass
    return pd.NA


def map_yesno(series: pd.Series) -> pd.Series:
    return pd.Series(series.map(parse_yesno_cell), index=series.index, dtype="Int64")


def pick_yesno(df: pd.DataFrame, names: list[str]) -> pd.Series:
    return map_yesno(pick_series(df, names))


def or_flags(df: pd.DataFrame, groups) -> pd.Series:
    """
    OR over groups of yes/no columns. Each group can be a single name or a list of names.
    Missing columns are tolerated.
    """
    norm_groups = [list(g) if isinstance(g, (list, tuple, set)) else [g] for g in groups]
    series_list = [pick_yesno(df, nm_list) for nm_list in norm_groups]
    if not series_list:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")
    out = series_list[0].copy()
    for s in series_list[1:]:
        out = (out.fillna(0) | s.fillna(0)).astype("Int64")
    all_na = series_list[0].isna()
    for s in series_list[1:]:
        all_na = all_na & s.isna()
    return out.mask(all_na, pd.NA)


def recode_sex(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    return s.map({
        "M": 0, "MALE": 0, "MAN": 0, "H": 0,
        "F": 1, "FEMALE": 1, "WOMAN": 1, "V": 1, "W": 1
    }).astype("Int64")


def recode_age(series: pd.Series) -> pd.Series:
    a = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=a.index, dtype="Int64")
    out.loc[a.notna() & (a < 15)] = 0
    out.loc[a.notna() & (a >= 15) & (a <= 39)] = 1
    out.loc[a.notna() & (a >= 40) & (a <= 59)] = 2
    out.loc[a.notna() & (a >= 60)] = 3
    return out


def recode_bmi(series: pd.Series) -> pd.Series:
    b = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=b.index, dtype="Int64")
    out.loc[b.notna() & (b < 18.5)] = 0
    out.loc[b.notna() & (b >= 18.5) & (b < 25)] = 1
    out.loc[b.notna() & (b >= 25)   & (b < 30)] = 2
    out.loc[b.notna() & (b >= 30)]  = 3
    return out


def recode_hbi_binary(series: pd.Series, threshold: int = 5) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=x.index, dtype="Int64")
    out.loc[x.notna() & (x >= threshold)] = 1
    out.loc[x.notna() & (x < threshold)] = 0
    return out


def map_responder(series: pd.Series) -> pd.Series:
    def _parse(x):
        s = "" if x is None else str(x).strip()
        if s == "":
            return pd.NA
        sl = s.lower()
        if sl in {"yes", "ja", "true", "present", "pos", "1"}:
            return 1
        if sl in {"no", "nee", "false", "neg", "off", "0"}:
            return 0
        if re.search(r"\bnon[-\s_]?responder", sl):
            return 0
        if re.search(r"\bresponder\b", sl):
            return 1
        try:
            f = float(sl)
            if f == 0.0:
                return 0
            if f == 1.0:
                return 1
        except Exception:
            pass
        return pd.NA
    return pd.Series(series.map(_parse), index=series.index, dtype="Int64")


def recode_disease_duration(series: pd.Series) -> pd.Series:
    d = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=d.index, dtype="Int64")
    out.loc[d.notna() & (d < 1)] = 0
    out.loc[d.notna() & (d >= 1) & (d < 5)] = 1
    out.loc[d.notna() & (d >= 5) & (d < 10)] = 2
    out.loc[d.notna() & (d >= 10)] = 3
    return out


def recode_calprotectin_category(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    cat = pd.Series(pd.NA, index=x.index, dtype="Int64")
    cat.loc[x.notna() & (x < 50)] = 0
    cat.loc[x.notna() & (x >= 50) & (x < 250)] = 1
    cat.loc[x.notna() & (x >= 250) & (x < 1000)] = 2
    cat.loc[x.notna() & (x >= 1000)] = 3
    return cat


def recode_montreal_behavior_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    s = s.str.replace(r"\s*\(.*\)\s*$", "", regex=True)
    s = s.replace({"B1": "1", "B2": "2", "B3": "3"})
    num = s.str.extract(r"^([123])", expand=False)
    return pd.to_numeric(num, errors="coerce").astype("Int64")


def recode_montreal_location_numeric(series: pd.Series) -> pd.Series:
    """
    Normalize Montreal location to 1/2/3.

    Works with:
    - bare numbers: "1", "2", "3"
    - first digit in strings: "2 or 7" -> 2
    - L-codes: "L2" -> 2
    - words: "ileal", "colonic", "ileo-colonic"
    """
    s = series.astype(str).str.strip()
    su = s.str.upper()

    out = pd.Series(pd.NA, index=s.index, dtype="Int64")

    # 1) bare numeric (1/2/3)
    num_direct = pd.to_numeric(s, errors="coerce")
    mask = num_direct.isin([1, 2, 3])
    out.loc[mask] = num_direct[mask].astype("Int64")

    # 2) first digit in text (e.g., "2 or 7")
    need = out.isna()
    if need.any():
        first_digit = su.str.extract(r"([123])", expand=False)
        ok = need & first_digit.notna()
        out.loc[ok] = pd.to_numeric(first_digit[ok], errors="coerce").astype("Int64")

    # 3) L1/L2/L3
    need = out.isna()
    if need.any():
        lcode = su.str.extract(r"L\s*([123])", expand=False)
        ok = need & lcode.notna()
        out.loc[ok] = pd.to_numeric(lcode[ok], errors="coerce").astype("Int64")

    # 4) words
    need = out.isna()
    if need.any():
        tmp = pd.Series(pd.NA, index=s.index, dtype="Int64")
        tmp.loc[su.str.contains(r"\bILEO[- ]?COL", regex=True, na=False)] = 3
        tmp.loc[su.str.contains(r"\bCOLONIC\b",    regex=True, na=False)] = 2
        tmp.loc[su.str.contains(r"\bILEAL\b",      regex=True, na=False)] = 1
        out.loc[need] = tmp[need]

    return out


def derive_montreal_location(df: pd.DataFrame) -> pd.Series:
    """Primary from V1_MontrealLCD; fallback to L1/L2/L3 flags if present."""
    primary = pick_series(df, ["V1_MontrealLCD", "V1 Montreal LCD", "V1_MontrealLCD "])
    loc = recode_montreal_location_numeric(primary)

    need = loc.isna()
    if need.any():
        l1 = pick_yesno(df, ["V1_L1", "V1_Montreal_L1", "Montreal_L1", "L1", "V1 L1"])
        l2 = pick_yesno(df, ["V1_L2", "V1_Montreal_L2", "Montreal_L2", "L2", "V1 L2"])
        l3 = pick_yesno(df, ["V1_L3", "V1_Montreal_L3", "Montreal_L3", "L3", "V1 L3"])
        out2 = pd.Series(pd.NA, index=df.index, dtype="Int64")
        s1, s2, s3 = l1.fillna(0), l2.fillna(0), l3.fillna(0)
        ssum = s1 + s2 + s3
        out2[(s1 == 1) & (ssum == 1)] = 1
        out2[(s2 == 1) & (ssum == 1)] = 2
        out2[(s3 == 1) & (ssum == 1)] = 3
        out2[ssum >= 2] = 3  # mixed -> L3
        loc = loc.fillna(out2)

    return loc


def extract_first_int(val, default=0) -> int:
    """Return the first integer found in the string, or `default` if none."""
    m = re.search(r"\d+", str(val))
    return int(m.group(0)) if m else default


def drop_garbage_rows(df0: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where STUDY_ID is empty or does not start with START,
    and drop rows that are completely NA (ignoring Bristol which may be 0).
    """
    df = df0.copy()
    if "STUDY_ID" in df.columns:
        sid = clean_str_na(df["STUDY_ID"])
        keep = sid.notna() & sid.str.match(r"^START", na=False)
        df = df.loc[keep].copy()

    ignore = [c for c in ["Bristol_stool_scale"] if c in df.columns]
    base = df.drop(columns=ignore, errors="ignore")
    df = df.loc[~base.isna().all(axis=1)].reset_index(drop=True)
    return df


# -------------------- builders --------------------

def build_core_and_extended(df_in: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_headers(df_in)

    # ---------- CORE ----------
    core = pd.DataFrame(index=df.index)
    for c in ["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID"]:
        core[c] = pick_series(df, [c])

    core["Age"] = recode_age(pick_series(df, ["V1_AgeatFecalSampling"]))
    core["Sex"] = recode_sex(pick_series(df, ["V1_Sex"]))
    core["BMI"] = recode_bmi(pick_series(df, ["V1_BMI"]))

    core["Smoking"]         = pick_yesno(df, ["V1_Currentsmoker"])
    core["Antibiotics_3m"]  = pick_yesno(df, ["V1_AntibioticsWithin3months"])
    core["PPI_use"]         = pick_yesno(df, ["V1_PPI_yes_or_no"])
    core["Steroids_ongoing"] = or_flags(df, [
        "V1_Prednisone_yes_or_no",
        "V1_Budesonide_yes_or_no",
        "V1_Beclometason_yes_or_no",
        "V1_steroid_inhaler_yes_or_no",
    ])

    # Immuno_ongoing = OR over immuno/biologics (columns may or may not exist; that's OK)
    core["Immuno_ongoing"] = or_flags(df, [
        ["V1_Infliximab_yes_or_no", "V1_Adalimumab_yes_or_no",
         "V1_Golimumab_yes_or_no", "V1_Certolizumab_yes_or_no"],
        ["V1_Ustekinumab_yes_or_no"],  # Vedolizumab not in your header list; leave only existent ones
        ["V1_Methotrexate_yes_or_no", "V1_Azathioprin_yes_or_no", "V1_6_MP_yes_or_no",
         "V1_Mercaptopurine_yes_or_no", "V1_Tioguanine_yes_or_no",
         "V1_Tacrolimus_yes_or_no", "V1_Immunosuppressants_yes_or_no"]
    ]).fillna(0).astype("Int64")

    # Bristol: first number; default 0
    br_raw = pick_series(df, ["V1_BristolStoolChart", "V1_Bristolstoolchart", "V1_BristolStoolChart "])
    core["Bristol_stool_scale"] = br_raw.apply(lambda v: extract_first_int(v, default=0)).astype("Int64")

    core = core[[
        "STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID", "Age", "Sex", "BMI",
        "Smoking", "Antibiotics_3m", "PPI_use", "Steroids_ongoing", "Bristol_stool_scale",
        "Immuno_ongoing"
    ]]

    # ---------- EXTENDED ----------
    ext = core.drop(columns=["Immuno_ongoing"]).copy()

    ext["Disease_duration_years"] = recode_disease_duration(
        pick_series(df, [
            "V1_DiseaseDurationYears", "V1_DiseaseDuration_Years", "V1_DiseaseDuration",
            "V1_DurationYears", "V1 Disease Duration Years", "DiseaseDurationYears"
        ])
    )
    ext["Montreal_phenotype"] = derive_montreal_location(df)
    ext["Montreal_L4"]        = pick_yesno(df, ["V1_L4inCD"])
    ext["Montreal_behavior"]  = recode_montreal_behavior_numeric(
        pick_series(df, ["V1_MontrealBCD", "Montreal_behavior", "V1 Montreal BCD"])
    )
    ext["Perianal_disease"]   = pick_yesno(df, ["V1_PinCD"])
    ext["HBI_baseline"]       = recode_hbi_binary(
        pick_series(df, ["V1_HarveyBradshawScore", "V1_HBI", "V1_HarveyBradshawScore "]),
        threshold=5
    )
    ext["Calprotectin_baseline"] = recode_calprotectin_category(
        pick_series(df, [
            "V1_FecalCalprotectine", "V1_Fecalcalprotectine", "V1_FecalCalprotectin",
            "V1 Fecal Calprotectine", "V1_Fecalcal", "V1_FecalCal"
        ])
    )
    ext["Any_resection"]     = or_flags(df, [
        "V1_Resections_yes_or_no", "V1_ResectionAny_yes_or_no",
        "V1_ResectionIlealAny_yes_or_no", "V1_ResectionColonicAny_yes_or_no"
    ])
    ext["Anti_TNF_current"]  = or_flags(df, [
        "V1_Infliximab_yes_or_no", "V1_Adalimumab_yes_or_no",
        "V1_Golimumab_yes_or_no", "V1_Certolizumab_yes_or_no"
    ])
    ext["5ASA_current"]      = pick_yesno(df, ["V1_5_ASA_yes_or_no"])
    ext["Responder"]         = map_responder(pick_series(df, ["Responder_study", "Responder", "Responder study"]))

    ext_cols = list(core.columns[:-1]) + [
        "Disease_duration_years", "Montreal_phenotype", "Montreal_L4", "Montreal_behavior",
        "Perianal_disease", "HBI_baseline", "Calprotectin_baseline",
        "Any_resection", "Anti_TNF_current", "5ASA_current", "Responder"
    ]
    ext = ext[ext_cols]

    # ---------- ID cleanup & dropping garbage ----------
    for c in ["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID"]:
        core[c] = clean_str_na(core[c])
        ext[c]  = clean_str_na(ext[c])
        if c.endswith("_sample_ID"):
            core[c] = core[c].str.upper()
            ext[c]  = ext[c].str.upper()

    # fill empty Oral_sample_ID with "NA"
    core["Oral_sample_ID"] = core["Oral_sample_ID"].fillna("NA")
    ext["Oral_sample_ID"]  = ext["Oral_sample_ID"].fillna("NA")

    # Drop trailing/garbage rows (incl. the blank last line you saw)
    core = drop_garbage_rows(core)
    ext  = drop_garbage_rows(ext)

    return core, ext


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Build Crohn core & extended metadata CSVs.")
    ap.add_argument("--input",  required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="crohn_metadata")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = read_excel_row_oriented(args.input)
    core, ext = build_core_and_extended(df_raw)

    (outdir / f"{args.prefix}_core.csv").write_text(core.to_csv(index=False))
    (outdir / f"{args.prefix}_extended.csv").write_text(ext.to_csv(index=False))

    print(f"[OK] wrote: {outdir / (args.prefix + '_core.csv')}  (rows={len(core)})")
    print(f"[OK] wrote: {outdir / (args.prefix + '_extended.csv')}   (rows={len(ext)})")


if __name__ == "__main__":
    main()
