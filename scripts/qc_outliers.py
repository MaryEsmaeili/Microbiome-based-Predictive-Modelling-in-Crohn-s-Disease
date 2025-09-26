#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd


def norm_id(x):
    if pd.isna(x):
        return x
    s = str(x).strip()
    return s.replace(" ", "").replace("\t", "")


def ensure_sample(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c.lower() for c in df.columns]
    df = df.copy()
    if "sample" in cols:
        sc = df.columns[cols.index("sample")]
        df[sc] = df[sc].apply(norm_id)
        return df.rename(columns={sc: "Sample"})
    if "sample_id" in cols:
        sc = df.columns[cols.index("sample_id")]
        df[sc] = df[sc].apply(norm_id)
        return df.rename(columns={sc: "Sample"})
    # اگر ستونی به اسم نمونه نبود، از ایندکس می‌سازیم
    df = df.reset_index(drop=False)
    df = df.rename(columns={"index": "Sample"})
    df["Sample"] = df["Sample"].apply(norm_id)
    return df


def read_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_sample(df)


def compute_l2_norms(df_wide: pd.DataFrame) -> pd.Series:
    """L2 norm per sample over numeric feature columns (excluding 'Sample')."""
    num = df_wide.drop(columns=[c for c in df_wide.columns if c.lower() == "sample"], errors="ignore")
    # فقط ستون‌های عددی
    num = num.select_dtypes(include=[np.number])
    # اگر خالی شد، خطا بده
    if num.shape[1] == 0:
        raise ValueError("No numeric feature columns found to compute norms.")
    arr = num.to_numpy(dtype=float)
    norms = np.linalg.norm(arr, axis=1)
    return pd.Series(norms, index=df_wide.index, name="l2_norm")


def iqr_threshold(series: pd.Series, mult: float = 1.5):
    q1 = float(np.nanpercentile(series, 25))
    q3 = float(np.nanpercentile(series, 75))
    iqr = q3 - q1
    lower = q1 - mult * iqr
    upper = q3 + mult * iqr
    return {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper}


def detect_outliers_l2(df_wide: pd.DataFrame, label: str, mult: float = 1.5):
    norms = compute_l2_norms(df_wide)
    thr = iqr_threshold(norms, mult=mult)
    mask_hi = norms > thr["upper"]
    mask_lo = norms < thr["lower"]  # معمولاً برای L2 کم کاربرد است اما می‌گذاریم
    out_mask = mask_hi | mask_lo
    out = pd.DataFrame({
        "Sample": df_wide["Sample"].values,
        f"norm_{label}": norms.values,
        f"outlier_{label}": out_mask.values
    })
    # برای راحتی، آستانه‌ها را هم اضافه می‌کنیم (تکراری در هر سطر تا گزارش ساده بماند)
    out[f"{label}_q1"] = thr["q1"]
    out[f"{label}_q3"] = thr["q3"]
    out[f"{label}_iqr"] = thr["iqr"]
    out[f"{label}_lower_thr"] = thr["lower"]
    out[f"{label}_upper_thr"] = thr["upper"]
    return out, thr


def attach_meta_for_report(df_wide: pd.DataFrame, meta_path: str | None):
    if not meta_path:
        return None
    try:
        meta = pd.read_csv(meta_path)
    except Exception:
        try:
            meta = pd.read_excel(meta_path)
        except Exception:
            return None
    meta = ensure_sample(meta)
    # فقط ستون‌های مهم را نگه داریم اگر موجودند
    keep = ["Sample", "disease", "Group", "site", "Site"]
    keep = [c for c in keep if c in meta.columns]
    return meta[keep].copy() if keep else meta[["Sample"]].copy()


def write_report(report_path: Path,
                 genus_thr: dict, species_thr: dict,
                 n_g, n_s, excl_list: list[str],
                 meta_g_before: pd.DataFrame | None,
                 meta_g_after: pd.DataFrame | None,
                 meta_s_before: pd.DataFrame | None,
                 meta_s_after: pd.DataFrame | None):
    lines = []
    lines.append("=== QC Outlier Report ===")
    lines.append("")
    lines.append("-> Thresholds (L2 + IQR rule, multiplier=1.5)")
    lines.append(f"GENUS : Q1={genus_thr['q1']:.6f}, Q3={genus_thr['q3']:.6f}, IQR={genus_thr['iqr']:.6f}, "
                 f"Lower={genus_thr['lower']:.6f}, Upper={genus_thr['upper']:.6f}")
    lines.append(f"SPECIES: Q1={species_thr['q1']:.6f}, Q3={species_thr['q3']:.6f}, IQR={species_thr['iqr']:.6f}, "
                 f"Lower={species_thr['lower']:.6f}, Upper={species_thr['upper']:.6f}")
    lines.append("")
    lines.append(f"Original samples: genus={n_g}, species={n_s}")
    lines.append(f"Excluded samples (union): {len(excl_list)}")
    lines.append("List: " + (", ".join(excl_list) if excl_list else "(none)"))
    lines.append("")

    def breakdown(title, meta_df):
        if meta_df is None or "Sample" not in meta_df.columns:
            return []
        out = [title]
        # disease / Group
        if "disease" in meta_df.columns:
            out.append("  disease counts:")
            out.append(str(meta_df["disease"].value_counts(dropna=False)))
        elif "Group" in meta_df.columns:
            out.append("  Group counts:")
            out.append(str(meta_df["Group"].value_counts(dropna=False)))
        # site
        if "site" in meta_df.columns:
            out.append("  site counts:")
            out.append(str(meta_df["site"].value_counts(dropna=False)))
        elif "Site" in meta_df.columns:
            out.append("  Site counts:")
            out.append(str(meta_df["Site"].value_counts(dropna=False)))
        out.append("")
        return out

    lines += breakdown("Before QC (GENUS):", meta_g_before)
    lines += breakdown("After  QC (GENUS):", meta_g_after)
    lines += breakdown("Before QC (SPECIES):", meta_s_before)
    lines += breakdown("After  QC (SPECIES):", meta_s_after)

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="QC outliers using L2+IQR and write norms file.")
    p.add_argument("--in-genus", required=True)
    p.add_argument("--in-species", required=True)
    p.add_argument("--meta", required=False, default=None)
    p.add_argument("--out-genus", required=True)
    p.add_argument("--out-species", required=True)
    p.add_argument("--out-exclusion", required=True)
    p.add_argument("--out-report", required=True)
    p.add_argument("--out-norms", required=False,
                   default="results/qc/qc_norms.csv")
    p.add_argument("--iqr-mult", type=float, default=1.5)
    args = p.parse_args()

    outdir = Path(args.out_report).parent
    outdir.mkdir(parents=True, exist_ok=True)

    g = read_matrix(args.in_genus)
    s = read_matrix(args.in_species)

    # برای گزارش: اگر متا داریم، قبل از QC مرج می‌کنیم
    meta = attach_meta_for_report(g, args.meta)
    meta_g_before = None
    meta_s_before = None
    if meta is not None:
        meta_g_before = g[["Sample"]].merge(meta, on="Sample", how="left")
        meta_s_before = s[["Sample"]].merge(meta, on="Sample", how="left")

    # کشف آوت‌لایر
    genus_out, genus_thr = detect_outliers_l2(g, label="genus", mult=args.iqr_mult)
    species_out, species_thr = detect_outliers_l2(s, label="species", mult=args.iqr_mult)

    # Union
    merged = genus_out.merge(species_out[["Sample", "outlier_species", "norm_species",
                                          "species_q1", "species_q3", "species_iqr",
                                          "species_lower_thr", "species_upper_thr"]],
                             on="Sample", how="outer")
    merged["outlier_any"] = merged["outlier_genus"].fillna(False) | merged["outlier_species"].fillna(False)

    # خروجی لیست حذف
    excluded = merged.loc[merged["outlier_any"], "Sample"].dropna().astype(str).tolist()
    Path(args.out_exclusion).write_text("\n".join(excluded), encoding="utf-8")

    # ماتریس‌های QC (حذف یونین آوت‌لایر)
    keep_g = ~g["Sample"].astype(str).isin(excluded)
    keep_s = ~s["Sample"].astype(str).isin(excluded)
    g_qc = g.loc[keep_g].copy()
    s_qc = s.loc[keep_s].copy()
    g_qc.to_csv(args.out_genus, index=False)
    s_qc.to_csv(args.out_species, index=False)

    # برای گزارش: بعد از QC
    meta_g_after = None
    meta_s_after = None
    if meta is not None:
        meta_g_after = g_qc[["Sample"]].merge(meta, on="Sample", how="left")
        meta_s_after = s_qc[["Sample"]].merge(meta, on="Sample", how="left")

    # گزارش متنی با آستانه‌ها
    write_report(Path(args.out_report),
                 genus_thr, species_thr,
                 len(g), len(s), excluded,
                 meta_g_before, meta_g_after,
                 meta_s_before, meta_s_after)

    # فایل تحلیلی نُرم‌ها
    # ستون Sample اول باشد
    cols = ["Sample",
            "norm_genus", "outlier_genus", "genus_q1", "genus_q3", "genus_iqr", "genus_lower_thr", "genus_upper_thr",
            "norm_species", "outlier_species", "species_q1", "species_q3", "species_iqr", "species_lower_thr", "species_upper_thr",
            "outlier_any"]
    merged = merged[cols]
    Path(args.out_norms).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_norms, index=False)

    # چاپ خلاصه کوتاه برای لاگ
    print(json.dumps({
        "in_genus": args.in_genus,
        "in_species": args.in_species,
        "n_genus": len(g),
        "n_species": len(s),
        "excluded_n": len(excluded),
        "excluded_samples": excluded,
        "genus_thresholds": genus_thr,
        "species_thresholds": species_thr,
        "out_genus": args.out_genus,
        "out_species": args.out_species,
        "out_exclusion": args.out_exclusion,
        "out_report": args.out_report,
        "out_norms": args.out_norms
    }, ensure_ascii=False))

if __name__ == "__main__":
    sys.exit(main())
