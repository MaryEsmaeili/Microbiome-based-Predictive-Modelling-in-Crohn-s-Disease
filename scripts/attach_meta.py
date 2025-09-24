#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, re, json
from pathlib import Path
import pandas as pd, numpy as np

def parse_args():
    p = argparse.ArgumentParser("Attach meta (Group/Site) to wide matrix; robust auto-detect + optional fills.")
    p.add_argument("--wide", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--id-col-wide", default=None)
    p.add_argument("--id-col-meta", default=None)
    p.add_argument("--group-col", default=None)
    p.add_argument("--site-col", default=None)
    # NEW:
    p.add_argument("--fill-na-group", default=None, help="If set, fill missing Group with this value (e.g. Healthy).")
    p.add_argument("--fill-na-site", default=None, help="If set, fill missing Site with this value (e.g. NA).")
    return p.parse_args()

def load_table(path):
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=None, engine="python")
        return df
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")

def normalize_id(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "").replace("_", "").upper()
    return s

def auto_pick_id_col(df, prefer=None):
    if prefer and prefer in df.columns: return prefer
    for c in ["Sample_ID","Sample","sample","ID","Id","id","sample_id","SampleID","sampleID"]:
        if c in df.columns: return c
    if df.shape[1] >= 2: return df.columns[0]
    return df.reset_index().columns[0]

def derive_group_series(meta, col):
    s = meta[col].copy()
    if pd.api.types.is_numeric_dtype(s):
        return s.map(lambda v: "Crohn" if v == 1 else ("Healthy" if v == 0 else np.nan))
    def map_text(v):
        if pd.isna(v): return np.nan
        t = str(v).strip().lower()
        if t in {"crohn","cd","case","patient","ibd","uc"} or "crohn" in t or "case" in t or "patient" in t:
            return "Crohn"
        if t in {"healthy","control","hc"} or "health" in t or "control" in t:
            return "Healthy"
        return np.nan
    return s.map(map_text)

def auto_pick_group(meta, prefer=None):
    if prefer and prefer in meta.columns:
        return derive_group_series(meta, prefer), prefer
    for c in ["Group","group","Diagnosis","diagnosis","Status","status","disease","Disease"]:
        if c in meta.columns:
            return derive_group_series(meta, c), c
    return pd.Series([np.nan]*len(meta), index=meta.index), None

def auto_pick_site(meta, prefer=None):
    if prefer and prefer in meta.columns:
        return meta[prefer], prefer
    for c in ["Site","site","Tissue","tissue","BodySite","bodysite"]:
        if c in meta.columns:
            return meta[c], c
    return pd.Series([np.nan]*len(meta), index=meta.index), None

def main():
    a = parse_args()
    wide = load_table(a.wide)
    meta = load_table(a.meta)
    if wide.index.name is not None and wide.index.name not in wide.columns:
        wide = wide.reset_index()

    wide_id = auto_pick_id_col(wide, a.id_col_wide)
    meta_id = auto_pick_id_col(meta, a.id_col_meta)
    wide[wide_id] = wide[wide_id].map(normalize_id)
    meta[meta_id] = meta[meta_id].map(normalize_id)

    g_series, g_col = auto_pick_group(meta, a.group_col)
    s_series, s_col = auto_pick_site(meta, a.site_col)
    meta_sub = pd.DataFrame({meta_id: meta[meta_id], "Group": g_series, "Site": s_series})

    merged = pd.merge(wide, meta_sub, left_on=wide_id, right_on=meta_id, how="left", validate="m:1")
    if meta_id in merged.columns: merged = merged.drop(columns=[meta_id])

    # NEW: fill NA if asked
    if a.fill_na_group is not None:
        before = merged["Group"].isna().sum()
        merged["Group"] = merged["Group"].fillna(a.fill_na_group)
        after = merged["Group"].isna().sum()
        print(f"[attach_meta] fill-na-group: {before} → {after}")

    if a.fill_na_site is not None and "Site" in merged.columns:
        before = merged["Site"].isna().sum()
        merged["Site"] = merged["Site"].fillna(a.fill_na_site)
        after = merged["Site"].isna().sum()
        print(f"[attach_meta] fill-na-site: {before} → {after}")

    n_rows = len(merged)
    n_group_filled = merged["Group"].notna().sum()
    site_counts = merged["Site"].fillna("NA").astype(str).str.lower().value_counts().to_dict()

    print(f"[attach_meta] wide_id={wide_id} | meta_id={meta_id}")
    print(f"[attach_meta] rows: {n_rows}, Group filled: {n_group_filled}")
    print(f"[attach_meta] Site unique={len(site_counts)} | {site_counts}")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    side = {
        "wide_id": wide_id, "meta_id": meta_id,
        "group_col_used": g_col, "site_col_used": s_col,
        "n_rows": int(n_rows), "n_group_filled": int(n_group_filled),
        "site_counts": site_counts,
    }
    with open(out.with_suffix(".attach_meta.json"), "w") as f:
        json.dump(side, f, indent=2)
    print(f"[attach_meta] saved: {out}")
    print(f"[attach_meta] meta summary: {out.with_suffix('.attach_meta.json')}")

if __name__ == "__main__":
    main()
