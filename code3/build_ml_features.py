#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build ML feature matrices from pct_all.csv + meta
- Site-specific CLR matrices (Oral / Fecal)
- Z-scored versions
- Delta features (Fecal - Oral) for Crohn pairs
- Design table (labels + covariates)
"""

import os, argparse
import numpy as np
import pandas as pd

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def clr_transform(pct, pseudocount=1e-6):
    X = (pct.astype(float)/100.0) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)
    return logX.sub(gm, axis=1)

def norm_meta(meta):
    m = meta.copy()
    m.columns = [c.strip() for c in m.columns]
    lower = {c.lower(): c for c in m.columns}
    def grab(keys):
        for k in keys:
            if k in lower: return lower[k]
        return None
    cols = {}
    cols["sample_id"]      = grab(["sample_id","id","sid","sampleid","sample"])
    cols["site"]           = grab(["site","body_site","location"])
    cols["disease"]        = grab(["disease","status","group"])
    cols["ppi_use"]        = grab(["ppi_use","ppi","ppi3m","ppi_current"])
    cols["age"]            = grab(["age"])
    cols["sex"]            = grab(["sex","gender"])
    cols["bmi"]            = grab(["bmi"])
    cols["antibiotics_3m"] = grab(["antibiotics_3m","abx_3m","antibiotics_last3m","abx3m"])
    cols["smoking"]        = grab(["smoking","smoker"])
    # optional outcomes
    cols["treatment_response"] = grab(["treatment_response","response","remission_12w","remission","outcome"])
    keep = {k:v for k,v in cols.items() if v is not None}
    out = m[list(keep.values())].copy()
    out.columns = list(keep.keys())
    # tidy types
    def to01(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().lower()
        if s in {"1","true","yes","y","crohn","cd","case","ibd"}: return 1
        if s in {"0","false","no","n","healthy","control","hc","non-ibd","nonibd"}: return 0
        try:
            v = int(float(s)); 
            if v in (0,1): return v
        except: pass
        return np.nan
    if "disease" in out: out["disease"] = out["disease"].map(to01)
    if "ppi_use" in out: out["ppi_use"] = out["ppi_use"].map(to01)
    if "antibiotics_3m" in out: out["antibiotics_3m"] = out["antibiotics_3m"].map(to01)
    if "smoking" in out: out["smoking"] = out["smoking"].map(to01)
    if "site" in out:
        out["site"] = out["site"].astype(str).str.strip().str.capitalize().replace({"Faecal":"Fecal"})
        out.loc[~out["site"].isin(["Oral","Fecal"]), "site"] = np.nan
    for c in ["age","bmi"]:
        if c in out: out[c] = pd.to_numeric(out[c], errors="coerce")
    if "sex" in out: out["sex"] = out["sex"].astype(str).str.strip().str.capitalize()
    return out

def zscore(df):
    mu = df.mean(axis=0)
    sd = df.std(axis=0, ddof=0).replace(0, np.nan)
    return (df - mu)/sd

def build_delta(clr, meta, pairs_csv):
    if (pairs_csv is None) or (not os.path.exists(pairs_csv)):
        return pd.DataFrame(), pd.DataFrame()
    pairs_raw = pd.read_csv(pairs_csv)
    cols = [c.lower() for c in pairs_raw.columns]
    def pick(*opts):
        return next((pairs_raw.columns[cols.index(o.lower())] for o in opts if o.lower() in cols), None)
    oc = pick("Oral_ID","Oral","OC","oral_id","oral")
    fc = pick("Fecal_ID","Fecal","FC","fecal_id","fecal")
    if oc is None or fc is None:
        return pd.DataFrame(), pd.DataFrame()
    meta_crohn = meta[meta["disease"]==1].copy()
    crohn_ids = set(meta_crohn["sample_id"].astype(str))
    pairs = pairs_raw[[oc,fc]].dropna().astype(str).values.tolist()
    pairs = [(o,f) for (o,f) in pairs if o in crohn_ids and f in crohn_ids and o in clr.columns and f in clr.columns]
    if len(pairs) < 5:
        return pd.DataFrame(), pd.DataFrame()
    # Δ = Fecal - Oral (ستون=taxon، ردیف=pair_index)
    rows = []
    idx  = []
    for i,(o,f) in enumerate(pairs):
        d = clr[f] - clr[o]
        rows.append(d)
        idx.append({"pair_id": i, "oral_id": o, "fecal_id": f})
    Xdelta = pd.DataFrame(rows, index=[r["pair_id"] for r in idx])
    idxdf  = pd.DataFrame(idx)
    return Xdelta, idxdf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct-all", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--rank", required=True, choices=["genus","species"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prevalence-min", type=float, default=0.05,
                    help="حداقل نسبت نمونه‌ها که حضور غیرصفر داشته باشند (برای انتخاب فیچر؛ پیش‌فرض 5%)")
    args = ap.parse_args()

    out_rank = os.path.join(args.outdir, args.rank)
    ensure_dir(out_rank)

    pct = pd.read_csv(args.pct_all, index_col=0)
    pct.columns = pct.columns.astype(str)
    clr = clr_transform(pct, 1e-6)      # taxa × samples
    clr_T = clr.T                       # samples × taxa
    clr_T.index.name = "Sample_ID"

    meta_raw = pd.read_csv(args.meta)
    meta = norm_meta(meta_raw)
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta = meta[meta["sample_id"].isin(clr.columns.astype(str))].copy()

    # Site splits
    for site in ["Oral","Fecal"]:
        ids = meta.loc[meta["site"]==site, "sample_id"].astype(str)
        X = clr_T.loc[ids].copy()
        # prevalence filter (در این مرحله فقط برای z نسخه اعمال می‌کنیم؛ خام را کامل نگه می‌داریم)
        X.to_csv(os.path.join(out_rank, f"X_{site.lower()}_clr.csv"))
        Xz = zscore(X)
        # فیلتر فیچرهای خیلی نایاب
        prev = (X.values!=0).sum(axis=0)/max(1, X.shape[0])
        keep = prev >= args.prevalence_min
        Xz.loc[:, keep].to_csv(os.path.join(out_rank, f"X_{site.lower()}_clr_z.csv"))

    # design table (labels+covars)
    y = meta.copy()
    y = y.rename(columns={"sample_id":"Sample_ID"})
    y.to_csv(os.path.join(out_rank, "y_design.csv"), index=False)

    # feature stats (برای گزارش)
    pres = (clr_T.values!=0).sum(axis=0)/max(1, clr_T.shape[0])
    var  = clr_T.var(axis=0).values
    stats = pd.DataFrame({"taxon": clr_T.columns, "prevalence": pres, "variance": var})
    stats.to_csv(os.path.join(out_rank, "feature_stats.csv"), index=False)

    # Δ-features for Crohn pairs
    Xdelta, idxdf = build_delta(clr, meta, args.pairs)
    if not Xdelta.empty:
        Xdelta.to_csv(os.path.join(out_rank, "X_delta_crohn.csv"))
        idxdf.to_csv(os.path.join(out_rank, "delta_index.csv"), index=False)

    print("[INFO] ML features built →", out_rank)

if __name__ == "__main__":
    main()
