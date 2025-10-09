#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Differential abundance with covariates on CLR features (site-specific).

Inputs:
  --features : results/taxa_compare/{rank}/ml_{rank}_wide_clr.csv
               columns: Sample, Group, Site, <taxa...> (numeric CLR)
  --meta     : data/meta/model_table_pooled.csv
               id col: one of [Sample, Sample_ID, sample_id, ID]
               disease (optional): 0/1 or text
               covariates (optional): Age, Sex, BMI, Smoking, Antibiotics_3m,
                                      PPI_use, Steroids_ongoing, Immuno_ongoing
  --exclude  : optional exclusion list (one sample ID per line)
  --site     : oral|fecal   (case-insensitive, exact equality)
  --rank     : genus|species
  --outdir   : results/da

Outputs (always created, even if empty):
  results/da/da_{site}_with_covariates_{rank}.csv            (logistic)
  results/da/da_{site}_with_covariates_{rank}_linear.csv     (linear)
  results/da/volcano_{site}_with_covariates_{rank}.png
  results/da/volcano_{site}_with_covariates_{rank}_linear.png
  results/da/da_runmeta_{site}_{rank}.json                   (debug info)
"""

import os, argparse, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------- utilities -----------------------------

ALLOWED_COVARS = [
    "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use",
    "Steroids_ongoing","Immuno_ongoing"
]

SITE_SYNONYMS = {
    "oral":  {"oral","saliva","mouth","buccal"},
    "fecal": {"fecal","faecal","stool","faeces","feces"},
}

def to_lower_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def ensure_binary_disease(series: pd.Series) -> pd.Series:
    """Map disease labels to 0/1. Accepts 0/1 or strings. Always returns a Pandas Series."""
    z = series.astype(str).str.strip().str.lower()
    case_tokens = {"crohn","cd","case","ibd","1"}
    ctrl_tokens = {"healthy","control","ctr","ctl","hc","non-ibd","nonibd","nibd","0"}
    y_txt = np.where(z.isin(case_tokens), 1,
             np.where(z.isin(ctrl_tokens), 0, np.nan))

    y_num = pd.to_numeric(series, errors="coerce")
    # اگر خود ستون 0/1 بود همان را نگه می‌داریم، وگرنه از نگاشت متنی استفاده می‌کنیم
    y = np.where(y_num.isin([0,1]), y_num, y_txt)

    # مهم: حتماً به Series برگردان با همان index
    return pd.Series(pd.to_numeric(y, errors="coerce"), index=series.index, dtype="float64")

def as01(col: pd.Series) -> pd.Series:
    """Convert common yes/no, male/female encodings to 0/1 where possible."""
    m = {"male":1,"m":1,"female":0,"f":0,"yes":1,"y":1,"true":1,"on":1,
         "no":0,"n":0,"false":0,"off":0,"current":1,"former":1,"never":0}
    if col.dtype == object:
        c = to_lower_str(col).map(lambda x: m.get(x, np.nan))
        # if mapping succeeded for a reasonable fraction, use it
        if pd.isna(c).mean() <= 0.6:
            return c
    return pd.to_numeric(col, errors="coerce")

def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu, sd = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:  # constant → return zeros
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float); n = p.size
    order = np.argsort(p); q = np.empty(n, float); prev = 1.0
    for i, idx in enumerate(order[::-1], 1):
        rank = n - i + 1
        val = min(prev, p[idx] * n / rank)
        q[idx] = val; prev = val
    return q

def volcano(df, xcol, qcol, out_png, title, qthr=0.05):
    plt.figure(figsize=(11,6))
    if df is None or df.empty:
        plt.title(title)
        plt.text(0.5,0.5,"No data", ha="center")
        plt.axis("off")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        return
    x = pd.to_numeric(df[xcol], errors="coerce").values
    q = pd.to_numeric(df[qcol], errors="coerce").clip(lower=1e-300).values
    y = -np.log10(q)
    sig = q <= qthr
    plt.scatter(x[~sig], y[~sig], s=26, alpha=.35, label="non-sig")
    plt.scatter(x[sig & (x>0)], y[sig & (x>0)], s=30, alpha=.95, label="sig up")
    plt.scatter(x[sig & (x<0)], y[sig & (x<0)], s=30, alpha=.95, label="sig down")
    plt.axhline(-np.log10(qthr), ls="--", color="0.5"); plt.axvline(0, ls="--", color="0.5")
    plt.xlabel(xcol); plt.ylabel("-log10(q)"); plt.title(title); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out_png, dpi=320); plt.close()


def read_exclusions(path):
    if not path or not os.path.exists(path): return set()
    with open(path, "r") as f:
        return {ln.strip().split()[0] for ln in f if ln.strip()}

def pick_covariates(cols):
    return [c for c in ALLOWED_COVARS if c in cols]

def taxa_columns(df: pd.DataFrame) -> list[str]:
    """Everything numeric except known meta columns becomes a taxon."""
    known = {"Sample","Group","Site","disease"} | set(ALLOWED_COVARS)
    cands = [c for c in df.columns if c not in known]
    # keep strictly numeric columns
    keep = []
    for c in cands:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            keep.append(c)
    return keep

def keep_prevalent(col: pd.Series, min_nonzero_frac=0.10) -> bool:
    arr = pd.to_numeric(col, errors="coerce").fillna(0).values
    return (np.count_nonzero(arr) / max(len(arr),1)) >= min_nonzero_frac and np.nanstd(arr) > 0

def fit_logistic(X: np.ndarray, y: np.ndarray):
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    try:
        return Logit(y, X).fit(disp=0, maxiter=200)
    except PerfectSeparationError:
        try:
            return Logit(y, X).fit_regularized(disp=0)
        except Exception:
            return None
    except Exception:
        try:
            return Logit(y, X).fit_regularized(disp=0)
        except Exception:
            return None

def fit_linear(X: np.ndarray, y: np.ndarray):
    try:
        return OLS(y, X).fit()
    except Exception:
        return None

def site_mask(series: pd.Series, site: str) -> pd.Series:
    vals = to_lower_str(series)
    allowed = SITE_SYNONYMS[site]
    return vals.isin(allowed) | (vals == site)

# ----------------------------- main runner -----------------------------

def run_site(df_all: pd.DataFrame, site: str, rank: str, outdir: str, debug: dict):
    base = os.path.join(outdir, f"da_{site}_with_covariates_{rank}")

    # filter by site (exact, case-insensitive, with synonyms)
    if "Site" not in df_all.columns:
        sel = pd.Series([True]*len(df_all), index=df_all.index)
    else:
        sel = site_mask(df_all["Site"], site)
    D = df_all.loc[sel].copy()

    # disease vector
    if "disease" in D.columns:
        y01 = ensure_binary_disease(D["disease"])
    elif "Group" in D.columns:
        y01 = ensure_binary_disease(D["Group"])
    else:
        y01 = pd.Series(np.nan, index=D.index)

    debug.update({
        "site": site,
        "rank": rank,
        "n_rows_after_site_filter": int(D.shape[0]),
        "disease_counts_mapped": {
            "0": int((y01==0).sum()),
            "1": int((y01==1).sum()),
            "nan": int(pd.isna(y01).sum())
        }
    })

    # covariates (only allowed set, if present)
    cov_names = pick_covariates(D.columns)
    C = pd.DataFrame(index=D.index)
    for c in cov_names:
        v = as01(D[c])
        # standardize non-binary (nunique>2)
        if pd.api.types.is_numeric_dtype(v) and pd.Series(v).nunique(dropna=True) > 2:
            v = zscore(v)
        # impute with median (safe for statsmodels)
        m = pd.to_numeric(v, errors="coerce")
        med = np.nanmedian(m) if np.isfinite(np.nanmedian(m)) else 0.0
        C[c] = m.fillna(med)

    debug["covariates_used"] = cov_names

    # taxa list after numeric check + prevalence
    taxa_raw = taxa_columns(D)
    taxa = [t for t in taxa_raw if keep_prevalent(D[t])]
    debug["n_taxa_after_filter"] = len(taxa)

    n_pos = int((y01==1).sum()); n_neg = int((y01==0).sum())

    # ---- Early exit: not enough data to fit → still write ALL outputs properly ----
    if (n_pos == 0) or (n_neg == 0) or (len(taxa) == 0) or (D.shape[0] < 20):
        empty = pd.DataFrame(columns=["taxon","effect","se","p","q","n"])
        empty.to_csv(base + ".csv", index=False)
        empty.to_csv(base + "_linear.csv", index=False)
        volcano(empty, "effect", "q", base.replace("da_","volcano_") + ".png",
                f"{site.capitalize()} (logistic, {rank})")
        volcano(empty, "effect", "q", base.replace("da_","volcano_") + "_linear.png",
                f"{site.capitalize()} (linear, {rank})")
        with open(os.path.join(outdir, f"da_runmeta_{site}_{rank}.json"), "w") as f:
            json.dump(debug, f, indent=2)
        return

    # ---------------- Logistic: disease ~ z(taxon) + covariates ----------------
    rows = []
    for t in taxa:
        tx = zscore(D[t])
        X = pd.concat([C, tx.rename("taxon")], axis=1)
        X = add_constant(X, has_constant="add")
        valid = np.isfinite(y01.to_numpy()) & np.all(np.isfinite(X.values), axis=1)
        if valid.sum() < 20: 
            continue
        model = fit_logistic(X.values[valid], y01.to_numpy()[valid])

        if model is None or "taxon" not in X.columns:
            continue
        t_idx = list(X.columns).index("taxon")
        try:
            coef = float(model.params[t_idx]); se = float(model.bse[t_idx]); p = float(model.pvalues[t_idx])
            rows.append({"taxon":t, "effect":coef, "se":se, "p":p, "n":int(valid.sum())})
        except Exception:
            continue
    logi = pd.DataFrame(rows)
    if not logi.empty:
        logi["q"] = bh_fdr(logi["p"].values)
        logi = logi.sort_values("q")
    else:
        logi = pd.DataFrame(columns=["taxon","effect","se","p","q","n"])
    logi.to_csv(base + ".csv", index=False)
    volcano(logi, "effect", "q", base.replace("da_","volcano_") + ".png",
            f"{site.capitalize()} (logistic, {rank})")

    # ---------------- Linear: taxon ~ disease01 + covariates -------------------
    rows = []
    dvec = pd.Series(y01.to_numpy(), index=D.index, name="disease01")
    for t in taxa:
        ty = pd.to_numeric(D[t], errors="coerce")
        X = pd.concat([C, dvec], axis=1)
        X = add_constant(X, has_constant="add")
        valid = np.isfinite(ty.values) & np.all(np.isfinite(X.values), axis=1)
        if valid.sum() < 20:
            continue
        model = fit_linear(X.values[valid], ty.values[valid])
        if model is None:
            continue
        di = list(X.columns).index("disease01")
        try:
            coef = float(model.params[di]); se = float(model.bse[di]); p = float(model.pvalues[di])
            rows.append({"taxon":t, "effect":coef, "se":se, "p":p, "n":int(valid.sum())})
        except Exception:
            continue
    lin = pd.DataFrame(rows)
    if not lin.empty:
        lin["q"] = bh_fdr(lin["p"].values)
        lin = lin.sort_values("q")
    else:
        lin = pd.DataFrame(columns=["taxon","effect","se","p","q","n"])
    lin.to_csv(base + "_linear.csv", index=False)
    volcano(lin, "effect", "q", base.replace("da_","volcano_") + "_linear.png",
            f"{site.capitalize()} (linear, {rank})")

    # ---------------- Debug meta ----------------
    with open(os.path.join(outdir, f"da_runmeta_{site}_{rank}.json"), "w") as f:
        json.dump(debug, f, indent=2)

# ----------------------------- CLI -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--meta",     required=True)
    ap.add_argument("--exclude",  default=None)
    ap.add_argument("--site",     choices=["oral","fecal"], required=True)
    ap.add_argument("--rank",     choices=["genus","species"], required=True)
    ap.add_argument("--outdir",   required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    X = pd.read_csv(args.features)             # Sample, Group, Site, taxa...
    M = pd.read_csv(args.meta)                 # ids + covariates
    excl = read_exclusions(args.exclude)
    if excl:
        X = X[~X["Sample"].astype(str).isin(excl)].copy()

    # ----- Join strategy -----
    # If meta has 'Sample' and overlaps with X['Sample'] → use inner join.
    # Else if no overlap, proceed WITHOUT covariates (we still can run DA).
    join_used = "none"
    if "Sample" in M.columns and X["Sample"].astype(str).isin(M["Sample"].astype(str)).any():
        df = X.merge(M, on="Sample", how="inner", suffixes=("","_meta"))
        join_used = "Sample"
    else:
        # Keep X; copy only a disease column from meta if clearly 0/1 and aligned by index? Not safe.
        # So: features + (no covariates from meta). If M has disease overall, it likely is on different cohort.
        df = X.copy()

    debug = {
        "features": args.features,
        "meta": args.meta,
        "exclude_used": bool(excl),
        "join_key": join_used,
        "n_rows_features": int(X.shape[0]),
        "n_rows_meta": int(M.shape[0]),
        "n_rows_after_merge": int(df.shape[0]),
    }

    # If meta had a 'disease' column but we didn't join, we cannot safely align it → ignore.
    # If we DID join and disease exists only in meta side, rename into main:
    if "disease" not in df.columns and "disease_meta" in df.columns:
        df = df.rename(columns={"disease_meta":"disease"})

    # Ensure disease exists (fallback to Group)
    if "disease" not in df.columns and "Group" in df.columns:
        df["disease"] = df["Group"]

    # Run for the requested site/rank
    run_site(df, args.site, args.rank, args.outdir, debug)

if __name__ == "__main__":
    main()
