#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DA with/without covariates (per-rank) using CLR from pct_all.csv
-----------------------------------------------------------------
Inputs:
  --pct-all: taxa × samples table of percentages from CORE (concatenated across groups)
  --meta: pooled metadata with at least [Sample_ID, Site (Oral/Fecal), disease (0/1)]
          Optional covariates: [Age, Sex, BMI, PPI_use, Antibiotics_3m, Smoking]
  --pairs: optional CSV of matched Crohn IDs for paired OC↔FC Wilcoxon.
           Supported column name pairs (case-insensitive):
             (Oral_ID, Fecal_ID), (Oral, Fecal), (OC, FC), (oral_id, fecal_id)
  --rank: 'genus' or 'species' (for filenames/labels)
  --outdir: output directory (we will write into <outdir>/<rank>/)
  --heat-topk: top-K labels on volcano plots

Outputs (in <outdir>/<rank>/):
  da_oral_unadj_{rank}.csv
  da_fecal_unadj_{rank}.csv
  da_paired_{rank}.csv          (only if --pairs given and usable)
  da_all_adj_{rank}.csv
  da_oral_adj_{rank}.csv
  da_fecal_adj_{rank}.csv
  volcano_oral_unadj_{rank}.png
  volcano_fecal_unadj_{rank}.png
  volcano_all_adj_{rank}.png
  volcano_oral_adj_{rank}.png
  volcano_fecal_adj_{rank}.png
"""

from __future__ import annotations

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import mannwhitneyu, wilcoxon

# Statsmodels for adjusted models
import statsmodels.formula.api as smf

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _save_placeholder_png(out_png: str, title: str):
    _ensure_dir_for(out_png)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.42, "No data or no testable features.", ha="center", va="center", fontsize=10)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

def _volcano_or_placeholder(df, x_col: str, q_col: str, name_col: str, out_png: str, title: str, k_onplot_each: int):
    try:
        import pandas as pd
        if (df is None) or (isinstance(df, pd.DataFrame) and df.empty) or (x_col not in df.columns) or (q_col not in df.columns):
            _save_placeholder_png(out_png, title)
            return
        volcano_plot(
            df.copy(),
            x=x_col,
            q=q_col,
            name_col=name_col,
            out_png=out_png,
            title=title,
            k_onplot_each=k_onplot_each
        )
    except Exception:
        _save_placeholder_png(out_png, title)


# ---------------- Utilities ----------------

def bh_qvalues(pvals):
    """Benjamini–Hochberg FDR."""
    p = np.asarray(pvals, dtype=float); n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float); prev = 1.0
    for i, idx in enumerate(order[::-1], start=1):
        rank = n - i + 1
        val = min(prev, p[idx] * n / rank)
        q[idx] = val; prev = val
    return q

def clr_transform(pct_taxa_by_samples: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """CLR per sample; input are percentages (0..100) taxa × samples."""
    X = (pct_taxa_by_samples.astype(float) / 100.0) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)  # per-sample mean of logs
    return logX.sub(gm, axis=1)

def prettify_taxon(label: str, rank: str) -> str:
    if label is None: return None
    r = (rank or "").lower()
    if r == "genus" and label.startswith("g__"):
        name = label.replace("g__", "").replace("_", " ")
        return " ".join(w.capitalize() for w in name.split())
    if r == "species" and label.startswith("s__"):
        name = label.replace("s__", "").replace("_", " ")
        toks = name.split()
        if toks:
            toks[0] = toks[0].capitalize()
            toks[1:] = [t.lower() for t in toks[1:]]
            return " ".join(toks)
    return label

def volcano_plot(df, x, sig_col, name_col, out_png, title, alpha=0.05, k_onplot_each=10, metric_label="q"):
    X = df[x].astype(float).values
    S = df[sig_col].astype(float).clip(lower=1e-300).values
    Y = -np.log10(S)
    names = df[name_col].astype(str).values if name_col in df.columns else df.index.astype(str).values
    is_sig = S <= alpha
    score = np.abs(X) * (Y + 1)
    up_idx = np.where(is_sig & (X > 0))[0]; dn_idx = np.where(is_sig & (X < 0))[0]
    up_sel = up_idx[np.argsort(-score[up_idx])[:k_onplot_each]]
    dn_sel = dn_idx[np.argsort(-score[dn_idx])[:k_onplot_each]]
    label_idxs = np.concatenate([up_sel, dn_sel])

    fig = plt.figure(figsize=(11, 6.5)); ax = fig.add_subplot(111)
    ax.scatter(X[~is_sig], Y[~is_sig], s=22, alpha=0.35, color="#999999", zorder=1)
    ax.scatter(X[is_sig & (X > 0)], Y[is_sig & (X > 0)], s=24, alpha=0.9, color="#1B4F72", zorder=2)
    ax.scatter(X[is_sig & (X < 0)], Y[is_sig & (X < 0)], s=24, alpha=0.9, color="#7FB3D5", zorder=2)
    ax.axhline(-np.log10(alpha), ls="--", lw=1, color="gray", alpha=0.8)
    ax.axvline(0, ls="--", lw=1, color="gray", alpha=0.8)
    ax.grid(True, ls=":", lw=0.6, alpha=0.4)
    ax.set_title(title); ax.set_ylabel(f"-log10({metric_label})")

    # label repulsion
    def _repel(ax, xs, ys, texts, fs=9, pad=0.015, iters=200):
        xy = np.vstack([xs, ys]).T.astype(float)
        for _ in range(iters):
            bumped = False
            for i in range(len(texts)):
                for j in range(i+1, len(texts)):
                    dx = xy[i,0]-xy[j,0]; dy = xy[i,1]-xy[j,1]
                    if abs(dx) < pad and abs(dy) < pad:
                        sgn = 1 if (dy == 0 and (i%2)==0) else (np.sign(dy) or 1)
                        xy[i,1] += sgn*pad; xy[j,1] -= sgn*pad
                        bumped = True
            if not bumped:
                break
        for (x1,y1,txt) in zip(xy[:,0], xy[:,1], texts):
            ax.text(x1, y1, txt, fontsize=fs, ha="left", va="bottom",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=3)
        for (x0,y0),(x1,y1) in zip(np.vstack([xs, ys]).T, xy):
            ax.annotate("", xy=(x0,y0), xytext=(x1,y1),
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.25", alpha=0.7), zorder=3)

    if len(label_idxs):
        _repel(ax, X[label_idxs], Y[label_idxs], [names[i] for i in label_idxs])

    plt.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

# ---------------- Meta helpers ----------------

def _norm_cols_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns; map common synonyms to canonical names."""
    m = meta.copy()
    m.columns = [c.strip() for c in m.columns]
    lower_map = {c.lower(): c for c in m.columns}
    # Canonical keys we care about (case-insensitive)
    aliases = {
        "sample_id": ["sample_id", "id", "sid", "sampleid", "sample"],
        "site": ["site", "body_site", "location"],
        "disease": ["disease", "status", "group"],
        "age": ["age"],
        "sex": ["sex", "gender"],
        "bmi": ["bmi"],
        "ppi_use": ["ppi_use", "ppi", "ppi3m", "ppi_current"],
        "antibiotics_3m": ["antibiotics_3m", "abx_3m", "antibiotics_last3m", "abx3m"],
        "smoking": ["smoking", "smoker"]
    }
    out = {}
    for canon, keys in aliases.items():
        found = next((lower_map[k] for k in keys if k in lower_map), None)
        if found is not None:
            out[canon] = m[found]
    # Build final dataframe with whatever we have
    cols = []
    for k in ["sample_id","site","disease","age","sex","bmi","ppi_use","antibiotics_3m","smoking"]:
        if k in out: cols.append(k)
    return pd.DataFrame({k: out[k] for k in cols})

def _to_int01(s):
    """Map possibly messy disease/PPI/etc. to 0/1 ints if possible."""
    if s is None: return np.nan
    x = str(s).strip().lower()
    if x in {"1","true","yes","y","crohn","cd","case","ibd"}: return 1
    if x in {"0","false","no","n","healthy","control","ctl","ctr","hc","non-ibd","nonibd"}: return 0
    try:
        v = int(float(x)); 
        if v in (0,1): return v
    except Exception:
        pass
    return np.nan

def _coerce_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce disease/PPI/... to numeric, standardize Site values."""
    out = df.copy()
    if "disease" in out.columns:
        out["disease"] = out["disease"].map(_to_int01)
    if "ppi_use" in out.columns:
        out["ppi_use"] = out["ppi_use"].map(_to_int01)
    if "antibiotics_3m" in out.columns:
        out["antibiotics_3m"] = out["antibiotics_3m"].map(_to_int01)
    if "smoking" in out.columns:
        out["smoking"] = out["smoking"].map(_to_int01)
    if "site" in out.columns:
        out["site"] = out["site"].astype(str).str.strip().str.capitalize().replace({"Faecal":"Fecal"})
        out.loc[~out["site"].isin(["Oral","Fecal"]), "site"] = np.nan
    for c in ["age","bmi"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "sex" in out.columns:
        out["sex"] = out["sex"].astype(str).str.strip().str.capitalize()
        # Keep categories as-is; statsmodels will dummy-code strings.
    return out

# ---------------- DA (unadjusted) ----------------

def mwu_unadjusted(clr: pd.DataFrame, meta_df: pd.DataFrame, site: str, rank: str) -> pd.DataFrame:
    """For a given site ('Oral' or 'Fecal'), MWU CLR of Crohn (1) vs Healthy (0)."""
    df = clr.T  # rows = samples
    df.index.name = "Sample_ID"
    m = meta_df.dropna(subset=["site","disease"]).copy()
    m = m[m["site"] == site]
    # keep only overlapping samples
    inter = df.index.intersection(m["sample_id"].astype(str))
    if len(inter) == 0:
        return pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","delta_clr","n_1","n_0"])
    df = df.loc[inter]
    m = m.set_index("sample_id").loc[inter]

    rows = []
    pvals = []
    for tax in clr.index:
        x = df[tax].values
        lab = m["disease"].values
        a = x[lab == 1]  # Crohn
        b = x[lab == 0]  # Healthy
        if (a.size == 0) or (b.size == 0):
            p = 1.0; U = np.nan; d = np.nan
        else:
            try:
                U, p = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                U, p = np.nan, 1.0
            d = float(np.nanmean(a) - np.nanmean(b))
        rows.append({
            "taxon": tax,
            "pretty_taxon": prettify_taxon(tax, rank),
            "test": f"MWU_{site}",
            "stat": float(U) if np.isfinite(U) else np.nan,
            "p": float(p),
            "delta_clr": d,
            "n_1": int((lab == 1).sum()),
            "n_0": int((lab == 0).sum())
        })
        pvals.append(p)
    out = pd.DataFrame(rows)
    if len(out):
        out["q"] = bh_qvalues(out["p"].values)
    cols = ["taxon","pretty_taxon","test","stat","p","q","delta_clr","n_1","n_0"]
    return out.reindex(columns=cols)

def wilcoxon_paired_oc_fc(clr: pd.DataFrame, meta_df: pd.DataFrame, pairs_df: pd.DataFrame, rank: str) -> pd.DataFrame:
    """
    Paired Wilcoxon OC↔FC in Crohn only. pairs_df provides IDs for Oral/Fecal columns.
    """
    # Identify column names
    colnames = [c.lower() for c in pairs_df.columns]
    def _pick(*cands):
        return next((pairs_df.columns[colnames.index(c.lower())] for c in cands if c.lower() in colnames), None)
    oc_col = _pick("Oral_ID","Oral","OC","oral_id","oral")
    fc_col = _pick("Fecal_ID","Fecal","FC","fecal_id","fecal")
    if oc_col is None or fc_col is None:
        warnings.warn("[pairs] Could not find Oral/Fecal columns; skipping paired DA.")
        return pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","delta_median","n_pairs"])

    # Restrict to Crohn
    crohn_ids = set(meta_df.loc[meta_df["disease"] == 1, "sample_id"].astype(str))
    pairs = pairs_df[[oc_col, fc_col]].dropna().astype(str).values.tolist()
    pairs = [(o,f) for (o,f) in pairs if o.strip() and f.strip()]
    pairs = [(o,f) for (o,f) in pairs if (o in crohn_ids and f in crohn_ids)]
    if len(pairs) < 5:
        warnings.warn(f"[pairs] Not enough Crohn pairs (n={len(pairs)}); skipping paired DA.")
        return pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","delta_median","n_pairs"])

    # Check presence in CLR columns
    samples = list(clr.columns.astype(str))
    present = [(o,f) for (o,f) in pairs if (o in samples and f in samples)]
    if len(present) < 5:
        warnings.warn(f"[pairs] Not enough matched columns (n={len(present)}); skipping paired DA.")
        return pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","delta_median","n_pairs"])

    oc_ids = [o for (o,_) in present]
    fc_ids = [f for (_,f) in present]

    rows = []
    pvals = []
    for tax in clr.index:
        x = clr.loc[tax, oc_ids].astype(float).values
        y = clr.loc[tax, fc_ids].astype(float).values
        dif = y - x
        dif = dif[~np.isnan(dif)]
        if dif.size == 0 or np.allclose(dif, 0.0):
            W, p = np.nan, 1.0
            dm = np.nan
        else:
            try:
                W, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                W, p = np.nan, 1.0
            dm = float(np.nanmedian(y) - np.nanmedian(x))
        rows.append({
            "taxon": tax,
            "pretty_taxon": prettify_taxon(tax, rank),
            "test": "Wilcoxon_OC_vs_FC_Crohn",
            "stat": float(W) if np.isfinite(W) else np.nan,
            "p": float(p),
            "delta_median": dm,
            "n_pairs": int(len(present))
        })
        pvals.append(p)
    out = pd.DataFrame(rows)
    if len(out):
        out["q"] = bh_qvalues(out["p"].values)
    cols = ["taxon","pretty_taxon","test","stat","p","q","delta_median","n_pairs"]
    return out.reindex(columns=cols)

# ---------------- DA (adjusted: OLS per taxon) ----------------

def _fit_ols_per_taxon(long_df: pd.DataFrame, formula: str, rank: str, label: str) -> pd.DataFrame:
    """
    long_df has columns: ['Sample_ID','taxon','CLR','disease','site', ... covariates]
    formula examples:
      CLR ~ disease + site + ppi_use + age + sex + bmi + antibiotics_3m + smoking
    Returns per-taxon rows with beta/p/q for main effects (esp. disease).
    """
    results = []
    taxa = long_df["taxon"].unique().tolist()
    for tax in taxa:
        d = long_df[long_df["taxon"] == tax].dropna(subset=["CLR","disease"])
        if d.shape[0] < 8:
            continue
        try:
            model = smf.ols(formula=formula, data=d).fit()
            # Extract params and p-values for key terms
            coefs = model.params
            pvals = model.pvalues

            # We will report disease coefficient if present; otherwise the first available of interest
            keys_of_interest = [k for k in coefs.index if k.startswith("disease")]
            if not keys_of_interest and "disease" in coefs.index:
                keys_of_interest = ["disease"]

            # If no disease term (e.g., site-specific model where disease present?), we still dump all betas
            for term in coefs.index:
                if term == "Intercept": 
                    continue
                results.append({
                    "taxon": tax,
                    "pretty_taxon": prettify_taxon(tax, rank),
                    "model": label,
                    "term": term,
                    "beta": float(coefs[term]),
                    "p": float(pvals.get(term, np.nan)),
                    "n": int(d.shape[0])
                })
        except Exception as e:
            # keep going, but note the failure
            results.append({
                "taxon": tax,
                "pretty_taxon": prettify_taxon(tax, rank),
                "model": label,
                "term": "__fit_failed__",
                "beta": np.nan,
                "p": np.nan,
                "n": int(d.shape[0])
            })
    out = pd.DataFrame(results)
    if not out.empty:
        # BH per model label × term? Commonly we want q for 'disease' terms; but compute global BH per label.
        out["q"] = out.groupby("model")["p"].transform(lambda s: bh_qvalues(s.values))
    return out

# ---------------- Main ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Differential Abundance with/without covariates (CLR-based)")
    ap.add_argument("--pct-all", required=True, help="pct_all.csv (taxa × samples, percentages) from CORE")
    ap.add_argument("--meta", required=True, help="pooled meta CSV")
    ap.add_argument("--pairs", default=None, help="optional pairs CSV for Crohn OC↔FC paired test")
    ap.add_argument("--rank", required=True, choices=["genus","species"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--heat-topk", type=int, default=10, help="top-K labels per side on volcano plots")
    ap.add_argument("--volcano-metric", choices=["q","p"], default="q")
    ap.add_argument("--alpha", type=float, default=0.05)

    return ap.parse_args()

def main():
    args = parse_args()
    metric = args.volcano_metric  # "q" یا "p"
    alpha  = args.alpha

    # -------- prepare outdirs --------
    outdir_rank = os.path.join(args.outdir, args.rank)
    os.makedirs(outdir_rank, exist_ok=True)

    # -------- define all volcano output paths (used both for placeholders and final plots) --------
    fn_volc_oral_unadj  = os.path.join(outdir_rank, f"volcano_oral_unadj_{args.rank}.png")
    fn_volc_fecal_unadj = os.path.join(outdir_rank, f"volcano_fecal_unadj_{args.rank}.png")
    fn_volc_all_adj     = os.path.join(outdir_rank, f"volcano_all_adj_{args.rank}.png")
    fn_volc_oral_adj    = os.path.join(outdir_rank, f"volcano_oral_adj_{args.rank}.png")
    fn_volc_fecal_adj   = os.path.join(outdir_rank, f"volcano_fecal_adj_{args.rank}.png")

    # --- PRE-CREATE ALL VOLCANO PLACEHOLDERS (guarantee files exist for Snakemake) ---
    _save_placeholder_png(fn_volc_oral_unadj,  f"Volcano (Oral Crohn−Healthy, unadjusted) — {args.rank}")
    _save_placeholder_png(fn_volc_fecal_unadj, f"Volcano (Fecal Crohn−Healthy, unadjusted) — {args.rank}")
    _save_placeholder_png(fn_volc_all_adj,     f"Volcano (Adjusted ALL; term=disease) — {args.rank}")
    _save_placeholder_png(fn_volc_oral_adj,    f"Volcano (Adjusted Oral; term=disease) — {args.rank}")
    _save_placeholder_png(fn_volc_fecal_adj,   f"Volcano (Adjusted Fecal; term=disease) — {args.rank}")

    # -------- load inputs --------
    pct_all = pd.read_csv(args.pct_all, index_col=0)
    pct_all.columns = pct_all.columns.astype(str)

    # CLR transform (taxa × samples)
    clr = clr_transform(pct_all, pseudocount=1e-6)

    # load & normalize meta
    raw_meta = pd.read_csv(args.meta)
    meta0 = _norm_cols_meta(raw_meta)
    if not {"sample_id","disease","site"}.issubset(meta0.columns):
        raise ValueError("[meta] meta must include at least Sample_ID, Site, disease (0/1).")
    meta = _coerce_covariates(meta0)
    meta["sample_id"] = meta["sample_id"].astype(str)

    # keep only samples present in clr
    keep = meta["sample_id"].isin(clr.columns.astype(str))
    meta = meta.loc[keep].copy()
    if meta.empty:
        raise ValueError("[meta] No overlapping Sample_ID with pct_all columns.")

    # ---------- Unadjusted (MWU per site) ----------
    da_oral_unadj  = mwu_unadjusted(clr, meta, site="Oral",  rank=args.rank)
    da_fecal_unadj = mwu_unadjusted(clr, meta, site="Fecal", rank=args.rank)

    # BH اگر نبود
    if not da_oral_unadj.empty and "q" not in da_oral_unadj.columns and "p" in da_oral_unadj.columns:
        da_oral_unadj["q"] = bh_qvalues(da_oral_unadj["p"])
    if not da_fecal_unadj.empty and "q" not in da_fecal_unadj.columns and "p" in da_fecal_unadj.columns:
        da_fecal_unadj["q"] = bh_qvalues(da_fecal_unadj["p"])

    da_oral_unadj.to_csv(os.path.join(outdir_rank, f"da_oral_unadj_{args.rank}.csv"), index=False)
    da_fecal_unadj.to_csv(os.path.join(outdir_rank, f"da_fecal_unadj_{args.rank}.csv"), index=False)

    # volcano (unadjusted) با متریک انتخابی
    if not da_oral_unadj.empty:
        volcano_plot(da_oral_unadj, x="delta_clr", sig_col=metric, name_col="pretty_taxon",
                     out_png=fn_volc_oral_unadj,
                     title=f"Volcano (Oral Crohn−Healthy, unadjusted) — {args.rank}",
                     alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric)
    if not da_fecal_unadj.empty:
        volcano_plot(da_fecal_unadj, x="delta_clr", sig_col=metric, name_col="pretty_taxon",
                     out_png=fn_volc_fecal_unadj,
                     title=f"Volcano (Fecal Crohn−Healthy, unadjusted) — {args.rank}",
                     alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric)

    # ---------- Paired (optional) ----------
    if args.pairs and os.path.exists(args.pairs):
        pairs_df = pd.read_csv(args.pairs)
        da_pair = wilcoxon_paired_oc_fc(clr, meta, pairs_df, rank=args.rank)
        da_pair.to_csv(os.path.join(outdir_rank, f"da_paired_{args.rank}.csv"), index=False)
    else:
        pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","delta_median","n_pairs"])\
          .to_csv(os.path.join(outdir_rank, f"da_paired_{args.rank}.csv"), index=False)

    # ---------- Adjusted models (OLS per taxon) ----------
    # long table for modeling
    clr_T = clr.T
    clr_T.index.name = "Sample_ID"
    long = clr_T.stack().reset_index()
    long.columns = ["Sample_ID", "taxon", "CLR"]
    long["Sample_ID"] = long["Sample_ID"].astype(str)

    # join meta
    mm = meta.rename(columns={"sample_id":"Sample_ID"})
    df_long = long.merge(mm, on="Sample_ID", how="inner")

    # choose terms
    base_terms = []
    if "disease" in df_long.columns: base_terms.append("disease")
    if "site" in df_long.columns:    base_terms.append("site")
    for c in ["ppi_use","age","sex","bmi","antibiotics_3m","smoking"]:
        if c in df_long.columns: base_terms.append(c)
    formula_all = "CLR ~ " + " + ".join(base_terms) if base_terms else "CLR ~ 1"

    da_all_adj = _fit_ols_per_taxon(df_long, formula=formula_all, rank=args.rank, label="ALL")
    # تضمین q
    if not da_all_adj.empty and "q" not in da_all_adj.columns and "p" in da_all_adj.columns:
        da_all_adj["q"] = bh_qvalues(da_all_adj["p"])
    da_all_adj.to_csv(os.path.join(outdir_rank, f"da_all_adj_{args.rank}.csv"), index=False)

    # volcano (ALL adjusted؛ فقط ترم‌های disease)
    if not da_all_adj.empty and "term" in da_all_adj.columns:
        da_all_disease = da_all_adj[da_all_adj["term"].str.startswith("disease", na=False)].copy()
        if not da_all_disease.empty:
            if "q" not in da_all_disease.columns and "p" in da_all_disease.columns:
                da_all_disease["q"] = bh_qvalues(da_all_disease["p"])
            volcano_plot(da_all_disease, x="beta", sig_col=metric, name_col="pretty_taxon",
                         out_png=fn_volc_all_adj,
                         title=f"Volcano (Adjusted ALL; term=disease) — {args.rank}",
                         alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric)

    # site-specific adjusted models
    for site in ["Oral","Fecal"]:
        sub = df_long[df_long["site"] == site].copy()
        out_csv = os.path.join(outdir_rank, f"da_{site.lower()}_adj_{args.rank}.csv")
        if sub.empty or sub["disease"].dropna().nunique() < 2:
            pd.DataFrame(columns=["taxon","pretty_taxon","model","term","beta","p","q","n"]).to_csv(out_csv, index=False)
            continue
        terms = [t for t in base_terms if t != "site"]
        formula_site = "CLR ~ " + " + ".join(terms) if terms else "CLR ~ 1"
        da_site = _fit_ols_per_taxon(sub, formula=formula_site, rank=args.rank, label=f"{site}")
        if not da_site.empty and "q" not in da_site.columns and "p" in da_site.columns:
            da_site["q"] = bh_qvalues(da_site["p"])
        da_site.to_csv(out_csv, index=False)

        ds = da_site[da_site["term"].str.startswith("disease", na=False)].copy()
        if not ds.empty:
            if "q" not in ds.columns and "p" in ds.columns:
                ds["q"] = bh_qvalues(ds["p"])
            out_png = fn_volc_oral_adj if site == "Oral" else fn_volc_fecal_adj
            volcano_plot(ds, x="beta", sig_col=metric, name_col="pretty_taxon",
                         out_png=out_png,
                         title=f"Volcano (Adjusted {site}; term=disease) — {args.rank}",
                         alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric)

    print("[INFO] DA completed →", outdir_rank)

if __name__ == "__main__":
    main()
