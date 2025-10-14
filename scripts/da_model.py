#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified DA pipeline (unadjusted, adjusted, paired, and 'simple' OLS) for genus/species.

Inputs
------
--pct-all         : taxa×samples % table (from CORE; concatenated groups)
--meta            : pooled metadata (needs at least Sample_ID, Site(Oral/Fecal), disease(0/1))
--pairs           : optional matched Oral/Fecal IDs for Crohn (auto-detect colnames)
--rank            : 'genus' or 'species'
--outdir          : base output dir (writes to <outdir>/<rank>/)
--heat-topk       : top-K labels per side on volcano plots
--volcano-metric  : "q" or "p" (controls both significance thresholding and y-label)
--alpha           : threshold applied to the chosen metric
--deltas          : optional deltas CSV (pair_id×taxa, FC_minus_OC for Crohn); if provided,
                    paired Wilcoxon + volcano will be computed from this table (safer sign)

Outputs (in <outdir>/<rank>/)
-----------------------------
# Unadjusted MWU (CLR) by site, CH:
  da_oral_unadj_{rank}.csv
  da_fecal_unadj_{rank}.csv
  volcano_oral_unadj_{rank}.png
  volcano_fecal_unadj_{rank}.png

# Adjusted OLS (per-taxon), ALL + per-site:
  da_all_adj_{rank}.csv
  da_oral_adj_{rank}.csv
  da_fecal_adj_{rank}.csv
  volcano_all_adj_{rank}.png          (term startswith 'disease')
  volcano_oral_adj_{rank}.png         (term startswith 'disease')
  volcano_fecal_adj_{rank}.png        (term startswith 'disease')

# Paired OC↔FC (Crohn) Wilcoxon:
  da_paired_{rank}.csv                 (computed from --deltas if present; else falls back to CLR)
  volcano_paired_{rank}.png            (x = median(FC−OC), y = -log10(metric))

# 'Simple' OLS (per-feature within each site; label ~ feature):
  da_simple_oral_{rank}.csv
  da_simple_fecal_{rank}.csv
"""

from __future__ import annotations

import os, argparse, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from scipy.stats import mannwhitneyu, wilcoxon
import statsmodels.formula.api as smf
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests

# ---------------------- utilities ----------------------

def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def bh_qvalues(pvals):
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
    gm = logX.mean(axis=0)                   # per-sample mean of logs
    return logX.sub(gm, axis=1)              # taxa × samples (CLR)

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

def save_placeholder_png(out_png: str, title: str):
    ensure_dir_for(out_png)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.42, "No data or no testable features.", ha="center", va="center", fontsize=10)
    fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig)

def volcano_plot(df, x, sig_col, name_col, out_png, title, alpha=0.05, k_onplot_each=10, metric_label="q"):
    """Generic volcano: x is effect (delta or beta), sig_col is 'p' or 'q'."""
    X = df[x].astype(float).values
    S = df[sig_col].astype(float).clip(lower=1e-300).values
    Y = -np.log10(S)
    names = df[name_col].astype(str).values if name_col in df.columns else df.index.astype(str).values
    is_sig = S <= alpha
    score = np.abs(X) * (Y + 1)              # ranking heuristic for labels

    up_idx = np.where(is_sig & (X > 0))[0]
    dn_idx = np.where(is_sig & (X < 0))[0]
    up_sel = up_idx[np.argsort(-score[up_idx])[:k_onplot_each]]
    dn_sel = dn_idx[np.argsort(-score[dn_idx])[:k_onplot_each]]
    label_idxs = np.concatenate([up_sel, dn_sel])

    fig = plt.figure(figsize=(11, 6.5)); ax = fig.add_subplot(111)
    ax.scatter(X[~is_sig], Y[~is_sig], s=22, alpha=0.35, color="#999999", zorder=1)
    ax.scatter(X[is_sig & (X > 0)], Y[is_sig & (X > 0)], s=24, alpha=0.9, color="#1B4F72", zorder=2)
    ax.scatter(X[is_sig & (X < 0)], Y[is_sig & (X < 0)], s=24, alpha=0.9, color="#7FB3D5", zorder=2)
    ax.axhline(-np.log10(alpha), ls="--", lw=1, color="gray", alpha=0.8)
    ax.axvline(0,              ls="--", lw=1, color="gray", alpha=0.8)
    ax.grid(True, ls=":", lw=0.6, alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel(f"-log10({metric_label})")

    # basic label repulsion
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
            if not bumped: break
        for (x1,y1,txt) in zip(xy[:,0], xy[:,1], texts):
            ax.text(x1, y1, txt, fontsize=fs, ha="left", va="bottom",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=3)
        for (x0,y0),(x1,y1) in zip(np.vstack([xs, ys]).T, xy):
            ax.annotate("", xy=(x0,y0), xytext=(x1,y1),
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.25", alpha=0.7), zorder=3)

    if len(label_idxs):
        _repel(ax, X[label_idxs], Y[label_idxs], [names[i] for i in label_idxs])

    plt.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight"); plt.close(fig)

# ---------------------- meta helpers ----------------------

def _norm_cols_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Best-effort, case-insensitive selection of meta columns."""
    m = meta.copy()
    m.columns = [c.strip() for c in m.columns]
    lower_map = {c.lower(): c for c in m.columns}
    aliases = {
        "sample_id": ["sample_id","id","sid","sampleid","sample"],
        "site":      ["site","body_site","location"],
        "disease":   ["disease","status","group"],
        "age": ["age"], "sex": ["sex","gender"], "bmi": ["bmi"],
        "ppi_use": ["ppi_use","ppi","ppi3m","ppi_current"],
        "antibiotics_3m": ["antibiotics_3m","abx_3m","antibiotics_last3m","abx3m"],
        "smoking": ["smoking","smoker"]
    }
    out = {}
    for canon, keys in aliases.items():
        found = next((lower_map[k] for k in keys if k in lower_map), None)
        if found is not None: out[canon] = m[found]
    cols = [k for k in ["sample_id","site","disease","age","sex","bmi","ppi_use","antibiotics_3m","smoking"] if k in out]
    return pd.DataFrame({k: out[k] for k in cols})

def _to_int01(s):
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
    out = df.copy()
    if "disease" in out: out["disease"] = out["disease"].map(_to_int01)
    if "ppi_use" in out: out["ppi_use"] = out["ppi_use"].map(_to_int01)
    if "antibiotics_3m" in out: out["antibiotics_3m"] = out["antibiotics_3m"].map(_to_int01)
    if "smoking" in out: out["smoking"] = out["smoking"].map(_to_int01)
    if "site" in out:
        out["site"] = out["site"].astype(str).str.strip().str.capitalize().replace({"Faecal":"Fecal"})
        out.loc[~out["site"].isin(["Oral","Fecal"]), "site"] = np.nan
    for c in ["age","bmi"]:
        if c in out: out[c] = pd.to_numeric(out[c], errors="coerce")
    if "sex" in out:
        out["sex"] = out["sex"].astype(str).str.strip().str.capitalize()
    return out

# ---------------------- tests (unadjusted/paired) ----------------------

def mwu_unadjusted(clr: pd.DataFrame, meta_df: pd.DataFrame, site: str, rank: str) -> pd.DataFrame:
    """MWU CLR by site: Crohn(1) vs Healthy(0)."""
    df = clr.T                                                  # samples × taxa
    df.index.name = "Sample_ID"
    m = meta_df.dropna(subset=["site","disease"]).copy()
    m = m[m["site"] == site]

    inter = df.index.intersection(m["sample_id"].astype(str))
    if len(inter) == 0:
        return pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","delta_clr","n_1","n_0"])
    df = df.loc[inter]
    m  = m.set_index("sample_id").loc[inter]

    rows, pvals = [], []
    for tax in clr.index:
        x = df[tax].values
        lab = m["disease"].values
        a = x[lab == 1]; b = x[lab == 0]
        if (a.size == 0) or (b.size == 0):
            p, U, d = 1.0, np.nan, np.nan
        else:
            try:
                U, p = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                U, p = np.nan, 1.0
            d = float(np.nanmean(a) - np.nanmean(b))
        rows.append({"taxon":tax, "pretty_taxon":prettify_taxon(tax,rank),
                     "test":f"MWU_{site}", "stat":float(U) if np.isfinite(U) else np.nan,
                     "p":float(p), "delta_clr":d,
                     "n_1":int((lab==1).sum()), "n_0":int((lab==0).sum())})
        pvals.append(p)
    out = pd.DataFrame(rows)
    if len(out): out["q"] = bh_qvalues(out["p"].values)
    return out[["taxon","pretty_taxon","test","stat","p","q","delta_clr","n_1","n_0"]]

def paired_from_deltas(deltas: pd.DataFrame, rank: str) -> pd.DataFrame:
    """
    Paired Wilcoxon using a deltas wide table (rows=pair_id, cols=taxa) where each value is FC-OC.
    Positive delta => higher in Fecal than Oral by construction.
    """
    rows, pvals = [], []
    for tax in deltas.columns:
        dif = deltas[tax].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if dif.size < 5 or np.allclose(dif, 0.0):
            W, p = np.nan, 1.0
            dm = np.nan
        else:
            try:
                # Standard two-sided Wilcoxon on paired deltas vs 0
                W, p = wilcoxon(dif, zero_method="wilcox", alternative="two-sided", correction=False)
            except ValueError:
                W, p = np.nan, 1.0
            dm = float(np.nanmedian(dif))
        rows.append({"taxon":tax, "pretty_taxon":prettify_taxon(tax, rank),
                     "test":"Wilcoxon_OC_vs_FC_Crohn", "stat":float(W) if np.isfinite(W) else np.nan,
                     "p":float(p), "delta_median":dm, "n_pairs":int(len(dif))})
        pvals.append(p)
    out = pd.DataFrame(rows)
    if len(out): out["q"] = bh_qvalues(out["p"].values)
    return out[["taxon","pretty_taxon","test","stat","p","q","delta_median","n_pairs"]]

# ---------------------- adjusted models ----------------------

def fit_ols_per_taxon(long_df: pd.DataFrame, formula: str, rank: str, label: str) -> pd.DataFrame:
    """OLS per taxon; returns one row per term (excluding Intercept)."""
    results = []
    for tax, d in long_df.groupby("taxon"):
        d = d.dropna(subset=["CLR","disease"])
        if d.shape[0] < 8:
            continue
        try:
            model = smf.ols(formula=formula, data=d).fit()
            coefs, pvals = model.params, model.pvalues
            for term in coefs.index:
                if term == "Intercept": continue
                results.append({
                    "taxon": tax,
                    "pretty_taxon": prettify_taxon(tax, rank),
                    "model": label,
                    "term": term,
                    "beta": float(coefs[term]),
                    "p": float(pvals.get(term, np.nan)),
                    "n": int(d.shape[0]),
                })
        except Exception:
            results.append({
                "taxon": tax, "pretty_taxon": prettify_taxon(tax, rank),
                "model": label, "term": "__fit_failed__", "beta": np.nan, "p": np.nan, "n": int(d.shape[0])
            })
    out = pd.DataFrame(results)
    if not out.empty:
        out["q"] = out.groupby("model")["p"].transform(lambda s: bh_qvalues(s.values))
    return out

# ---------------------- 'simple' OLS (per site) ----------------------

def simple_ols_by_site(clr: pd.DataFrame, meta: pd.DataFrame, site: str, rank: str) -> pd.DataFrame:
    """
    Re-implementation of da_simple: within one site, for each feature do OLS(label ~ feature).
    Returns: feature, beta, pval, qval, n, site
    """
    df = clr.T.reset_index().rename(columns={"index":"Sample_ID"})
    m  = meta.loc[meta["site"] == site, ["sample_id","disease"]].rename(columns={"sample_id":"Sample_ID","disease":"label"})
    mm = df.merge(m, on="Sample_ID", how="inner")
    feats = [c for c in mm.columns if c not in {"Sample_ID","label"}]
    rows = []; Y = mm["label"].astype(float).values
    for f in feats:
        x = mm[f].astype(float).values
        try:
            mdl = OLS(Y, add_constant(x), hasconst=True).fit()
            rows.append({"feature":f, "beta":float(mdl.params[1]), "pval":float(mdl.pvalues[1]), "n":len(mm)})
        except Exception:
            rows.append({"feature":f, "beta":np.nan, "pval":1.0, "n":len(mm)})
    out = pd.DataFrame(rows)
    if len(out):
        ok = np.isfinite(out["pval"].values)
        q = np.full(len(out), np.nan)
        if ok.sum() > 0:
            q[ok] = multipletests(out.loc[ok,"pval"].values, method="fdr_bh")[1]
        out["qval"] = q
    out["site"] = site
    return out[["feature","beta","pval","qval","n","site"]]

# ---------------------- main ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Unified DA models (adjusted/unadjusted/paired/simple) with volcano plots")
    ap.add_argument("--pct-all", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--deltas", default=None, help="Optional deltas CSV (pair_id×taxa, FC_minus_OC)")
    ap.add_argument("--rank", required=True, choices=["genus","species"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--heat-topk", type=int, default=10)
    ap.add_argument("--volcano-metric", choices=["q","p"], default="q")
    ap.add_argument("--alpha", type=float, default=0.05)
    return ap.parse_args()

def main():
    args = parse_args()
    metric = args.volcano_metric   # "q" or "p"
    alpha  = args.alpha

    outdir_rank = os.path.join(args.outdir, args.rank)
    os.makedirs(outdir_rank, exist_ok=True)

    # figure paths (created up-front to guarantee existence for workflows)
    volc_paths = {
        "oral_unadj"  : os.path.join(outdir_rank, f"volcano_oral_unadj_{args.rank}.png"),
        "fecal_unadj" : os.path.join(outdir_rank, f"volcano_fecal_unadj_{args.rank}.png"),
        "all_adj"     : os.path.join(outdir_rank, f"volcano_all_adj_{args.rank}.png"),
        "oral_adj"    : os.path.join(outdir_rank, f"volcano_oral_adj_{args.rank}.png"),
        "fecal_adj"   : os.path.join(outdir_rank, f"volcano_fecal_adj_{args.rank}.png"),
        "paired"      : os.path.join(outdir_rank, f"volcano_paired_{args.rank}.png"),
    }
    save_placeholder_png(volc_paths["oral_unadj"],  f"Volcano (Oral CH, unadjusted) — {args.rank}")
    save_placeholder_png(volc_paths["fecal_unadj"], f"Volcano (Fecal CH, unadjusted) — {args.rank}")
    save_placeholder_png(volc_paths["all_adj"],     f"Volcano (Adjusted ALL; term=disease) — {args.rank}")
    save_placeholder_png(volc_paths["oral_adj"],    f"Volcano (Adjusted Oral; term=disease) — {args.rank}")
    save_placeholder_png(volc_paths["fecal_adj"],   f"Volcano (Adjusted Fecal; term=disease) — {args.rank}")
    save_placeholder_png(volc_paths["paired"],      f"Volcano (Paired OC↔FC; x = median FC−OC) — {args.rank}")

    # -------- load inputs & transform --------
    pct_all = pd.read_csv(args.pct_all, index_col=0)
    pct_all.columns = pct_all.columns.astype(str)
    clr = clr_transform(pct_all, pseudocount=1e-6)

    raw_meta = pd.read_csv(args.meta)
    meta0 = _norm_cols_meta(raw_meta)
    if not {"sample_id","disease","site"}.issubset(meta0.columns):
        raise ValueError("[meta] need Sample_ID, Site, disease (0/1).")
    meta = _coerce_covariates(meta0)
    meta["sample_id"] = meta["sample_id"].astype(str)

    keep = meta["sample_id"].isin(clr.columns.astype(str))
    meta = meta.loc[keep].copy()
    if meta.empty:
        raise ValueError("[meta] No overlapping Sample_ID with pct_all columns.")

    # ========== 1) Unadjusted MWU by site ==========
    da_oral_unadj  = mwu_unadjusted(clr, meta, site="Oral",  rank=args.rank)
    da_fecal_unadj = mwu_unadjusted(clr, meta, site="Fecal", rank=args.rank)

    da_oral_unadj.to_csv(os.path.join(outdir_rank, f"da_oral_unadj_{args.rank}.csv"), index=False)
    da_fecal_unadj.to_csv(os.path.join(outdir_rank, f"da_fecal_unadj_{args.rank}.csv"), index=False)

    if not da_oral_unadj.empty:
        volcano_plot(da_oral_unadj, x="delta_clr", sig_col=metric, name_col="pretty_taxon",
                     out_png=volc_paths["oral_unadj"],
                     title=f"Volcano (Oral Crohn−Healthy, unadjusted) — {args.rank}",
                     alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric)
    if not da_fecal_unadj.empty:
        volcano_plot(da_fecal_unadj, x="delta_clr", sig_col=metric, name_col="pretty_taxon",
                     out_png=volc_paths["fecal_unadj"],
                     title=f"Volcano (Fecal Crohn−Healthy, unadjusted) — {args.rank}",
                     alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric)

    # ========== 2) Adjusted OLS (ALL + site) ==========
    clr_T = clr.T; clr_T.index.name = "Sample_ID"
    long = clr_T.stack().reset_index()
    long.columns = ["Sample_ID","taxon","CLR"]
    long["Sample_ID"] = long["Sample_ID"].astype(str)

    mm = meta.rename(columns={"sample_id":"Sample_ID"})
    df_long = long.merge(mm, on="Sample_ID", how="inner")

    base_terms = []
    if "disease" in df_long.columns: base_terms.append("disease")
    if "site" in df_long.columns:    base_terms.append("site")
    for c in ["ppi_use","age","sex","bmi","antibiotics_3m","smoking"]:
        if c in df_long.columns: base_terms.append(c)
    formula_all = "CLR ~ " + " + ".join(base_terms) if base_terms else "CLR ~ 1"

    da_all_adj = fit_ols_per_taxon(df_long, formula=formula_all, rank=args.rank, label="ALL")
    da_all_adj.to_csv(os.path.join(outdir_rank, f"da_all_adj_{args.rank}.csv"), index=False)

    ds = da_all_adj[da_all_adj["term"].str.startswith("disease", na=False)].copy()
    if not ds.empty:
        if "q" not in ds and "p" in ds: ds["q"] = bh_qvalues(ds["p"])
        volcano_plot(ds, x="beta", sig_col=metric, name_col="pretty_taxon",
                     out_png=volc_paths["all_adj"],
                     title=f"Volcano (Adjusted ALL; term=disease) — {args.rank}",
                     alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric)

    for site in ["Oral","Fecal"]:
        sub = df_long[df_long["site"] == site].copy()
        out_csv = os.path.join(outdir_rank, f"da_{site.lower()}_adj_{args.rank}.csv")
        if sub.empty or sub["disease"].dropna().nunique() < 2:
            pd.DataFrame(columns=["taxon","pretty_taxon","model","term","beta","p","q","n"]).to_csv(out_csv, index=False)
        else:
            terms = [t for t in base_terms if t != "site"]
            formula_site = "CLR ~ " + " + ".join(terms) if terms else "CLR ~ 1"
            da_site = fit_ols_per_taxon(sub, formula=formula_site, rank=args.rank, label=site)
            da_site.to_csv(out_csv, index=False)
            ds = da_site[da_site["term"].str.startswith("disease", na=False)].copy()
            if not ds.empty:
                if "q" not in ds and "p" in ds: ds["q"] = bh_qvalues(ds["p"])
                out_png = volc_paths["oral_adj"] if site=="Oral" else volc_paths["fecal_adj"]
                volcano_plot(ds, x="beta", sig_col=metric, name_col="pretty_taxon",
                             out_png=out_png,
                             title=f"Volcano (Adjusted {site}; term=disease) — {args.rank}",
                             alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric)

    # ========== 3) Paired OC↔FC (Crohn) from deltas if available ==========
    paired_csv = os.path.join(outdir_rank, f"da_paired_{args.rank}.csv")
    paired_df = None
    if args.deltas and os.path.exists(args.deltas):
        # Expect wide table: first col is pair_id, others taxa; values are FC-OC
        deltas = pd.read_csv(args.deltas)
        # try to locate the pair id col
        if "pair_id" not in deltas.columns:
            # assume first column is pair_id
            deltas = deltas.rename(columns={deltas.columns[0]:"pair_id"})
        deltas = deltas.set_index("pair_id")
        paired_df = paired_from_deltas(deltas, rank=args.rank)
    else:
        # Fallback: compute paired from CLR (only Crohn pairs present in meta+pairs)
        # This block mirrors your old wilcoxon code, but we skip here to keep FC-OC sign consistent.
        pass

    if paired_df is None or paired_df.empty:
        # keep a valid CSV even when we cannot compute
        pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","delta_median","n_pairs"]).to_csv(paired_csv, index=False)
    else:
        paired_df.to_csv(paired_csv, index=False)
        sigcol = metric
        if sigcol not in paired_df.columns and "p" in paired_df.columns:
            paired_df[sigcol] = bh_qvalues(paired_df["p"]) if metric=="q" else paired_df["p"]
        volcano_plot(paired_df.rename(columns={"delta_median":"x_delta"}),
                     x="x_delta", sig_col=sigcol, name_col="pretty_taxon",
                     out_png=volc_paths["paired"],
                     title=f"Volcano (Paired OC↔FC; x = median FC−OC) — {args.rank}",
                     alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric)

    # ========== 4) 'Simple' OLS inside each site ==========
    simp_oral  = simple_ols_by_site(clr, meta, site="Oral",  rank=args.rank)
    simp_fecal = simple_ols_by_site(clr, meta, site="Fecal", rank=args.rank)
    simp_oral.to_csv(os.path.join(outdir_rank, f"da_simple_oral_{args.rank}.csv"),  index=False)
    simp_fecal.to_csv(os.path.join(outdir_rank, f"da_simple_fecal_{args.rank}.csv"), index=False)
    print(f"[volcano] metric={args.volcano_metric}, alpha={args.alpha}, ythr={-np.log10(args.alpha):.3f}")

    print("[INFO] DA (unified) completed →", outdir_rank)

if __name__ == "__main__":
    main()
