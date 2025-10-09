#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PPI & interactions per taxon (CLR OLS)
--------------------------------------
This script fits per-taxon OLS models on CLR-transformed abundances to estimate:
- Main PPI effect (ALL, Oral-only, Fecal-only)
- disease:ppi_use interaction (ALL)
- disease:site interaction (ALL; useful QC/biology)
It writes tidy coefficient tables and volcano plots. If a table is empty,
placeholder PNGs are created so Snakemake won't fail on missing files.
"""

from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import statsmodels.formula.api as smf

# ------------- small utils -------------

def ensure_dir(path: str) -> None:
    """Create directory (parents) if not existing."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def bh_qvalues(pvals) -> np.ndarray:
    """Benjamini–Hochberg FDR for a 1-D iterable of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    # iterate from largest p to smallest
    for i, idx in enumerate(order[::-1], start=1):
        rank = n - i + 1
        val = min(prev, p[idx] * n / rank)
        q[idx] = val
        prev = val
    return q

def clr_transform(pct_taxa_by_samples: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """Return taxa×samples CLR from percentage table (0..100)."""
    X = (pct_taxa_by_samples.astype(float) / 100.0) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)  # per-sample mean log
    return logX.sub(gm, axis=1)

def prettify_taxon(label: str, rank: str) -> str:
    """Nice display names for MetaPhlAn-like labels."""
    if label is None:
        return None
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

def save_placeholder_png(out_png: str, title: str) -> None:
    """Write a neutral placeholder PNG."""
    ensure_dir(os.path.dirname(out_png))
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.42, "No data or no testable features.", ha="center", va="center", fontsize=10)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

def volcano(df, x, sig_col, name_col, out_png, title, alpha=0.05, k_onplot_each=10, metric_label="q"):
    ensure_dir(os.path.dirname(out_png))
    if (df is None) or df.empty or (x not in df.columns) or (sig_col not in df.columns):
        save_placeholder_png(out_png, title); return
    X = pd.to_numeric(df[x], errors="coerce").values
    S = pd.to_numeric(df[sig_col], errors="coerce").clip(lower=1e-300).values
    Y = -np.log10(S)
    names = df[name_col].astype(str).values if name_col in df.columns else df.index.astype(str).values
    is_sig = S <= alpha
    score = np.abs(X) * (Y + 1)
    up_idx = np.where(is_sig & (X > 0))[0]; dn_idx = np.where(is_sig & (X < 0))[0]
    up_sel = up_idx[np.argsort(-score[up_idx])[:k_onplot_each]]
    dn_sel = dn_idx[np.argsort(-score[dn_idx])[:k_onplot_each]]
    label_idxs = np.concatenate([up_sel, dn_sel])

    fig = plt.figure(figsize=(10.5, 6.2)); ax = fig.add_subplot(111)
    ax.scatter(X[~is_sig], Y[~is_sig], s=22, alpha=0.35, color="#999999", zorder=1)
    ax.scatter(X[is_sig & (X > 0)], Y[is_sig & (X > 0)], s=24, alpha=0.9, color="#1B4F72", zorder=2)
    ax.scatter(X[is_sig & (X < 0)], Y[is_sig & (X < 0)], s=24, alpha=0.9, color="#7FB3D5", zorder=2)
    ax.axhline(-np.log10(alpha), ls="--", lw=1, color="gray", alpha=0.8)
    ax.axvline(0, ls="--", lw=1, color="gray", alpha=0.8)
    ax.grid(True, ls=":", lw=0.6, alpha=0.4)
    ax.set_title(title); ax.set_ylabel(f"-log10({metric_label})")
    for i in label_idxs:
        ax.text(X[i], Y[i], names[i], fontsize=9, ha="left", va="bottom",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=3)
    plt.tight_layout(); fig.savefig(out_png, dpi=320, bbox_inches="tight"); plt.close(fig)

# ------------- meta helpers -------------

def norm_cols_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names; map common synonyms to canonical keys."""
    m = meta.copy()
    m.columns = [c.strip() for c in m.columns]
    lower_map = {c.lower(): c for c in m.columns}
    aliases = {
        "sample_id": ["sample_id", "id", "sid", "sampleid", "sample"],
        "site": ["site","body_site","location"],
        "disease": ["disease","status","group"],
        "age": ["age"],
        "sex": ["sex","gender"],
        "bmi": ["bmi"],
        "ppi_use": ["ppi_use","ppi","ppi3m","ppi_current"],
        "antibiotics_3m": ["antibiotics_3m","abx_3m","antibiotics_last3m","abx3m"],
        "smoking": ["smoking","smoker"]
    }
    out = {}
    for canon, keys in aliases.items():
        found = next((lower_map[k] for k in keys if k in lower_map), None)
        if found is not None:
            out[canon] = m[found]
    cols = [k for k in ["sample_id","site","disease","age","sex","bmi","ppi_use","antibiotics_3m","smoking"] if k in out]
    return pd.DataFrame({k: out[k] for k in cols})

def to_int01(x):
    """Map value to 0/1 if possible (used for binary covariates)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"1","true","yes","y","crohn","cd","case","ibd"}:
        return 1
    if s in {"0","false","no","n","healthy","control","hc","non-ibd","nonibd"}:
        return 0
    try:
        v = int(float(s))
        if v in (0,1):
            return v
    except Exception:
        pass
    return np.nan

def coerce_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric/binary; standardize 'site' levels."""
    out = df.copy()
    if "disease" in out: out["disease"] = out["disease"].map(to_int01)
    if "ppi_use" in out: out["ppi_use"] = out["ppi_use"].map(to_int01)
    if "antibiotics_3m" in out: out["antibiotics_3m"] = out["antibiotics_3m"].map(to_int01)
    if "smoking" in out: out["smoking"] = out["smoking"].map(to_int01)
    if "site" in out:
        out["site"] = out["site"].astype(str).str.strip().str.capitalize().replace({"Faecal":"Fecal"})
        out.loc[~out["site"].isin(["Oral","Fecal"]), "site"] = np.nan
    for c in ["age","bmi"]:
        if c in out: out[c] = pd.to_numeric(out[c], errors="coerce")
    if "sex" in out: out["sex"] = out["sex"].astype(str).str.strip().str.capitalize()
    return out

# ------------- modeling core -------------

def fit_ols_per_taxon(long_df: pd.DataFrame, formula: str, rank: str, model_label: str,
                      min_n: int = 8) -> pd.DataFrame:
    """
    Fit OLS per taxon with given formula (patsy). Returns tidy table:
    [taxon, pretty_taxon, model, term, beta, p, q, n]
    q-values are BH-corrected within model_label (over terms pooled).
    """
    rows = []
    for tax, d in long_df.groupby("taxon", sort=False):
        dd = d.dropna(subset=["CLR"])
        # need disease at least whenever it's present in formula
        must_have = [c for c in ["disease","ppi_use","site"] if c in dd.columns and (c in formula)]
        dd = dd.dropna(subset=must_have) if must_have else dd
        if dd.shape[0] < min_n:
            continue
        try:
            m = smf.ols(formula=formula, data=dd).fit()
            coefs = m.params
            pvals = m.pvalues
            for term in coefs.index:
                if term == "Intercept":
                    continue
                rows.append({
                    "taxon": tax,
                    "pretty_taxon": prettify_taxon(tax, rank),
                    "model": model_label,
                    "term": term,
                    "beta": float(coefs[term]),
                    "p": float(pvals.get(term, np.nan)),
                    "n": int(dd.shape[0]),
                })
        except Exception:
            # silent skip or verbose row; here we add a row for diagnostics
            rows.append({
                "taxon": tax,
                "pretty_taxon": prettify_taxon(tax, rank),
                "model": model_label,
                "term": "__fit_failed__",
                "beta": np.nan,
                "p": np.nan,
                "n": int(dd.shape[0]),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = out.groupby("model")["p"].transform(lambda s: bh_qvalues(s.values))
    return out

# ------------- CLI & main -------------

def parse_args():
    ap = argparse.ArgumentParser(description="Per-taxon PPI & interaction analysis (CLR OLS).")
    ap.add_argument("--pct-all", required=True, help="taxa×samples percentage table from CORE (pct_all.csv)")
    ap.add_argument("--meta", required=True, help="pooled metadata CSV with Sample_ID, Site, disease, PPI")
    ap.add_argument("--rank", required=True, choices=["genus","species"], help="label prettification and paths")
    ap.add_argument("--outdir", required=True, help="output root directory (we write into <outdir>/<rank>/)")
    ap.add_argument("--heat-topk", type=int, default=12, help="labels per side on volcano plots")
    ap.add_argument("--volcano-metric", choices=["q","p"], default="q",
                    help="Metric used on volcano y-axis (-log10 of this). Default: q")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance threshold for the volcano guide line. Default: 0.05")

    return ap.parse_args()

def main():
    args = parse_args()
    out_rank = os.path.join(args.outdir, args.rank)
    ensure_dir(out_rank)

    # ---- 1) Load tables & CLR
    pct = pd.read_csv(args.pct_all, index_col=0)
    pct.columns = pct.columns.astype(str)  # make sure sample IDs are strings
    clr = clr_transform(pct, pseudocount=1e-6)  # taxa × samples
    clr_T = clr.T
    clr_T.index.name = "Sample_ID"

    # ---- 2) Metadata normalization & coercion
    meta_raw = pd.read_csv(args.meta)
    meta0 = norm_cols_meta(meta_raw)
    req = {"sample_id","site","disease"}
    missing = req - set(meta0.columns)
    if missing:
        raise ValueError(f"[meta] Missing required columns: {sorted(list(missing))}")
    meta = coerce_covariates(meta0)
    meta["sample_id"] = meta["sample_id"].astype(str)

    # Keep only overlapping samples
    keep = meta["sample_id"].isin(clr.columns.astype(str))
    meta = meta.loc[keep].copy()
    if meta.empty:
        raise ValueError("[meta] No overlapping Sample_ID with pct_all columns.")

    # ---- 3) Long format join
    long = clr_T.stack().reset_index()
    long.columns = ["Sample_ID","taxon","CLR"]
    long["Sample_ID"] = long["Sample_ID"].astype(str)
    mm = meta.rename(columns={"sample_id":"Sample_ID"})
    long = long.merge(mm, on="Sample_ID", how="inner")
    if long.empty:
        raise ValueError("No overlap between abundance and meta after join.")

    # ---- 4) Write design counts (audit)
    design_rows = []
    for label, sub in [("ALL", long), ("Oral", long[long["site"]=="Oral"]), ("Fecal", long[long["site"]=="Fecal"])]:
        m = sub.dropna(subset=["disease","ppi_use"])
        design_rows.append({
            "scope": label,
            "n_rows": len(sub),
            "n_with_disease_ppi": len(m),
            "n_unique_samples": sub["Sample_ID"].nunique(),
            "n_crohn": int((sub["disease"]==1).sum()),
            "n_healthy": int((sub["disease"]==0).sum()),
            "n_ppi_yes": int((sub["ppi_use"]==1).sum()) if "ppi_use" in sub.columns else 0,
            "n_ppi_no": int((sub["ppi_use"]==0).sum()) if "ppi_use" in sub.columns else 0,
        })
    pd.DataFrame(design_rows).to_csv(os.path.join(out_rank, "ppi_design_counts.csv"), index=False)

    # ---- 5) ALL model (with interactions)
    rhs_parts = []
    if "disease" in long.columns: rhs_parts.append("disease")
    if "site" in long.columns:    rhs_parts.append("site")
    if "ppi_use" in long.columns: rhs_parts.append("ppi_use")
    # interactions of interest
    if set(["disease","site"]).issubset(long.columns):    rhs_parts.append("disease:site")
    if set(["disease","ppi_use"]).issubset(long.columns): rhs_parts.append("disease:ppi_use")
    # optional covariates
    for c in ["age","sex","bmi","antibiotics_3m","smoking"]:
        if c in long.columns:
            rhs_parts.append(c)
    formula_all = "CLR ~ " + (" + ".join(rhs_parts) if rhs_parts else "1")

    eff_all = fit_ols_per_taxon(long, formula=formula_all, rank=args.rank, model_label="ALL", min_n=12)
    eff_all.to_csv(os.path.join(out_rank, "ppi_effects_all.csv"), index=False)

    # ---- Volcanoes (ALL)
    volc_specs = [
        ("ppi_use",            "volcano_ppi_all.png",            "Volcano (ALL) — PPI effect"),
        ("disease:ppi_use",    "volcano_int_disease_ppi.png",    "Volcano (ALL) — disease:PPI interaction"),
        ("disease:site",       "volcano_int_disease_site.png",   "Volcano (ALL) — disease:site interaction"),
    ]
    metric = args.volcano_metric  # "q" or "p"
    alpha  = args.alpha
    for term_prefix, png, title in [
        ("ppi_use",         "volcano_ppi_all.png",          "Volcano (ALL) — PPI effect"),
        ("disease:ppi_use", "volcano_int_disease_ppi.png",  "Volcano (ALL) — disease:PPI interaction"),
        ("disease:site",    "volcano_int_disease_site.png", "Volcano (ALL) — disease:site interaction"),
    ]:
        sub = eff_all[eff_all["term"].astype(str).str.startswith(term_prefix, na=False)].copy()
        if sub.empty:
            save_placeholder_png(os.path.join(out_rank, png), title)
        else:
            sub["q"] = bh_qvalues(sub["p"].values)
            volcano(sub, x="beta", sig_col=metric, name_col="pretty_taxon",
                    out_png=os.path.join(out_rank, png), title=title,
                    alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric)

    # ---- 6) Site-specific models (Oral / Fecal) to see PPI within site
    for site in ["Oral","Fecal"]:
        sub = long[long["site"] == site].copy()
        out_csv = os.path.join(out_rank, f"ppi_effects_{site.lower()}.csv")
        out_png = os.path.join(out_rank, f"volcano_ppi_{site.lower()}.png")
        if sub.empty or sub["disease"].dropna().nunique() < 2:
            # write empty CSV and placeholder plot
            pd.DataFrame(columns=["taxon","pretty_taxon","model","term","beta","p","q","n"]).to_csv(out_csv, index=False)
            save_placeholder_png(out_png, f"Volcano ({site}) — PPI effect")
            continue

        terms = [t for t in ["disease","ppi_use","age","sex","bmi","antibiotics_3m","smoking"] if t in sub.columns]
        formula_site = "CLR ~ " + (" + ".join(terms) if terms else "1")

        eff_site = fit_ols_per_taxon(sub, formula=formula_site, rank=args.rank, model_label=site, min_n=10)
        eff_site.to_csv(out_csv, index=False)

        sub_v = eff_site[eff_site["term"].astype(str).str.startswith("ppi_use", na=False)].copy()
        if sub_v.empty:
            save_placeholder_png(out_png, f"Volcano ({site}) — PPI effect")
        else:
            sub_v["q"] = bh_qvalues(sub_v["p"].values)
            volcano(sub_v, x="beta", sig_col=metric, name_col="pretty_taxon",
                    out_png=out_png, title=f"Volcano ({site}) — PPI effect",
                    alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric)

    print("[INFO] PPI/interaction analysis done →", out_rank)


if __name__ == "__main__":
    main()
